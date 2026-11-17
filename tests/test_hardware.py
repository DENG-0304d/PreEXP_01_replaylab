import inspect
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Barrier, Thread
import unittest

from replaylab.artifacts import read_artifact
from replaylab.contracts import ArtifactKind, FormatError
from test_preflight import HARDWARE, ROOT
from test_contracts import put_json


TEMPLATES = ROOT / "deploy" / "kubernetes"


class HardwareLifecycleTests(unittest.TestCase):
    def setUp(self):
        from replaylab.hardware import BenchControl, MemoryLeaseStore, RunnerIdentity, SyntheticInjector

        self.store = MemoryLeaseStore()
        self.store.initialize()
        self.injector = SyntheticInjector(max_runtime_ns=20)
        self.bench = BenchControl(self.store, self.injector, recovery_subject="operator")
        self.first = RunnerIdentity("runner-a", "pod-a")
        self.second = RunnerIdentity("runner-b", "pod-b")

    def test_only_one_concurrent_owner_and_redeployment_preserves_lease(self):
        barrier = Barrier(2)
        results = []

        def compete(owner):
            barrier.wait()
            results.append(self.bench.acquire(owner))

        threads = [Thread(target=compete, args=(owner,)) for owner in (self.first, self.second)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(sorted(result.status for result in results), ["blocked", "pass"])
        winner, = [result for result in results if result.status == "pass"]
        self.assertEqual(self.store.initialize(), winner.snapshot)
        self.assertEqual(self.store.read().holder, winner.snapshot.holder)
        self.assertEqual(self.bench.acquire(winner.snapshot.holder).reason, "lease_held")

    def test_attempt_isolation_and_idempotence_before_and_after_stop(self):
        from replaylab.hardware import AttemptIdentity, CleanupState

        with TemporaryDirectory() as directory:
            root = Path(directory)
            attempt = AttemptIdentity(self.first, "run-a", "try-a", root / "a")
            other = AttemptIdentity(self.second, "run-b", "try-b", root / "b")
            self.assertEqual(self.bench.register_attempt(attempt).status, "pass")
            self.assertEqual(self.bench.register_attempt(other).status, "pass")
            self.assertTrue((root / "a").is_dir() and (root / "b").is_dir())
            collision = AttemptIdentity(self.second, "new", "other", root / "a")
            self.assertEqual(self.bench.register_attempt(collision).status, "blocked")
            token = self.bench.acquire(self.first).token
            self.assertEqual(self.bench.reset(token, other).status, "blocked")
            self.assertEqual(self.bench.reset(token, attempt).status, "pass")
            self.assertEqual(self.bench.start_injector(token, attempt, deadline_ns=21).status, "blocked")
            self.assertEqual(self.bench.start_injector(token, attempt, deadline_ns=10).reason, "injector_started")
            self.assertEqual(self.bench.start_injector(token, attempt, deadline_ns=15).reason, "already_started")
            self.assertEqual(self.injector.record(attempt).deadline_ns, 10)
            failed = self.bench.cleanup(token, attempt, state=CleanupState(False, True, True))
            self.assertEqual(self.bench.release(token, cleanup=failed).status, "blocked")
            self.assertEqual(self.bench.acquire(self.second).status, "blocked")
            self.assertEqual(self.bench.start_injector(token, attempt, deadline_ns=10).reason, "already_stopped")
            clean = self.bench.cleanup(token, attempt, state=CleanupState(True, True, True))
            self.assertEqual(self.bench.release(token, cleanup=clean).status, "pass")
            self.assertIsNone(self.store.read().holder)
            self.assertEqual(self.bench.acquire(self.second).status, "pass")
            self.assertEqual(self.bench.reset(token, attempt).status, "blocked")

    def test_hard_stop_outlives_runner_and_recovery_requires_authorized_evidence(self):
        from replaylab.hardware import AttemptIdentity, CleanupState, RecoveryEvidence
        from replaylab.preflight import EvidenceRef

        with TemporaryDirectory() as directory:
            attempt = AttemptIdentity(self.first, "run-a", "try-a", Path(directory) / "a")
            self.bench.register_attempt(attempt)
            held = self.bench.acquire(self.first)
            self.bench.reset(held.token, attempt)
            self.bench.start_injector(held.token, attempt, deadline_ns=10)
            refs = (EvidenceRef("mock-recovery.json", "0" * 64, "cleanup"),)
            state = CleanupState(True, True, True)
            proof = RecoveryEvidence("operator", self.first, True, True, state, refs)
            self.injector.advance(9)
            self.assertIsNone(self.injector.record(attempt).stopped_at_ns)
            self.assertEqual(self.bench.recover(held.snapshot, evidence=proof).reason, "injector_still_active")
            self.injector.advance(15)
            record = self.injector.record(attempt)
            self.assertEqual((record.stopped_at_ns, record.stop_reason), (10, "hard_deadline"))
            self.assertEqual(self.bench.acquire(self.second).reason, "lease_held")
            self.assertEqual(self.bench.recover(held.snapshot, evidence=RecoveryEvidence(
                "runner-a", self.first, True, True, state, refs)).reason, "recovery_subject_not_authorized")
            self.assertEqual(self.bench.recover(held.snapshot, evidence=RecoveryEvidence(
                "operator", self.first, False, True, state, refs)).reason, "recovery_evidence_incomplete")
            self.assertEqual(self.bench.recover(held.snapshot, evidence=RecoveryEvidence(
                "operator", self.second, True, True, state, refs)).reason, "recovery_evidence_incomplete")
            self.assertEqual(self.bench.recover(held.snapshot, evidence=proof).status, "pass")
            self.assertEqual(self.bench.acquire(self.first).reason, "runner_attempt_reused")
            self.assertEqual(self.bench.acquire(self.second).status, "pass")
            self.assertEqual(self.bench.recover(held.snapshot, evidence=proof).reason, "lease_holder_changed")
            self.assertEqual(self.store.read().holder, self.second)

    def test_conflict_rechecks_current_holder_for_acquire_release_and_recovery(self):
        from unittest.mock import patch
        from replaylab.hardware import AttemptIdentity, CleanupState, RecoveryEvidence, RunnerIdentity
        from replaylab.preflight import EvidenceRef

        original_update = self.store.compare_and_set

        def conflict_with(owner):
            def update(expected, new_owner):
                original_update(expected, owner)
                return None
            return update

        with patch.object(self.store, "compare_and_set", side_effect=conflict_with(self.second)):
            self.assertEqual(self.bench.acquire(self.first).reason, "lease_held")
        self.assertEqual(self.store.read().holder, self.second)

        with TemporaryDirectory() as directory:
            original_update(self.store.read(), None)
            attempt = AttemptIdentity(self.first, "run-a", "try-a", Path(directory) / "a")
            self.bench.register_attempt(attempt)
            lease = self.bench.acquire(self.first)
            self.bench.reset(lease.token, attempt)
            state = CleanupState(True, True, True)
            report = self.bench.cleanup(lease.token, attempt, state=state)

            def conflict_then_update(expected, owner):
                if self.store.read() == lease.snapshot:
                    original_update(expected, expected.holder)
                    return None
                return original_update(expected, owner)

            with patch.object(self.store, "compare_and_set", side_effect=conflict_then_update):
                self.assertEqual(self.bench.release(lease.token, cleanup=report).status, "pass")

        next_runner = RunnerIdentity("runner-next", "pod-next")
        lease = self.bench.acquire(next_runner)
        proof = RecoveryEvidence("operator", next_runner, True, True, state, (EvidenceRef("demo", "0" * 64, "stop"),))
        with patch.object(self.store, "compare_and_set", side_effect=conflict_with(self.second)):
            self.assertEqual(self.bench.recover(lease.snapshot, evidence=proof).reason, "lease_holder_changed")
        self.assertEqual(self.store.read().holder, self.second)


class DeploymentTests(unittest.TestCase):
    def declared_study(self) -> dict:
        data = json.loads((HARDWARE / "study.json").read_bytes())
        kubernetes = data["environment"]["kubernetes"]
        image = "registry.example/app@sha256:" + "1" * 64
        kubernetes.update(
            namespace="lab", node_name="node-resource", node_hostname_label="node-label",
            bench_lease_name="bench", gateway_image_digest=image, runner_image_digest=image, backend_image_digest=image,
            gateway_deployment_name="gateway", gateway_service_name="gateway", job_name="replay-001",
            backend_args=["--model", "/model"], backend_port=8080, backend_startup_probe_path="/health",
            backend_readiness_probe_path="/ready", gateway_port=8081, gateway_startup_probe_path="/healthz",
            gateway_readiness_probe_path="/readyz", output_mount_path="/output", output_pvc_name="captures",
            output_storage_class="storage", output_storage_size="5Gi",
            recovery_subject="operator@example.org",
        )
        for replica in "AB":
            kubernetes["replicas"][replica].update(
                deployment_name=f"backend-{replica.lower()}", service_name=f"vllm-{replica.lower()}",
                container_name="vllm",
            )
        return data

    def test_render_uses_scheduler_job_suspension_and_separate_lease_bootstrap(self):
        from replaylab.hardware import render_kubernetes

        with TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "study.json"
            put_json(config, self.declared_study())
            paths = render_kubernetes(read_artifact(config, ArtifactKind.STUDY), template_dir=TEMPLATES,
                                      out_dir=root / "rendered")
            rendered = {path.name: json.loads(path.read_bytes()) for path in paths}
            self.assertEqual(rendered["lease-bootstrap.json"]["kind"], "Lease")
            objects = rendered["bench.json"]["items"]
            index = {(row["kind"], row["metadata"]["name"]): row for row in objects}
            self.assertFalse(any(row["kind"] == "Lease" for row in objects))
            a = index["Deployment", "backend-a"]
            b = index["Deployment", "backend-b"]
            gateway = index["Deployment", "gateway"]
            job = index["Job", "replay-001"]
            for resource in (a, b, gateway, job):
                spec = resource["spec"]["template"]["spec"]
                self.assertEqual(spec["nodeSelector"], {"kubernetes.io/hostname": "node-label"})
                self.assertNotIn("nodeName", spec)
            for backend, name in ((a, "vllm-a"), (b, "vllm-b")):
                service = index["Service", name]
                self.assertEqual(backend["spec"]["replicas"], 1)
                spec = backend["spec"]["template"]
                self.assertEqual(service["spec"]["selector"], spec["metadata"]["labels"])
                self.assertEqual(spec["spec"]["containers"][0]["resources"]["limits"]["nvidia.com/gpu"], 1)
            self.assertNotEqual(index["Service", "vllm-a"]["spec"]["selector"],
                                index["Service", "vllm-b"]["spec"]["selector"])
            self.assertEqual({name: job["spec"][name] for name in ("suspend", "parallelism", "completions", "backoffLimit")},
                             {"suspend": True, "parallelism": 1, "completions": 1, "backoffLimit": 0})
            for resource in (gateway, job):
                pod = resource["spec"]["template"]["spec"]
                self.assertNotIn("resources", pod["containers"][0])
                self.assertEqual(pod["volumes"][0]["persistentVolumeClaim"]["claimName"], "captures")
            self.assertEqual(job["spec"]["template"]["spec"]["restartPolicy"], "Never")
            self.assertIn(("PersistentVolumeClaim", "captures"), index)
            lease_permissions = [rule for row in objects if row["kind"] == "Role" for rule in row["rules"]
                                 if "leases" in rule["resources"]]
            self.assertEqual({(tuple(rule["verbs"]), tuple(rule["resourceNames"])) for rule in lease_permissions},
                             {(("get", "update"), ("bench",)), (("get",), ("bench",))})

    def test_null_template_blocks_render_and_malformed_value_locates_field(self):
        from replaylab.hardware import DeploymentBlocked, render_kubernetes

        with TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root / "rendered"
            with self.assertRaises(DeploymentBlocked) as blocked:
                render_kubernetes(read_artifact(HARDWARE / "study.json", ArtifactKind.STUDY),
                                  template_dir=TEMPLATES, out_dir=destination)
            self.assertIn("environment.kubernetes.node_hostname_label", blocked.exception.missing_fields)
            self.assertFalse(destination.exists())
            data = self.declared_study()
            data["environment"]["kubernetes"]["backend_port"] = True
            config = root / "bad.json"
            put_json(config, data)
            with self.assertRaises(FormatError) as invalid:
                render_kubernetes(read_artifact(config, ArtifactKind.STUDY), template_dir=TEMPLATES,
                                  out_dir=destination)
            self.assertEqual((invalid.exception.file, invalid.exception.field),
                             (config, "environment.kubernetes.backend_port"))
            self.assertGreater(invalid.exception.line, 1)
            self.assertFalse(destination.exists())

    def test_real_execution_contract_has_frozen_branch_and_live_is_only_an_interface(self):
        from replaylab.hardware import BlockedHardware, FrozenActionBranch, HardwareBlocked
        from replaylab.live import LiveReplay

        self.assertEqual(tuple(inspect.signature(BlockedHardware.run).parameters), ("self", "attempt", "branch"))
        self.assertTrue({"condition_id", "role", "delay_ns", "input_trace_path", "study_path", "plan_path", "plan_sha256"}
                        <= FrozenActionBranch.__dataclass_fields__.keys())
        self.assertIn("独立", inspect.getdoc(LiveReplay.run))
        self.assertTrue(LiveReplay._is_protocol)
        backend = BlockedHardware()
        for action, args in ((backend.run, (None, None)), (backend.sample, (None,))):
            with self.assertRaisesRegex(HardwareBlocked, "BLOCKED_UNTIL_HARDWARE"):
                action(*args)
