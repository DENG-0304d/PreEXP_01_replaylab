import json
from pathlib import Path
import tempfile
import unittest

from replaylab.artifacts import load_manifest_bundle, read_artifact
from replaylab.contracts import ArtifactKind
from test_contracts import put_json


FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "synthetic"


def fixture_study(path, **mock_options):
    study = json.loads((FIXTURE / "study.json").read_bytes())
    for name in ("input_trace_path", "warmup_trace_path", "prehistory_trace_path", "inputs_path"):
        study["replay"][name] = str(FIXTURE / study["replay"][name])
    study["observation"]["health_profile_path"] = str(FIXTURE / study["observation"]["health_profile_path"])
    scenario = json.loads((FIXTURE / "inputs" / "mock_scenario.json").read_bytes())
    scenario.update(mock_options)
    scenario_path = path.with_name("scenario.json")
    put_json(scenario_path, scenario)
    study["replay"]["mock_scenario_path"] = str(scenario_path)
    put_json(path, study)
    return path


class RunnerTests(unittest.TestCase):
    def execute(self, root, **options):
        from replaylab.runner import run_study

        config = fixture_study(root / "study.json", **options)
        plan = FIXTURE / "plan.json"
        out = root / "runs"
        out.mkdir()
        result = run_study(read_artifact(config, ArtifactKind.STUDY),
                           read_artifact(plan, ArtifactKind.PLAN), out_dir=out)
        return result, config, out

    def failures(self, clean):
        from replaylab.runner import frozen_branch_order

        plan = json.loads((FIXTURE / "plan.json").read_bytes())
        study = json.loads((FIXTURE / "study.json").read_bytes())
        block = plan["blocks"][0]
        role = frozen_branch_order(block, study["replay"]["rotation_seed"])[0]["role"]
        return [{"pair_id": block["pair_id"], "role": role, "at_ns": 105_000_000,
                 "cleanup_confirmed": clean, "diagnostic_retry": clean}]

    def test_failed_slot_with_successful_cleanup_and_retry_stays_failed(self):
        with tempfile.TemporaryDirectory() as directory:
            result, config, out = self.execute(Path(directory), failures=self.failures(True))
            self.assertFalse(result.cleanup_blocked)
            manifest = json.loads((out / "manifest.json").read_bytes())
            first = manifest["runs"][0]
            self.assertEqual(first["run_status"], "failed")
            self.assertIn("simulated_runner_abort", first["reason"])
            self.assertTrue(all(row["run_status"] == "complete" for row in manifest["runs"][1:]))
            attempts = json.loads((out / "attempts.json").read_bytes())["diagnostic_retries"]
            [retry] = attempts
            self.assertEqual(retry["retry_of_run_id"], first["run_id"])
            self.assertNotEqual(retry["independent_start"]["attempt_id"], first["independent_start"]["attempt_id"])
            self.assertEqual(retry["run_status"], "complete")
            from hashlib import sha256
            path = out / manifest["attempts_path"]
            self.assertEqual(sha256(path.read_bytes()).hexdigest(), manifest["attempts_sha256"])
            for ref in retry["files"].values():
                self.assertEqual(sha256((out / ref["path"]).read_bytes()).hexdigest(), ref["sha256"])
            bundle = load_manifest_bundle(out / "manifest.json", config)
            self.assertNotIn(retry["run_id"], {item.record["run_id"] for item in bundle.runs})
            self.assertEqual(bundle.excluded_runs[0]["run_id"], first["run_id"])

    def test_unconfirmed_cleanup_keeps_all_later_slots_blocked(self):
        with tempfile.TemporaryDirectory() as directory:
            result, _, out = self.execute(Path(directory), failures=self.failures(False))
            self.assertTrue(result.cleanup_blocked)
            manifest = json.loads((out / "manifest.json").read_bytes())
            self.assertEqual(manifest["runs"][0]["run_status"], "failed")
            self.assertEqual({(row["run_status"], row["reason"]) for row in manifest["runs"][1:]},
                             {("blocked", "previous_cleanup_unconfirmed")})

    def test_timeout_cleans_only_attempt_owned_resources(self):
        with tempfile.TemporaryDirectory() as directory:
            [failure] = self.failures(True)
            failure.update(reason="simulated_runner_timeout", diagnostic_retry=False)
            result, _, out = self.execute(Path(directory), failures=[failure])
            first = result.manifest["runs"][0]
            self.assertEqual((first["run_status"], first["reason"]), ("failed", "simulated_runner_timeout"))
            diag = out / first["files"]["diagnostics.jsonl"]["path"]
            checks = [row["payload"] for row in read_artifact(diag, ArtifactKind.DIAGNOSTICS).data
                      if row["kind"] == "cleanup_check"]
            [check] = checks
            self.assertEqual(check["attempt_id"], first["independent_start"]["attempt_id"])
            self.assertEqual((check["confirmed"], check["process_owned"], check["injection_owned"]),
                             (True, False, False))
            self.assertEqual(result.complete_n, 4)

    def test_open_loop_same_time_order_and_source(self):
        with tempfile.TemporaryDirectory() as directory:
            _, config, out = self.execute(Path(directory))
            bundle = load_manifest_bundle(out / "manifest.json", config)
            self.assertEqual(len(bundle.runs), 5)
            self.assertEqual({item.record["source_kind"] for item in bundle.runs}, {"synthetic"})
            self.assertEqual(len({item.record["independent_start"]["attempt_id"] for item in bundle.runs}), 5)
            for item in bundle.runs:
                events = item.artifacts[ArtifactKind.EVENTS].data
                by_request = {row["payload"]["request_id"]: row for row in events if row["type"] == "arrival"}
                schedule = item.artifacts[ArtifactKind.ARRIVALS].data
                for row in schedule:
                    self.assertEqual(by_request[row["request_id"]]["payload"]["client_sent_ns"],
                                     row["scheduled_arrival_ns"])
                at = [row["type"] for row in events if row["gw_at_ns"] == 100_000_000]
                if item.record["role"] == "immediate":
                    self.assertLess(at.index("inflight_snapshot"), at.index("rule_armed"))
                    self.assertLess(at.index("rule_armed"), at.index("tick"))
                outcomes = {row["request_id"]: row for row in item.artifacts[ArtifactKind.OUTCOMES].data}
                self.assertLess(by_request["healthy-01"]["gw_at_ns"], outcomes["healthy-00"]["terminal_ns"])

    def test_frozen_health_table_matches_independent_calibration_samples(self):
        from replaylab.progress import CalibrationRun, CalibrationSample, FrozenHealthProfile, ProgressConfig, build_health_profile

        study = read_artifact(FIXTURE / "study.json", ArtifactKind.STUDY)
        trace = read_artifact(FIXTURE / study.data["observation"]["calibration_evidence_path"],
                              ArtifactKind.DIAGNOSTICS)
        samples = tuple(CalibrationSample(row["payload"]["replica_id"], row["payload"]["work_key"],
                                          row["payload"]["duration_ns"]) for row in trace.data)
        runs = [CalibrationRun(trace.data[0]["run_id"], trace.data[0]["payload"]["attempt_id"],
                               "calibration", "synthetic", True, samples)]
        generated = build_health_profile(runs, ProgressConfig.from_study(study))
        frozen = FrozenHealthProfile.from_document(read_artifact(
            FIXTURE / study.data["observation"]["health_profile_path"], ArtifactKind.HEALTH_PROFILE,
        ))
        self.assertEqual(generated, frozen)
