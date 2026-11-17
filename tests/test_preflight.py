from dataclasses import replace
import json
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from test_contracts import put_json, study
from test_runner import FIXTURE
import test_shadow


ROOT = FIXTURE.parents[1]
HARDWARE = ROOT / "fixtures" / "hardware"


class PreflightTests(unittest.TestCase):
    def invoke(self, *args):
        return subprocess.run([sys.executable, "-B", "-m", "replaylab", *map(str, args)], cwd=ROOT,
                              capture_output=True, text=True, encoding="utf-8", check=False)

    def test_t15_explicit_missing_prerequisites_and_backend_not_connected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = study()
            data["source_kind"] = "real"
            data["environment"].update(gpu=None, model=None, authorization="denied")
            config = root / "study.json"
            put_json(config, data)
            for name, extra in [("preflight", []), ("run", ["--plan", FIXTURE / "plan.json"])]:
                out = root / name
                result = self.invoke(name, "--config", config, *extra, "--out", out)
                self.assertEqual(result.returncode, 3)
                status = json.loads((out / "status.json").read_bytes())
                self.assertEqual(status["reason"], "BLOCKED_UNTIL_HARDWARE")
                self.assertTrue({"environment.gpu", "environment.model", "environment.authorization"}
                                <= set(status["missing_fields"]))
                summary = json.loads((out / "summary.json").read_bytes())
                self.assertEqual(summary["backend_status"], "not_connected")
                self.assertEqual(summary["action_replay_status"], "blocked")
                self.assertFalse((out / "manifest.json").exists())
            complete = study()
            complete.update(source_kind="real")
            put_json(config, complete)
            out = root / "configured"
            result = self.invoke("run", "--config", config, "--plan", FIXTURE / "plan.json", "--out", out)
            self.assertEqual(result.returncode, 3)
            summary = json.loads((out / "summary.json").read_bytes())
            self.assertEqual((summary["backend_status"], summary["missing_fields"]), ("not_connected", []))
            self.assertEqual(summary["backend_reason"], "真实执行后端尚未接入")

    def test_t15_template_separates_prerequisites_from_unexecuted_probes(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "real"
            result = self.invoke("preflight", "--config", HARDWARE / "study.json", "--out", out)
            self.assertEqual(result.returncode, 3, result.stderr)
            report = json.loads((out / "preflight.json").read_bytes())
            checks = {row["name"]: row for row in report["checks"]}
            for check, field in {"gpu_allocation": "environment.gpu.model",
                                 "model_ready": "environment.model.name",
                                 "authorization": "environment.authorization"}.items():
                self.assertEqual(checks[check]["status"], "blocked")
                self.assertIn(field, checks[check]["missing_fields"])
                self.assertEqual(checks[check]["evidence_refs"], [])
            self.assertEqual((report["action_replay_status"], report["inflight_timing_status"],
                              report["device_capability"]), ("blocked", "blocked", "unavailable"))
            self.assertNotIn("router.router_id", report["missing_fields"])
            run_out = Path(directory) / "run"
            self.assertEqual(self.invoke("run", "--config", HARDWARE / "study.json", "--plan", FIXTURE / "plan.json",
                                         "--out", run_out).returncode, 3)
            run_report = json.loads((run_out / "preflight.json").read_bytes())
            self.assertTrue({"router.router_id", "replay.inputs_path"} <= set(run_report["missing_fields"]))
            complete = study()
            complete.update(source_kind="real")
            config = Path(directory) / "declared.json"
            put_json(config, complete)
            out = Path(directory) / "declared"
            self.assertEqual(self.invoke("preflight", "--config", config, "--out", out).returncode, 3)
            report = json.loads((out / "preflight.json").read_bytes())
            self.assertEqual({row["status"] for row in report["checks"]}, {"not_tested"})

    def test_t15_mock_reports_all_states_and_cannot_unlock_real_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = HARDWARE / "study.json"
            fixture = HARDWARE / "mock_preflight.json"
            result = self.invoke("preflight", "--config", config, "--mock-platform", fixture, "--out", root / "mock")
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads((root / "mock" / "preflight.json").read_bytes())
            self.assertEqual(report["source_kind"], "synthetic")
            self.assertEqual({row["source_kind"] for row in report["checks"]}, {"synthetic"})
            self.assertEqual({row["status"] for row in report["checks"]}, {"pass", "blocked", "not_tested"})
            executed = [row for row in report["checks"] if row["status"] in {"pass", "fail"}]
            for check in executed:
                ref, = check["evidence_refs"]
                self.assertEqual(ref["sha256"], hashlib.sha256(fixture.read_bytes()).hexdigest())
                self.assertEqual(ref["field"], f"checks.{check['name']}")
            self.assertEqual(report["device_capability"], "engine_progress_only")
            self.assertEqual(report["device_lead_status"], "unverified")
            self.assertEqual(report["action_replay_status"], "pass")
            self.assertEqual(report["inflight_timing_status"], "blocked")
            mixed_data = json.loads(fixture.read_bytes())
            mixed_data["checks"]["injection_locality"] = {"status": "fail", "reason": "simulated_b_over_tolerance"}
            mixed_fixture = root / "mixed.json"
            put_json(mixed_fixture, mixed_data)
            result = self.invoke("preflight", "--config", config, "--mock-platform", mixed_fixture,
                                 "--out", root / "mixed")
            self.assertEqual(result.returncode, 0, result.stderr)
            mixed = json.loads((root / "mixed" / "preflight.json").read_bytes())
            self.assertEqual({row["status"] for row in mixed["checks"]}, {"pass", "fail", "blocked", "not_tested"})
            failed, = [row for row in mixed["checks"] if row["status"] == "fail"]
            self.assertTrue(failed["evidence_refs"])
            self.assertEqual(mixed["action_replay_status"], "fail")
            result = self.invoke("run", "--config", config, "--plan", FIXTURE / "plan.json", "--out", root / "run")
            self.assertEqual(result.returncode, 3, result.stderr)
            status = json.loads((root / "run" / "summary.json").read_bytes())
            self.assertEqual((status["action_replay_status"], status["backend_reason"]),
                             ("blocked", "真实执行后端尚未接入"))
            self.assertFalse((root / "run" / "manifest.json").exists())

    def test_t15_verified_device_capability_still_needs_fresh_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = json.loads((HARDWARE / "mock_preflight.json").read_bytes())
            data["checks"]["device_verification"] = {"status": "pass", "reason": "simulated_valid_marker"}
            data["checks"]["evidence_freshness"] = {"status": "fail", "reason": "simulated_expired_on_receipt"}
            mock = root / "device.json"
            put_json(mock, data)
            result = self.invoke("preflight", "--config", HARDWARE / "study.json", "--mock-platform", mock,
                                 "--out", root / "out")
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads((root / "out" / "preflight.json").read_bytes())
            self.assertEqual((report["device_capability"], report["inflight_timing_status"]), ("device_verified", "fail"))
            self.assertEqual(report["action_replay_status"], "pass")
            self.assertIsNone(report["device_lead_ns"])
            self.assertEqual(report["device_lead_reason"], "evidence_freshness_unverified")

    def test_t15_bad_mock_configuration_is_located(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mock = root / "invalid.json"
            put_json(mock, {"schema_version": 2, "source_kind": "real", "checks": {}})
            result = self.invoke("preflight", "--config", HARDWARE / "study.json", "--mock-platform", mock,
                                 "--out", root / "out")
            self.assertEqual(result.returncode, 2)
            error = json.loads((root / "out" / "status.json").read_bytes())["error"]
            self.assertEqual((error["file"], error["line"], error["field"]), (str(mock), 3, "source_kind"))
            self.assertIn("synthetic", error["reason"])

    def test_t15_synthetic_preflight_and_engine_only_observation_boundary(self):
        from replaylab.shadow import replay_shadow

        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "preflight"
            result = self.invoke("preflight", "--config", FIXTURE / "study.json", "--out", out)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads((out / "preflight.json").read_bytes())
            self.assertEqual((report["source_kind"], report["device_capability"]), ("synthetic", "unavailable"))
            self.assertEqual(report["inflight_timing_status"], "synthetic_only")
        fixture = test_shadow.ShadowTests()
        trace = replace(fixture.trace(fixture.timeline()), device_verified=False)
        observed = replay_shadow(trace, window=fixture.window(source_kind="real"), config=fixture.config())
        self.assertEqual(observed.baseline.alert_usable_ns, 100)
        self.assertEqual(observed.lead_reason, "device_timing_unverified")
        self.assertIsNone(observed.lead_ns)
