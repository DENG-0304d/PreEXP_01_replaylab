import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from test_artifacts import digest, make_bundle
from test_contracts import put_json
from test_pairing import paired_package
from test_runner import FIXTURE
from test_shadow import enable_shadow


ROOT = Path(__file__).resolve().parents[1]


class CliTests(unittest.TestCase):
    def invoke(self, *args, seed="1"):
        return subprocess.run(
            [sys.executable, "-B", "-m", "replaylab", *map(str, args)], cwd=ROOT,
            env={**os.environ, "PYTHONHASHSEED": seed}, capture_output=True,
            encoding="utf-8", errors="replace", check=False,
        )

    def test_all_commands_reject_missing_out_without_attempt_directory(self):
        for command in ["preflight", "run", "shadow", "paired"]:
            with self.subTest(command=command):
                result = self.invoke(command)
                self.assertEqual(result.returncode, 2)
                self.assertIn("--out", result.stderr)

    def test_existing_output_and_argument_error_in_new_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            existing = root / "existing"
            existing.mkdir()
            status = existing / "status.json"
            status.write_bytes(b"keep")
            result = self.invoke("preflight", "--out", existing)
            self.assertEqual(result.returncode, 2)
            self.assertEqual(status.read_bytes(), b"keep")
            result = self.invoke("run", "--out", root / "new", "--invalid")
            self.assertEqual(result.returncode, 2)
            self.assertEqual(json.loads((root / "new" / "status.json").read_bytes())["exit_code"], 2)

    def test_paired_without_shadow_accepts_an_existing_frozen_package(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, _, manifest = paired_package(root)
            out = root / "paired"
            result = self.invoke("paired", "--config", config, "--manifest", manifest, "--out", out)
            self.assertEqual(result.returncode, 0, result.stderr)
            summary = json.loads((out / "summary.json").read_bytes())
            self.assertEqual(summary["execution_status"], "pairing_complete")
            self.assertIsNone(summary["research_status"])
            self.assertTrue((out / "metrics.csv").is_file())

    def test_bad_shared_config_writes_status_for_each_command(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bad = root / "bad.json"
            put_json(bad, {"schema_version": 1})
            for command, extra in [
                ("preflight", []), ("run", ["--plan", bad]),
                ("shadow", ["--manifest", bad]), ("paired", ["--manifest", bad]),
            ]:
                with self.subTest(command=command):
                    out = root / command
                    result = self.invoke(command, "--config", bad, *extra, "--out", out)
                    self.assertEqual(result.returncode, 2)
                    error = json.loads((out / "status.json").read_bytes())["error"]
                    self.assertEqual(set(error), {"file", "line", "field", "reason"})

    def test_paired_preserves_immediate_reference_after_delayed_tamper(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, _, manifest = paired_package(root, roles=("reference", "immediate", "delayed"))
            target = manifest.parent / "r3" / "inputs.jsonl"
            target.write_bytes(target.read_bytes() + b" ")
            out = root / "paired"
            result = self.invoke("paired", "--config", config, "--manifest", manifest, "--out", out)
            self.assertEqual(result.returncode, 0, result.stderr)
            summary = json.loads((out / "summary.json").read_bytes())
            self.assertEqual(summary["imported_run_ids"], ["r1", "r2"])
            self.assertEqual(summary["excluded_runs"][0]["branch"]["role"], "delayed")
            self.assertEqual(summary["execution_status"], "pairing_complete")
            self.assertEqual((summary["valid_contrast_n"], summary["bad_contrast_n"]), (1, 2))
            by_name = {row["contrast"]: row for row in summary["contrasts"]}
            self.assertEqual(by_name["immediate-reference"]["delta"]["goodput_delta"], 0)
            self.assertEqual(by_name["delayed-reference"]["issues"][0]["code"], "branch_import_failed")
            self.assertIsNone(summary["research_status"])

    def test_shared_bad_config_and_run_local_exclusion_exit_codes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, _, manifest = make_bundle(root)
            bad = root / "bad.json"
            put_json(bad, {"schema_version": 1})
            result = self.invoke("preflight", "--config", bad, "--out", root / "bad")
            self.assertEqual(result.returncode, 2)
            status = json.loads((root / "bad" / "status.json").read_bytes())
            self.assertEqual(status["error"]["field"], "schema_version")

            (manifest.parent / "r1" / "arrivals.csv").unlink()
            result = self.invoke("paired", "--config", config, "--manifest", manifest, "--out", root / "one")
            self.assertEqual(result.returncode, 4, result.stderr)
            summary = json.loads((root / "one" / "summary.json").read_bytes())
            self.assertEqual(summary["imported_run_count"], 1)
            self.assertEqual(summary["excluded_runs"][0]["run_id"], "r1")
            self.assertEqual(len(summary["planned_branches"]), 3)
            self.assertEqual(summary["research_status"], "no_valid_pairs")

            (manifest.parent / "r2" / "arrivals.csv").unlink()
            result = self.invoke("shadow", "--config", config, "--manifest", manifest, "--out", root / "zero")
            self.assertEqual(result.returncode, 3, result.stderr)
            result = self.invoke("paired", "--config", config, "--manifest", manifest, "--out", root / "zero-paired")
            self.assertEqual(result.returncode, 3, result.stderr)
            summary = json.loads((root / "zero-paired" / "summary.json").read_bytes())
            self.assertEqual(summary["planned_contrast_n"], 3)
            self.assertEqual(summary["valid_contrast_n"], 0)

    def test_paired_shared_plan_digest_failure_is_exit_2(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, plan_path, manifest = paired_package(root)
            plan_path.write_bytes(plan_path.read_bytes() + b" ")
            out = root / "invalid"
            result = self.invoke("paired", "--config", config, "--manifest", manifest, "--out", out)
            self.assertEqual(result.returncode, 2)
            report = json.loads((out / "status.json").read_bytes())
            self.assertEqual(report["error"]["field"], "plan_sha256")

    def test_rehashed_arrivals_still_must_match_across_branches(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, _, manifest = paired_package(root, roles=("reference", "immediate", "delayed"))
            path = manifest.parent / "r3" / "arrivals.csv"
            path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))
            data = json.loads(manifest.read_bytes())
            data["runs"][2]["files"]["arrivals.csv"]["sha256"] = digest(path)
            put_json(manifest, data)
            out = root / "paired"
            result = self.invoke("paired", "--config", config, "--manifest", manifest, "--out", out)
            self.assertEqual(result.returncode, 0, result.stderr)
            summary = json.loads((out / "summary.json").read_bytes())
            self.assertEqual(summary["imported_run_count"], 3)
            self.assertEqual(summary["valid_contrast_n"], 1)
            bad = next(row for row in summary["contrasts"] if row["contrast"] == "delayed-reference")
            self.assertIn("arrival_hash_mismatch", {item["code"] for item in bad["issues"]})

    def test_two_importable_runs_with_arm_jitter_have_no_valid_contrast(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, _, manifest = paired_package(root)
            path = manifest.parent / "r2" / "events.jsonl"
            rows = [json.loads(line) for line in path.read_bytes().splitlines()]
            arm = next(row for row in rows if row["type"] == "rule_armed")
            arm["gw_at_ns"] += 1
            arm["event_at_ns"] += 1
            path.write_bytes(b"".join(json.dumps(row).encode() + b"\n" for row in rows))
            data = json.loads(manifest.read_bytes())
            data["runs"][1]["files"]["events.jsonl"]["sha256"] = digest(path)
            put_json(manifest, data)
            out = root / "paired"
            result = self.invoke("paired", "--config", config, "--manifest", manifest, "--out", out)
            self.assertEqual(result.returncode, 4, result.stderr)
            summary = json.loads((out / "summary.json").read_bytes())
            self.assertEqual((summary["imported_run_count"], summary["valid_contrast_n"]), (2, 0))
            self.assertIn("arm_jitter_exceeded", {item["code"] for item in summary["contrasts"][0]["issues"]})

    def test_paired_shadow_import_checks_sibling_timeline_and_recomputes_csv_fields(self):
        from replaylab.artifacts import read_artifact, write_artifact
        from replaylab.contracts import ArtifactKind, WINDOW_KEYS
        from replaylab.reporting import write_report
        from test_scoring import arrival, events, outcome

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, plan_path, manifest = paired_package(root)
            cfg = json.loads(config.read_bytes())
            cfg["pairing"].update(bootstrap_samples=128, bootstrap_interpolation="linear",
                                  min_independent_blocks=1)
            put_json(config, cfg)
            original = json.loads(plan_path.read_bytes())
            original["blocks"][0]["study_split"] = "validation"
            put_json(plan_path, original)
            index = json.loads(manifest.read_bytes())
            index.update(config_sha256=digest(config), plan_sha256=digest(plan_path))
            for record in index["runs"]:
                run_id = record["run_id"]
                record["study_split"] = "validation"
                specs = [(0, "replica_epoch", dict(replica_id="B", epoch=1)),
                         (10, "route_state", dict(replica_id="B", epoch=1, state_version=1, sampled_ns=10,
                                                  ready=True, pending_prefill_tokens=0, unfinished_requests=0)),
                         (10, "arrival", dict(request_id="q", client_sent_ns=10)), (100, "tick", {})]
                if record["role"] == "reference":
                    specs.append((10, "dispatch_decision", dict(request_id="q", default_target="A", actual_target="A",
                        eligible=False, selected=False, b_gate="not_applicable", reason="inactive",
                        rule_active=False, route_state_refs={}, cache_hint_tokens=0)))
                else:
                    specs.append((10, "rule_armed", dict(rule_id="fixture", activation_id="a", reason="planned")))
                files = {
                    ArtifactKind.ARRIVALS: [arrival(scheduled=10)],
                    ArtifactKind.OUTCOMES: [outcome(run_id, status="rejected" if run_id == "r1" else "ok", at=10)],
                    ArtifactKind.EVENTS: events(run_id, *specs),
                }
                for kind, rows in files.items():
                    path = manifest.parent / run_id / kind
                    write_artifact(path, kind, rows)
                    record["files"][kind]["sha256"] = digest(path)
            put_json(manifest, index)

            shadow = root / "shadow"
            shadow.mkdir()
            key = ("fixture", digest(config), digest(plan_path), "synthetic", "validation", "example", "p", "a")
            refs = dict(kind="shadow_window", run_id="r1", comparison="same_threshold_or",
                        baseline_audit_ids=["baseline"], inflight_audit_ids=["inflight"])
            window = {**dict(zip(WINDOW_KEYS, key)), "baseline_intended_arm_ns": 10, "inflight_intended_arm_ns": 10,
                      "baseline_caught_state": "caught", "inflight_goodput_delta": 999, "coverage_cell": "both_caught",
                      "evidence_refs_json": [refs]}
            windows = write_report(shadow, "windows.csv", [window])

            def run(name):
                out = root / name
                result = self.invoke("paired", "--config", config, "--manifest", manifest,
                                     "--shadow", windows, "--out", out)
                self.assertEqual(result.returncode, 0, result.stderr)
                return read_artifact(out / "windows.csv", ArtifactKind.WINDOWS).data[0]

            no_timeline = run("unverified")
            self.assertEqual((no_timeline["baseline_caught_state"], no_timeline["coverage_cell"]), ("unknown", None))
            self.assertEqual(no_timeline["inflight_goodput_delta"], 1)
            audit = [
                dict(audit_id=group, run_id="r1", origin="shadow", window_key=list(key),
                     comparison="same_threshold_or", signal_group=group, intended_arm_ns=10,
                     type="candidate", state="pass", gw_at_ns=10, gw_seq=4, request_id="q",
                     unacked_prefill_tokens=0, unacked_requests=0, evidence_event_ids=["event-2", "event-3", "event-4"])
                for group in ("baseline", "inflight")
            ]
            write_report(shadow, "timeline.jsonl", audit)
            self.assertEqual(run("verified")["coverage_cell"], "both_caught")
            audit[1]["evidence_event_ids"][0] = "missing"
            write_report(shadow, "timeline.jsonl", audit)
            self.assertIsNone(run("broken-reference")["coverage_cell"])

            out = root / "wrong-comparison"
            result = self.invoke("paired", "--config", config, "--manifest", manifest, "--shadow", windows,
                                 "--shadow-comparison", "same_health_budget", "--out", out)
            self.assertEqual(result.returncode, 2)
            self.assertEqual(json.loads((out / "status.json").read_bytes())["error"]["field"], "evidence_refs_json")

            window["plan_sha256"] = "f" * 64
            write_report(shadow, "windows.csv", [window])
            out = root / "wrong-plan"
            result = self.invoke("paired", "--config", config, "--manifest", manifest, "--shadow", windows,
                                 "--out", out)
            self.assertEqual(result.returncode, 2)
            self.assertEqual(json.loads((out / "status.json").read_bytes())["error"]["field"], "plan_sha256")

    def test_missing_inference_fields_keep_valid_paired_report_and_exit_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, _, manifest = paired_package(root)
            cfg = json.loads(config.read_bytes())
            cfg["pairing"].pop("seed")
            cfg["pairing"].pop("min_independent_blocks")
            put_json(config, cfg)
            index = json.loads(manifest.read_bytes())
            index["config_sha256"] = digest(config)
            put_json(manifest, index)
            out = root / "paired"
            result = self.invoke("paired", "--config", config, "--manifest", manifest, "--out", out)
            self.assertEqual(result.returncode, 0, result.stderr)
            summary = json.loads((out / "summary.json").read_bytes())
            self.assertEqual(summary["valid_contrast_n"], 1)
            self.assertIn("pairing.seed", summary["inference_missing_fields"])
            report = next(group["full_pool"] for group in summary["effect_groups"]
                          if group["contrast"] == "immediate-reference")
            self.assertEqual((report["mean"], report["interval"]), (0, None))

    def test_hardware_null_blocks_execution_but_allows_offline_contract_check(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, plan_path, manifest = make_bundle(root)
            data = json.loads(config.read_bytes())
            data["source_kind"] = "real"
            data["environment"]["gpu"] = None
            put_json(config, data)
            index = json.loads(manifest.read_bytes())
            index["config_sha256"] = digest(config)
            put_json(manifest, index)
            enable_shadow(config, manifest)
            for command, extra in [("preflight", []), ("run", ["--plan", plan_path])]:
                result = self.invoke(command, "--config", config, *extra, "--out", root / command)
                self.assertEqual(result.returncode, 3, result.stderr)
                status = json.loads((root / command / "status.json").read_bytes())
                self.assertEqual(status["reason"], "BLOCKED_UNTIL_HARDWARE")
                self.assertIn("environment.gpu", status["missing_fields"])
            result = self.invoke("shadow", "--config", config, "--manifest", manifest, "--out", root / "offline")
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_t01_all_cli_outputs_and_manifest_hashes_are_byte_reproducible(self):
        from hashlib import sha256

        expected_dir = FIXTURE / "expected"
        expected = json.loads((expected_dir / "t01.json").read_bytes())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            roots = [root / "first", root / "second"]
            config = FIXTURE / "study.json"
            for output, seed in zip(roots, ("1", "23")):
                commands = [
                    ("preflight", []), ("run", ["--plan", FIXTURE / "plan.json"]),
                    ("shadow", ["--manifest", output / "run" / "manifest.json"]),
                    ("paired", ["--manifest", output / "run" / "manifest.json",
                                "--shadow", output / "shadow" / "windows.csv"]),
                ]
                for command, arguments in commands:
                    result = self.invoke(command, "--config", config, *arguments, "--out", output / command, seed=seed)
                    self.assertEqual(result.returncode, 0, (command, result.stderr))
            files = [sorted(path.relative_to(output).as_posix() for path in output.rglob("*") if path.is_file())
                     for output in roots]
            self.assertEqual(files[0], files[1])
            for relative in files[0]:
                self.assertEqual((roots[0] / relative).read_bytes(), (roots[1] / relative).read_bytes(), relative)
            actual_digests = {relative: sha256((roots[0] / relative).read_bytes()).hexdigest() for relative in files[0]}
            self.assertEqual(actual_digests, json.loads((expected_dir / "file_sha256.json").read_bytes()))

            for output in roots:
                package = output / "run"
                manifest = json.loads((package / "manifest.json").read_bytes())
                self.assertEqual(manifest["config_sha256"], digest(config))
                self.assertEqual(manifest["plan_sha256"], digest(package / manifest["plan_path"]))
                self.assertEqual(manifest["attempts_sha256"], digest(package / manifest["attempts_path"]))
                self.assertEqual([row["role"] for row in manifest["runs"]], expected["branch_order"])
                for row in manifest["runs"]:
                    self.assertEqual(row["source_kind"], "synthetic")
                    self.assertEqual(row["device_capability"], "unavailable")
                    for ref in row["files"].values():
                        self.assertEqual(digest(package / ref["path"]), ref["sha256"])
                summary = json.loads((output / "paired" / "summary.json").read_bytes())
                contrasts = {row["contrast"]: row["delta"]["goodput_delta"] for row in summary["contrasts"]}
                self.assertEqual(contrasts, expected["contrasts"])
                self.assertEqual(summary["aa"]["deltas"], expected["aa_deltas"])
                self.assertEqual(summary["independent_action_block_n"], 1)
                self.assertEqual(summary["research_sample_n"], 0)
                self.assertIsNone(summary["research_status"])
                self.assertEqual(summary["research_reason"], "no_real_validation_actions")
                [window] = summary["shadow_windows"]["windows"]
                self.assertEqual(window["baseline"]["match"]["state"], "UNTESTED_DELAY")
                self.assertEqual(window["baseline"]["match"]["neighbors_ns"], [0, 100000000])
                self.assertEqual(window["inflight"]["match"]["state"], "MATCHED")
                self.assertEqual(window["inflight"]["caught_state"], "caught")
                self.assertEqual(window["inflight"]["match"]["observations"][0]["goodput_delta"], 10)
                self.assertIsNone(window["coverage_cell"])
                ref = next(row for row in manifest["runs"] if row["role"] == "reference")
                ledger_ids = {json.loads(line)["event_id"] for line in
                              (package / ref["files"]["events.jsonl"]["path"]).read_bytes().splitlines()}
                audits = [json.loads(line) for line in (output / "shadow" / "timeline.jsonl").read_bytes().splitlines()]
                self.assertTrue(any(row["type"] == "candidate" and row["state"] == "pass" for row in audits))
                self.assertTrue(all(set(row["evidence_event_ids"]) <= ledger_ids for row in audits))

            # A nontrivial T03 replay: mutate offline completion truth after capturing both firing signals.
            first = roots[0]
            manifest_path = first / "run" / "manifest.json"
            manifest = json.loads(manifest_path.read_bytes())
            ref = next(row for row in manifest["runs"] if row["role"] == "reference")
            diag = first / "run" / ref["files"]["diagnostics.jsonl"]["path"]
            row = dict(run_id=ref["run_id"], kind="device_completion_truth", record_id="changed",
                       payload=dict(source_kind="synthetic", end_ns=99000000))
            diag.write_bytes(diag.read_bytes() + json.dumps(row, separators=(",", ":")).encode() + b"\n")
            ref["files"]["diagnostics.jsonl"]["sha256"] = digest(diag)
            put_json(manifest_path, manifest)
            changed = root / "after-offline-change"
            result = self.invoke("shadow", "--config", config, "--manifest", manifest_path, "--out", changed, seed="17")
            self.assertEqual(result.returncode, 0, result.stderr)
            for name in ("windows.csv", "timeline.jsonl"):
                self.assertEqual((changed / name).read_bytes(), (first / "shadow" / name).read_bytes())
            before = json.loads((first / "shadow" / "summary.json").read_bytes())
            after = json.loads((changed / "summary.json").read_bytes())
            self.assertEqual(before["shadow"], after["shadow"])
