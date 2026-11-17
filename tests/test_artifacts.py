import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from test_contracts import plan, put_json, study


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_bundle(root, *, roles=("reference", "immediate")):
    config = root / "study.json"
    data_dir = root / "package"
    data_dir.mkdir()
    plan_path = data_dir / "original-plan.json"
    put_json(config, study())
    put_json(plan_path, plan())
    runs = []
    for run_id, role in enumerate(roles, 1):
        run_id = f"r{run_id}"
        files = {
            "arrivals.csv": "request_id,scheduled_arrival_ns,in_eval,ttft_deadline_ns,terminal_deadline_ns,input_id,prompt_tokens,max_output_tokens\nq,1,1,15,20,i,1,1\n",
            "inputs.jsonl": '{"input_id":"i","prompt":"hi","sampling":{}}\n',
            "health_profile.csv": "replica_id,work_key,healthy_median_ns,gap_limit_ns,inflight_limit_ns,calibration_count\nA,one,1,2,3,4\n",
            "events.jsonl": f'{{"run_id":"{run_id}","event_id":"e","type":"tick","event_at_ns":10,"gw_at_ns":10,"gw_seq":1,"payload":{{}}}}\n',
            "outcomes.csv": f"run_id,request_id,terminal_status,actual_arrival_ns,first_token_ns,last_token_ns,output_tokens,terminal_ns\n{run_id},q,ok,1,2,2,1,3\n",
            "diagnostics.jsonl": "",
        }
        run_dir = data_dir / run_id
        run_dir.mkdir()
        for filename, content in files.items():
            (run_dir / filename).write_bytes(content.encode("utf-8"))
        replica = dict(queued_ids=[], active_ids=[], warm_prefix_set_id="warm",
                       cached_tokens_total=0, pending_prefill_tokens=0, unfinished_requests=0)
        put_json(run_dir / "anchor_state.json", {
            "anchor_id": "a", "anchor_at_ns": 10, "phase": "before_intervention",
            "b_existing_request_ids": [], "A": replica, "B": replica,
        })
        runs.append({
            "run_id": run_id, "source_kind": "synthetic", "condition_id": "example",
            "pair_id": "p", "anchor_id": "a", "anchor_at_ns": 10, "role": role,
            "planned_delay_ns": {"reference": None, "immediate": 0, "delayed": 4}[role],
            "study_split": "discovery", "run_status": "complete", "arrival_trace_id": "trace",
            "prehistory_id": "pre", "eval_cohort_id": "cohort", "slo_id": "slo",
            "router_id": "fixture", "router_digest": "0" * 64, "diversion_rule_id": "fixture",
            "intervention_digest": "1" * 64, "clock_id": "sim",
            "independent_start": {"attempt_id": run_id, "replica_epochs": {"A": 1, "B": 1}, "started_at_ns": 0},
            "files": {path.name: {"path": f"{run_id}/{path.name}", "sha256": digest(path)}
                      for path in sorted(run_dir.iterdir())},
        })
    manifest = data_dir / "manifest.json"
    put_json(manifest, {"study_id": "fixture", "schema_version": 2, "config_sha256": digest(config),
                        "plan_path": plan_path.name, "plan_sha256": digest(plan_path), "runs": runs})
    return config, plan_path, manifest


class ArtifactTests(unittest.TestCase):
    def test_raw_plan_hash_uses_original_bytes_and_manifest_location(self):
        from replaylab.artifacts import load_manifest_bundle, resolve_manifest_path
        from replaylab.contracts import FormatError

        with tempfile.TemporaryDirectory() as directory:
            config, plan_path, manifest = make_bundle(Path(directory))
            self.assertEqual(resolve_manifest_path(manifest, plan_path.name), plan_path)
            self.assertEqual(len(load_manifest_bundle(manifest, config).runs), 2)
            plan_path.write_bytes(plan_path.read_bytes() + b" ")
            with self.assertRaises(FormatError) as caught:
                load_manifest_bundle(manifest, config)
            self.assertEqual((caught.exception.file, caught.exception.field), (manifest, "plan_sha256"))
            self.assertIn("SHA-256", caught.exception.reason)

    def test_one_bad_run_keeps_other_run_and_original_plan(self):
        from replaylab.artifacts import load_manifest_bundle

        with tempfile.TemporaryDirectory() as directory:
            config, _, manifest = make_bundle(Path(directory))
            target = manifest.parent / "r1" / "arrivals.csv"
            target.write_bytes(target.read_bytes()[:-1] + b" ")
            bundle = load_manifest_bundle(manifest, config)
            self.assertEqual([run.record["run_id"] for run in bundle.runs], ["r2"])
            self.assertEqual(bundle.plan.data, plan())
            [excluded] = bundle.excluded_runs
            self.assertEqual(excluded["run_id"], "r1")
            self.assertEqual(excluded["branch"]["role"], "reference")
            self.assertEqual(excluded["error"]["field"], "runs[0].files['arrivals.csv'].sha256")

    def test_matched_digest_does_not_hide_invalid_run_file(self):
        from replaylab.artifacts import load_manifest_bundle
        with tempfile.TemporaryDirectory() as directory:
            config, _, manifest = make_bundle(Path(directory))
            target = manifest.parent / "r1" / "events.jsonl"
            target.write_bytes(target.read_bytes().replace(b'"gw_seq":1', b'"gw_seq":true'))
            index = json.loads(manifest.read_text())
            index["runs"][0]["files"]["events.jsonl"]["sha256"] = digest(target)
            put_json(manifest, index)
            bundle = load_manifest_bundle(manifest, config)
            self.assertEqual(len(bundle.runs), 1)
            self.assertEqual(bundle.excluded_runs[0]["error"]["field"], "gw_seq")

    def test_additional_manifest_file_refs_check_binary_bytes(self):
        from replaylab.artifacts import load_manifest_bundle

        with tempfile.TemporaryDirectory() as directory:
            config, _, manifest = make_bundle(Path(directory))
            target = manifest.parent / "r1" / "raw.bin"
            target.write_bytes(b"\xff\x00")
            index = json.loads(manifest.read_bytes())
            index["runs"][0]["files"]["raw.bin"] = {"path": "r1/raw.bin", "sha256": digest(target)}
            put_json(manifest, index)
            self.assertEqual(len(load_manifest_bundle(manifest, config).runs), 2)
            target.write_bytes(b"\xff\x01")
            [rejected] = load_manifest_bundle(manifest, config).excluded_runs
            self.assertEqual(rejected["error"]["field"], "runs[0].files['raw.bin'].sha256")
