import json
from pathlib import Path
import tempfile
import unittest


def study():
    return {
        "schema_version": 2,
        "environment": {
            "model": {"name": "fixture", "revision": "1"},
            "tokenizer": {"name": "fixture", "revision": "1"},
            "vllm": {"version": "synthetic", "build_id": "fixture"},
            "gpu": {"model": "synthetic", "device_ids": ["a", "b"]},
            "replicas": {"A": {"endpoint": "mock:a"}, "B": {"endpoint": "mock:b"}},
            "gateway_version": "1", "collector_version": "1",
        },
        "replay": {
            "input_trace_path": "arrivals.csv", "warmup_trace_path": "warmup.csv",
            "prehistory_trace_path": "prehistory.csv", "clock_id": "sim",
            "rotation_seed": 7, "send_jitter_ns": 0, "arm_jitter_ns": 0,
            "observation_end_ns": 100,
        },
        "observation": {
            "tick_ns": 1, "work_ttl_ns": 5, "completion_ttl_ns": 5,
            "residual_ttl_ns": 5, "evidence_ttl_ns": 5, "alpha": 0.5,
            "threshold": 2, "health_profile_path": "health_profile.csv",
            "workload_buckets": {"decode": [1]}, "conditional_thresholds": {},
            "healthy_alert_budget": 0, "healthy_redirect_budget": 0,
        },
        "router": {
            "router_id": "fixture", "router_digest": "0" * 64, "state_ttl_ns": 5,
            "prefill_cost_ns_per_token": 1, "decode_load_cost_ns": 1,
            "speed_factor_min": 0.5, "speed_factor_max": 2,
            "reservation_ack_policy": "versioned", "reservation_timeout_ns": 5,
        },
        "diversion": {
            "rule_id": "fixture", "rule_seed": 0, "ratio_ppm": 1000000,
            "duration_ns": 4, "max_redirects": 1, "max_b_unfinished": 2,
            "max_b_prefill_tokens": 5, "b_state_ttl_ns": 5,
        },
        "slo": {"slo_id": "slo", "threshold_source": "fixture", "deadline_basis": "scheduled_arrival_ns"},
        "pairing": {
            "prehistory_tolerances": {}, "delta_goodput_requests": 1,
            "min_independent_blocks": 2, "interval_algorithm": "percentile_bootstrap",
            "confidence_level": 0.95, "seed": 8, "capture_perturbation_budget": {},
            "primary_contrast": "immediate-reference",
        },
        "injection": {
            "scope": "A", "implementation_id": "fixture", "level": "low",
            "start_offset_ns": -1, "stop_offset_ns": 8,
            "cleanup_timeout_ns": 10, "condition_id": "example",
        },
    }


def plan():
    return {
        "schema_version": 2,
        "blocks": [{
            "condition_id": "example", "pair_id": "p", "anchor_id": "a",
            "anchor_at_ns": 10, "study_split": "discovery", "kind": "action",
            "branches": [
                {"role": "reference", "delay_ns": None},
                {"role": "immediate", "delay_ns": 0},
                {"role": "delayed", "delay_ns": 4},
            ],
        }],
    }


def put_json(path, data):
    path.write_bytes((json.dumps(data, indent=2) + "\n").encode("utf-8"))


class ContractTests(unittest.TestCase):
    def test_step_start_work_version_is_optional_but_boolean_is_invalid(self):
        from replaylab.artifacts import read_artifact
        from replaylab.contracts import ArtifactKind, FormatError

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            row = dict(run_id="r", event_id="s", type="step_start", event_at_ns=0,
                       gw_at_ns=0, gw_seq=1, payload=dict(replica_id="A", epoch=1,
                       step_id="s", start_ns=0, boundary_id="gpu"))
            put = lambda: path.write_bytes((json.dumps(row) + "\n").encode())
            put()
            read_artifact(path, ArtifactKind.EVENTS)
            row["payload"]["work_version"] = True
            put()
            with self.assertRaises(FormatError) as caught:
                read_artifact(path, ArtifactKind.EVENTS)
            self.assertEqual((caught.exception.line, caught.exception.field), (1, "payload.work_version"))
    def test_extra_diagnostic_time_must_be_integer_ns(self):
        from replaylab.artifacts import read_artifact
        from replaylab.contracts import ArtifactKind, FormatError

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "diagnostics.jsonl"
            path.write_bytes(b'{"run_id":"r","record_id":"x","kind":"injection","payload":{"stop_ns":true}}\n')
            with self.assertRaises(FormatError) as caught:
                read_artifact(path, ArtifactKind.DIAGNOSTICS)
            self.assertEqual((caught.exception.line, caught.exception.field), (1, "payload.stop_ns"))

    def test_located_config_version_missing_field_and_boolean_time(self):
        from replaylab.artifacts import read_artifact
        from replaylab.contracts import ArtifactKind, FormatError

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "study.json"
            for field, edit, needle, reason in [
                ("schema_version", lambda d: d.update(schema_version=1), '"schema_version"', "2"),
                ("observation.tick_ns", lambda d: d["observation"].update(tick_ns=True), '"tick_ns"', "integer"),
                ("observation.work_ttl_ns", lambda d: d["observation"].pop("work_ttl_ns"), '"observation"', "required"),
            ]:
                with self.subTest(field=field):
                    data = study()
                    edit(data)
                    put_json(path, data)
                    line = next(i for i, row in enumerate(path.read_text().splitlines(), 1) if needle in row)
                    with self.assertRaises(FormatError) as caught:
                        read_artifact(path, ArtifactKind.STUDY)
                    error = caught.exception
                    self.assertEqual((error.file, error.line, error.field), (path, line, field))
                    self.assertIn(reason, error.reason)

    def test_plan_delay_rules_and_boolean_anchor(self):
        from replaylab.artifacts import read_artifact
        from replaylab.contracts import ArtifactKind, FormatError

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plan.json"
            for field, value in [("anchor_at_ns", True), ("blocks[0].branches[2].delay_ns", 0)]:
                data = plan()
                if field == "anchor_at_ns":
                    data["blocks"][0][field] = value
                    field = f"blocks[0].{field}"
                else:
                    data["blocks"][0]["branches"][2]["delay_ns"] = value
                put_json(path, data)
                with self.assertRaises(FormatError) as caught:
                    read_artifact(path, ArtifactKind.PLAN)
                self.assertEqual(caught.exception.field, field)
            put_json(path, plan())
            self.assertEqual(read_artifact(path, ArtifactKind.PLAN).data, plan())

    def test_action_block_can_plan_only_the_two_required_roles(self):
        from replaylab.artifacts import read_artifact
        from replaylab.contracts import ArtifactKind

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plan.json"
            data = plan()
            data["blocks"][0]["branches"].pop()
            put_json(path, data)
            self.assertEqual(read_artifact(path, ArtifactKind.PLAN).data, data)

    def test_jsonl_blank_is_reported_at_its_physical_line(self):
        from replaylab.artifacts import read_artifact
        from replaylab.contracts import ArtifactKind, FormatError

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "inputs.jsonl"
            path.write_bytes(b'{"input_id":"i","prompt":"hi","sampling":{}}\n\n{"input_id":"j"}\n')
            with self.assertRaises(FormatError) as caught:
                read_artifact(path, ArtifactKind.INPUTS)
            error = caught.exception
            self.assertEqual((error.file, error.line, error.field), (path, 2, "$"))
            self.assertIn("blank", error.reason)

    def test_events_validate_each_run_in_original_order_and_allow_retransmission(self):
        from replaylab.artifacts import read_artifact
        from replaylab.contracts import ArtifactKind, FormatError

        def event(run, sequence, at):
            return {"run_id": run, "event_id": "repeat", "type": "tick",
                    "event_at_ns": 0, "gw_seq": sequence, "gw_at_ns": at, "payload": {}}

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            prefix = [event("a", 2, 8), event("b", 1, 2)]
            for last, invalid in [
                (event("a", 4, 8), None), (event("a", 2, 8), "gw_seq"),
                (event("a", 1, 9), "gw_seq"), (event("a", 4, 7), "gw_at_ns"),
            ]:
                path.write_bytes(b"".join(json.dumps(row).encode() + b"\n" for row in [*prefix, last]))
                if invalid:
                    with self.assertRaises(FormatError) as caught:
                        read_artifact(path, ArtifactKind.EVENTS)
                    self.assertEqual((caught.exception.line, caught.exception.field), (3, invalid))
                else:
                    self.assertEqual(len(read_artifact(path, ArtifactKind.EVENTS).data), 3)

    def test_csv_uniqueness_and_strict_boolean(self):
        from replaylab.artifacts import read_artifact
        from replaylab.contracts import ArtifactKind, FormatError

        header = "request_id,scheduled_arrival_ns,in_eval,ttft_deadline_ns,terminal_deadline_ns,input_id,prompt_tokens,max_output_tokens\n"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "arrivals.csv"
            for body, line, field in [
                ('"a\nb",1,1,2,3,i,0,1\nr,2,true,3,4,i,0,1\n', 4, "in_eval"),
                ("r,1,1,2,3,i,0,1\nr,2,0,3,4,i,0,1\n", 3, "request_id"),
            ]:
                path.write_bytes((header + body).encode())
                with self.assertRaises(FormatError) as caught:
                    read_artifact(path, ArtifactKind.ARRIVALS)
                self.assertEqual((caught.exception.line, caught.exception.field), (line, field))
