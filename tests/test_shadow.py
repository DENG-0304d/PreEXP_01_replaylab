from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from replaylab.artifacts import read_artifact
from replaylab.contracts import ArtifactKind, LocatedDocument
from test_artifacts import digest, make_bundle
from test_contracts import put_json, study
from test_gateway_order import complete, decision, epoch, event, work
from test_inflight_evidence import snapshot, start
from test_progress import make_tracker


def shadow_data():
    data = study()
    data["observation"].update(
        evidence_ttl_ns=10, work_ttl_ns=20, completion_ttl_ns=40, residual_ttl_ns=20,
        clock_bounds=dict(age_error_ns=0, duration_error_ns=0, lead_error_ns=0),
    )
    data["replay"]["observation_end_ns"] = 125
    data["router"].update(state_ttl_ns=20, reservation_timeout_ns=30)
    data["diversion"].update(b_state_ttl_ns=20, max_b_unfinished=1, max_b_prefill_tokens=3)
    return data


def enable_shadow(config, manifest):
    data = json.loads(config.read_bytes())
    data["observation"]["clock_bounds"] = dict(age_error_ns=0, duration_error_ns=0, lead_error_ns=0)
    put_json(config, data)
    index = json.loads(manifest.read_bytes())
    index["config_sha256"] = digest(config)
    put_json(manifest, index)


def route(seq, at, replica, pending=0):
    return event("route_state", seq, at, replica_id=replica, epoch=1, state_version=1,
                 sampled_ns=at, ready=True, pending_prefill_tokens=pending, unfinished_requests=0)


def original_decision(seq, at, request_id="q"):
    row = decision(seq, at)
    row["payload"]["request_id"] = request_id
    return row


class ShadowTests(unittest.TestCase):
    def config(self, data=None, *, comparison="same_threshold_or"):
        from replaylab.shadow import ShadowConfig

        return ShadowConfig.from_study(LocatedDocument(Path("study.json"), data or shadow_data(), {}),
                                       comparison=comparison)

    def window(self, **changes):
        from replaylab.shadow import ShadowWindow

        window = ShadowWindow("study", "0" * 64, "1" * 64, "synthetic", "validation",
                              "case", "pair", "anchor", 100, "frozen", "rule", ("definition",))
        return replace(window, **changes)

    def trace(self, events, *, healthy_intervals=None):
        from replaylab.shadow import CommonTrace

        return CommonTrace("run", LocatedDocument(Path("events.jsonl"), events, {}, row_oriented=True),
                           {"q": 1, "q2": 1}, make_tracker().health_profile,
                           healthy_intervals=healthy_intervals)

    def timeline(self):
        return [
            epoch(1, 0), epoch(2, 0, replica="B"), work(3, 0), start(4, 0),
            route(5, 100, "A"), route(6, 100, "B", pending=2),
            complete(7, 100, endpoint=100, step="completed", duration=30), snapshot(8, 100, 100),
            event("tick", 9, 100), event("arrival", 10, 100, request_id="q", client_sent_ns=100),
            original_decision(11, 100), event("arrival", 12, 101, request_id="q2", client_sent_ns=101),
            original_decision(13, 101, "q2"),
        ]

    def test_t13_real_inflight_or_and_public_progress_tie_without_virtual_capacity(self):
        from replaylab.shadow import replay_shadow

        result = replay_shadow(self.trace(self.timeline(), healthy_intervals=((100, 110),)),
                               window=self.window(), config=self.config())
        baseline, advanced = result.baseline, result.inflight
        self.assertEqual((baseline.alert_usable_ns, advanced.alert_usable_ns, result.lead_ns), (100, 100, 0))
        self.assertIn("e7", baseline.evidence_event_ids)
        self.assertIn("e7", advanced.evidence_event_ids)
        self.assertIn("e8", advanced.evidence_event_ids)
        self.assertNotIn("e8", baseline.evidence_event_ids)
        self.assertEqual((baseline.candidate_n, advanced.candidate_n), (2, 2))
        self.assertEqual((baseline.healthy_trigger_n, advanced.healthy_trigger_n), (1, 1))
        self.assertEqual([row["request_id"] for row in result.timeline if row["type"] == "candidate"
                          and row["signal_group"] == "inflight" and row["state"] == "pass"], ["q", "q2"])

    def test_read_at_dispatch_before_later_same_tick_done_and_do_not_backfill(self):
        from replaylab.shadow import replay_shadow

        rows = self.timeline()[:6] + [
            event("tick", 7, 100), event("arrival", 8, 100, request_id="q", client_sent_ns=100),
            snapshot(9, 100, 100), original_decision(10, 100),
            snapshot(11, 100, 100, status="DONE_VERIFIED"), event("tick", 12, 101),
        ]
        result = replay_shadow(self.trace(rows), window=self.window(), config=self.config())
        self.assertIsNone(result.baseline.alert_usable_ns)
        self.assertEqual((result.inflight.alert_cursor.gw_seq, result.inflight.candidate_n), (10, 1))
        self.assertEqual([row["state"] for row in result.timeline if row["type"] == "signal"
                          and row["signal_group"] == "inflight"], ["UNKNOWN", "TRIGGER", "CLEAR"])

    def test_t09_single_branch_preflight_extracts_original_send_and_uses_planned_offset(self):
        from replaylab.shadow import BranchTrace, check_action_candidate, inspect_action_branch

        window = self.window()
        record = dict(run_id="run", source_kind="synthetic", study_split="validation", condition_id="case",
                      anchor_id="anchor", anchor_at_ns=100, intervention_digest="frozen",
                      diversion_rule_id="rule", run_status="complete", role="delayed",
                      planned_delay_ns=10, anchor_definition_id="definition")
        decision_row = original_decision(2, 113)
        decision_row["payload"].update(actual_target="B", reason="redirected")
        rows = [event("rule_armed", 1, 111, rule_id="rule", activation_id="a", reason="armed"),
                decision_row, event("dispatch_sent", 3, 115, request_id="q", target="B", reservation_id="r")]
        fact = inspect_action_branch(BranchTrace(record, rows))
        self.assertEqual((fact.actual_arm_ns, fact.first_redirect_ns), (111, 115))
        for altered, intended, limit, expected in [
            (fact, 110, 1, "preflight_pass"), (fact, 111, 1, "planned_delay_mismatch"),
            (fact, 110, 0, "arm_jitter_exceeded"),
            (replace(fact, condition_id="other"), 110, 1, "condition_mismatch"),
            (replace(fact, run_status="failed"), 110, 1, "run_not_complete"),
        ]:
            self.assertEqual(check_action_candidate(window, intended_arm_ns=intended, candidate=altered,
                                                    arm_jitter_ns=limit).reason, expected)

    def test_t03_offline_truth_change_does_not_change_shadow_windows_or_timeline(self):
        from replaylab.artifacts import load_manifest_bundle
        from replaylab.shadow import write_shadow_reports

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, _, manifest = make_bundle(root)
            enable_shadow(config, manifest)
            outputs = []
            for number, contents in enumerate([b'{"run_id":"r1","kind":"done","record_id":"d"}\n', b""]):
                target = manifest.parent / "r1" / "diagnostics.jsonl"
                target.write_bytes(contents)
                data = json.loads(manifest.read_bytes())
                data["runs"][0]["files"]["diagnostics.jsonl"]["sha256"] = digest(target)
                put_json(manifest, data)
                out = root / str(number)
                out.mkdir()
                write_shadow_reports(load_manifest_bundle(manifest, config),
                                     read_artifact(config, ArtifactKind.STUDY), out_dir=out)
                outputs.append([(out / name).read_bytes() for name in ("windows.csv", "timeline.jsonl")])
            self.assertEqual(outputs[0], outputs[1])

    def test_equal_budget_reads_frozen_strategy_parameters_and_keeps_modes_separate(self):
        from replaylab.shadow import replay_shadow

        data = shadow_data()
        data["observation"]["alert_policies"] = {
            "base": {"threshold": 9, "gap_limits": {"A": {"light": 40}}},
            "device": {"threshold": 10, "gap_limits": {"A": {"light": 40}},
                       "inflight_limits": {"A": {"light": 99}}},
        }
        data["observation"]["shadow_same_budget"] = {
            "baseline_policy_id": "base", "inflight_policy_id": "device",
            "budget": {"unit": "trigger_count", "limit": 1},
            "calibration_run_ids": ["cal-a"], "validation_split": "validation",
        }
        equivalent = self.config(data, comparison="same_health_budget")
        self.assertEqual(equivalent.policy_ids, ("base", "device"))
        result = replay_shadow(self.trace(self.timeline()), window=self.window(), config=equivalent)
        self.assertIsNone(result.baseline.alert_usable_ns)
        self.assertEqual(result.inflight.alert_usable_ns, 100)
        self.assertIsNone(result.lead_ns)
        self.assertEqual(result.comparison, "same_health_budget")
