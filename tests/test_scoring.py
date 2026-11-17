from pathlib import Path
import unittest

from replaylab.artifacts import ImportedRun
from replaylab.contracts import ArtifactKind, LocatedDocument
from test_contracts import study


def located(name, data):
    return LocatedDocument(Path(name), data, {}, name.endswith((".csv", ".jsonl")))


def arrival(request="q", scheduled=0, ttft=20, deadline=35, *, evaluate=True):
    return dict(request_id=request, scheduled_arrival_ns=scheduled, in_eval=evaluate,
                ttft_deadline_ns=ttft, terminal_deadline_ns=deadline,
                input_id="input", prompt_tokens=2, max_output_tokens=4)


def outcome(run, request="q", *, status="ok", at=0, first=20, last=21, terminal=35, tokens=2):
    return dict(run_id=run, request_id=request, terminal_status=status, actual_arrival_ns=at,
                first_token_ns=first, last_token_ns=last, terminal_ns=terminal, output_tokens=tokens)


def events(run_id, *specs):
    return [
        dict(run_id=run_id, event_id=f"event-{sequence}", type=kind, event_at_ns=at,
             gw_at_ns=at, gw_seq=sequence, payload=payload)
        for sequence, (at, kind, payload) in enumerate(sorted(specs, key=lambda spec: spec[0]), 1)
    ]


def dispatch(target, rid):
    return ("dispatch_sent", dict(request_id=rid, target=target, reservation_id=f"reservation-{rid}"))


def run_fixture(run_id, *, role="reference", delay=None, arrivals=None, outcomes=None, ledger=None,
                b_existing=(), started=0, split="discovery", pair="p", source="synthetic"):
    if arrivals is None:
        arrivals = [arrival()]
    if outcomes is None:
        outcomes = [outcome(run_id, a["request_id"], at=a["scheduled_arrival_ns"]) for a in arrivals]
    if ledger is None:
        specs = [(a["scheduled_arrival_ns"], "arrival", dict(request_id=a["request_id"],
                 client_sent_ns=a["scheduled_arrival_ns"])) for a in arrivals]
        specs.append((100, "tick", {}))
        if role in {"immediate", "delayed"}:
            specs.append((10 + delay, "rule_armed", dict(rule_id="fixture", activation_id="a", reason="planned")))
        ledger = events(run_id, *specs)
    sides = dict(queued_ids=[], active_ids=[], warm_prefix_set_id="warm", cached_tokens_total=0,
                 pending_prefill_tokens=0, unfinished_requests=0)
    state = dict(anchor_id="a", anchor_at_ns=10, phase="before_intervention",
                 b_existing_request_ids=list(b_existing), A=dict(sides), B=dict(sides))
    rec = dict(
        run_id=run_id, role=role, planned_delay_ns=delay, run_status="complete", source_kind=source,
        condition_id="example", pair_id=pair, anchor_id="a", anchor_at_ns=10, study_split=split,
        arrival_trace_id="trace", prehistory_id="pre", eval_cohort_id="cohort", slo_id="slo",
        router_id="fixture", router_digest="0" * 64, diversion_rule_id="fixture",
        intervention_digest="1" * 64, clock_id="sim",
        independent_start=dict(attempt_id=run_id, started_at_ns=started, replica_epochs={"A": 1, "B": 1}),
        files={name: dict(path=f"{run_id}/{name}", sha256=sha)
               for name, sha in [("arrivals.csv", "a" * 64), ("inputs.jsonl", "b" * 64)]},
    )
    return ImportedRun(rec, {
        ArtifactKind.ARRIVALS: located("arrivals.csv", arrivals),
        ArtifactKind.OUTCOMES: located("outcomes.csv", outcomes),
        ArtifactKind.EVENTS: located("events.jsonl", ledger),
        ArtifactKind.ANCHOR_STATE: located("anchor_state.json", state),
        ArtifactKind.DIAGNOSTICS: located("diagnostics.jsonl", []),
    })


class ScoringTests(unittest.TestCase):
    def config(self, *, generation=None):
        from replaylab.scoring import ScoringConfig

        data = study()
        if generation is not None:
            data["slo"]["b_generation_guardrail"] = dict(threshold_ns=generation, max_new_violations=1)
        return ScoringConfig.from_study(located("study.json", data))

    def test_t08_equality_late_send_ok_missing_first_and_known_failures(self):
        from replaylab.scoring import score_request

        a = arrival()
        success = outcome("r")
        self.assertEqual(score_request(a, success).state, "pass")
        late = outcome("r", at=15, first=21, terminal=36)
        self.assertEqual(score_request(a, late).state, "fail")
        self.assertEqual(score_request(a, outcome("r", first=None)).state, "unknown")
        self.assertEqual(score_request(a, None).state, "unknown")
        self.assertEqual(score_request(a, outcome("r", status="unresolved")).state, "unknown")
        for status in ("rejected", "timeout", "cancelled", "backend_error"):
            with self.subTest(status=status):
                self.assertEqual(score_request(a, outcome("r", status=status, first=2)).state, "fail")

    def test_t08_partial_known_goodput_and_short_actual_observation(self):
        from replaylab.scoring import score_run

        rows = [arrival(key) for key in ("good", "missing", "no-token", "timeout")]
        results = [outcome("r", "good"), outcome("r", "no-token", first=None),
                   outcome("r", "timeout", status="timeout")]
        run = run_fixture("r", arrivals=rows, outcomes=results, ledger=events("r", (50, "tick", {})))
        scored = score_run(run, self.config())
        self.assertEqual((scored.goodput, scored.known_goodput, scored.timeout_n), (None, 1, 1))
        self.assertEqual(set(scored.unknown_request_ids), {"missing", "no-token"})
        self.assertIn("observation_incomplete", scored.incomplete_reasons)

    def test_t08_existing_b_and_direction_generation_is_independent_of_main_slo(self):
        from replaylab.scoring import compare_scores, score_run

        ids = ("r1", "r2", "finished", "at-anchor", "short", "gen")
        rows = [arrival(key) for key in ids]
        earlier = set(ids) - {"r2", "at-anchor"}
        groups = ("r1", "short", "gen")
        results = []
        for run_id in ("delayed", "immediate"):
            ledger = events(run_id, *((0, "arrival", dict(request_id=key, client_sent_ns=0)) for key in ids),
                            *((5 if key in earlier else 10, *dispatch("B", key)) for key in ids),
                            (100, "tick", {}))
            answers = [outcome(run_id, key, first=6, last=7, terminal=10 if key == "finished" else 35)
                       for key in ids]
            answers[4].update(output_tokens=1)
            if run_id == "immediate":
                answers[0].update(terminal_status="backend_error")
                answers[1].update(terminal_status="rejected")
                answers[5].update(last_token_ns=10)
            run = run_fixture(run_id, arrivals=rows, outcomes=answers, ledger=ledger, b_existing=groups)
            results.append(score_run(run, self.config(generation=2)))
        ref, candidate = results
        delta = compare_scores(ref, candidate)
        self.assertEqual((ref.goodput, candidate.goodput), (6, 4))
        self.assertEqual(candidate.b_existing_ids, frozenset(groups))
        self.assertEqual(delta.b_new_violation_ids, ("r1",))
        self.assertEqual((delta.b_recovered_ids, delta.b_violation_delta), ((), 1))
        self.assertEqual(delta.generation_new_violation_ids, ("gen", "r1"))
        self.assertEqual(delta.generation_unknown_ids, ("short",))
        self.assertEqual(candidate.requests["short"].state, "pass")
        self.assertEqual(candidate.generation["short"].state, "unknown")

    def test_possible_existing_b_with_unknown_terminal_cannot_verify_group(self):
        from replaylab.scoring import score_run

        run = run_fixture("r", outcomes=[outcome("r", status="backend_error", terminal=None)],
                          ledger=events("r", (5, *dispatch("B", "q")), (100, "tick", {})))
        scored = score_run(run, self.config())
        self.assertEqual(scored.goodput, 0)
        self.assertEqual(scored.b_unknown_ids, ("q",))
        self.assertIn("b_cohort_unverified", scored.incomplete_reasons)

    def test_claimed_existing_b_with_missing_send_evidence_is_unverified(self):
        from replaylab.scoring import score_run

        scored = score_run(run_fixture("r", b_existing=("q",)), self.config())
        self.assertEqual(scored.b_unknown_ids, ("q",))
        self.assertFalse(scored.b_cohort_verified)
