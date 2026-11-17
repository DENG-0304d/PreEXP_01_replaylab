"""Recalculate request and B-cohort scores from recorded timestamps."""

from collections.abc import Mapping
from dataclasses import dataclass

from .artifacts import ImportedRun
from .contracts import POS_INT, TEXT, UINT, ArtifactKind, LocatedDocument, choice
from .ledger import GatewayEvent, GatewayLedger
from .progress import _required


_FAILED = frozenset({"rejected", "timeout", "cancelled", "backend_error"})


@dataclass(frozen=True)
class ScoringConfig:
    slo_id: str
    observation_end_ns: int
    generation_threshold_ns: int | None
    generation_max_new_violations: int | None

    @classmethod
    def from_study(cls, study: LocatedDocument) -> "ScoringConfig":
        slo_id = _required(study, ("slo", "slo_id"), TEXT)
        _required(study, ("slo", "deadline_basis"), choice("scheduled_arrival_ns"))
        end = _required(study, ("replay", "observation_end_ns"), UINT)
        guardrail = study.data["slo"].get("b_generation_guardrail")
        prefix = ("slo", "b_generation_guardrail")
        threshold = maximum = None
        if guardrail is not None:
            threshold = _required(study, (*prefix, "threshold_ns"), POS_INT)
            maximum = _required(study, (*prefix, "max_new_violations"), UINT)
        return cls(slo_id, end, threshold, maximum)


@dataclass(frozen=True)
class RequestScore:
    request_id: str
    state: str
    reason: str
    terminal_status: str | None
    scheduled_arrival_ns: int
    ttft_deadline_ns: int
    terminal_deadline_ns: int
    actual_arrival_ns: int | None
    first_token_ns: int | None
    terminal_ns: int | None


def score_request(arrival: Mapping[str, object], outcome: Mapping[str, object] | None) -> RequestScore:
    row = outcome if outcome is not None else {}
    status = row.get("terminal_status")
    first, terminal = row.get("first_token_ns"), row.get("terminal_ns")
    ttft_deadline, terminal_deadline = arrival["ttft_deadline_ns"], arrival["terminal_deadline_ns"]
    if status in _FAILED:
        state, reason = "fail", status
    elif status != "ok":
        state, reason = "unknown", "missing_outcome" if outcome is None else "unresolved"
    elif first is None or terminal is None:
        state, reason = "unknown", "missing_first_token" if first is None else "missing_terminal_time"
    elif terminal < first:
        state, reason = "unknown", "terminal_before_first_token"
    elif first > ttft_deadline or terminal > terminal_deadline:
        state, reason = "fail", "slo_deadline_missed"
    else:
        state, reason = "pass", "slo_met"
    return RequestScore(
        arrival["request_id"], state, reason, status, arrival["scheduled_arrival_ns"],
        ttft_deadline, terminal_deadline, row.get("actual_arrival_ns"), first, terminal,
    )


@dataclass(frozen=True)
class GenerationScore:
    state: str
    reason: str
    interval_total_ns: int | None = None
    interval_count: int | None = None


def _score_generation(outcome: dict | None, threshold_ns: int) -> GenerationScore:
    row = outcome or {}
    status = row.get("terminal_status")
    if status in _FAILED:
        return GenerationScore("fail", status)
    if status != "ok":
        return GenerationScore("unknown", "missing_or_unresolved_outcome")
    tokens = row["output_tokens"]
    first, last = row["first_token_ns"], row["last_token_ns"]
    if tokens is None or first is None or last is None:
        return GenerationScore("unknown", "missing_generation_fields")
    if tokens < 2:
        return GenerationScore("unknown", "fewer_than_two_tokens")
    if last < first:
        return GenerationScore("unknown", "last_token_before_first")
    duration, count = last - first, tokens - 1
    if duration > threshold_ns * count:
        return GenerationScore("fail", "generation_threshold_exceeded", duration, count)
    return GenerationScore("pass", "generation_threshold_met", duration, count)


@dataclass(frozen=True)
class RunScore:
    requests: Mapping[str, RequestScore]
    eval_ids: frozenset[str]
    goodput: int | None
    known_goodput: int
    unmet_n: int
    rejected_n: int
    timeout_n: int
    unknown_request_ids: tuple[str, ...]
    b_existing_ids: frozenset[str]
    b_unknown_ids: tuple[str, ...]
    b_cohort_verified: bool
    b_violation_n: int | None
    generation: Mapping[str, GenerationScore]
    generation_configured: bool
    observed_until_ns: int | None
    required_until_ns: int
    client_sends: Mapping[str, int]
    gateway_receipts: Mapping[str, int]
    arms: tuple[GatewayEvent, ...]
    redirected_n: int
    blocked_n: int
    incomplete_reasons: tuple[str, ...]


def score_run(run: ImportedRun, config: ScoringConfig) -> RunScore:
    arrivals = run.artifacts[ArtifactKind.ARRIVALS].data
    outcomes = {row["request_id"]: row for row in run.artifacts[ArtifactKind.OUTCOMES].data}
    events = run.artifacts[ArtifactKind.EVENTS]
    anchor = run.artifacts[ArtifactKind.ANCHOR_STATE].data
    ledger = GatewayLedger(run.record["run_id"])
    client_sends, gateway_receipts, arms = {}, {}, []
    for index, row in enumerate(events.data):
        update = ledger.apply(row, location=events.location((index,)))
        accepted = update.event
        if accepted is None:
            continue
        if accepted.kind == "arrival":
            request_id = accepted.payload["request_id"]
            if request_id in client_sends:
                raise accepted.error("payload.request_id", "request has multiple gateway arrival records")
            client_sends[request_id] = accepted.payload["client_sent_ns"]
            gateway_receipts[request_id] = accepted.cursor.gw_at_ns
        elif accepted.kind == "rule_armed":
            arms.append(accepted)

    requests = {row["request_id"]: score_request(row, outcomes.get(row["request_id"])) for row in arrivals}
    eval_ids = frozenset(row["request_id"] for row in arrivals if row["in_eval"])
    evaluated = [requests[key] for key in sorted(eval_ids)]
    known_goodput = sum(row.state == "pass" for row in evaluated)
    unmet_n = sum(row.state == "fail" for row in evaluated)
    unknown = tuple(row.request_id for row in evaluated if row.state == "unknown")
    latest_deadline = max((max(row.ttft_deadline_ns, row.terminal_deadline_ns) for row in evaluated), default=0)
    required_until = max(config.observation_end_ns, latest_deadline)
    view = ledger.view()
    observed_until = run.record.get("observed_until_ns")
    incomplete = []
    if observed_until is not None and not UINT.accepts(observed_until):
        incomplete.append("invalid_observation_coverage")
        observed_until = None
    elif observed_until is None and view.cursor is not None:
        observed_until = view.cursor.gw_at_ns
    if unknown:
        incomplete.append("evaluation_outcome_unknown")
    if observed_until is None or observed_until < required_until:
        incomplete.append("observation_incomplete")
    goodput = known_goodput if not incomplete else None

    before = run.record["anchor_at_ns"]
    b_ids, uncertain_b = set(), set()
    redirected = blocked = 0
    for key, facts in view.dispatches.items():
        if facts.decision is not None:
            payload = facts.decision.payload
            if (payload["default_target"] == "A" and payload["actual_target"] == "A"
                    and payload["eligible"] and payload["selected"] and payload["b_gate"] in {"deny", "unknown"}):
                blocked += 1
        if facts.sent is None or facts.sent.payload["target"] != "B":
            continue
        if (facts.decision is not None and facts.decision.payload["default_target"] == "A"
                and facts.decision.payload["actual_target"] == "B"):
            redirected += 1
        if facts.sent.cursor.gw_at_ns >= before:
            continue
        row = outcomes.get(key)
        terminal = None if row is None or row["terminal_status"] == "unresolved" else row["terminal_ns"]
        if terminal is None or terminal < facts.sent.cursor.gw_at_ns:
            uncertain_b.add(key)
        elif terminal > before:
            b_ids.add(key)
    b_existing = frozenset(b_ids)
    declared = anchor["b_existing_request_ids"]
    uncertain_b.update(key for key in declared
                       if key not in view.dispatches or view.dispatches[key].sent is None)
    if uncertain_b:
        incomplete.append("b_cohort_unverified")
    elif len(declared) != len(set(declared)) or set(declared) != b_ids:
        incomplete.append("b_cohort_mismatch")
    if not b_existing <= eval_ids or not set(declared) <= eval_ids:
        incomplete.append("b_cohort_outside_eval")
    b_verified = not any(reason.startswith("b_cohort_") for reason in incomplete)
    b_results = [requests[key] for key in sorted(b_existing & eval_ids)]
    b_violations = (sum(row.state == "fail" for row in b_results)
                    if b_verified and all(row.state != "unknown" for row in b_results) else None)
    generation = {}
    if config.generation_threshold_ns is not None:
        generation = {key: _score_generation(outcomes.get(key), config.generation_threshold_ns)
                      for key in sorted(b_existing)}
    return RunScore(
        requests, eval_ids, goodput, known_goodput, unmet_n,
        sum(row.terminal_status == "rejected" for row in evaluated),
        sum(row.terminal_status == "timeout" for row in evaluated),
        unknown, b_existing, tuple(sorted(uncertain_b)), b_verified,
        b_violations, generation, config.generation_threshold_ns is not None,
        observed_until, required_until, client_sends, gateway_receipts,
        tuple(arms), redirected, blocked, tuple(incomplete),
    )


@dataclass(frozen=True)
class ScoreDelta:
    goodput_delta: int
    b_new_violation_ids: tuple[str, ...]
    b_recovered_ids: tuple[str, ...]
    b_violation_delta: int
    generation_new_violation_ids: tuple[str, ...] | None
    generation_recovered_ids: tuple[str, ...] | None
    generation_unknown_ids: tuple[str, ...] | None
    generation_violation_delta: int | None


def _changes(reference: Mapping, candidate: Mapping, cohort: frozenset[str]) -> tuple[tuple[str, ...], ...]:
    new, recovered, unknown = [], [], []
    for key in sorted(cohort):
        left, right = reference[key].state, candidate[key].state
        if "unknown" in (left, right):
            unknown.append(key)
        elif (left, right) == ("pass", "fail"):
            new.append(key)
        elif (left, right) == ("fail", "pass"):
            recovered.append(key)
    return tuple(new), tuple(recovered), tuple(unknown)


def compare_scores(reference: RunScore, candidate: RunScore) -> ScoreDelta:
    if (reference.goodput is None or candidate.goodput is None or reference.eval_ids != candidate.eval_ids
            or reference.b_existing_ids != candidate.b_existing_ids
            or not reference.b_cohort_verified or not candidate.b_cohort_verified):
        raise ValueError("comparison requires complete, matching evaluation and B cohorts")
    cohort = reference.b_existing_ids
    new, recovered, _ = _changes(reference.requests, candidate.requests, cohort)
    gen_new = gen_recovered = gen_unknown = gen_delta = None
    if reference.generation_configured and candidate.generation_configured:
        gen_new, gen_recovered, gen_unknown = _changes(reference.generation, candidate.generation, cohort)
        if not gen_unknown:
            gen_delta = len(gen_new) - len(gen_recovered)
    return ScoreDelta(candidate.goodput - reference.goodput, new, recovered, len(new) - len(recovered),
                      gen_new, gen_recovered, gen_unknown, gen_delta)
