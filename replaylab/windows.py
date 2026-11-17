"""Exact action-slot association and coverage verified against shadow audit evidence."""

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass

from .artifacts import ImportedBundle, ImportedRun
from .contracts import TEXT, UINT, WINDOW_KEYS, ArtifactKind, FormatError, LocatedDocument
from .diversion import DiversionConfig, stable_selected
from .ledger import GatewayLedger
from .pairing import ContrastValidation, PairingResult
from .statistics import (
    BlockEffect, EffectReport, FrozenScope, StatisticsResult, frozen_scope, independent_block,
    report_number, summarize_blocks,
)


COMPARISONS = ("same_threshold_or", "same_health_budget")
_WINDOW_ROLES = {"action": "reference", "shadow_only": "shadow"}
_CELLS = {("caught", "caught"): "both_caught", ("missed", "caught"): "inflight_only",
          ("caught", "missed"): "baseline_only", ("missed", "missed"): "both_missed"}


@dataclass(frozen=True)
class SignalEvidence:
    alert_usable_ns: int | None
    intended_arm_ns: int | None
    assessment: str = "unknown"
    reason: str | None = None
    eligible_n: int | None = None
    blocked_n: int | None = None
    audit_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class ShadowWindowInput:
    key: tuple[str, ...]
    anchor_at_ns: int
    scope: FrozenScope | None
    comparison: str
    baseline: SignalEvidence
    inflight: SignalEvidence
    source_run_id: str | None = None
    reason: str | None = None

    @property
    def fields(self) -> dict:
        return dict(zip(WINDOW_KEYS, self.key))


@dataclass(frozen=True)
class ActionObservation:
    block: tuple[str, ...]
    reference_run_id: str
    run_id: str
    actual_arm_ns: int
    first_redirect_ns: int | None
    goodput_delta: int


@dataclass(frozen=True)
class ActionSlot:
    scope: FrozenScope | None
    contrast: ContrastValidation
    attempts: tuple[tuple[str, str], ...]
    observation: ActionObservation | None


@dataclass(frozen=True)
class ActionIndex:
    slots: tuple[ActionSlot, ...]
    statistics: StatisticsResult


def _block_key(record: dict) -> tuple:
    return record.get("condition_id"), record.get("pair_id"), record.get("anchor_id"), record.get("study_split")


def _manifest_runs(bundle: ImportedBundle) -> tuple[dict, ...]:
    return tuple(record for record in bundle.manifest.data.get("runs", ())
                 if isinstance(record, dict) and all(TEXT.accepts(value) for value in _block_key(record)))


def _observed_action(row: ContrastValidation) -> ActionObservation | None:
    if row.delta is None or row.candidate.score is None or len(row.candidate.score.arms) != 1:
        return None
    arm = row.candidate.score.arms[0]
    ledger = GatewayLedger(row.candidate.run_id)
    doc = row.candidate.run.artifacts[ArtifactKind.EVENTS]
    for i, event in enumerate(doc.data):
        ledger.apply(event, location=doc.location((i,)))
    first = min((
        item.sent.cursor.gw_at_ns for item in ledger.view().dispatches.values()
        if item.sent is not None and item.decision is not None
        and arm.cursor.gw_seq < item.decision.cursor.gw_seq < item.sent.cursor.gw_seq
        and item.decision.payload["default_target"] == "A" and item.decision.payload["actual_target"] == "B"
        and item.sent.payload["target"] == "B"
    ), default=None)
    return ActionObservation(independent_block(row), row.reference.run_id, row.candidate.run_id,
                             arm.cursor.gw_at_ns, first, row.delta.goodput_delta)


def build_action_index(
    pairing: PairingResult, statistics: StatisticsResult, *, bundle: ImportedBundle | None = None,
) -> ActionIndex:
    """Keep the planned reference contrasts, including those without runs or valid scores."""
    records = defaultdict(list)
    if bundle is not None:
        for item in _manifest_runs(bundle):
            records[_block_key(item)].append(item)
    slots = []
    for row in pairing.contrasts:
        plan = row.planned
        if plan.kind != "action" or plan.reference_role != "reference":
            continue
        key = (*plan.block[1:], plan.study_split)
        basis = row.reference or row.candidate
        scope = None if basis is None else frozen_scope(pairing.study_id, basis.record)
        if scope is None:
            known = {frozen_scope(pairing.study_id, record) for record in records[key]} - {None}
            if len(known) == 1:
                scope = known.pop()
        attempts = tuple(sorted((record["run_id"], record["run_status"] if TEXT.accepts(record.get("run_status")) else "unknown")
                                for record in records[key] if TEXT.accepts(record.get("run_id"))
                                and record.get("role") == plan.candidate_role
                                and record.get("planned_delay_ns") == plan.candidate_delay_ns))
        slots.append(ActionSlot(scope, row, attempts, _observed_action(row)))
    return ActionIndex(tuple(slots), statistics)


@dataclass(frozen=True)
class DelayMatch:
    state: str
    delay_ns: int | None = None
    neighbors_ns: tuple[int | None, int | None] | None = None
    observations: tuple[ActionObservation, ...] = ()
    exploration: tuple[ActionObservation, ...] = ()
    effect: EffectReport | None = None
    reasons: tuple[str, ...] = ()
    attempts: tuple[tuple[str, str], ...] = ()
    excluded: tuple[dict, ...] = ()

    def report(self) -> dict:
        return {
            "state": self.state, "delay_ns": self.delay_ns, "neighbors_ns": self.neighbors_ns,
            "observations": [asdict(item) for item in self.observations],
            "exploration": [asdict(item) for item in self.exploration],
            "effect": None if self.effect is None else self.effect.report(),
            "reasons": self.reasons, "attempts": self.attempts, "excluded": self.excluded,
        }


def _related_slots(window: ShadowWindowInput, index: ActionIndex) -> tuple[ActionSlot, ...]:
    fields = window.fields
    return tuple(slot for slot in index.slots if (
        (slot.scope is not None and slot.scope == window.scope)
        or ((slot.scope is None or slot.contrast.planned.pair_id == fields["pair_id"])
            and slot.contrast.planned.study_id == fields["study_id"]
            and slot.contrast.planned.condition_id == fields["condition_id"]
            and slot.contrast.planned.anchor_id == fields["anchor_id"]
            and slot.contrast.planned.anchor_at_ns == window.anchor_at_ns)
    ))


def _effect(window: ShadowWindowInput, delay: int, index: ActionIndex) -> EffectReport | None:
    for group in index.statistics.groups:
        if (group.scope == window.scope and group.study_split == "validation" and group.delay_ns == delay
                and group.contrast in {"immediate-reference", "delayed-reference"}):
            return group.effect
    return None


def match_delay(window: ShadowWindowInput, *, intended_arm_ns: int, index: ActionIndex) -> DelayMatch:
    delay = intended_arm_ns - window.anchor_at_ns
    if delay < 0:
        return DelayMatch("INVALID_ARM", delay, reasons=("intended_arm_before_anchor",))
    if window.scope is None:
        return DelayMatch("SCOPE_UNVERIFIED", delay, reasons=("frozen_window_scope_unverified",))
    slots = _related_slots(window, index)
    exact = [slot for slot in slots if slot.contrast.planned.delay_ns == delay]
    if not exact:
        delays = {slot.contrast.planned.delay_ns for slot in slots}
        neighbors = (max((ns for ns in delays if ns < delay), default=None),
                     min((ns for ns in delays if ns > delay), default=None))
        return DelayMatch("UNTESTED_DELAY", delay, neighbors, reasons=("no_planned_delay",))
    observations, exploration, excluded = [], [], []
    reasons = set()
    for slot in exact:
        if slot.scope != window.scope or slot.observation is None:
            scope_reason = None
            if slot.scope != window.scope:
                scope_reason = "planned_scope_unverified" if slot.scope is None else "frozen_scope_mismatch"
                reasons.add(scope_reason)
            reasons.update(item.code for item in slot.contrast.issues)
            excluded.append(dict(pair_id=slot.contrast.planned.pair_id, study_split=slot.contrast.planned.study_split,
                                 scope_reason=scope_reason, issues=[asdict(item) for item in slot.contrast.issues]))
            continue
        if slot.contrast.planned.study_split == window.fields["study_split"] == "validation":
            observations.append(slot.observation)
        else:
            exploration.append(slot.observation)
    attempts = tuple(sorted({attempt for slot in exact for attempt in slot.attempts}))
    observations.sort(key=lambda item: (item.block, item.run_id))
    exploration.sort(key=lambda item: (item.block, item.run_id))
    if not observations:
        state = "EXPLORATORY_ONLY" if exploration else "PLANNED_NO_VALID_PAIR"
        if exploration:
            reasons.add("validation_match_required")
        return DelayMatch(state, delay, exploration=tuple(exploration), reasons=tuple(sorted(reasons)),
                          attempts=attempts, excluded=tuple(excluded))
    effect = _effect(window, delay, index)
    if effect is None or effect.interval is None:
        reasons.add(effect.reason if effect is not None else "effect_unverified")
    return DelayMatch("MATCHED", delay, observations=tuple(observations), exploration=tuple(exploration),
                      effect=effect, reasons=tuple(sorted(reasons)), attempts=attempts, excluded=tuple(excluded))


@dataclass(frozen=True)
class GroupCoverage:
    signal: SignalEvidence
    match: DelayMatch
    caught_state: str
    reason: str | None

    def report(self) -> dict:
        return {"signal": asdict(self.signal), "match": self.match.report(),
                "caught_state": self.caught_state, "reason": self.reason}


def _group(window: ShadowWindowInput, signal: SignalEvidence, index: ActionIndex) -> GroupCoverage:
    arm = signal.intended_arm_ns
    match = DelayMatch("NO_ARM") if arm is None else match_delay(window, intended_arm_ns=arm, index=index)
    if match.state == "UNTESTED_DELAY":
        return GroupCoverage(signal, match, "untested", "UNTESTED_DELAY")
    if signal.assessment == "unknown":
        return GroupCoverage(signal, match, "unknown", signal.reason or "dispatch_evidence_unverified")
    if arm is not None and match.state != "MATCHED":
        return GroupCoverage(signal, match, "unknown", match.state)
    if signal.assessment == "no_candidate":
        return GroupCoverage(signal, match, "missed", signal.reason or "no_executable_candidate")
    effect = match.effect
    if effect is not None and effect.status == "supported":
        return GroupCoverage(signal, match, "caught", None)
    if effect is not None and effect.status == "meaningful_gain_excluded":
        return GroupCoverage(signal, match, "missed", "meaningful_gain_excluded_at_delay")
    return GroupCoverage(signal, match, "unknown", effect.reason if effect is not None else "effect_unverified")


def _opportunity(window: ShadowWindowInput, index: ActionIndex) -> tuple[str, tuple[int, ...]]:
    slots = _related_slots(window, index)
    valid_split = [slot for slot in slots if slot.contrast.planned.study_split == "validation"]
    delays = sorted({slot.contrast.planned.delay_ns for slot in valid_split})
    if window.scope is None or window.fields["study_split"] != "validation":
        return "INSUFFICIENT_EVIDENCE", ()
    if not delays:
        return "UNTESTED", ()
    checks = {delay: _effect(window, delay, index) for delay in delays}
    verified = tuple(delay for delay, effect in checks.items() if effect is not None and effect.status == "supported")
    if verified:
        return "VERIFIED_OPPORTUNITY", verified
    if all(effect is not None and effect.status == "meaningful_gain_excluded" for effect in checks.values()):
        return "MEANINGFUL_GAIN_EXCLUDED", ()
    return "INSUFFICIENT_EVIDENCE", ()


def _common_effect(baseline: GroupCoverage, inflight: GroupCoverage, index: ActionIndex) -> EffectReport:
    left = {(item.block, item.reference_run_id): item for item in baseline.match.observations}
    right = {(item.block, item.reference_run_id): item for item in inflight.match.observations}
    effects = [BlockEffect(block, right[block, ref].goodput_delta - left[block, ref].goodput_delta)
               for block, ref in sorted(left.keys() & right.keys())]
    return summarize_blocks(effects, index.statistics.config, validation=True)


@dataclass(frozen=True)
class WindowResult:
    window: ShadowWindowInput
    baseline: GroupCoverage
    inflight: GroupCoverage
    opportunity_state: str
    verified_delays_ns: tuple[int, ...]
    coverage_cell: str | None
    common_effect: EffectReport | None

    def report(self) -> dict:
        common = None
        if self.common_effect is not None:
            common = self.common_effect.report()
            # The absolute gain threshold applies to action versus reference.
            del common["status"], common["stable_harm"]
            if self.common_effect.interval is not None:
                common["reason"] = None
            elif not self.common_effect.n:
                common["reason"] = "no_common_independent_blocks"
        return {**self.window.fields, "comparison": self.window.comparison, "source_run_id": self.window.source_run_id,
                "reason": self.window.reason, "baseline": self.baseline.report(), "inflight": self.inflight.report(),
                "opportunity_state": self.opportunity_state, "verified_delays_ns": self.verified_delays_ns,
                "coverage_cell": self.coverage_cell, "common_effect": common, "closed_loop_goodput_delta": None}

    def csv_row(self) -> dict:
        result = {**self.window.fields, "opportunity_state": self.opportunity_state, "coverage_cell": self.coverage_cell,
                  "goodput_delta_between_groups": None if self.common_effect is None else report_number(self.common_effect.mean)}
        reasons = [self.window.reason] if self.window.reason else []
        refs = [{"kind": "shadow_window", "comparison": self.window.comparison, "run_id": self.window.source_run_id}]
        for name, group in (("baseline", self.baseline), ("inflight", self.inflight)):
            signal, match = group.signal, group.match
            result.update({f"{name}_{key}": value for key, value in {
                "alert_usable_ns": signal.alert_usable_ns, "intended_arm_ns": signal.intended_arm_ns,
                "eligible_n": signal.eligible_n, "blocked_n": signal.blocked_n, "caught_state": group.caught_state,
                "matched_delay_ns": match.delay_ns if match.observations else None,
                "matched_run_ids_json": [item.run_id for item in match.observations] or None,
                "actual_arm_times_json": [item.actual_arm_ns for item in match.observations] or None,
                "first_redirect_times_json": [item.first_redirect_ns for item in match.observations] or None,
                "goodput_delta": (report_number(match.effect.mean)
                                  if match.effect is not None and match.effect.interval is not None else None),
            }.items()})
            if group.reason:
                reasons.append(f"{name}:{group.reason}")
            refs[0][f"{name}_audit_ids"] = list(signal.audit_ids)
        result.update(reason=";".join(reasons) or None, evidence_refs_json=refs)
        return result


@dataclass(frozen=True)
class WindowAnalysis:
    windows: tuple[WindowResult, ...]

    def summary(self) -> dict:
        verified = [row for row in self.windows if row.opportunity_state == "VERIFIED_OPPORTUNITY"]
        cells = Counter(row.coverage_cell for row in verified if row.coverage_cell is not None)
        return {
            "total_window_n": len(self.windows), "verified_opportunity_n": len(verified),
            "four_cell_n": sum(cells.values()), "coverage_cells": {cell: cells[cell] for cell in sorted(_CELLS.values())},
            "opportunity_counts": dict(Counter(row.opportunity_state for row in self.windows)),
            "baseline_coverage_counts": dict(Counter(row.baseline.caught_state for row in verified)),
            "inflight_coverage_counts": dict(Counter(row.inflight.caught_state for row in verified)),
            "windows": [row.report() for row in self.windows],
        }


def analyze_windows(windows: tuple[ShadowWindowInput, ...], *, index: ActionIndex) -> WindowAnalysis:
    output = []
    for window in windows:
        baseline = _group(window, window.baseline, index)
        inflight = _group(window, window.inflight, index)
        opportunity, delays = _opportunity(window, index)
        cell = _CELLS.get((baseline.caught_state, inflight.caught_state)) if opportunity == "VERIFIED_OPPORTUNITY" else None
        common = _common_effect(baseline, inflight, index) if cell == "both_caught" else None
        output.append(WindowResult(window, baseline, inflight, opportunity, delays, cell, common))
    return WindowAnalysis(tuple(output))


def _authorized_origins(bundle: ImportedBundle, block: dict, source: str) -> list[ImportedRun]:
    return [run for run in bundle.runs if _block_key(run.record) == _block_key(block)
            and run.record["role"] == _WINDOW_ROLES[block["kind"]]
            and run.record["source_kind"] == source and run.record["anchor_at_ns"] == block["anchor_at_ns"]]


def _receipts(origin: ImportedRun) -> dict | None:
    ledger = GatewayLedger(origin.record["run_id"])
    location = origin.artifacts[ArtifactKind.EVENTS]
    result = {}
    try:
        for index, row in enumerate(location.data):
            update = ledger.apply(row, location=location.location((index,)))
            if update.event is not None:
                result[update.event.event_id] = update.event
    except FormatError:
        return None
    return result


def _linked_events(audit: dict, original: dict) -> tuple | None:
    refs = audit.get("evidence_event_ids")
    if type(refs) is not list or any(not TEXT.accepts(ref) or ref not in original for ref in refs):
        return None
    return tuple(original[ref] for ref in refs)


def _audited_decision(audit: dict, linked: tuple, start: int, stop: int):
    now, seq, request = audit.get("gw_at_ns"), audit.get("gw_seq"), audit.get("request_id")
    if not UINT.accepts(now) or not UINT.accepts(seq) or not start <= now < stop:
        return None
    matches = [event for event in linked if event.kind == "dispatch_decision"
               and event.cursor.gw_at_ns == now and event.cursor.gw_seq == seq
               and event.payload["request_id"] == request]
    return matches[0] if len(matches) == 1 else None


def _candidate_state(
    audit: dict, linked: tuple, decision, origin: ImportedRun, original: dict,
    anchor: int, anchor_id: str, config: DiversionConfig, arm: int | None,
) -> str | None:
    now, seq, request = decision.cursor.gw_at_ns, decision.cursor.gw_seq, decision.payload["request_id"]
    if any(event.cursor.gw_seq > seq for event in linked):
        return None
    if arm is None or decision.payload["default_target"] != "A" or config.max_redirects == 0:
        return "ineligible"
    arrivals = [event for event in original.values() if event.kind == "arrival" and event.payload["request_id"] == request
                and event.cursor.gw_seq < seq]
    if len(arrivals) != 1 or arrivals[0] not in linked:
        return None
    if arrivals[0].cursor.gw_at_ns < anchor:
        return "ineligible"
    if not stable_selected(config.rule_seed, anchor_id, request, config.ratio_ppm):
        return "not_selected"
    snapshots = [event for event in original.values() if event.kind == "route_state"
                 and event.payload["replica_id"] == "B" and event.cursor.gw_seq < seq
                 and event.payload["sampled_ns"] <= event.cursor.gw_at_ns]
    epochs = [event for event in original.values() if event.kind == "replica_epoch"
              and event.payload["replica_id"] == "B" and event.cursor.gw_seq < seq]
    if not epochs:
        return None
    epoch = epochs[-1].payload["epoch"]
    snapshot = max((event for event in snapshots if event.payload["epoch"] == epoch),
                   key=lambda event: (event.payload["state_version"], event.payload["sampled_ns"],
                                      -event.cursor.gw_seq), default=None)
    if snapshot is None or snapshot not in linked:
        return None
    p = snapshot.payload
    snapshot_fields = ("ready", "unfinished_requests", "pending_prefill_tokens")
    if any(event.payload["epoch"] == epoch and event.payload["state_version"] == p["state_version"]
           and any(event.payload[key] != p[key] for key in snapshot_fields) for event in snapshots):
        return None
    if not 0 <= now - p["sampled_ns"] <= config.b_state_ttl_ns:
        return None
    if not p["ready"]:
        return "deny"
    prompts = {row["request_id"]: row["prompt_tokens"] for row in origin.artifacts[ArtifactKind.ARRIVALS].data}
    if request not in prompts:
        return None
    new = _new_tokens(decision, prompts[request])
    if p["unfinished_requests"] + 1 > config.max_b_unfinished or p["pending_prefill_tokens"] + new > config.max_b_prefill_tokens:
        return "deny"
    begin = next(event.cursor.gw_seq for event in epochs if event.payload["epoch"] == epoch)
    bounds = _original_b_hold_bound(original, snapshot, snapshots, prompts, begin, seq)
    reserved_p, reserved_n = audit.get("unacked_prefill_tokens"), audit.get("unacked_requests")
    if bounds is None or not UINT.accepts(reserved_p) or not UINT.accepts(reserved_n):
        return None
    if reserved_p < bounds[0] or reserved_n < bounds[1]:
        return None
    if (p["unfinished_requests"] + reserved_n + 1 <= config.max_b_unfinished
            and p["pending_prefill_tokens"] + reserved_p + new <= config.max_b_prefill_tokens):
        return "pass"
    return None


def _new_tokens(decision, prompt: int) -> int:
    raw = decision.payload["cache_hint_tokens"]
    hint = raw.get("B") if hasattr(raw, "get") else raw
    return prompt - (min(max(hint, 0), prompt) if type(hint) is int else 0)


def _original_b_hold_bound(
    original: dict, current, snapshots: list, prompts: dict, begin: int, seq: int,
) -> tuple[int, int] | None:
    """Bound unconfirmed load from original decisions and explicit releases or ACKs."""
    held, sent, versions = {}, {}, {}
    for snapshot in snapshots:
        if snapshot.payload["epoch"] == current.payload["epoch"]:
            versions.setdefault(snapshot.payload["state_version"], snapshot)
    for event in original.values():
        if not begin < event.cursor.gw_seq < seq:
            continue
        p = event.payload
        if event.kind == "dispatch_decision" and p["actual_target"] == "B":
            if p["request_id"] not in prompts:
                return None
            held[p["request_id"]] = _new_tokens(event, prompts[p["request_id"]])
        elif event.kind == "dispatch_sent" and p["target"] == "B":
            if p["request_id"] not in prompts:
                return None
            held.setdefault(p["request_id"], prompts[p["request_id"]])
            sent[p["request_id"]] = event
        elif event.kind in {"reservation_release", "dispatch_ack"} and p["replica_id"] == "B":
            request = p["request_id"]
            receipt = sent.get(request)
            if receipt is None or receipt.payload["reservation_id"] != p["reservation_id"]:
                continue
            if event.kind == "reservation_release" and p["reason"] != "snapshot_confirmed":
                held.pop(request, None)
            elif event.kind == "dispatch_ack":
                named = versions.get(p["state_version"])
                if (named is not None and named.cursor.gw_seq > receipt.cursor.gw_seq
                        and p["state_version"] <= current.payload["state_version"]
                        and receipt.cursor.gw_at_ns <= named.payload["sampled_ns"] <= current.payload["sampled_ns"]):
                    held.pop(request, None)
    return sum(held.values()), len(held)


def _parse_signal(
    row: dict, name: str, *, provenance: dict | None, timeline: dict | None, key: tuple[str, ...],
    comparison: str, origin: ImportedRun | None, original: dict | None, anchor: int,
    observation_end_ns: int | None, diversion: DiversionConfig | None,
) -> SignalEvidence:
    signal = {field: row.get(f"{name}_{field}") for field in
              ("alert_usable_ns", "intended_arm_ns", "eligible_n", "blocked_n")}
    if timeline is None:
        return SignalEvidence(**signal, reason="timeline_missing")
    if provenance is None:
        return SignalEvidence(**signal, reason="shadow_comparison_unverified")
    if origin is None or original is None:
        return SignalEvidence(**signal, reason="shadow_origin_unverified")
    audit_ids = provenance.get(f"{name}_audit_ids")
    if type(audit_ids) is not list or not audit_ids or any(not TEXT.accepts(item) for item in audit_ids):
        return SignalEvidence(**signal, reason="dispatch_evidence_unverified")
    arm = signal["intended_arm_ns"]
    if observation_end_ns is None or diversion is None:
        return SignalEvidence(**signal, reason="shadow_window_parameters_missing")
    start = anchor if arm is None else arm
    stop = observation_end_ns if arm is None else min(observation_end_ns, arm + diversion.duration_ns)
    if (not anchor <= start < stop
            or (arm is not None and signal["alert_usable_ns"] is not None and signal["alert_usable_ns"] > arm)):
        return SignalEvidence(**signal, reason="invalid_shadow_observation_window")

    covered, executable = set(), False
    closure = False
    for audit_id in audit_ids:
        audit = timeline.get(audit_id)
        if (audit is None or audit.get("origin") != "shadow" or audit.get("comparison") != comparison
                or audit.get("run_id") != origin.record["run_id"] or audit.get("window_key") != list(key)
                or audit.get("signal_group") != name or audit.get("intended_arm_ns") != arm):
            return SignalEvidence(**signal, reason="dispatch_evidence_unverified")
        linked = _linked_events(audit, original)
        if linked is None:
            return SignalEvidence(**signal, reason="dispatch_evidence_unverified")
        if audit.get("type") == "coverage":
            closure |= (audit.get("state") == "complete" and audit.get("coverage_start_ns") == start
                        and audit.get("coverage_end_ns") == stop
                        and any(event.kind == "tick" and event.cursor.gw_at_ns >= stop for event in linked))
            continue
        decision = _audited_decision(audit, linked, start, stop)
        if decision is None or audit.get("type") != "candidate":
            return SignalEvidence(**signal, reason="dispatch_evidence_unverified")
        state = audit.get("state")
        assessed = _candidate_state(audit, linked, decision, origin, original, anchor, key[-1], diversion, arm)
        if assessed is None or assessed != state:
            return SignalEvidence(**signal, reason="dispatch_condition_unknown")
        if state == "pass":
            executable = True
        covered.add(decision.event_id)
    if executable:
        return SignalEvidence(**signal, assessment="executable", audit_ids=tuple(audit_ids))
    expected = {event.event_id for event in original.values() if event.kind == "dispatch_decision"
                and start <= event.cursor.gw_at_ns < stop}
    if closure and covered == expected:
        return SignalEvidence(**signal, assessment="no_candidate", reason="no_executable_candidate",
                              audit_ids=tuple(audit_ids))
    return SignalEvidence(**signal, reason="observation_coverage_unverified")


def read_shadow_windows(
    document: LocatedDocument, *, bundle: ImportedBundle, timeline: LocatedDocument | None = None,
    comparison: str = "same_threshold_or", observation_end_ns: int | None = None,
    diversion: DiversionConfig | None = None, expected_source: str | None = None,
) -> tuple[ShadowWindowInput, ...]:
    """Bind windows to the frozen input bundle and audit ids to companion timeline rows."""
    if comparison not in COMPARISONS:
        raise ValueError("unknown shadow comparison")
    manifest = bundle.manifest.data
    study_id = manifest["study_id"]
    blocks = {_block_key(block): block for block in bundle.plan.data["blocks"] if block["kind"] in _WINDOW_ROLES}
    raw_records = _manifest_runs(bundle)
    declared_sources = defaultdict(set)
    for record in raw_records:
        if record.get("source_kind") in ("real", "synthetic"):
            declared_sources[_block_key(record)].add(record["source_kind"])
    audits = None
    if timeline is not None:
        audits = {}
        for audit in timeline.data:
            if isinstance(audit, dict) and TEXT.accepts(audit.get("audit_id")):
                audit_id = audit["audit_id"]
                audits[audit_id] = audit if audit_id not in audits else None
    output = {}
    for i, row in enumerate(document.data):
        for field in ("study_id", "config_sha256", "plan_sha256"):
            input_value, expected = row[field], manifest[field]
            same = input_value == expected if field == "study_id" else input_value.lower() == expected.lower()
            if not same:
                raise document.error((i, field), "shadow window does not match the imported manifest")
        block = blocks.get(_block_key(row))
        if block is None:
            raise document.error((i, "anchor_id"), "shadow window is outside the predeclared plan")
        source = row["source_kind"]
        if declared_sources[_block_key(block)] and source not in declared_sources[_block_key(block)]:
            raise document.error((i, "source_kind"), "shadow source does not match the planned block records")
        provenance = [ref for ref in (row.get("evidence_refs_json") or ())
                      if isinstance(ref, dict) and ref.get("kind") == "shadow_window"]
        if len(provenance) > 1 or (provenance and provenance[0].get("comparison") != comparison):
            raise document.error((i, "evidence_refs_json"), "shadow provenance or comparison does not match")
        origins = _authorized_origins(bundle, block, source)
        registered = origins[0] if len(origins) == 1 else None
        if provenance and provenance[0].get("run_id") is not None:
            origins = [item for item in origins if item.record["run_id"] == provenance[0]["run_id"]]
        origin = origins[0] if len(origins) == 1 else None
        basis = origin or registered
        scope = None if basis is None else frozen_scope(study_id, basis.record)
        key = tuple(manifest[name] if name in {"config_sha256", "plan_sha256"} else row[name] for name in WINDOW_KEYS)
        if key in output:
            raise document.error((i, "anchor_id"), "duplicate normalized shadow window key")
        original = None if origin is None or audits is None else _receipts(origin)
        baseline, inflight = (
            _parse_signal(row, name, provenance=provenance[0] if provenance else None, timeline=audits, key=key,
                          comparison=comparison, origin=origin, original=original, anchor=block["anchor_at_ns"],
                          observation_end_ns=observation_end_ns, diversion=diversion)
            for name in ("baseline", "inflight")
        )
        output[key] = ShadowWindowInput(key, block["anchor_at_ns"], scope, comparison, baseline, inflight,
                                        None if basis is None else basis.record["run_id"],
                                        None if origin is not None else "shadow_origin_unverified")
    for block_key, block in blocks.items():
        sources = declared_sources[block_key] or ({expected_source} if expected_source is not None else set())
        for source in sorted(sources):
            key = (study_id, manifest["config_sha256"], manifest["plan_sha256"], source, block["study_split"],
                   block["condition_id"], block["pair_id"], block["anchor_id"])
            if key not in output:
                origins = _authorized_origins(bundle, block, source)
                origin = origins[0] if len(origins) == 1 else None
                scope = None if origin is None else frozen_scope(study_id, origin.record)
                signal = SignalEvidence(None, None, reason="missing_shadow_window")
                output[key] = ShadowWindowInput(key, block["anchor_at_ns"], scope, comparison, signal, signal,
                                                None if origin is None else origin.record["run_id"],
                                                "missing_shadow_window")
    return tuple(output[key] for key in sorted(output))
