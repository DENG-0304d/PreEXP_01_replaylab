"""Replay both signals over one gateway ledger without changing its decisions."""

from dataclasses import dataclass, field, replace
from hashlib import sha256
from pathlib import Path

from .artifacts import ImportedBundle, json_text
from .contracts import TEXT, UINT, WINDOW_KEYS, ArtifactKind, LocatedDocument, check, field_name
from .diversion import DiversionConfig, evaluate_b_gate, stable_selected
from .inflight import ClockBounds, InflightConfig, InflightThresholds, InflightTracker
from .ledger import EvidenceGap, GatewayCursor, GatewayLedger
from .progress import AlertThresholds, Criterion, FrozenHealthProfile, ProgressBlocked, ProgressConfig, ProgressTracker, _required
from .reporting import write_report
from .router import GatewayRouter, RouterConfig, RoutingRequest
from .statistics import frozen_scope
from .windows import COMPARISONS


@dataclass(frozen=True)
class ShadowConfig:
    comparison: str
    observation_end_ns: int
    progress: ProgressConfig
    router: RouterConfig
    diversion: DiversionConfig
    inflight: InflightConfig
    clock_bounds: ClockBounds | None
    policy_ids: tuple[str, str]
    thresholds: tuple[AlertThresholds, AlertThresholds]
    inflight_thresholds: InflightThresholds | None = None
    calibration_run_ids: tuple[str, ...] = ()
    budget: dict | None = None

    @classmethod
    def from_study(cls, study: LocatedDocument, *, comparison: str = "same_threshold_or") -> "ShadowConfig":
        if comparison not in COMPARISONS:
            raise ValueError("unknown comparison mode")
        progress = ProgressConfig.from_study(study)
        raw = study.data["observation"]
        clock = (ClockBounds.from_document(study, ("observation", "clock_bounds"))
                 if raw.get("clock_bounds") is not None else None)
        policies = ("shared_public", "shared_public_plus_inflight")
        thresholds = (progress.alert_thresholds, progress.alert_thresholds)
        device, calibration, budget = None, (), None
        if comparison == "same_health_budget":
            key = ("observation", "shadow_same_budget")
            policies = tuple(_required(study, (*key, name), TEXT)
                             for name in ("baseline_policy_id", "inflight_policy_id"))
            if policies[0] == policies[1]:
                raise study.error((*key, "inflight_policy_id"), "health-budget strategies need distinct frozen IDs")
            split = _required(study, (*key, "validation_split"), TEXT)
            if split != "validation":
                raise study.error((*key, "validation_split"), "expected independent validation split")
            calibration = tuple(_required(study, (*key, "calibration_run_ids"), [TEXT]))
            if not calibration or len(set(calibration)) != len(calibration):
                raise study.error((*key, "calibration_run_ids"), "expected distinct calibration run IDs")
            budget = dict(_required(study, (*key, "budget"), dict))
            if not budget:
                raise ProgressBlocked([field_name((*key, "budget"))])
            thresholds = tuple(AlertThresholds.from_study(study, policy_id=policy) for policy in policies)
            device = InflightThresholds.from_study(study, policy_id=policies[1])
        return cls(comparison, _required(study, ("replay", "observation_end_ns"), UINT), progress,
                   RouterConfig.from_study(study), DiversionConfig.from_study(study),
                   InflightConfig.from_study(study), clock, policies, thresholds, device, calibration, budget)


@dataclass(frozen=True)
class ShadowWindow:
    study_id: str
    config_sha256: str
    plan_sha256: str
    source_kind: str
    study_split: str
    condition_id: str
    pair_id: str
    anchor_id: str
    anchor_at_ns: int
    intervention_digest: str | None
    diversion_rule_id: str | None
    anchor_definition: tuple[str, ...] = ()

    @property
    def key(self) -> tuple:
        return tuple(getattr(self, name) for name in WINDOW_KEYS)


@dataclass(frozen=True)
class CommonTrace:
    run_id: str
    events: LocatedDocument
    prompts: dict[str, int]
    health: FrozenHealthProfile
    healthy_intervals: tuple[tuple[int, int], ...] | None = None
    clock_bounds: ClockBounds | None = None
    device_verified: bool = True
    gaps: tuple[EvidenceGap, ...] = ()


@dataclass
class SignalResult:
    alert_cursor: GatewayCursor | None = None
    evidence_event_ids: tuple[str, ...] = ()
    eligible_n: int = 0
    blocked_n: int = 0
    candidate_n: int = 0
    healthy_trigger_n: int | None = None
    audit_ids: list[str] = field(default_factory=list)
    previous_state: str = "UNKNOWN"
    uncertain: bool = False

    @property
    def alert_usable_ns(self) -> int | None:
        return None if self.alert_cursor is None else self.alert_cursor.gw_at_ns

    @property
    def intended_arm_ns(self) -> int | None:
        return self.alert_usable_ns


@dataclass(frozen=True)
class ShadowResult:
    comparison: str
    baseline: SignalResult
    inflight: SignalResult
    timeline: tuple[dict, ...]
    lead_ns: int | None
    lead_reason: str | None
    reason: str | None


def _hints(payload: dict) -> dict:
    value = payload["cache_hint_tokens"]
    return {replica: value.get(replica) if hasattr(value, "get") else value for replica in ("A", "B")}


class _OriginalRoute:
    def __init__(self, router: GatewayRouter, prompts: dict[str, int]) -> None:
        self.router = router
        self.prompts = prompts
        self.arrivals = {}
        self.pending = {}
        self.wire_refs = {}
        self.unverified = set()

    def quote(self, event):
        key = event.payload["request_id"]
        arrival = self.arrivals.get(key)
        if arrival is None or key not in self.prompts:
            return None
        return self.router.quote(RoutingRequest(key, arrival.cursor.gw_at_ns, self.prompts[key]),
                                 cursor=event.cursor, cache_hints=_hints(event.payload))

    def consume(self, update, quote) -> None:
        event = update.event
        if event is None:
            return
        payload, cursor = event.payload, event.cursor
        if update.epoch_changed:
            self.unverified.discard(payload["replica_id"])
        if event.kind == "arrival":
            self.arrivals[payload["request_id"]] = event
        elif event.kind == "dispatch_decision":
            target = payload["actual_target"]
            if target is not None:
                if quote is None or quote.replicas[target].state is None or not quote.replicas[target].state.payload["ready"]:
                    self.unverified.add(target)
                else:
                    reserved = self.router.reserve(quote, target=target, cursor=cursor)
                    self.pending[payload["request_id"]] = reserved.reservation_id
        elif event.kind == "dispatch_sent":
            local = self.pending.get(payload["request_id"])
            reservation = self.router.reservation(local) if local else None
            if reservation is None or reservation.target != payload["target"]:
                self.unverified.add(payload["target"])
            else:
                self.router.mark_sent(local, cursor=cursor)
                self.wire_refs[(payload["request_id"], payload["reservation_id"])] = local
        elif event.kind in {"dispatch_ack", "reservation_release"}:
            local = self.wire_refs.get((payload["request_id"], payload["reservation_id"]))
            if local is not None:
                event = replace(event, payload={**payload, "reservation_id": local})
                update = replace(update, event=event)
        self.router.consume(update)


def _merge(a, b) -> str:
    return "TRIGGER" if "TRIGGER" in (a, b) else "CLEAR" if "CLEAR" in (a, b) else "UNKNOWN"


def _candidate(event, quote, original: _OriginalRoute, window: ShadowWindow, config: ShadowConfig, armed: bool):
    payload = event.payload
    key = payload["request_id"]
    refs = [event.event_id]
    arrival = original.arrivals.get(key)
    if arrival is not None:
        refs.append(arrival.event_id)
    if not armed or payload["default_target"] != "A" or config.diversion.max_redirects == 0:
        return "ineligible", "inactive_or_default_target", refs, {}
    if quote is None or arrival is None:
        return "unknown", "request_evidence_missing", refs, {}
    if arrival.cursor.gw_at_ns < window.anchor_at_ns:
        return "ineligible", "pre_anchor", refs, {}
    if not stable_selected(config.diversion.rule_seed, window.anchor_id, key, config.diversion.ratio_ppm):
        return "not_selected", "not_selected", refs, {}
    if "B" in original.unverified:
        return "unknown", "original_b_reservations_unverified", refs, {}
    gate = evaluate_b_gate(quote, config.diversion)
    b = quote.replicas["B"]
    details = {}
    if b.state is not None:
        refs.append(b.state.event_id)
        details = dict(unacked_requests=b.n - b.state.payload["unfinished_requests"],
                       unacked_prefill_tokens=b.p - b.state.payload["pending_prefill_tokens"])
    return gate.state, gate.reason, refs, details


def replay_shadow(trace: CommonTrace, *, window: ShadowWindow, config: ShadowConfig) -> ShadowResult:
    clocks = trace.clock_bounds if window.source_kind == "real" else trace.clock_bounds or config.clock_bounds
    if window.source_kind == "synthetic" and clocks is None:
        raise ProgressBlocked(["observation.clock_bounds"])
    progress = ProgressTracker(config.progress, trace.health)
    inflight = InflightTracker(config.inflight, trace.health, clock_bounds=clocks, device_verified=trace.device_verified)
    original = _OriginalRoute(GatewayRouter(config.router, progress), trace.prompts)
    ledger = GatewayLedger(trace.run_id)
    results = [SignalResult(healthy_trigger_n=0 if trace.healthy_intervals is not None else None) for _ in range(2)]
    names = ("baseline", "inflight")
    timeline = []
    identity = sha256(json_text([*window.key, config.comparison]).encode()).hexdigest()[:16]

    def append(cursor, kind, name, **data):
        audit_id = f"{identity}-{len(timeline) + 1}"
        timeline.append(dict(audit_id=audit_id, run_id=trace.run_id, origin="shadow", window_key=list(window.key),
                             comparison=config.comparison, signal_group=name, gw_at_ns=cursor.gw_at_ns,
                             gw_seq=cursor.gw_seq, audit_seq=len(timeline) + 1, type=kind, **data))
        return audit_id

    anchor_seen = False
    epochs = {}
    ticks = []
    for index, row in enumerate(trace.events.data):
        update = ledger.apply(row, location=trace.events.location((index,)))
        cursor = update.cursor
        event = update.event
        at = cursor.gw_at_ns
        timeline.append({**row, "origin": "run", "window_key": list(window.key), "evidence_event_ids": [row["event_id"]]})
        if update.epoch_changed:
            epochs[event.payload["replica_id"]] = cursor.gw_seq
        with original.router.transaction(cursor):
            original.router.expire(cursor=cursor)
            is_dispatch = event is not None and event.kind == "dispatch_decision"
            is_tick = event is not None and event.kind == "tick"
            if is_tick:
                ticks.append(event)
                anchor_seen |= at == window.anchor_at_ns
            quote = original.quote(event) if is_dispatch else None
            if (is_dispatch or is_tick) and at < config.observation_end_ns:
                context = dict(now_ns=at, before_gw_seq=cursor.gw_seq)
                public = [progress.view("A", **context, thresholds=limits, retrospective_gaps=trace.gaps)
                          for limits in config.thresholds]
                device = inflight.view("A", **context, thresholds=config.inflight_thresholds)
                if any(gap.replica_id in (None, "A") and "completion" in gap.streams
                       and max(gap.first_gw_seq, epochs.get("A", -1) + 1) <= min(gap.last_gw_seq, cursor.gw_seq - 1)
                       for gap in trace.gaps):
                    device = Criterion("UNKNOWN", "completion_history_gap", device.evidence)
                for i, (name, result) in enumerate(zip(names, results)):
                    readings = [public[i].residual, public[i].gap]
                    if i and trace.device_verified:
                        readings.append(device)
                    state = public[i].state if len(readings) == 2 else _merge(public[i].state, device.state)
                    relevant = [item for item in readings if item.state == state]
                    evidence = {event.cursor.gw_seq: event.event_id for item in relevant for event in item.evidence}
                    ids = tuple(evidence[key] for key in sorted(evidence))
                    if (result.healthy_trigger_n is not None and state == "TRIGGER" and result.previous_state != state
                            and any(begin <= at < end for begin, end in trace.healthy_intervals)):
                        result.healthy_trigger_n += 1
                    result.previous_state = state
                    if anchor_seen and state == "UNKNOWN" and result.alert_cursor is None:
                        result.uncertain = True
                    if anchor_seen and state == "TRIGGER" and result.alert_cursor is None:
                        result.alert_cursor, result.evidence_event_ids = cursor, ids
                    append(cursor, "signal", name, state=state, reason=[item.reason for item in readings],
                           evidence_event_ids=list(ids), policy_id=config.policy_ids[i])
                    if not is_dispatch or not anchor_seen:
                        continue
                    arm = result.intended_arm_ns
                    if arm is not None and at >= arm + config.diversion.duration_ns:
                        continue
                    candidate_state, reason, refs, extra = _candidate(event, quote, original, window, config, arm is not None)
                    result.eligible_n += int(candidate_state in {"pass", "deny", "not_selected", "unknown"})
                    result.blocked_n += int(candidate_state in {"deny", "unknown"})
                    result.candidate_n += int(candidate_state == "pass")
                    result.uncertain |= candidate_state == "unknown"
                    result.audit_ids.append(append(
                        cursor, "candidate", name, state=candidate_state, reason=reason,
                        request_id=event.payload["request_id"], intended_arm_ns=arm,
                        evidence_event_ids=refs, **extra,
                    ))
            original.consume(update, quote)
        public_update = progress.consume(update)
        device_update = inflight.consume(update, public_update)
        if row["type"] in {"step_start", "inflight_snapshot", "step_complete"} and device_update.reason:
            append(cursor, "inflight_receipt", "inflight", state=row["payload"].get("status"),
                   step_id=row["payload"].get("step_id"), reason=device_update.reason,
                   evidence_event_ids=[row["event_id"]])
    # A final, unarmed window may use its whole observation range. Once armed, earlier
    # unarmed candidate rows are excluded from the certificate for that action window.
    for name, result in zip(names, results):
        arm = result.intended_arm_ns
        active_audits = {row["audit_id"] for row in timeline if row["type"] == "candidate"
                         and row["signal_group"] == name and row["intended_arm_ns"] == arm}
        result.audit_ids[:] = [audit_id for audit_id in result.audit_ids if audit_id in active_audits]
        start = window.anchor_at_ns if arm is None else arm
        stop = config.observation_end_ns if arm is None else min(config.observation_end_ns, arm + config.diversion.duration_ns)
        closing = next((tick for tick in ticks if tick.cursor.gw_at_ns >= stop), None)
        if anchor_seen and not result.uncertain and not trace.gaps and closing is not None and start < stop:
            result.audit_ids.append(append(closing.cursor, "coverage", name, state="complete",
                                           intended_arm_ns=arm, coverage_start_ns=start, coverage_end_ns=stop,
                                           evidence_event_ids=[closing.event_id]))
    first, second = (result.alert_usable_ns for result in results)
    lead = None
    cause = "signal_not_observed"
    if not trace.device_verified:
        cause = "device_timing_unverified"
    elif first is not None and second is not None:
        difference = first - second
        if difference == 0:
            lead, cause = 0, None
        elif clocks is None or abs(difference) <= clocks.lead_error_ns:
            cause = "clock_lead_uncertain"
        else:
            lead, cause = difference, None
    return ShadowResult(config.comparison, *results, tuple(timeline), lead, cause,
                        None if anchor_seen else "anchor_order_unknown")


@dataclass(frozen=True)
class BranchTrace:
    record: dict
    events: list[dict]


@dataclass(frozen=True)
class ObservedActionFact:
    run_id: str
    source_kind: str
    study_split: str
    condition_id: str
    anchor_id: str
    anchor_at_ns: int
    anchor_definition: tuple[str, ...]
    intervention_digest: str | None
    diversion_rule_id: str | None
    run_status: str
    role: str
    planned_delay_ns: int | None
    actual_arm_ns: int | None
    first_redirect_ns: int | None


def inspect_action_branch(branch: BranchTrace) -> ObservedActionFact:
    rec = branch.record
    ledger = GatewayLedger(rec["run_id"])
    arms = []
    for row in branch.events:
        accepted = ledger.apply(row).event
        if accepted is not None and accepted.kind == "rule_armed" and accepted.payload["rule_id"] == rec.get("diversion_rule_id"):
            arms.append(accepted)
    arm = arms[0] if len(arms) == 1 else None
    first = min((item.sent.cursor.gw_at_ns for item in ledger.view().dispatches.values()
                 if arm is not None and item.sent is not None and item.decision is not None
                 and arm.cursor.gw_seq < item.decision.cursor.gw_seq < item.sent.cursor.gw_seq
                 and item.decision.payload["default_target"] == "A" and item.decision.payload["actual_target"] == "B"
                 and item.sent.payload["target"] == "B"), default=None)
    scope = frozen_scope("unused", rec)
    definition = scope.anchor_definition if scope is not None else ((rec["anchor_definition_id"],)
                                                                    if rec.get("anchor_definition_id") else ())
    return ObservedActionFact(
        *(rec[name] for name in ("run_id", "source_kind", "study_split", "condition_id", "anchor_id", "anchor_at_ns")),
        definition, rec.get("intervention_digest"), rec.get("diversion_rule_id"), rec["run_status"],
        rec["role"], rec.get("planned_delay_ns"), None if arm is None else arm.cursor.gw_at_ns, first,
    )


@dataclass(frozen=True)
class CandidateCheck:
    eligible: bool
    reason: str


def check_action_candidate(window: ShadowWindow, *, intended_arm_ns: int, candidate: ObservedActionFact,
                           arm_jitter_ns: int) -> CandidateCheck:
    checks = [
        (candidate.run_status == "complete", "run_not_complete"),
        (candidate.source_kind == window.source_kind, "source_mismatch"),
        (candidate.study_split == window.study_split, "split_mismatch"),
        (candidate.condition_id == window.condition_id, "condition_mismatch"),
        ((candidate.anchor_id, candidate.anchor_at_ns, candidate.anchor_definition)
         == (window.anchor_id, window.anchor_at_ns, window.anchor_definition), "anchor_mismatch"),
        (candidate.intervention_digest == window.intervention_digest, "frozen_condition_mismatch"),
        (candidate.diversion_rule_id == window.diversion_rule_id, "action_mismatch"),
        (candidate.role in {"immediate", "delayed"} and candidate.planned_delay_ns == intended_arm_ns - window.anchor_at_ns,
         "planned_delay_mismatch"),
        (candidate.actual_arm_ns is not None, "actual_arm_missing"),
    ]
    for passed, reason in checks:
        if not passed:
            return CandidateCheck(False, reason)
    if abs(candidate.actual_arm_ns - intended_arm_ns) > arm_jitter_ns:
        return CandidateCheck(False, "arm_jitter_exceeded")
    return CandidateCheck(True, "preflight_pass")


def _block_key(row: dict) -> tuple:
    return tuple(row.get(name) for name in ("condition_id", "pair_id", "anchor_id", "study_split"))


def _intervals(record: dict) -> tuple[tuple[int, int], ...] | None:
    rows = record.get("healthy_intervals")
    if not isinstance(rows, list):
        return None
    if any(not isinstance(item, list) or len(item) != 2 or not all(UINT.accepts(t) for t in item)
           or item[0] >= item[1] for item in rows):
        return None
    return tuple(tuple(item) for item in rows)


def write_shadow_reports(bundle: ImportedBundle, study: LocatedDocument, *, out_dir: Path,
                         comparison: str = "same_threshold_or") -> dict:
    config = ShadowConfig.from_study(study, comparison=comparison)
    manifest = bundle.manifest.data
    if config.calibration_run_ids:
        calibration = [run.record for run in bundle.runs if run.record["run_id"] in config.calibration_run_ids
                       and run.record["study_split"] == "calibration"]
        attempts = {item["independent_start"]["attempt_id"] for item in calibration}
        validation_attempts = {run.record["independent_start"]["attempt_id"] for run in bundle.runs
                               if run.record["study_split"] == "validation"}
        if len(attempts) != len(config.calibration_run_ids) or attempts & validation_attempts:
            raise ProgressBlocked(["observation.shadow_same_budget.calibration_run_ids"])
    rows, timeline, reports = [], [], []
    for block in bundle.plan.data["blocks"]:
        role = {"action": "reference", "shadow_only": "shadow"}.get(block["kind"])
        if role is None:
            continue
        matching = [item for item in bundle.runs if _block_key(item.record) == _block_key(block)]
        sources = {item.record["source_kind"] for item in matching}
        sources.update(item["source_kind"] for item in manifest["runs"] if isinstance(item, dict)
                       and _block_key(item) == _block_key(block) and item.get("source_kind") in ("synthetic", "real"))
        for source in sorted(sources or {study.data.get("source_kind") or "synthetic"}):
            references = [item for item in matching if item.record["source_kind"] == source and item.record["role"] == role]
            item = references[0] if len(references) == 1 else None
            rec = item.record if item else {}
            scope = frozen_scope(manifest["study_id"], rec)
            window = ShadowWindow(
                manifest["study_id"], manifest["config_sha256"], manifest["plan_sha256"], source, block["study_split"],
                block["condition_id"], block["pair_id"], block["anchor_id"], block["anchor_at_ns"],
                rec.get("intervention_digest"), rec.get("diversion_rule_id"), scope.anchor_definition if scope else (),
            )
            row = dict(zip(WINDOW_KEYS, window.key))
            eligibility = config.comparison == "same_threshold_or" or block["study_split"] == "validation"
            if item is None or not eligibility:
                row["reason"] = ("validation_split_required" if not eligibility else
                                 "ambiguous_reference_trace" if references else "missing_reference_trace")
                reports.append(dict(window_key=window.key, reason=row["reason"]))
            else:
                clock = None
                if source == "real" and isinstance(rec.get("clock_bounds"), dict):
                    i = next(i for i, run in enumerate(manifest["runs"]) if run.get("run_id") == rec["run_id"])
                    clock = ClockBounds.from_document(bundle.manifest, ("runs", i, "clock_bounds"))
                device_verified = (source == "synthetic" or rec.get("device_capability") == "device_verified")
                trace = CommonTrace(
                    rec["run_id"], item.artifacts[ArtifactKind.EVENTS],
                    {req["request_id"]: req["prompt_tokens"] for req in item.artifacts[ArtifactKind.ARRIVALS].data},
                    FrozenHealthProfile.from_document(item.artifacts[ArtifactKind.HEALTH_PROFILE]),
                    _intervals(rec), clock, device_verified,
                )
                result = replay_shadow(trace, window=window, config=config)
                provenance = dict(kind="shadow_window", run_id=trace.run_id, comparison=comparison)
                for name, signal in (("baseline", result.baseline), ("inflight", result.inflight)):
                    row.update({f"{name}_{field}": getattr(signal, field) for field in
                                ("alert_usable_ns", "intended_arm_ns", "eligible_n", "blocked_n")})
                    provenance[f"{name}_audit_ids"] = signal.audit_ids
                row.update(evidence_refs_json=[provenance], reason=result.reason)
                timeline.extend(result.timeline)
                reports.append(dict(window_key=window.key, run_id=trace.run_id, lead_ns=result.lead_ns,
                                    lead_reason=result.lead_reason, reason=result.reason,
                                    healthy_trigger_n=dict(baseline=result.baseline.healthy_trigger_n,
                                                           inflight=result.inflight.healthy_trigger_n)))
            rows.append(row)
    actions = [inspect_action_branch(BranchTrace(run.record, run.artifacts[ArtifactKind.EVENTS].data))
               for run in bundle.runs if run.record["role"] in {"immediate", "delayed"}]
    write_report(out_dir, ArtifactKind.WINDOWS.value, rows)
    write_report(out_dir, ArtifactKind.TIMELINE.value, timeline)
    return dict(comparison=comparison, policy_ids=config.policy_ids, calibration_run_ids=config.calibration_run_ids,
                budget=config.budget, window_n=len(rows), windows=reports,
                observed_actions=[dict(run_id=fact.run_id, actual_arm_ns=fact.actual_arm_ns,
                                       first_redirect_ns=fact.first_redirect_ns) for fact in actions])
