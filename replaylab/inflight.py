"""Device attestations visible in gateway order, separate from public progress."""

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType

from .contracts import POS_INT, REPLICA, UINT, Key, LocatedDocument, check
from .ledger import GatewayCursor, GatewayEvent, LedgerUpdate
from .progress import Bucket, FrozenHealthProfile, ProgressUpdate, State, _required


_Step = tuple[str, int, str]
_MARKER_REASONS = {
    "not_recorded": "marker_not_recorded", "wrong_epoch": "marker_epoch_mismatch",
    "coverage_unknown": "coverage_unknown", "query_error": "query_error",
}


@dataclass(frozen=True)
class ClockBounds:
    """Absolute error bounds for each comparison, in nanoseconds."""

    age_error_ns: int
    duration_error_ns: int
    lead_error_ns: int

    def __post_init__(self) -> None:
        if not all(UINT.accepts(value) for value in (self.age_error_ns, self.duration_error_ns, self.lead_error_ns)):
            raise ValueError("clock bounds must be nonnegative integers in nanoseconds")

    @classmethod
    def from_document(cls, doc: LocatedDocument, key: Key) -> "ClockBounds":
        return cls(*(_required(doc, (*key, name), UINT) for name in
                     ("age_error_ns", "duration_error_ns", "lead_error_ns")))


@dataclass(frozen=True)
class InflightConfig:
    evidence_ttl_ns: int

    @classmethod
    def from_study(cls, study: LocatedDocument) -> "InflightConfig":
        return cls(_required(study, ("observation", "evidence_ttl_ns"), POS_INT))


@dataclass(frozen=True)
class InflightThresholds:
    limits: Mapping[Bucket, int]

    def __post_init__(self) -> None:
        object.__setattr__(self, "limits", MappingProxyType(dict(self.limits)))

    @classmethod
    def from_study(cls, study: LocatedDocument, *, policy_id: str) -> "InflightThresholds":
        root = ("observation", "alert_policies", policy_id, "inflight_limits")
        limits = {}
        for replica, rows in _required(study, root, dict).items():
            check(study, replica, REPLICA, (*root, replica))
            check(study, rows, dict, (*root, replica))
            for work_key in rows:
                limits[replica, work_key] = _required(study, (*root, replica, work_key), POS_INT)
        return cls(limits)


@dataclass(frozen=True)
class InflightUpdate:
    receipt: GatewayEvent | None
    reason: str | None = None


@dataclass(frozen=True)
class IterationView:
    step_id: str
    snapshot: GatewayEvent | None
    state: State
    reason: str
    evidence: tuple[GatewayEvent, ...] = ()

    @property
    def raw_status(self) -> str | None:
        return None if self.snapshot is None else self.snapshot.payload["status"]


@dataclass(frozen=True)
class InflightView:
    state: State
    iterations: tuple[IterationView, ...]
    evidence: tuple[GatewayEvent, ...]
    reason: str


@dataclass
class _Iteration:
    starts: dict[str, GatewayEvent] = field(default_factory=dict)
    snapshot: GatewayEvent | None = None
    ambiguous: bool = False
    seal: GatewayEvent | None = None
    waiting_done: list[GatewayEvent] = field(default_factory=list)


def _key(event: GatewayEvent) -> _Step:
    payload = event.payload
    return payload["replica_id"], payload["epoch"], payload["step_id"]


def _marker_reason(event: GatewayEvent) -> str | None:
    p = event.payload
    if p["status"] == "UNVERIFIED":
        return p["validation_reason"] or _MARKER_REASONS.get(p["marker_status"], "unverified")
    if p["marker_status"] != "recorded":
        return _MARKER_REASONS.get(p["marker_status"], "marker_unverified")
    return None


class InflightTracker:
    """Advance through one ledger prefix. A verified device DONE never updates the EMA."""

    def __init__(
        self, config: InflightConfig, health_profile: FrozenHealthProfile, *,
        clock_bounds: ClockBounds | None, device_verified: bool = True,
    ) -> None:
        self.config = config
        self.health = health_profile
        self.clock_bounds = clock_bounds
        self.device_verified = device_verified
        self._cursor: GatewayCursor | None = None
        self._iterations: dict[_Step, _Iteration] = {}
        self._work: dict[tuple[str, int, int], GatewayEvent] = {}
        self._work_conflicts: set[tuple[str, int, int]] = set()

    def consume(self, update: LedgerUpdate, public_progress: ProgressUpdate) -> InflightUpdate:
        cursor = update.cursor
        prior = self._cursor
        if public_progress.cursor != cursor or (prior is not None and (
            cursor.run_id != prior.run_id or cursor.gw_at_ns < prior.gw_at_ns
            or cursor.gw_seq < prior.gw_seq or (cursor.gw_seq == prior.gw_seq and update.gap is None)
        )):
            raise ValueError("inflight must consume its public progress update in gateway order")
        self._cursor = cursor
        event = update.event
        if event is None:
            return InflightUpdate(update.receipt, update.reason)
        if update.epoch_changed:
            replica = event.payload["replica_id"]
            self._iterations = {key: value for key, value in self._iterations.items() if key[0] != replica}
            self._work = {key: value for key, value in self._work.items() if key[0] != replica}
            self._work_conflicts = {key for key in self._work_conflicts if key[0] != replica}
        elif event.kind == "work_state":
            payload = event.payload
            if payload["sampled_ns"] <= cursor.gw_at_ns:
                key = payload["replica_id"], payload["epoch"], payload["work_version"]
                first = self._work.setdefault(key, event)
                if first.payload["work_key"] != payload["work_key"]:
                    self._work_conflicts.add(key)
        elif public_progress.completed_step is not None:
            iteration = self._iterations.setdefault(public_progress.completed_step, _Iteration())
            if iteration.seal is None:
                iteration.seal = event
                iteration.waiting_done.clear()
            return InflightUpdate(event, "public_completion")
        elif event.kind == "step_start":
            iteration = self._iterations.setdefault(_key(event), _Iteration())
            iteration.starts[event.event_id] = event
            for pending in iteration.waiting_done:
                if pending.payload["start_event_id"] == event.event_id and self._done_matches(iteration, pending):
                    iteration.seal = pending
                    iteration.waiting_done.clear()
                    return InflightUpdate(event, "device_completion_bound")
        elif event.kind == "inflight_snapshot":
            return InflightUpdate(event, self._snapshot(event))
        return InflightUpdate(event)

    @staticmethod
    def _done_matches(iteration: _Iteration, event: GatewayEvent) -> bool:
        start = iteration.starts.get(event.payload["start_event_id"])
        return (start is not None and start.payload["boundary_id"] == event.payload["boundary_id"]
                and event.payload["sample_ns"] >= start.payload["start_ns"])

    def _snapshot(self, event: GatewayEvent) -> str:
        iteration = self._iterations.setdefault(_key(event), _Iteration())
        if iteration.seal is not None:
            return "iteration_sealed"
        p = event.payload
        if self.device_verified and p["status"] == "DONE_VERIFIED" and _marker_reason(event) is None:
            if self._done_matches(iteration, event):
                iteration.snapshot = event
                iteration.seal = event
                iteration.waiting_done.clear()
                return "device_completed"
            if p["start_event_id"] not in iteration.starts:
                iteration.waiting_done.append(event)

        previous = iteration.snapshot
        if previous is not None:
            old = previous.payload
            if p["sample_ns"] != old["sample_ns"]:
                order = p["sample_ns"] - old["sample_ns"]
            elif "source_seq" in p and "source_seq" in old:
                order = p["source_seq"] - old["source_seq"]
            else:
                order = event.event_at_ns - previous.event_at_ns
            if order < 0:
                return "older_snapshot"
            if order == 0:
                if p == old:
                    return "duplicate_snapshot"
                iteration.ambiguous = True
                return "snapshot_order_unknown"
        iteration.snapshot = event
        iteration.ambiguous = False
        return "snapshot_received"

    def _judge(
        self, key: _Step, iteration: _Iteration, now_ns: int, thresholds: InflightThresholds | None,
    ) -> IterationView:
        snap = iteration.snapshot
        view = IterationView(key[2], snap, "UNKNOWN", "missing_snapshot", () if snap is None else (snap,))
        if iteration.seal is not None:
            seal = iteration.seal
            reason = "device_completed" if seal.kind == "inflight_snapshot" else "public_completion"
            return replace(view, state="CLEAR", reason=reason, evidence=(seal,))
        if snap is None:
            return view
        if iteration.ambiguous:
            return replace(view, reason="snapshot_order_unknown")
        p = snap.payload
        marker_reason = _marker_reason(snap)
        if marker_reason is not None:
            return replace(view, reason=marker_reason)
        if not self.device_verified:
            return replace(view, reason="device_unverified")
        start = iteration.starts.get(p["start_event_id"])
        if start is None:
            return replace(view, reason="missing_step_start")
        view = replace(view, evidence=(start, snap))
        if start.payload["boundary_id"] != p["boundary_id"]:
            return replace(view, reason="boundary_mismatch")
        if p["work_version"] is None or start.payload.get("work_version") is None:
            return replace(view, reason="missing_work_binding")
        if start.payload["work_version"] != p["work_version"]:
            return replace(view, reason="work_version_mismatch")
        version = (*key[:2], p["work_version"])
        work = self._work.get(version)
        if version in self._work_conflicts:
            return replace(view, reason="conflicting_work_version")
        if work is None:
            return replace(view, reason="missing_work_version")
        view = replace(view, evidence=(start, work, snap))
        bucket_key = key[0], work.payload["work_key"]
        bucket = self.health.buckets.get(bucket_key)
        if bucket is None:
            return replace(view, reason="missing_health_bucket")
        if self.clock_bounds is None:
            return replace(view, reason="clock_alignment_unknown")
        bounds = self.clock_bounds
        age = now_ns - p["sample_ns"]
        low, high = age - bounds.age_error_ns, age + bounds.age_error_ns
        if low > self.config.evidence_ttl_ns:
            return replace(view, reason="evidence_expired")
        if high < 0:
            return replace(view, reason="sample_in_future")
        if low < 0 or high > self.config.evidence_ttl_ns:
            return replace(view, reason="clock_age_uncertain")
        limit = bucket.inflight_limit_ns if thresholds is None else thresholds.limits.get(bucket_key)
        if limit is None:
            return replace(view, reason="missing_inflight_threshold")
        elapsed = p["sample_ns"] - start.payload["start_ns"]
        low, high = elapsed - bounds.duration_error_ns, elapsed + bounds.duration_error_ns
        if high < 0:
            return replace(view, reason="sample_before_start")
        if low < 0 or low <= limit < high:
            return replace(view, reason="clock_duration_uncertain")
        if low > limit:
            return replace(view, state="TRIGGER", reason="inflight_above_threshold")
        return replace(view, state="CLEAR", reason="inflight_at_or_below_threshold")

    def view(
        self, replica_id: str, *, now_ns: int, before_gw_seq: int | None = None,
        thresholds: InflightThresholds | None = None,
    ) -> InflightView:
        cursor = self._cursor
        if (not UINT.accepts(now_ns) or (cursor is not None and now_ns < cursor.gw_at_ns)
                or (before_gw_seq is not None and (
                    not UINT.accepts(before_gw_seq) or (cursor is not None and before_gw_seq <= cursor.gw_seq)
                ))):
            raise ValueError("inflight judgement must be at or after the consumed prefix")
        cases = tuple(self._judge(key, value, now_ns, thresholds)
                      for key, value in sorted(self._iterations.items()) if key[0] == replica_id)
        trigger = [case for case in cases if case.state == "TRIGGER"]
        active = [case for case in cases if case.reason not in {"device_completed", "public_completion"}]
        clear = [case for case in (active or cases) if case.state == "CLEAR"]
        chosen = trigger or clear or list(active or cases)
        evidence = {ref.cursor.gw_seq: ref for case in chosen for ref in case.evidence}
        state = "TRIGGER" if trigger else "CLEAR" if clear else "UNKNOWN"
        return InflightView(state, cases, tuple(evidence[seq] for seq in sorted(evidence)),
                            chosen[0].reason if chosen else "no_inflight_evidence")
