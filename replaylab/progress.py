"""Shared completion and workload evidence for the progress baseline."""

from bisect import bisect_right
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from decimal import Context, Decimal, ROUND_HALF_EVEN, localcontext
from types import MappingProxyType
from typing import Literal

from .artifacts import json_text
from .contracts import (
    ALPHA, POS_INT, POS_NUMBER, REPLICA, TEXT, UINT, VERSION, ArtifactKind, Key,
    LocatedDocument, Spec, check, field_name, validate_canonical,
)
from .ledger import EvidenceGap, GatewayCursor, GatewayEvent, LedgerUpdate


State = Literal["TRIGGER", "CLEAR", "UNKNOWN"]
Bucket = tuple[str, str]
_EMA_CONTEXT = Context(prec=34, rounding=ROUND_HALF_EVEN)
_COMPLETION_FIELDS = ("boundary_id", "end_ns", "duration_ns", "work_key")


class ProgressBlocked(ValueError):
    def __init__(self, missing_fields: Iterable[str]) -> None:
        self.missing_fields = tuple(sorted(set(missing_fields)))
        super().__init__("missing frozen progress inputs: " + ", ".join(self.missing_fields))


def _required(study: LocatedDocument, key: Key, spec: Spec) -> object:
    value = study.data
    for index, name in enumerate(key):
        if value is None:
            raise ProgressBlocked([field_name(key[:index])])
        check(study, value, dict, key[:index])
        value = value.get(name)
    if value is None:
        raise ProgressBlocked([field_name(key)])
    check(study, value, spec, key)
    return value


def _positive_decimal(study: LocatedDocument, key: Key) -> Decimal:
    return Decimal(str(_required(study, key, POS_NUMBER)))


@dataclass(frozen=True)
class AlertThresholds:
    residual_threshold: Decimal
    gap_limits_ns: Mapping[Bucket, int] = field(default_factory=lambda: MappingProxyType({}))

    @classmethod
    def from_study(cls, study: LocatedDocument, *, policy_id: str | None = None) -> "AlertThresholds":
        """Read alert overrides only. EMA, freshness, and routing remain shared."""
        if policy_id is None:
            return cls(_positive_decimal(study, ("observation", "threshold")))
        prefix = ("observation", "alert_policies", policy_id)
        residual = _positive_decimal(study, (*prefix, "threshold"))
        limits = {}
        for replica, rows in _required(study, (*prefix, "gap_limits"), dict).items():
            key = (*prefix, "gap_limits", replica)
            check(study, replica, REPLICA, key)
            check(study, rows, dict, key)
            for work_key in rows:
                limits[replica, work_key] = _required(study, (*key, work_key), POS_INT)
        return cls(residual, MappingProxyType(limits))


@dataclass(frozen=True)
class BucketLimits:
    gap_limit_ns: int
    inflight_limit_ns: int


@dataclass(frozen=True)
class ProgressConfig:
    work_ttl_ns: int
    completion_ttl_ns: int
    residual_ttl_ns: int
    alpha: Decimal
    speed_factor_min: Decimal
    speed_factor_max: Decimal
    workload_buckets: Mapping[str, tuple[int, ...]]
    calibration_limits: Mapping[Bucket, BucketLimits]
    alert_thresholds: AlertThresholds

    @classmethod
    def from_study(cls, study: LocatedDocument) -> "ProgressConfig":
        check(study, study.data, {"schema_version": VERSION})
        buckets = _required(study, ("observation", "workload_buckets"), dict)
        dimensions = {}
        for name, bounds in sorted(buckets.items()):
            key = ("observation", "workload_buckets", name)
            check(study, name, TEXT, key)
            check(study, bounds, [UINT], key)
            for index, (left, right) in enumerate(zip(bounds, bounds[1:]), 1):
                if left >= right:
                    raise study.error((*key, index), "bucket boundaries must be strictly increasing")
            dimensions[name] = tuple(bounds)

        limits = {}
        prefix = ("observation", "conditional_thresholds")
        for replica, rows in _required(study, prefix, dict).items():
            key = (*prefix, replica)
            check(study, replica, REPLICA, key)
            check(study, rows, dict, key)
            for work_key in rows:
                check(study, work_key, TEXT, (*key, work_key))
                limits[replica, work_key] = BucketLimits(
                    _required(study, (*key, work_key, "gap_limit_ns"), POS_INT),
                    _required(study, (*key, work_key, "inflight_limit_ns"), POS_INT),
                )
        low = _positive_decimal(study, ("router", "speed_factor_min"))
        high = _positive_decimal(study, ("router", "speed_factor_max"))
        if low > high:
            raise study.error(("router", "speed_factor_max"), "must be at least speed_factor_min")
        return cls(
            work_ttl_ns=_required(study, ("observation", "work_ttl_ns"), POS_INT),
            completion_ttl_ns=_required(study, ("observation", "completion_ttl_ns"), POS_INT),
            residual_ttl_ns=_required(study, ("observation", "residual_ttl_ns"), POS_INT),
            alpha=Decimal(str(_required(study, ("observation", "alpha"), ALPHA))),
            speed_factor_min=low,
            speed_factor_max=high,
            workload_buckets=MappingProxyType(dimensions),
            calibration_limits=MappingProxyType(limits),
            alert_thresholds=AlertThresholds.from_study(study),
        )


def bucket_workload(values: Mapping[str, int], config: ProgressConfig) -> str | None:
    """Place a value equal to a boundary in the higher bucket."""
    buckets = []
    for name, bounds in config.workload_buckets.items():
        value = values.get(name)
        if type(value) is not int or value < 0:
            return None
        buckets.append([name, bisect_right(bounds, value)])
    return json_text(buckets)


@dataclass(frozen=True)
class HealthBucket:
    healthy_median_ns: int
    gap_limit_ns: int
    inflight_limit_ns: int
    calibration_count: int


@dataclass(frozen=True)
class FrozenHealthProfile:
    buckets: Mapping[Bucket, HealthBucket]

    def __post_init__(self) -> None:
        object.__setattr__(self, "buckets", MappingProxyType(dict(self.buckets)))

    @classmethod
    def from_document(cls, document: LocatedDocument) -> "FrozenHealthProfile":
        validate_canonical(ArtifactKind.HEALTH_PROFILE, document)
        return cls({
            (row["replica_id"], row["work_key"]): HealthBucket(
                row["healthy_median_ns"], row["gap_limit_ns"], row["inflight_limit_ns"], row["calibration_count"],
            ) for row in document.data
        })

    def to_rows(self) -> list[dict]:
        return [
            dict(replica_id=replica, work_key=work_key, healthy_median_ns=bucket.healthy_median_ns,
                 gap_limit_ns=bucket.gap_limit_ns, inflight_limit_ns=bucket.inflight_limit_ns,
                 calibration_count=bucket.calibration_count)
            for (replica, work_key), bucket in sorted(self.buckets.items())
        ]


@dataclass(frozen=True)
class CalibrationSample:
    replica_id: str
    work_key: str
    duration_ns: int


@dataclass(frozen=True)
class CalibrationRun:
    run_id: str
    attempt_id: str
    study_split: str
    source_kind: str
    healthy: bool
    samples: tuple[CalibrationSample, ...]


def _integer_median(durations: list[int]) -> int:
    durations.sort()
    middle = len(durations) // 2
    if len(durations) % 2:
        return durations[middle]
    whole, remainder = divmod(durations[middle - 1] + durations[middle], 2)
    return whole + (whole % 2 if remainder else 0)


def build_health_profile(runs: Iterable[CalibrationRun], config: ProgressConfig) -> FrozenHealthProfile:
    """Build only from separate healthy calibration attempts of the same source kind."""
    samples: dict[Bucket, list[int]] = {}
    run_ids, attempts, sources = set(), set(), set()
    for run in runs:
        if run.study_split != "calibration" or run.healthy is not True:
            continue
        if not run.run_id or run.run_id in run_ids:
            raise ValueError("expected a distinct nonempty calibration run_id")
        if not run.attempt_id or run.attempt_id in attempts:
            raise ValueError("expected a distinct nonempty calibration attempt_id")
        if run.source_kind not in {"real", "synthetic"}:
            raise ValueError("invalid calibration source_kind")
        sources.add(run.source_kind)
        if len(sources) > 1:
            raise ValueError("calibration runs must share one source_kind")
        run_ids.add(run.run_id)
        attempts.add(run.attempt_id)
        for sample in run.samples:
            if sample.replica_id not in {"A", "B"} or not sample.work_key or not POS_INT.accepts(sample.duration_ns):
                raise ValueError(f"invalid healthy calibration sample in run {run.run_id}")
            samples.setdefault((sample.replica_id, sample.work_key), []).append(sample.duration_ns)
    missing = [
        field_name(("observation", "conditional_thresholds", *key, name))
        for key in samples if key not in config.calibration_limits
        for name in ("gap_limit_ns", "inflight_limit_ns")
    ]
    if missing:
        raise ProgressBlocked(missing)
    return FrozenHealthProfile({
        key: HealthBucket(
            _integer_median(durations), config.calibration_limits[key].gap_limit_ns,
            config.calibration_limits[key].inflight_limit_ns, len(durations),
        ) for key, durations in samples.items()
    })


@dataclass(frozen=True)
class ProgressUpdate:
    cursor: GatewayCursor
    reason: str | None = None
    completed_step: tuple[str, int, str] | None = None


@dataclass(frozen=True)
class Criterion:
    state: State
    reason: str
    evidence: tuple[GatewayEvent, ...] = ()
    known_gaps: tuple[EvidenceGap, ...] = ()


@dataclass(frozen=True)
class PublicProgressView:
    cursor: GatewayCursor | None
    now_ns: int
    before_gw_seq: int | None
    epoch: int | None
    work: GatewayEvent | None
    completion: GatewayEvent | None
    last_used: GatewayEvent | None
    ema: Decimal | None
    residual: Criterion
    gap: Criterion
    state: State

    @property
    def completion_end_ns(self) -> int | None:
        return None if self.completion is None else self.completion.payload["end_ns"]

    @property
    def last_used_end_ns(self) -> int | None:
        return None if self.last_used is None else self.last_used.payload["end_ns"]


@dataclass(frozen=True)
class SpeedFactor:
    value: Decimal
    reason: str
    evidence: tuple[GatewayEvent, ...] = ()
    known_gaps: tuple[EvidenceGap, ...] = ()


@dataclass
class _ReplicaProgress:
    epoch: int | None = None
    epoch_seq: int = -1
    work: GatewayEvent | None = None
    completion: GatewayEvent | None = None
    last_used: GatewayEvent | None = None
    ema: Decimal | None = None
    completions: dict[str, GatewayEvent] = field(default_factory=dict)


class ProgressTracker:
    """Project one ledger prefix into shared evidence, with caller-supplied decision time."""

    def __init__(self, config: ProgressConfig, health_profile: FrozenHealthProfile) -> None:
        self.config = config
        self.health_profile = health_profile
        self._cursor: GatewayCursor | None = None
        self._replicas = {replica: _ReplicaProgress() for replica in ("A", "B")}
        self._gaps: list[EvidenceGap] = []

    def consume(self, update: LedgerUpdate) -> ProgressUpdate:
        cursor = update.cursor
        previous = self._cursor
        if previous is not None and (
            cursor.run_id != previous.run_id or cursor.gw_at_ns < previous.gw_at_ns
            or cursor.gw_seq < previous.gw_seq or (cursor.gw_seq == previous.gw_seq and update.gap is None)
        ):
            raise ValueError("progress updates must follow one ledger in gateway receipt order")

        reason = update.reason
        completed = None
        event = update.public_event
        if update.gap is not None:
            self._gaps.append(update.gap)
        elif event is not None:
            replica = event.payload["replica_id"]
            epoch = event.payload["epoch"]
            state = self._replicas[replica]
            if update.epoch_changed:
                self._replicas[replica] = _ReplicaProgress(epoch=epoch, epoch_seq=cursor.gw_seq)
                reason = "epoch_changed"
            elif state.epoch != epoch:
                raise ValueError("progress consumption must start before the replica_epoch declaration")
            elif event.kind == "work_state":
                reason = self._work(state, event)
            elif event.kind == "step_complete":
                reason, completed = self._complete(state, event)
        self._cursor = cursor
        return ProgressUpdate(cursor, reason, completed)

    @staticmethod
    def _work(state: _ReplicaProgress, event: GatewayEvent) -> str:
        sampled = event.payload["sampled_ns"]
        if sampled > event.cursor.gw_at_ns:
            return "work_sample_in_future"
        if state.work is not None:
            old = state.work.payload
            version = event.payload["work_version"]
            if version < old["work_version"]:
                return "older_work_version"
            if version == old["work_version"] and sampled < old["sampled_ns"]:
                return "older_work_sample"
            if event.payload == old:
                return "duplicate_work_snapshot"
        state.work = event
        return "work_updated"

    def _complete(
        self, state: _ReplicaProgress, event: GatewayEvent,
    ) -> tuple[str, tuple[str, int, str] | None]:
        payload = event.payload
        step = payload["step_id"]
        previous = state.completions.get(step)
        if previous is not None:
            for name in _COMPLETION_FIELDS:
                if payload[name] != previous.payload[name]:
                    raise event.error(f"payload.{name}", f"conflicting completion for step_id {step!r}")
            return "duplicate_completion", None

        end_ns = payload["end_ns"]
        if end_ns > event.cursor.gw_at_ns:
            return "completion_end_in_future", None
        state.completions[step] = event
        completed = payload["replica_id"], payload["epoch"], step
        if state.completion is not None and end_ns < state.completion.payload["end_ns"]:
            return "older_completion", completed
        state.completion = event
        duration = payload["duration_ns"]
        if duration <= 0:
            return "invalid_duration", completed
        bucket = self.health_profile.buckets.get((payload["replica_id"], payload["work_key"]))
        if bucket is None:
            return "missing_health_bucket", completed
        with localcontext(_EMA_CONTEXT):
            residual = Decimal(duration) / Decimal(bucket.healthy_median_ns)
            alpha = self.config.alpha
            state.ema = residual if state.ema is None else alpha * residual + (1 - alpha) * state.ema
        state.last_used = event
        return "ema_updated", completed

    def _available_gaps(
        self, replica_id: str, now_ns: int, before_gw_seq: int | None,
        retrospective_gaps: Iterable[EvidenceGap],
    ) -> tuple[EvidenceGap, ...]:
        cursor = self._cursor
        if not UINT.accepts(now_ns) or (cursor is not None and now_ns < cursor.gw_at_ns):
            raise ValueError("now_ns must be an integer at or after the current gateway time")
        seq = -1 if cursor is None else cursor.gw_seq
        if before_gw_seq is not None:
            if not UINT.accepts(before_gw_seq) or before_gw_seq <= seq:
                raise ValueError("before_gw_seq must follow the consumed gateway prefix")
            seq = before_gw_seq - 1
        if cursor is None:
            return ()
        state = self._replicas[replica_id]
        known = (gap for gap in self._gaps if gap.known_at.gw_seq <= seq and gap.known_at.gw_at_ns <= now_ns)
        return tuple(dict.fromkeys(
            gap for gap in (*known, *retrospective_gaps)
            if gap.known_at.run_id == cursor.run_id and gap.replica_id in (None, replica_id)
            and max(gap.first_gw_seq, state.epoch_seq + 1) <= min(gap.last_gw_seq, seq)
        ))

    def _residual(
        self, state: _ReplicaProgress, now_ns: int, threshold: Decimal, gaps: tuple[EvidenceGap, ...],
    ) -> Criterion:
        evidence = () if state.last_used is None else (state.last_used,)
        missing = tuple(gap for gap in gaps if "completion" in gap.streams)
        if missing:
            return Criterion("UNKNOWN", "completion_history_gap", evidence, missing)
        if state.last_used is None:
            return Criterion("UNKNOWN", "no_residual")
        age = now_ns - state.last_used.payload["end_ns"]
        if not 0 <= age <= self.config.residual_ttl_ns:
            return Criterion("UNKNOWN", "residual_expired", evidence)
        if state.ema > threshold:
            return Criterion("TRIGGER", "residual_above_threshold", evidence)
        return Criterion("CLEAR", "residual_at_or_below_threshold", evidence)

    def _gap(
        self, replica: str, state: _ReplicaProgress, now_ns: int, thresholds: AlertThresholds,
        gaps: tuple[EvidenceGap, ...],
    ) -> Criterion:
        work = state.work
        evidence = () if work is None else (work,)
        work_gaps = tuple(
            gap for gap in gaps if "work" in gap.streams and (work is None or work.cursor.gw_seq <= gap.last_gw_seq)
        )
        if work_gaps:
            return Criterion("UNKNOWN", "work_history_gap", evidence, work_gaps)
        if work is None:
            return Criterion("UNKNOWN", "no_work")
        if not 0 <= now_ns - work.payload["sampled_ns"] <= self.config.work_ttl_ns:
            return Criterion("UNKNOWN", "work_expired", evidence)
        if not (work.payload["busy"] or work.payload["backlog"]):
            return Criterion("CLEAR", "idle", evidence)

        completion = state.completion
        complete_evidence = evidence if completion is None else (*evidence, completion)
        completion_gaps = tuple(gap for gap in gaps if "completion" in gap.streams)
        if completion_gaps:
            return Criterion("UNKNOWN", "completion_history_gap", complete_evidence, completion_gaps)
        key = replica, work.payload["work_key"]
        bucket = self.health_profile.buckets.get(key)
        if bucket is None:
            return Criterion("UNKNOWN", "missing_health_bucket", complete_evidence)
        if completion is None:
            return Criterion("UNKNOWN", "no_completion", evidence)
        age = now_ns - completion.payload["end_ns"]
        if not 0 <= age <= self.config.completion_ttl_ns:
            return Criterion("UNKNOWN", "completion_expired", complete_evidence)
        if age > thresholds.gap_limits_ns.get(key, bucket.gap_limit_ns):
            return Criterion("TRIGGER", "completion_gap_above_threshold", complete_evidence)
        return Criterion("CLEAR", "completion_gap_at_or_below_threshold", complete_evidence)

    def view(
        self, replica_id: str, *, now_ns: int, before_gw_seq: int | None = None,
        thresholds: AlertThresholds | None = None, retrospective_gaps: Iterable[EvidenceGap] = (),
    ) -> PublicProgressView:
        """Read only the consumed prefix; optionally attach gaps learned later in offline analysis."""
        gaps = self._available_gaps(replica_id, now_ns, before_gw_seq, retrospective_gaps)
        state = self._replicas[replica_id]
        thresholds = thresholds if thresholds is not None else self.config.alert_thresholds
        residual = self._residual(state, now_ns, thresholds.residual_threshold, gaps)
        gap = self._gap(replica_id, state, now_ns, thresholds, gaps)
        labels = residual.state, gap.state
        combined = "TRIGGER" if "TRIGGER" in labels else "CLEAR" if "CLEAR" in labels else "UNKNOWN"
        return PublicProgressView(
            self._cursor, now_ns, before_gw_seq, state.epoch, state.work, state.completion,
            state.last_used, state.ema, residual, gap, combined,
        )

    def speed_factor(
        self, replica_id: str, *, now_ns: int, before_gw_seq: int | None = None,
        retrospective_gaps: Iterable[EvidenceGap] = (),
    ) -> SpeedFactor:
        gaps = self._available_gaps(replica_id, now_ns, before_gw_seq, retrospective_gaps)
        state = self._replicas[replica_id]
        residual = self._residual(state, now_ns, self.config.alert_thresholds.residual_threshold, gaps)
        if residual.state == "UNKNOWN":
            return SpeedFactor(Decimal(1), residual.reason, residual.evidence, residual.known_gaps)
        factor = min(max(state.ema, self.config.speed_factor_min), self.config.speed_factor_max)
        return SpeedFactor(factor, "ema" if factor == state.ema else "ema_clamped", residual.evidence)
