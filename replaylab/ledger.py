"""Gateway receipts, replica generations, and the evidence visible at each cursor."""

from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType

from .artifacts import json_text
from .contracts import ArtifactKind, FieldLocation, FormatError, LocatedDocument, UINT, validate_canonical


_REPLICAS = ("A", "B")
_PUBLIC = frozenset({"replica_epoch", "work_state", "step_complete"})
_RESERVATION_EVENTS = frozenset({"dispatch_ack", "reservation_release"})
_EXPLICIT_EPOCH_EVENTS = _PUBLIC | {"route_state", "step_start", "inflight_snapshot"}
_RECEIPT_FIELDS = frozenset({"gw_at_ns", "gw_seq"})


@dataclass(frozen=True)
class GatewayCursor:
    run_id: str
    gw_seq: int
    gw_at_ns: int


@dataclass(frozen=True)
class GatewayEvent:
    cursor: GatewayCursor
    event_id: str
    kind: str
    event_at_ns: int
    payload: Mapping[str, object]
    location: FieldLocation

    def error(self, field: str, reason: str) -> FormatError:
        return FormatError(self.location.file, self.location.line, field, reason)


@dataclass(frozen=True)
class EvidenceGap:
    """Inclusive missing sequence range and the evidence prefix and time of discovery."""

    first_gw_seq: int
    last_gw_seq: int
    known_at: GatewayCursor
    streams: frozenset[str]
    reason: str
    replica_id: str | None = None

    def __post_init__(self) -> None:
        if (not all(UINT.accepts(value) for value in
                    (self.first_gw_seq, self.last_gw_seq, self.known_at.gw_seq, self.known_at.gw_at_ns))
                or not self.first_gw_seq <= self.last_gw_seq <= self.known_at.gw_seq):
            raise ValueError("gap range must be nonnegative, ordered, and at or before known_at")
        streams = frozenset(self.streams)
        if not streams or streams - {"work", "completion"}:
            raise ValueError("gap streams must select work and/or completion")
        if self.replica_id not in (None, *_REPLICAS) or not self.reason:
            raise ValueError("gap must have a reason and a scope of A, B, or both")
        object.__setattr__(self, "streams", streams)


@dataclass(frozen=True)
class LedgerUpdate:
    cursor: GatewayCursor
    receipt: GatewayEvent | None = None
    event: GatewayEvent | None = None
    reason: str | None = None
    epoch_changed: bool = False
    gap: EvidenceGap | None = None

    @property
    def public_event(self) -> GatewayEvent | None:
        if self.event is not None and self.event.kind in _PUBLIC:
            return self.event
        return None


@dataclass(frozen=True)
class DispatchFacts:
    decision: GatewayEvent | None = None
    sent: GatewayEvent | None = None
    sent_epoch: int | None = None


@dataclass(frozen=True)
class GatewayView:
    cursor: GatewayCursor | None
    epochs: Mapping[str, int]
    # Most recently received accepted event of each kind, scoped to the active epoch.
    epoch_events: Mapping[str, Mapping[str, GatewayEvent]]
    public_events: Mapping[str, Mapping[str, GatewayEvent]]
    dispatches: Mapping[str, DispatchFacts]
    gaps: tuple[EvidenceGap, ...]


def _freeze(value: object) -> object:
    if isinstance(value, dict):
        return MappingProxyType({name: _freeze(child) for name, child in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(child) for child in value)
    return value


class GatewayLedger:
    """Consume one run in file order and publish immutable receipt snapshots."""

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self._cursor: GatewayCursor | None = None
        self._seen: dict[str, str] = {}
        self._epochs: dict[str, int] = {}
        self._epoch_events: dict[str, dict[str, GatewayEvent]] = {replica: {} for replica in _REPLICAS}
        self._dispatches: dict[str, DispatchFacts] = {}
        self._gaps: list[EvidenceGap] = []

    def apply(
        self, event: Mapping[str, object], *, location: FieldLocation | None = None,
    ) -> LedgerUpdate:
        location = location or FieldLocation(Path("<gateway>"), 1, "$")
        record = dict(event)
        document = LocatedDocument(location.file, [record], {(0,): location.line}, row_oriented=True)
        validate_canonical(ArtifactKind.EVENTS, document)
        cursor = GatewayCursor(record["run_id"], record["gw_seq"], record["gw_at_ns"])
        receipt = GatewayEvent(
            cursor, record["event_id"], record["type"], record["event_at_ns"], _freeze(record["payload"]), location,
        )
        self._check_cursor(receipt)

        signature = json_text({key: value for key, value in record.items() if key not in _RECEIPT_FIELDS})
        previous = self._seen.get(receipt.event_id)
        if previous is not None:
            if previous != signature:
                raise receipt.error("event_id", "event ID reused with different source content")
            self._cursor = cursor
            return LedgerUpdate(cursor, receipt, reason="duplicate_event")

        replica = None
        epoch = None
        if receipt.kind in _EXPLICIT_EPOCH_EVENTS or receipt.kind in _RESERVATION_EVENTS:
            replica = receipt.payload["replica_id"]
        if receipt.kind in _EXPLICIT_EPOCH_EVENTS:
            epoch = receipt.payload["epoch"]
        reason = self._epoch_rejection(receipt, replica, epoch)
        epoch_changed = False
        if reason is None:
            if receipt.kind in {"dispatch_decision", "dispatch_sent"}:
                self._record_dispatch(receipt)
            if replica is not None:
                if receipt.kind == "replica_epoch":
                    epoch_changed = epoch != self._epochs.get(replica)
                    if epoch_changed:
                        self._epochs[replica] = epoch
                        self._epoch_events[replica].clear()
                self._epoch_events[replica][receipt.kind] = receipt

        self._seen[receipt.event_id] = signature
        self._cursor = cursor
        return LedgerUpdate(cursor, receipt, receipt if reason is None else None, reason, epoch_changed)

    def _check_cursor(self, receipt: GatewayEvent) -> None:
        cursor = receipt.cursor
        if cursor.run_id != self.run_id:
            raise receipt.error("run_id", "does not match the gateway ledger run")
        if self._cursor is not None:
            if cursor.gw_seq <= self._cursor.gw_seq:
                raise receipt.error("gw_seq", "must be strictly increasing in gateway receipt order")
            if cursor.gw_at_ns < self._cursor.gw_at_ns:
                raise receipt.error("gw_at_ns", "must be nondecreasing in gateway receipt order")

    def _epoch_rejection(self, event: GatewayEvent, replica: str | None, epoch: int | None) -> str | None:
        current = self._epochs.get(replica)
        if event.kind in _RESERVATION_EVENTS:
            # These envelopes omit epoch. Associate feedback with the epoch of the actual send.
            facts = self._dispatches.get(event.payload["request_id"])
            if (facts is None or facts.sent is None or facts.sent_epoch is None
                    or facts.sent.payload["target"] != replica
                    or facts.sent.payload["reservation_id"] != event.payload["reservation_id"]):
                return "reservation_epoch_unknown"
            epoch = facts.sent_epoch
        if epoch is None:
            return None
        if current is not None and epoch < current:
            return "stale_epoch"
        if event.kind != "replica_epoch" and (current is None or epoch > current):
            return "future_epoch"
        return None

    def _record_dispatch(self, event: GatewayEvent) -> None:
        request = event.payload["request_id"]
        facts = self._dispatches.get(request, DispatchFacts())
        if event.kind == "dispatch_decision":
            if facts.decision is not None:
                raise event.error("payload.request_id", "request already has a final dispatch decision")
            facts = replace(facts, decision=event)
        else:
            if facts.sent is not None:
                raise event.error("payload.request_id", "request already has a dispatch_sent receipt")
            facts = replace(facts, sent=event, sent_epoch=self._epochs.get(event.payload["target"]))
        self._dispatches[request] = facts

    def mark_gap(self, gap: EvidenceGap) -> LedgerUpdate:
        cursor = self._cursor
        if (cursor is None or gap.known_at.run_id != self.run_id
                or gap.known_at.gw_seq > cursor.gw_seq):
            raise ValueError("gap discovery must refer to a visible sequence in this run")
        self._gaps.append(gap)
        return LedgerUpdate(cursor, reason="evidence_gap", gap=gap)

    def view(self) -> GatewayView:
        epoch_events = {replica: MappingProxyType(events.copy()) for replica, events in self._epoch_events.items()}
        public_events = {
            replica: MappingProxyType({kind: event for kind, event in events.items() if kind in _PUBLIC})
            for replica, events in epoch_events.items()
        }
        return GatewayView(
            self._cursor, MappingProxyType(self._epochs.copy()), MappingProxyType(epoch_events),
            MappingProxyType(public_events), MappingProxyType(self._dispatches.copy()), tuple(self._gaps),
        )
