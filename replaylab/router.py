"""Default routing and local capacity reservations in gateway order."""

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, replace
from decimal import Context, Decimal, ROUND_HALF_EVEN, localcontext
from threading import RLock
from types import MappingProxyType

from .contracts import POS_INT, TEXT, UINT, VERSION, LocatedDocument, check
from .ledger import GatewayCursor, GatewayEvent, LedgerUpdate
from .progress import ProgressTracker, SpeedFactor, _required


_REPLICAS = ("A", "B")
_SCORE_CONTEXT = Context(prec=34, rounding=ROUND_HALF_EVEN)
_FEEDBACK = frozenset({"replica_epoch", "route_state", "dispatch_ack", "reservation_release"})


@dataclass(frozen=True)
class RouterConfig:
    state_ttl_ns: int
    prefill_cost_ns_per_token: int
    decode_load_cost_ns: int
    reservation_timeout_ns: int
    reservation_ack_policy: str

    @classmethod
    def from_study(cls, study: LocatedDocument) -> "RouterConfig":
        check(study, study.data, {"schema_version": VERSION})
        policy = _required(study, ("router", "reservation_ack_policy"), TEXT)
        if policy != "versioned":
            raise study.error(("router", "reservation_ack_policy"), "expected versioned snapshot policy")
        return cls(
            state_ttl_ns=_required(study, ("router", "state_ttl_ns"), POS_INT),
            prefill_cost_ns_per_token=_required(study, ("router", "prefill_cost_ns_per_token"), POS_INT),
            decode_load_cost_ns=_required(study, ("router", "decode_load_cost_ns"), POS_INT),
            reservation_timeout_ns=_required(study, ("router", "reservation_timeout_ns"), POS_INT),
            reservation_ack_policy=policy,
        )


@dataclass(frozen=True)
class RoutingRequest:
    request_id: str
    arrived_at_ns: int
    prompt_tokens: int

    def __post_init__(self) -> None:
        if not TEXT.accepts(self.request_id) or not UINT.accepts(self.arrived_at_ns) or not UINT.accepts(self.prompt_tokens):
            raise ValueError("routing requires a request ID, a nonnegative integer arrival and token count")


@dataclass(frozen=True)
class ReplicaQuote:
    state: GatewayEvent | None
    p: int
    n: int
    new_tokens: int
    cache_hint_tokens: int
    cache_hint_reason: str
    speed: SpeedFactor
    score: Decimal | None

    def fresh_at(self, now_ns: int, ttl_ns: int) -> bool:
        return self.state is not None and 0 <= now_ns - self.state.payload["sampled_ns"] <= ttl_ns


@dataclass(frozen=True)
class RoutingQuote:
    request: RoutingRequest
    cursor: GatewayCursor
    revision: int
    replicas: Mapping[str, ReplicaQuote]
    default_target: str | None


@dataclass(frozen=True)
class Reservation:
    reservation_id: str
    request_id: str
    target: str
    epoch: int
    prefill_tokens: int
    decided_at: GatewayCursor
    deadline_ns: int
    sent_at: GatewayCursor | None = None


@dataclass(frozen=True)
class ReservationRelease:
    reservation: Reservation
    cursor: GatewayCursor
    reason: str


class StaleQuote(ValueError):
    """A quote is usable only at its own cursor and capacity revision."""


class GatewayRouter:
    """Use complete, cumulative versioned snapshots within each replica epoch.

    An ACK asserts that its named snapshot includes the request. Later versions
    in that epoch reflect its admission or subsequent termination. Wait for the
    exact named snapshot before letting the current version take over accounting.
    """

    def __init__(self, config: RouterConfig, progress: ProgressTracker) -> None:
        if config.reservation_ack_policy != "versioned":
            raise ValueError("expected versioned snapshot policy")
        self.config = config
        self.progress = progress
        self._lock = RLock()
        self._cursor: GatewayCursor | None = None
        self._active_cursor: GatewayCursor | None = None
        self._revision = 0
        self._epochs: dict[str, int] = {}
        self._states: dict[str, GatewayEvent] = {}
        self._versions: dict[str, dict[int, GatewayEvent]] = {replica: {} for replica in _REPLICAS}
        self._reservations: dict[str, Reservation] = {}
        self._acks: dict[str, set[int]] = {}

    def _check_cursor(self, cursor: GatewayCursor) -> None:
        if not (TEXT.accepts(cursor.run_id) and UINT.accepts(cursor.gw_seq) and UINT.accepts(cursor.gw_at_ns)):
            raise ValueError("invalid gateway cursor")
        if self._active_cursor is not None:
            if self._active_cursor != cursor:
                raise ValueError("nested routing operations must share the transaction cursor")
        elif self._cursor is not None and (
            cursor.run_id != self._cursor.run_id or cursor.gw_seq <= self._cursor.gw_seq
            or cursor.gw_at_ns < self._cursor.gw_at_ns
        ):
            raise ValueError("routing commits must follow one run in gateway order")

    @contextmanager
    def transaction(self, cursor: GatewayCursor) -> Iterator[None]:
        """Serialize quote, capacity commit, and action bookkeeping at one cursor."""
        with self._lock:
            self._check_cursor(cursor)
            if self._active_cursor is not None:
                yield
                return
            self._active_cursor = cursor
            try:
                yield
                self._cursor = cursor
            finally:
                self._active_cursor = None

    def quote(
        self, request: RoutingRequest, *, cursor: GatewayCursor, cache_hints: Mapping[str, int | None],
    ) -> RoutingQuote:
        with self._lock:
            self._check_cursor(cursor)
            if request.arrived_at_ns > cursor.gw_at_ns:
                raise ValueError("request has not arrived at the gateway")
            replicas = {
                replica: self._quote_replica(replica, request.prompt_tokens, cursor, cache_hints.get(replica))
                for replica in _REPLICAS
            }
            available = [(row.score, replica) for replica, row in replicas.items() if row.score is not None]
            target = min(available)[1] if available else None
            return RoutingQuote(request, cursor, self._revision, MappingProxyType(replicas), target)

    def _quote_replica(
        self, replica: str, prompt_tokens: int, cursor: GatewayCursor, hint: int | None,
    ) -> ReplicaQuote:
        state = self._states.get(replica)
        p = 0 if state is None else state.payload["pending_prefill_tokens"]
        n = 0 if state is None else state.payload["unfinished_requests"]
        for reservation in self._reservations.values():
            if reservation.target == replica:
                p += reservation.prefill_tokens
                n += 1
        clamped = min(max(hint, 0), prompt_tokens) if type(hint) is int else 0
        hint_reason = "unknown_cache_hint" if type(hint) is not int else "cache_hint"
        if hint_reason == "cache_hint" and clamped != hint:
            hint_reason = "cache_hint_clamped"
        new_tokens = prompt_tokens - clamped
        speed = self.progress.speed_factor(replica, now_ns=cursor.gw_at_ns, before_gw_seq=cursor.gw_seq)
        row = ReplicaQuote(state, p, n, new_tokens, clamped, hint_reason, speed, None)
        if row.fresh_at(cursor.gw_at_ns, self.config.state_ttl_ns) and state.payload["ready"]:
            cost = self.config.prefill_cost_ns_per_token * (p + new_tokens) + self.config.decode_load_cost_ns * (n + 1)
            with localcontext(_SCORE_CONTEXT):
                return replace(row, score=speed.value * cost)
        return row

    def reserve(self, quote: RoutingQuote, *, target: str, cursor: GatewayCursor) -> Reservation:
        with self.transaction(cursor):
            if quote.cursor != cursor or quote.revision != self._revision:
                raise StaleQuote("quote predates the current snapshot or local reservations")
            row = quote.replicas.get(target)
            if row is None or row.state is None or row.state is not self._states.get(target) or not row.state.payload["ready"]:
                raise StaleQuote("target does not have the quoted ready snapshot")
            if any(r.request_id == quote.request.request_id for r in self._reservations.values()):
                raise ValueError("request already has a reservation")
            reservation_id = f"reservation-{cursor.gw_seq}"
            if reservation_id in self._reservations:
                raise ValueError("a gateway cursor can reserve capacity for only one request")
            reservation = Reservation(
                reservation_id, quote.request.request_id, target, row.state.payload["epoch"],
                row.new_tokens, cursor, cursor.gw_at_ns + self.config.reservation_timeout_ns,
            )
            self._reservations[reservation.reservation_id] = reservation
            self._revision += 1
            return reservation

    def reservation(self, reservation_id: str) -> Reservation | None:
        with self._lock:
            return self._reservations.get(reservation_id)

    def mark_sent(self, reservation_id: str, *, cursor: GatewayCursor) -> None:
        with self.transaction(cursor):
            reservation = self._reservations.get(reservation_id)
            if reservation is None or reservation.sent_at is not None or cursor.gw_at_ns >= reservation.deadline_ns:
                raise ValueError("reservation is not awaiting an on-time transport handoff")
            self._reservations[reservation_id] = replace(reservation, sent_at=cursor)

    def release(
        self, reservation_id: str, *, cursor: GatewayCursor, reason: str,
    ) -> ReservationRelease | None:
        with self.transaction(cursor):
            return self._release(reservation_id, cursor, reason)

    def _release(self, reservation_id: str, cursor: GatewayCursor, reason: str) -> ReservationRelease | None:
        reservation = self._reservations.pop(reservation_id, None)
        self._acks.pop(reservation_id, None)
        if reservation is None:
            return None
        self._revision += 1
        return ReservationRelease(reservation, cursor, reason)

    def next_deadline_ns(self) -> int | None:
        with self._lock:
            return min((reservation.deadline_ns for reservation in self._reservations.values()), default=None)

    def expire(self, *, cursor: GatewayCursor) -> tuple[ReservationRelease, ...]:
        with self.transaction(cursor):
            due = [key for key, reservation in self._reservations.items() if reservation.deadline_ns <= cursor.gw_at_ns]
            return tuple(self._release(key, cursor, "reservation_timeout") for key in due)

    def consume(self, update: LedgerUpdate) -> tuple[ReservationRelease, ...]:
        event = update.event
        if event is None or event.kind not in _FEEDBACK:
            return ()
        cursor = update.cursor
        replica = event.payload["replica_id"]
        with self.transaction(cursor):
            if update.epoch_changed:
                self._epochs[replica] = event.payload["epoch"]
                self._states.pop(replica, None)
                self._versions[replica].clear()
                self._revision += 1
                old = [key for key, reservation in self._reservations.items() if reservation.target == replica]
                return tuple(self._release(key, cursor, "epoch_changed") for key in old)
            if event.kind == "route_state":
                self._observe_state(replica, event)
            elif event.kind in {"dispatch_ack", "reservation_release"}:
                key = event.payload["reservation_id"]
                reservation = self._reservations.get(key)
                if not self._matches(reservation, event):
                    return ()
                if event.kind == "reservation_release":
                    if event.payload["reason"] == "snapshot_confirmed":
                        return self._confirmed(replica, cursor)
                    return (self._release(key, cursor, event.payload["reason"]),)
                version = event.payload["state_version"]
                self._acks.setdefault(key, set()).add(version)
            return self._confirmed(replica, cursor)

    def _matches(self, reservation: Reservation | None, event: GatewayEvent) -> bool:
        replica = event.payload["replica_id"]
        return (
            reservation is not None and reservation.sent_at is not None
            and reservation.target == replica and reservation.epoch == self._epochs.get(replica)
            and reservation.request_id == event.payload["request_id"]
        )

    def _observe_state(self, replica: str, event: GatewayEvent) -> None:
        payload = event.payload
        if payload["epoch"] != self._epochs.get(replica) or payload["sampled_ns"] > event.cursor.gw_at_ns:
            return
        version = payload["state_version"]
        previous = self._versions[replica].get(version)
        if previous is not None:
            for name in ("pending_prefill_tokens", "unfinished_requests", "ready"):
                if payload[name] != previous.payload[name]:
                    raise event.error(f"payload.{name}", "conflicting capacity for the same snapshot version")
        else:
            self._versions[replica][version] = event
        current = self._states.get(replica)
        if current is None or (version, payload["sampled_ns"]) > (
            current.payload["state_version"], current.payload["sampled_ns"]
        ):
            self._states[replica] = event
            self._revision += 1

    def _confirmed(self, replica: str, cursor: GatewayCursor) -> tuple[ReservationRelease, ...]:
        current = self._states.get(replica)
        if current is None:
            return ()
        versions = self._versions[replica]
        confirmed = []
        for key, acknowledgments in self._acks.items():
            reservation = self._reservations[key]
            if reservation.target != replica or not self._sampled_after_send(current, reservation):
                continue
            for version in acknowledgments:
                named = versions.get(version)
                if named is None or version > current.payload["state_version"]:
                    continue
                if (self._sampled_after_send(named, reservation)
                        and named.payload["sampled_ns"] <= current.payload["sampled_ns"]):
                    confirmed.append(key)
                    break
        return tuple(self._release(key, cursor, "snapshot_confirmed") for key in confirmed)

    @staticmethod
    def _sampled_after_send(event: GatewayEvent, reservation: Reservation) -> bool:
        sent = reservation.sent_at
        return (sent is not None and event.cursor.gw_seq > sent.gw_seq
                and event.payload["sampled_ns"] >= sent.gw_at_ns)
