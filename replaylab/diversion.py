"""Fixed diversion decisions, transport handoffs, and deadlines in gateway order."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from hashlib import sha256

from .artifacts import json_text
from .contracts import PPM, POS_INT, TEXT, UINT, VERSION, LocatedDocument, check
from .ledger import GatewayCursor, LedgerUpdate
from .progress import _required
from .router import GatewayRouter, ReservationRelease, RoutingQuote, RoutingRequest


@dataclass(frozen=True)
class DiversionConfig:
    rule_id: str
    rule_seed: int
    ratio_ppm: int
    duration_ns: int
    max_redirects: int
    max_b_unfinished: int
    max_b_prefill_tokens: int
    b_state_ttl_ns: int

    @classmethod
    def from_study(cls, study: LocatedDocument) -> "DiversionConfig":
        check(study, study.data, {"schema_version": VERSION})
        return cls(**{
            name: _required(study, ("diversion", name), spec)
            for name, spec in (
                ("rule_id", TEXT), ("rule_seed", int), ("ratio_ppm", PPM), ("duration_ns", POS_INT),
                ("max_redirects", UINT), ("max_b_unfinished", UINT), ("max_b_prefill_tokens", UINT),
                ("b_state_ttl_ns", POS_INT),
            )
        })


def stable_selected(rule_seed: int, anchor_id: str, request_id: str, ratio_ppm: int) -> bool:
    if not PPM.accepts(ratio_ppm):
        raise ValueError("ratio_ppm must be an integer in [0, 1000000]")
    wire = json_text([rule_seed, anchor_id, request_id]).encode("utf-8")
    prefix = int.from_bytes(sha256(wire).digest()[:8], "big")
    return prefix * 1_000_000 < ratio_ppm * (1 << 64)


@dataclass(frozen=True)
class GateDecision:
    state: str
    reason: str


def evaluate_b_gate(quote: RoutingQuote, config: DiversionConfig) -> GateDecision:
    b = quote.replicas["B"]
    if not b.fresh_at(quote.cursor.gw_at_ns, config.b_state_ttl_ns):
        return GateDecision("unknown", "b_stale")
    if not b.state.payload["ready"]:
        return GateDecision("deny", "b_unready")
    if b.n + 1 > config.max_b_unfinished or b.p + b.new_tokens > config.max_b_prefill_tokens:
        return GateDecision("deny", "b_limit")
    return GateDecision("pass", "redirected")


@dataclass(frozen=True)
class RuleTransition:
    """A transition fact. Derived facts take their own sequence when logged."""

    kind: str
    cursor: GatewayCursor
    effective_at_ns: int
    rule_id: str
    activation_id: str
    reason: str

    def payload(self) -> dict:
        return dict(rule_id=self.rule_id, activation_id=self.activation_id,
                    reason=self.reason, effective_at_ns=self.effective_at_ns)


@dataclass(frozen=True)
class DispatchDecision:
    quote: RoutingQuote
    actual_target: str | None
    eligible: bool
    selected: bool
    b_gate: str
    reason: str
    rule_active: bool
    reservation_id: str | None

    @property
    def default_target(self) -> str | None:
        return self.quote.default_target

    @property
    def request_id(self) -> str:
        return self.quote.request.request_id

    @property
    def redirected(self) -> bool:
        return self.default_target == "A" and self.actual_target == "B"

    def payload(self) -> dict:
        refs = {
            replica: dict(event_id=row.state.event_id, epoch=row.state.payload["epoch"],
                          state_version=row.state.payload["state_version"])
            for replica, row in self.quote.replicas.items() if row.state is not None
        }
        return dict(
            request_id=self.request_id, default_target=self.default_target, actual_target=self.actual_target,
            eligible=self.eligible, selected=self.selected, b_gate=self.b_gate, reason=self.reason,
            rule_active=self.rule_active, route_state_refs=refs,
            cache_hint_tokens={replica: row.cache_hint_tokens for replica, row in self.quote.replicas.items()},
            cache_hint_reasons={replica: row.cache_hint_reason for replica, row in self.quote.replicas.items()},
            speed_factor_reasons={replica: row.speed.reason for replica, row in self.quote.replicas.items()},
        )


@dataclass(frozen=True)
class SendResult:
    decision: DispatchDecision
    cursor: GatewayCursor
    sent: bool
    reason: str
    terminal_status: str | None = None

    def dispatch_sent_payload(self) -> dict:
        if not self.sent:
            raise ValueError("an unsent request has no dispatch_sent payload")
        return dict(request_id=self.decision.request_id, target=self.decision.actual_target,
                    reservation_id=self.decision.reservation_id)


@dataclass(frozen=True)
class TerminalOutcome:
    request_id: str
    cursor: GatewayCursor
    status: str
    reason: str


@dataclass(frozen=True)
class TimerResult:
    transition: RuleTransition | None
    releases: tuple[ReservationRelease, ...]


class DiversionController:
    """Run one anchor's action in the same gateway transaction as local capacity."""

    def __init__(
        self, config: DiversionConfig, router: GatewayRouter, *, anchor_id: str, anchor_at_ns: int,
    ) -> None:
        if not TEXT.accepts(anchor_id) or not UINT.accepts(anchor_at_ns):
            raise ValueError("expected an anchor ID and nonnegative integer anchor time")
        self.config = config
        self.router = router
        self.anchor_id = anchor_id
        self.anchor_at_ns = anchor_at_ns
        self._armed: RuleTransition | None = None
        self._disarmed: RuleTransition | None = None
        self._decisions: dict[str, DispatchDecision] = {}
        self._sends: dict[str, SendResult] = {}
        self._terminal: dict[str, TerminalOutcome] = {}
        self._pending_redirects: set[str] = set()
        self._in_handoff = False
        self._releases: list[ReservationRelease] = []
        self.redirected_n = 0
        self.blocked_n = 0
        self.first_redirect_ns: int | None = None

    @property
    def transitions(self) -> tuple[RuleTransition, ...]:
        return tuple(item for item in (self._armed, self._disarmed) if item is not None)

    @property
    def releases(self) -> tuple[ReservationRelease, ...]:
        return tuple(self._releases)

    @property
    def pending_redirect_n(self) -> int:
        return len(self._pending_redirects)

    def terminal_outcome(self, request_id: str) -> TerminalOutcome | None:
        return self._terminal.get(request_id)

    def arm(self, *, cursor: GatewayCursor, activation_id: str) -> RuleTransition:
        with self.router.transaction(cursor):
            if self._armed is not None:
                raise ValueError("this anchor has already been armed")
            if not TEXT.accepts(activation_id):
                raise ValueError("activation_id must be nonempty")
            self._advance(cursor)
            self._armed = RuleTransition(
                "rule_armed", cursor, cursor.gw_at_ns, self.config.rule_id, activation_id, "armed",
            )
            return self._armed

    def _rule_deadline(self) -> int | None:
        if self._armed is None or self._disarmed is not None:
            return None
        return self._armed.cursor.gw_at_ns + self.config.duration_ns

    def next_deadline_ns(self) -> int | None:
        deadlines = (self._rule_deadline(), self.router.next_deadline_ns())
        return min((time for time in deadlines if time is not None), default=None)

    def active_at(self, now_ns: int) -> bool:
        if self._armed is None:
            return False
        armed_at = self._armed.cursor.gw_at_ns
        end = armed_at + self.config.duration_ns
        if self._disarmed is not None:
            end = min(end, self._disarmed.effective_at_ns)
        return armed_at <= now_ns < end

    def _close(self, cursor: GatewayCursor, *, effective_at_ns: int, reason: str) -> RuleTransition:
        self._disarmed = RuleTransition(
            "rule_disarmed", cursor, effective_at_ns, self.config.rule_id, self._armed.activation_id, reason,
        )
        return self._disarmed

    def decide(
        self, request: RoutingRequest, *, cursor: GatewayCursor, cache_hints: Mapping[str, int | None],
    ) -> DispatchDecision:
        with self.router.transaction(cursor):
            if request.request_id in self._decisions:
                raise ValueError("request already has a final dispatch decision")
            self._advance(cursor)
            quote = self.router.quote(request, cursor=cursor, cache_hints=cache_hints)
            active = self.active_at(cursor.gw_at_ns)
            eligible, selected, gate, reason = self._evaluate(quote, active)
            target = "B" if reason == "redirected" else quote.default_target
            held = None if target is None else self.router.reserve(quote, target=target, cursor=cursor)
            decision = DispatchDecision(
                quote, target, eligible, selected, gate, reason, active,
                None if held is None else held.reservation_id,
            )
            self._decisions[request.request_id] = decision
            if decision.redirected:
                self._pending_redirects.add(request.request_id)
            if gate in {"deny", "unknown"}:
                self.blocked_n += 1
            if target is None:
                self._unsent(decision, cursor, "no_backend", "rejected")
            return decision

    def _evaluate(self, quote: RoutingQuote, active: bool) -> tuple[bool, bool, str, str]:
        default = quote.default_target
        if default is None:
            return False, False, "not_applicable", "no_backend"
        if default == "B":
            return False, False, "not_applicable", "default_b"
        if not active:
            return False, False, "not_applicable", "inactive"
        if quote.request.arrived_at_ns < self.anchor_at_ns:
            return False, False, "not_applicable", "pre_anchor"
        if not stable_selected(self.config.rule_seed, self.anchor_id, quote.request.request_id, self.config.ratio_ppm):
            return True, False, "not_applicable", "not_selected"
        if self.redirected_n + self.pending_redirect_n >= self.config.max_redirects:
            return True, True, "not_applicable", "redirect_slots_full"
        gate = evaluate_b_gate(quote, self.config)
        return True, True, gate.state, gate.reason

    def handoff(
        self, request_id: str, *, cursor: GatewayCursor, accept: Callable[[DispatchDecision], bool],
    ) -> SendResult:
        """Accept synchronously at the supplied cursor, without deferring the transfer."""
        with self.router.transaction(cursor):
            if self._in_handoff:
                raise ValueError("transport acceptance cannot start another handoff")
            decision = self._decisions[request_id]
            previous = self._sends.get(request_id)
            if previous is not None and previous.sent:
                raise ValueError("request already has a successful handoff")
            self._advance(cursor)
            if request_id in self._sends:
                return self._sends[request_id]
            if decision.redirected and not self.active_at(cursor.gw_at_ns):
                self._release_decision(decision, cursor, "rule_expired")
                return self._sends[request_id]
            held = self.router.reservation(decision.reservation_id)
            if held is None:
                return self._unsent(decision, cursor, "reservation_unavailable", "cancelled")
            self._in_handoff = True
            try:
                accepted = accept(decision)
            finally:
                self._in_handoff = False
            if type(accepted) is not bool:
                raise ValueError("transport acceptance must report a boolean")
            if not accepted:
                self._release_decision(decision, cursor, "transport_rejected", terminal_status="backend_error")
                return self._sends[request_id]

            self.router.mark_sent(held.reservation_id, cursor=cursor)
            result = SendResult(decision, cursor, True, "sent")
            self._sends[request_id] = result
            if decision.redirected:
                self._pending_redirects.remove(request_id)
                self.redirected_n += 1
                if self.first_redirect_ns is None:
                    self.first_redirect_ns = cursor.gw_at_ns
                if self.redirected_n == self.config.max_redirects:
                    self._close(cursor, effective_at_ns=cursor.gw_at_ns, reason="max_redirects")
            return result

    def _unsent(self, decision: DispatchDecision, cursor: GatewayCursor, reason: str, status: str) -> SendResult:
        self._pending_redirects.discard(decision.request_id)
        result = SendResult(decision, cursor, False, reason, status)
        self._sends[decision.request_id] = result
        self._terminal[decision.request_id] = TerminalOutcome(decision.request_id, cursor, status, reason)
        return result

    def _record_releases(self, releases: tuple[ReservationRelease, ...]) -> None:
        for release in releases:
            self._releases.append(release)
            request_id = release.reservation.request_id
            decision = self._decisions.get(request_id)
            if decision is not None and release.reservation.sent_at is None and request_id not in self._sends:
                self._unsent(decision, release.cursor, release.reason, "cancelled")

    def _release_decision(
        self, decision: DispatchDecision, cursor: GatewayCursor, reason: str, *, terminal_status: str = "cancelled",
    ) -> None:
        released = self.router.release(decision.reservation_id, cursor=cursor, reason=reason)
        if released is not None:
            self._releases.append(released)
        self._unsent(decision, cursor, reason, terminal_status)

    def _advance(self, cursor: GatewayCursor) -> TimerResult:
        first_release = len(self._releases)
        deadline = self._rule_deadline()
        transition = None
        if deadline is not None and cursor.gw_at_ns >= deadline:
            transition = self._close(cursor, effective_at_ns=deadline, reason="duration_elapsed")
            for request_id in sorted(self._pending_redirects, key=lambda key: self._decisions[key].quote.cursor.gw_seq):
                self._release_decision(self._decisions[request_id], cursor, "rule_expired")
        self._record_releases(self.router.expire(cursor=cursor))
        return TimerResult(transition, tuple(self._releases[first_release:]))

    def on_timer(self, *, cursor: GatewayCursor) -> TimerResult:
        with self.router.transaction(cursor):
            return self._advance(cursor)

    def consume(self, update: LedgerUpdate) -> tuple[ReservationRelease, ...]:
        event = update.event
        if event is None or event.kind not in {"replica_epoch", "route_state", "dispatch_ack", "reservation_release"}:
            return ()
        with self.router.transaction(update.cursor):
            timer = self._advance(update.cursor)
            released = self.router.consume(update)
            self._record_releases(released)
            return (*timer.releases, *released)

    def record_terminal(
        self, request_id: str, *, cursor: GatewayCursor, terminal_status: str,
    ) -> TerminalOutcome:
        with self.router.transaction(cursor):
            sent = self._sends.get(request_id)
            if sent is None or not sent.sent:
                raise ValueError("a backend terminal outcome requires a successful handoff")
            if terminal_status not in {"ok", "rejected", "timeout", "cancelled", "backend_error"}:
                raise ValueError("expected a resolved backend terminal status")
            if request_id in self._terminal:
                raise ValueError("request already has a terminal outcome")
            self._advance(cursor)
            outcome = TerminalOutcome(request_id, cursor, terminal_status, "backend_terminal")
            self._terminal[request_id] = outcome
            released = self.router.release(sent.decision.reservation_id, cursor=cursor, reason=terminal_status)
            if released is not None:
                self._releases.append(released)
            return outcome
