from decimal import Decimal
from pathlib import Path
import unittest

from replaylab.contracts import LocatedDocument
from replaylab.ledger import GatewayCursor, GatewayLedger
from replaylab.progress import ProgressBlocked
from test_contracts import study
from test_gateway_order import complete, epoch, event
from test_progress import make_tracker


class Harness:
    def __init__(self, **router_config):
        from replaylab.router import GatewayRouter, RouterConfig

        self.seq = 0
        self.ledger = GatewayLedger("run")
        self.progress = make_tracker()
        data = study()
        data["router"].update(state_ttl_ns=10**9, reservation_timeout_ns=10**9)
        data["router"].update(router_config)
        self.router = GatewayRouter(
            RouterConfig.from_study(LocatedDocument(Path("study.json"), data, {})), self.progress,
        )
        self.feedback = self.router.consume
        self.emit(epoch(self.cursor(0).gw_seq, 0))
        self.emit(epoch(self.cursor(0).gw_seq, 0, replica="B"))

    def cursor(self, now_ns):
        self.seq += 1
        return GatewayCursor("run", self.seq, now_ns)

    def emit(self, row):
        update = self.ledger.apply(row)
        self.progress.consume(update)
        return self.feedback(update)

    def receive(self, kind, at, **payload):
        cursor = self.cursor(at)
        return self.emit(event(kind, cursor.gw_seq, at, **payload))

    def state(self, replica, at, *, version=1, pending=0, unfinished=0, ready=True, generation=1, sampled=None):
        return self.receive("route_state", at, replica_id=replica, epoch=generation, state_version=version,
                            sampled_ns=at if sampled is None else sampled, ready=ready,
                            pending_prefill_tokens=pending, unfinished_requests=unfinished)

    def sent(self, reservation, at):
        cursor = self.cursor(at)
        self.router.mark_sent(reservation.reservation_id, cursor=cursor)
        self.emit(event("dispatch_sent", cursor.gw_seq, at, request_id=reservation.request_id,
                        target=reservation.target, reservation_id=reservation.reservation_id))

    def ack(self, reservation, at, version):
        return self.receive("dispatch_ack", at, request_id=reservation.request_id,
                            replica_id=reservation.target, reservation_id=reservation.reservation_id,
                            state_version=version, reason="accepted")


class RouterTests(unittest.TestCase):
    def test_score_tie_a_cache_hint_clamp_and_no_ema(self):
        from replaylab.router import RoutingRequest

        h = Harness()
        h.state("A", 1, pending=3, unfinished=2)
        h.state("B", 1, pending=3, unfinished=2)
        quote = h.router.quote(RoutingRequest("r", 1, 5), cursor=h.cursor(1), cache_hints={})
        self.assertEqual(quote.default_target, "A")
        for replica in quote.replicas.values():
            self.assertEqual((replica.score, replica.new_tokens, replica.cache_hint_reason),
                             (Decimal(11), 5, "unknown_cache_hint"))
            self.assertEqual((replica.speed.value, replica.speed.reason), (Decimal(1), "no_residual"))
        quote = h.router.quote(RoutingRequest("s", 1, 5), cursor=h.cursor(1), cache_hints={"A": -2, "B": 90})
        self.assertEqual((quote.default_target, quote.replicas["A"].new_tokens, quote.replicas["B"].new_tokens),
                         ("B", 5, 0))

    def test_default_state_ttl_ready_and_current_shared_speed(self):
        from replaylab.router import RoutingRequest

        h = Harness(state_ttl_ns=25)
        h.state("A", 100)
        h.state("B", 100)
        h.emit(complete(h.cursor(100).gw_seq, 100, endpoint=100, duration=30))
        req = RoutingRequest("r", 100, 3)
        q = h.router.quote(req, cursor=h.cursor(120), cache_hints={})
        self.assertEqual((q.default_target, q.replicas["A"].score), ("B", Decimal(8)))
        q = h.router.quote(req, cursor=h.cursor(121), cache_hints={})
        self.assertEqual((q.default_target, q.replicas["A"].speed.reason), ("A", "residual_expired"))
        self.assertEqual(h.router.quote(req, cursor=h.cursor(125), cache_hints={}).default_target, "A")
        self.assertIsNone(h.router.quote(req, cursor=h.cursor(126), cache_hints={}).default_target)
        h.state("A", 126, version=2, ready=False)
        self.assertIsNone(h.router.quote(req, cursor=h.cursor(126), cache_hints={}).default_target)

    def test_quote_at_same_state_version_expires_when_another_request_reserves(self):
        from replaylab.router import RoutingRequest, StaleQuote

        h = Harness()
        h.state("A", 0)
        h.state("B", 0)
        earlier, later = h.cursor(0), h.cursor(0)
        first = h.router.quote(RoutingRequest("first", 0, 1), cursor=earlier, cache_hints={})
        stale = h.router.quote(RoutingRequest("later", 0, 1), cursor=later, cache_hints={})
        h.router.reserve(first, target="B", cursor=earlier)
        with self.assertRaises(StaleQuote):
            h.router.reserve(stale, target="B", cursor=later)
        refresh = h.router.quote(RoutingRequest("later", 0, 1), cursor=later, cache_hints={})
        self.assertEqual(refresh.replicas["B"].state.payload["state_version"], 1)
        self.assertEqual((refresh.replicas["B"].p, refresh.replicas["B"].n), (1, 1))

    def test_release_on_expiry_and_required_config(self):
        from replaylab.router import RouterConfig, RoutingRequest

        data = study()
        data["router"]["reservation_timeout_ns"] = None
        with self.assertRaises(ProgressBlocked) as caught:
            RouterConfig.from_study(LocatedDocument(Path("study.json"), data, {}))
        self.assertEqual(caught.exception.missing_fields, ("router.reservation_timeout_ns",))
        h = Harness(reservation_timeout_ns=5)
        h.state("A", 0)
        cursor = h.cursor(1)
        quote = h.router.quote(RoutingRequest("q", 0, 1), cursor=cursor, cache_hints={})
        reservation = h.router.reserve(quote, target="A", cursor=cursor)
        self.assertEqual(h.router.next_deadline_ns(), 6)
        self.assertEqual(h.router.expire(cursor=h.cursor(5)), ())
        [released] = h.router.expire(cursor=h.cursor(6))
        self.assertEqual((released.reservation, released.reason), (reservation, "reservation_timeout"))
        self.assertIsNone(h.router.next_deadline_ns())

    def test_ack_for_presend_snapshot_cannot_remove_local_charge(self):
        from replaylab.router import RoutingRequest

        h = Harness()
        h.state("A", 1)
        cursor = h.cursor(2)
        quote = h.router.quote(RoutingRequest("q", 2, 1), cursor=cursor, cache_hints={})
        reservation = h.router.reserve(quote, target="A", cursor=cursor)
        h.sent(reservation, 2)
        h.ack(reservation, 3, 1)
        self.assertIsNotNone(h.router.reservation(reservation.reservation_id))
        h.state("A", 4, version=2, unfinished=1, pending=1)
        h.ack(reservation, 4, 2)
        self.assertIsNone(h.router.reservation(reservation.reservation_id))
