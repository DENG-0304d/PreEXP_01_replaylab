from pathlib import Path
import unittest
from unittest.mock import patch

from replaylab.contracts import LocatedDocument
from test_contracts import study
from test_gateway_order import epoch, event
from test_router import Harness


MS = 1_000_000


class ActionHarness(Harness):
    def __init__(self, *, anchor_at=0, router=None, **diversion):
        from replaylab.diversion import DiversionConfig, DiversionController

        super().__init__(**(router or {}))
        data = study()
        data["diversion"].update(duration_ns=250 * MS, b_state_ttl_ns=10**9, max_redirects=5,
                                 max_b_unfinished=100, max_b_prefill_tokens=100)
        data["diversion"].update(diversion)
        config = DiversionConfig.from_study(LocatedDocument(Path("study.json"), data, {}))
        self.action = DiversionController(config, self.router, anchor_id="anchor", anchor_at_ns=anchor_at)
        self.feedback = self.action.consume

    def arm(self, at):
        cursor = self.cursor(at)
        transition = self.action.arm(cursor=cursor, activation_id="act")
        self.emit(event("rule_armed", cursor.gw_seq, at, **transition.payload()))
        return transition

    def decide(self, request_id, at, *, arrived_at=None, prompt=1):
        from replaylab.router import RoutingRequest

        cursor = self.cursor(at)
        request = RoutingRequest(request_id, at if arrived_at is None else arrived_at, prompt)
        decision = self.action.decide(request, cursor=cursor, cache_hints={})
        self.emit(event("dispatch_decision", cursor.gw_seq, at, **decision.payload()))
        return decision

    def handoff(self, request_id, at, accept=lambda _: True):
        cursor = self.cursor(at)
        result = self.action.handoff(request_id, cursor=cursor, accept=accept)
        if result.sent:
            self.emit(event("dispatch_sent", cursor.gw_seq, at, **result.dispatch_sent_payload()))
        return result


class DiversionTests(unittest.TestCase):
    def fresh(self, **kwargs):
        h = ActionHarness(**kwargs)
        h.state("A", 0)
        h.state("B", 0)
        return h

    def test_t05_two_actual_arm_times_share_first_redirect_but_exit_at_different_deadlines(self):
        for arm, end, active_at_400 in [(100, 350, False), (200, 450, True)]:
            with self.subTest(arm=arm):
                h = self.fresh(anchor_at=100 * MS)
                self.assertEqual(h.arm(arm * MS).cursor.gw_at_ns, arm * MS)
                self.assertEqual(h.decide("q", 300 * MS).reason, "redirected")
                self.assertTrue(h.handoff("q", 300 * MS).sent)
                self.assertEqual(h.action.first_redirect_ns, 300 * MS)
                self.assertEqual(h.action.next_deadline_ns(), end * MS)
                self.assertEqual(h.action.active_at(400 * MS), active_at_400)
                result = h.action.on_timer(cursor=h.cursor(end * MS))
                self.assertEqual((result.transition.kind, result.transition.cursor.gw_at_ns),
                                 ("rule_disarmed", end * MS))
                self.assertEqual(h.action.active_at(400 * MS), active_at_400)

    def test_t05_decisions_and_sends_at_each_cutoff_boundary(self):
        for when, should_redirect in [(9, True), (10, False), (11, False)]:
            with self.subTest(decision_at=when):
                h = self.fresh(duration_ns=10)
                h.arm(0)
                decision = h.decide("q", when)
                self.assertEqual(decision.actual_target, "B" if should_redirect else "A")
                self.assertEqual(decision.reason, "redirected" if should_redirect else "inactive")
            with self.subTest(send_at=when):
                h = self.fresh(duration_ns=10)
                h.arm(0)
                decision = h.decide("q", 0)
                self.assertEqual(decision.actual_target, "B")  # Later gw_seq, same grid time.
                handed = []
                result = h.handoff("q", when, accept=lambda item: handed.append(item) or True)
                self.assertEqual(result.sent, should_redirect)
                self.assertEqual(handed, [decision] if should_redirect else [])
                self.assertEqual(h.action.redirected_n, int(should_redirect))
                self.assertEqual(h.action.first_redirect_ns, when if should_redirect else None)
                facts = h.ledger.view().dispatches["q"]
                self.assertEqual(facts.decision.payload["actual_target"], "B")
                if not should_redirect:
                    self.assertIsNone(facts.sent)
                    self.assertEqual((result.terminal_status, result.reason), ("cancelled", "rule_expired"))
                    self.assertIsNone(h.router.reservation(decision.reservation_id))
                    self.assertEqual(h.action.pending_redirect_n, 0)

    def test_t05_existing_first_redirect_survives_a_later_unaccepted_cutoff(self):
        h = self.fresh(duration_ns=10)
        h.arm(0)
        h.decide("first", 1)
        h.handoff("first", 2)
        h.decide("late", 9)
        self.assertEqual(h.handoff("late", 10).terminal_status, "cancelled")
        self.assertEqual((h.action.redirected_n, h.action.first_redirect_ns), (1, 2))

    def test_t06_b_gate_block_default_b_and_two_different_failure_paths(self):
        h = self.fresh(max_b_unfinished=1)
        h.state("B", 0, version=2, pending=2)
        h.arm(0)
        first = h.decide("first", 1)
        blocked = h.decide("blocked", 1)
        self.assertEqual((blocked.default_target, blocked.actual_target, blocked.b_gate, blocked.reason),
                         ("A", "A", "deny", "b_limit"))
        self.assertEqual((h.action.blocked_n, h.action.redirected_n, h.action.first_redirect_ns), (1, 0, None))
        fail = h.handoff("first", 2, lambda _: False)
        self.assertFalse(fail.sent)
        self.assertIsNone(h.router.reservation(first.reservation_id))
        second = h.decide("second", 3)
        self.assertTrue(h.handoff("second", 4).sent)
        h.action.record_terminal("second", cursor=h.cursor(5), terminal_status="rejected")
        self.assertEqual((h.action.redirected_n, h.action.first_redirect_ns, h.action.blocked_n), (1, 4, 1))
        self.assertIsNone(h.router.reservation(second.reservation_id))
        self.assertEqual(h.action.terminal_outcome("second").status, "rejected")
        self.assertIsNotNone(h.ledger.view().dispatches["second"].sent)
        h.state("A", 6, version=2, pending=99)
        default = h.decide("default", 6)
        self.assertEqual((default.default_target, default.actual_target, default.reason), ("B", "B", "default_b"))
        self.assertEqual((h.action.blocked_n, h.action.redirected_n), (1, 1))

    def test_t06_ack_and_snapshot_may_arrive_either_way_or_old_snapshot_after_new(self):
        for order in [("ack", "v2"), ("v2", "ack"), ("v3", "ack", "v2")]:
            with self.subTest(order=order):
                h = self.fresh(max_b_unfinished=1)
                h.arm(0)
                decision = h.decide("q", 1)
                h.handoff("q", 1)
                held = h.router.reservation(decision.reservation_id)
                for index, label in enumerate(order, 2):
                    if label == "ack":
                        h.ack(held, index, 2)
                    else:
                        version = int(label[1:])
                        h.state("B", index, version=version, unfinished=1, pending=1,
                                sampled=2 if version == 2 else index)
                    if index == 2 and len(order) > 1:
                        self.assertIsNotNone(h.router.reservation(decision.reservation_id))
                    if label == "ack" and order[0] == "v3":
                        self.assertIsNotNone(h.router.reservation(decision.reservation_id))
                self.assertIsNone(h.router.reservation(decision.reservation_id))
                next_decision = h.decide("next", 7)
                snapshot = next_decision.quote.replicas["B"]
                self.assertEqual((snapshot.p, snapshot.n), (1, 1))
                self.assertEqual(snapshot.state.payload["state_version"], 3 if order[0] == "v3" else 2)
                self.assertEqual(next_decision.reason, "b_limit")

    def test_t06_epoch_change_clears_pending_slot_and_keeps_decision(self):
        h = self.fresh(max_redirects=1)
        h.arm(0)
        original = h.decide("q", 1)
        h.emit(epoch(h.cursor(2).gw_seq, 2, value=2, replica="B"))
        self.assertIsNone(h.router.reservation(original.reservation_id))
        self.assertEqual(h.action.pending_redirect_n, 0)
        self.assertEqual(h.action.terminal_outcome("q").status, "cancelled")
        self.assertEqual(h.ledger.view().dispatches["q"].decision.payload["actual_target"], "B")
        h.state("B", 3, generation=2)
        self.assertEqual(h.decide("new", 3).reason, "redirected")

    def test_pending_cap_is_a_distinct_reason_until_a_send_reaches_the_cap(self):
        h = self.fresh(max_redirects=1, max_b_unfinished=1)
        h.state("B", 0, version=2, pending=2)
        h.arm(0)
        h.decide("hold", 1)
        other = h.decide("other", 1)
        self.assertEqual((other.reason, other.actual_target, other.selected, other.b_gate),
                         ("redirect_slots_full", "A", True, "not_applicable"))
        self.assertTrue(h.action.active_at(1))
        self.assertEqual(h.action.blocked_n, 0)
        h.handoff("hold", 2, lambda _: False)
        h.decide("next", 3)
        self.assertTrue(h.handoff("next", 3).sent)
        self.assertFalse(h.action.active_at(3))
        self.assertEqual(h.action.transitions[-1].reason, "max_redirects")
        exit_event = h.action.transitions[-1]
        sent = h.ledger.view().dispatches["next"].sent
        h.receive("rule_disarmed", 3, **exit_event.payload())
        self.assertGreater(h.ledger.view().cursor.gw_seq, sent.cursor.gw_seq)
        self.assertEqual(h.ledger.view().cursor.gw_at_ns, sent.cursor.gw_at_ns)

    def test_independent_b_ttl_pre_anchor_and_zero_cap(self):
        h = self.fresh(duration_ns=10, b_state_ttl_ns=2, router={"state_ttl_ns": 12}, anchor_at=2)
        h.state("B", 0, version=2, pending=2)
        h.arm(2)
        self.assertEqual(h.decide("old", 2, arrived_at=1).reason, "pre_anchor")
        decision = h.decide("stale", 3)
        self.assertIsNotNone(decision.quote.replicas["B"].score)
        self.assertEqual((decision.reason, decision.b_gate), ("b_stale", "unknown"))
        zero = self.fresh(max_redirects=0)
        zero.arm(0)
        q = zero.decide("zero", 0)
        self.assertEqual((q.actual_target, zero.action.redirected_n), ("A", 0))

    def test_timer_handles_unsent_and_sent_reservations_after_rule_exit(self):
        h = self.fresh(duration_ns=5, router={"reservation_timeout_ns": 10})
        h.arm(0)
        first = h.decide("sent", 1)
        h.handoff("sent", 2)
        h.state("A", 2, version=2, pending=99)
        default_b = h.decide("default-b", 2)
        self.assertEqual(default_b.reason, "default_b")
        self.assertEqual(h.action.next_deadline_ns(), 5)
        h.action.on_timer(cursor=h.cursor(5))
        self.assertEqual(h.action.next_deadline_ns(), 11)
        self.assertIsNotNone(h.router.reservation(first.reservation_id))
        h.action.on_timer(cursor=h.cursor(11))
        self.assertEqual(h.action.next_deadline_ns(), 12)
        self.assertIsNone(h.router.reservation(first.reservation_id))
        self.assertIsNone(h.action.terminal_outcome("sent"))
        h.action.on_timer(cursor=h.cursor(12))
        self.assertEqual(h.action.terminal_outcome("default-b").status, "cancelled")
        self.assertIsNone(h.action.next_deadline_ns())
        self.assertEqual(h.action.redirected_n, 1)

    def test_pending_timeout_can_precede_rule_deadline(self):
        h = self.fresh(duration_ns=20, max_redirects=1, router={"reservation_timeout_ns": 5})
        h.arm(0)
        original = h.decide("q", 1)
        self.assertEqual(h.action.next_deadline_ns(), 6)
        event = h.action.on_timer(cursor=h.cursor(6))
        self.assertIsNone(event.transition)
        self.assertEqual(event.releases[0].reason, "reservation_timeout")
        self.assertEqual(h.action.terminal_outcome("q").status, "cancelled")
        self.assertIsNone(h.router.reservation(original.reservation_id))
        self.assertEqual((h.action.pending_redirect_n, h.action.next_deadline_ns()), (0, 20))
        self.assertEqual(h.decide("other", 6).reason, "redirected")

    def test_ineligible_and_gate_failure_reasons_are_distinguishable(self):
        h = ActionHarness()
        h.arm(0)
        no_backend = h.decide("none", 0)
        self.assertEqual((no_backend.reason, no_backend.actual_target, no_backend.reservation_id),
                         ("no_backend", None, None))
        self.assertEqual(h.action.terminal_outcome("none").status, "rejected")
        h.state("A", 1)
        h.state("B", 1, ready=False)
        unready = h.decide("busy", 1)
        self.assertEqual((unready.reason, unready.b_gate, unready.actual_target), ("b_unready", "deny", "A"))
        excluded = self.fresh(ratio_ppm=0)
        excluded.arm(0)
        self.assertEqual(excluded.decide("outside", 1).reason, "not_selected")

    def test_one_final_decision_and_one_successful_transport_call(self):
        from replaylab.router import RoutingRequest

        h = self.fresh()
        h.arm(0)
        decision = h.decide("q", 1)
        handed = []
        h.handoff("q", 1, lambda item: handed.append(item) or True)
        with self.assertRaisesRegex(ValueError, "successful handoff"):
            h.handoff("q", 2, lambda item: handed.append(item) or True)
        with self.assertRaisesRegex(ValueError, "final dispatch decision"):
            h.action.decide(RoutingRequest("q", 1, 1), cursor=h.cursor(2), cache_hints={})
        self.assertEqual(handed, [decision])
        self.assertEqual(h.action.redirected_n, 1)

    def test_stable_selection_chinese_fixed_vector_and_strict_comparison(self):
        from replaylab.diversion import stable_selected

        ids = (17, "锚点", "请求一")
        self.assertFalse(stable_selected(*ids, 426746))
        self.assertTrue(stable_selected(*ids, 426747))
        self.assertFalse(stable_selected(*ids, 0))
        self.assertTrue(stable_selected(*ids, 1000000))
        with patch("replaylab.diversion.sha256") as digest:
            digest.return_value.digest.return_value = (1 << 63).to_bytes(8, "big") + bytes(24)
            self.assertFalse(stable_selected(*ids, 500000))
            self.assertTrue(stable_selected(*ids, 500001))
            self.assertEqual(digest.call_args.args[0], b'[17,"\\u951a\\u70b9","\\u8bf7\\u6c42\\u4e00"]')
