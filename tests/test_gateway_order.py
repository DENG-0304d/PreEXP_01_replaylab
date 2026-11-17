from decimal import Decimal
import json
from pathlib import Path
import tempfile
import unittest

from replaylab.artifacts import read_artifact
from replaylab.contracts import ArtifactKind, FieldLocation, FormatError


def event(kind, sequence, received, *, event_id=None, source=None, **payload):
    return {
        "run_id": "run", "event_id": event_id or f"e{sequence}", "type": kind,
        "event_at_ns": received if source is None else source, "gw_seq": sequence,
        "gw_at_ns": received, "payload": payload,
    }


def epoch(sequence, at, value=1, replica="A"):
    return event("replica_epoch", sequence, at, replica_id=replica, epoch=value)


def complete(sequence, received, *, endpoint, step="s", duration=10, key="light",
             generation=1, replica="A", boundary="gpu", event_id=None):
    return event("step_complete", sequence, received, source=endpoint, event_id=event_id,
                 replica_id=replica, epoch=generation, step_id=step, duration_ns=duration,
                 end_ns=endpoint, boundary_id=boundary, work_key=key)


def work(sequence, received, *, sampled=None, version=1, busy=True, backlog=False,
         key="light", replica="A", generation=1):
    return event("work_state", sequence, received, source=sampled, replica_id=replica,
                 epoch=generation, work_version=version, work_key=key, busy=busy, backlog=backlog,
                 sampled_ns=received if sampled is None else sampled)


def decision(sequence, at):
    return event("dispatch_decision", sequence, at, request_id="q", default_target="A",
                 actual_target="A", eligible=False, selected=False, b_gate="not_applicable",
                 reason="inactive", rule_active=False, route_state_refs={}, cache_hint_tokens=0)


class GatewayOrderTests(unittest.TestCase):
    def test_t02_original_order_controls_evidence_at_same_gateway_time(self):
        from replaylab.ledger import GatewayLedger
        from test_progress import make_tracker

        for completion_first in (False, True):
            with self.subTest(completion_first=completion_first):
                ledger, tracker = GatewayLedger("run"), make_tracker()
                tracker.consume(ledger.apply(epoch(1, 0)))
                tracker.consume(ledger.apply(work(2, 105)))
                completion_seq, dispatch_seq = (3, 4) if completion_first else (4, 3)
                completion = complete(completion_seq, 110, endpoint=100, duration=30)
                if completion_first:
                    tracker.consume(ledger.apply(completion))
                baseline = tracker.view("A", now_ns=110, before_gw_seq=dispatch_seq)
                other = tracker.view("A", now_ns=110, before_gw_seq=dispatch_seq)
                self.assertEqual(baseline, other)
                self.assertEqual(baseline.completion_end_ns, 100 if completion_first else None)
                self.assertEqual(baseline.ema, Decimal("3") if completion_first else None)
                self.assertEqual([ref.event_id for ref in baseline.residual.evidence],
                                 [completion["event_id"]] if completion_first else [])
                self.assertEqual(baseline.work.event_id, "e2")
                tracker.consume(ledger.apply(decision(dispatch_seq, 110)))
                if not completion_first:
                    tracker.consume(ledger.apply(completion))
                    self.assertIsNone(baseline.completion_end_ns)
                    self.assertEqual(tracker.view("A", now_ns=110).completion_end_ns, 100)

    def test_t02_dispatch_clock_expires_residual_without_an_intermediate_tick(self):
        from replaylab.ledger import GatewayLedger
        from test_progress import make_tracker

        ledger, tracker = GatewayLedger("run"), make_tracker()
        tracker.consume(ledger.apply(epoch(1, 0)))
        tracker.consume(ledger.apply(complete(2, 100, endpoint=100, duration=30)))
        before = tracker.view("A", now_ns=120, before_gw_seq=3)
        expired = tracker.view("A", now_ns=121, before_gw_seq=3)
        self.assertEqual(before.residual.state, "TRIGGER")
        self.assertEqual(expired.residual.reason, "residual_expired")
        speed = tracker.speed_factor("A", now_ns=121, before_gw_seq=3)
        self.assertEqual((speed.value, speed.reason), (Decimal(1), "residual_expired"))
        tracker.consume(ledger.apply(decision(3, 121)))
        self.assertEqual(expired.cursor.gw_seq, 2)

    def test_t02_order_errors_keep_file_and_field_and_allow_sequence_gaps(self):
        from replaylab.ledger import GatewayLedger

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            rows = [epoch(1, 5), event("tick", 4, 5), event("tick", 3, 5)]
            path.write_text("\n".join(map(json.dumps, rows)) + "\n", encoding="utf-8")
            with self.assertRaises(FormatError) as caught:
                read_artifact(path, ArtifactKind.EVENTS)
            self.assertEqual((caught.exception.file, caught.exception.line, caught.exception.field),
                             (path, 3, "gw_seq"))
            ledger = GatewayLedger("run")
            for row in rows[:2]:
                ledger.apply(row)
            for bad, field in [(rows[2], "gw_seq"), (event("tick", 5, 4), "gw_at_ns")]:
                with self.assertRaises(FormatError) as caught:
                    ledger.apply(bad, location=FieldLocation(path, 3, "$"))
                self.assertEqual((caught.exception.file, caught.exception.line, caught.exception.field),
                                 (path, 3, field))
            self.assertEqual(ledger.view().cursor.gw_seq, 4)

    def test_t13_two_readers_share_public_work_at_its_natural_publication(self):
        from replaylab.ledger import GatewayLedger
        from test_progress import make_tracker

        ledger, tracker = GatewayLedger("run"), make_tracker()
        tracker.consume(ledger.apply(epoch(1, 0)))
        tracker.consume(ledger.apply(complete(2, 100, endpoint=100)))
        before = tracker.view("A", now_ns=106, before_gw_seq=3)
        self.assertIsNone(before.work)
        private = event("step_start", 3, 106, replica_id="A", epoch=1,
                        step_id="next", start_ns=106, boundary_id="gpu")
        tracker.consume(ledger.apply(private))
        self.assertIsNone(tracker.view("A", now_ns=106).work)
        tracker.consume(ledger.apply(work(4, 106, sampled=105, version=7)))
        for reader in (tracker.view("A", now_ns=106), tracker.view("A", now_ns=106)):
            self.assertEqual(reader.gap.state, "TRIGGER")
            self.assertEqual(reader.work.payload["work_version"], 7)
            self.assertEqual((reader.work.event_id, reader.work.cursor.gw_at_ns, reader.work.cursor.gw_seq),
                             ("e4", 106, 4))
            self.assertEqual([ref.event_id for ref in reader.gap.evidence], ["e4", "e2"])

    def test_t14_event_id_retransmission_preserves_first_evidence_and_conflict_is_located(self):
        from replaylab.ledger import GatewayLedger
        from test_progress import make_tracker

        ledger, tracker = GatewayLedger("run"), make_tracker()
        tracker.consume(ledger.apply(epoch(1, 0)))
        first = complete(2, 100, endpoint=100, duration=30, event_id="same")
        tracker.consume(ledger.apply(first))
        duplicate = ledger.apply({**first, "gw_seq": 8, "gw_at_ns": 105})
        tracker.consume(duplicate)
        self.assertEqual(duplicate.reason, "duplicate_event")
        public = tracker.view("A", now_ns=105)
        self.assertEqual(public.cursor.gw_seq, 8)
        self.assertEqual(public.ema, Decimal(3))
        self.assertEqual(public.residual.evidence[0].cursor.gw_seq, 2)
        path = Path("events.jsonl")
        with self.assertRaises(FormatError) as caught:
            ledger.apply({**first, "gw_seq": 9, "gw_at_ns": 106, "event_at_ns": 99},
                         location=FieldLocation(path, 19, "$"))
        self.assertEqual((caught.exception.file, caught.exception.line, caught.exception.field),
                         (path, 19, "event_id"))
        self.assertEqual(ledger.view().cursor.gw_seq, 8)

    def test_epoch_isolation_keeps_dispatch_facts_and_rejects_old_ack(self):
        from replaylab.ledger import GatewayLedger
        from test_progress import make_tracker

        ledger, tracker = GatewayLedger("run"), make_tracker()
        for row in [
            epoch(1, 0), epoch(2, 0, replica="B"), complete(3, 100, endpoint=100),
            work(4, 100, replica="B"),
            event("route_state", 5, 100, replica_id="A", epoch=1, sampled_ns=100,
                  state_version=1, ready=True, pending_prefill_tokens=0, unfinished_requests=0),
            event("step_start", 6, 100, replica_id="A", epoch=1, step_id="p", boundary_id="gpu", start_ns=100),
            decision(7, 100),
            event("dispatch_sent", 8, 100, request_id="q", target="A", reservation_id="r"),
        ]:
            tracker.consume(ledger.apply(row))
        previous = ledger.view()
        with self.assertRaises(TypeError):
            previous.epoch_events["A"]["route_state"].payload["ready"] = False
        future = ledger.apply(work(9, 101, generation=2))
        tracker.consume(future)
        self.assertEqual(future.reason, "future_epoch")
        tracker.consume(ledger.apply(epoch(10, 102, value=2)))
        self.assertNotIn("route_state", ledger.view().epoch_events["A"])
        self.assertNotIn("step_start", ledger.view().epoch_events["A"])
        self.assertIn("route_state", previous.epoch_events["A"])
        self.assertEqual(ledger.view().dispatches["q"].sent.event_id, "e8")
        self.assertIsNone(tracker.view("A", now_ns=102).ema)
        self.assertEqual(tracker.view("B", now_ns=102).work.event_id, "e4")
        stale = ledger.apply(complete(11, 103, endpoint=103, step="old"))
        tracker.consume(stale)
        self.assertEqual(stale.reason, "stale_epoch")
        ack = ledger.apply(event("dispatch_ack", 12, 104, request_id="q", replica_id="A",
                                 reservation_id="r", state_version=1, reason="accepted"))
        tracker.consume(ack)
        self.assertEqual(ack.reason, "stale_epoch")
        self.assertIsNone(tracker.view("A", now_ns=104).completion_end_ns)

    def test_t02_gap_discovered_later_affects_offline_prefix_and_completion_speed(self):
        from replaylab.ledger import EvidenceGap, GatewayLedger
        from test_progress import make_tracker

        ledger, tracker = GatewayLedger("run"), make_tracker()
        tracker.consume(ledger.apply(epoch(1, 0)))
        tracker.consume(ledger.apply(complete(2, 100, endpoint=100, duration=30)))
        original = tracker.view("A", now_ns=105, before_gw_seq=5)
        tracker.consume(ledger.apply(decision(5, 105)))
        tracker.consume(ledger.apply(event("tick", 6, 106)))
        gap = EvidenceGap(3, 4, ledger.view().cursor, frozenset({"completion"}), "missing gateway rows", "A")
        tracker.consume(ledger.mark_gap(gap))
        self.assertEqual(original.residual.state, "TRIGGER")
        self.assertEqual(ledger.view().dispatches["q"].decision.event_id, "e5")
        self.assertEqual(tracker.speed_factor("A", now_ns=106).reason, "completion_history_gap")
        tracker.consume(ledger.apply(work(7, 107, busy=False)))
        self.assertEqual(tracker.view("A", now_ns=107).state, "CLEAR")

        replay_ledger, replay = GatewayLedger("run"), make_tracker()
        replay.consume(replay_ledger.apply(epoch(1, 0)))
        replay.consume(replay_ledger.apply(complete(2, 100, endpoint=100, duration=30)))
        before_gap = replay.view("A", now_ns=102, before_gw_seq=3, retrospective_gaps=(gap,))
        retrospective = replay.view("A", now_ns=105, before_gw_seq=5, retrospective_gaps=(gap,))
        self.assertEqual(before_gap.residual.state, "TRIGGER")
        self.assertEqual(retrospective.residual.state, "UNKNOWN")
        self.assertEqual(retrospective.residual.evidence, original.residual.evidence)

        tracker.consume(ledger.apply(epoch(8, 108, value=2)))
        tracker.consume(ledger.apply(complete(9, 109, endpoint=109, generation=2, duration=30)))
        self.assertEqual(tracker.speed_factor("A", now_ns=109).value, Decimal(2))

    def test_work_gap_only_invalidates_work_and_later_work_restores_it(self):
        from replaylab.ledger import EvidenceGap, GatewayLedger
        from test_progress import make_tracker

        ledger, tracker = GatewayLedger("run"), make_tracker()
        for row in [epoch(1, 0), complete(2, 100, endpoint=100, duration=30),
                    work(3, 100), event("tick", 5, 106)]:
            tracker.consume(ledger.apply(row))
        gap = EvidenceGap(4, 4, ledger.view().cursor, frozenset({"work"}), "lost work sample", "A")
        tracker.consume(ledger.mark_gap(gap))
        view = tracker.view("A", now_ns=106)
        self.assertEqual((view.residual.state, view.gap.reason, view.state),
                         ("TRIGGER", "work_history_gap", "TRIGGER"))
        tracker.consume(ledger.apply(work(6, 106, version=2)))
        self.assertEqual(tracker.view("A", now_ns=106).gap.state, "TRIGGER")

    def test_gap_known_after_last_receipt_needs_no_synthetic_tick(self):
        from replaylab.ledger import EvidenceGap, GatewayCursor, GatewayLedger
        from test_progress import make_tracker

        ledger, tracker = GatewayLedger("run"), make_tracker()
        for row in [epoch(1, 0), complete(2, 100, endpoint=100, duration=30), decision(5, 105)]:
            tracker.consume(ledger.apply(row))
        gap = EvidenceGap(3, 4, GatewayCursor("run", 5, 106),
                          frozenset({"completion"}), "discovered after capture", "A")
        tracker.consume(ledger.mark_gap(gap))
        self.assertEqual(ledger.view().cursor.gw_at_ns, 105)
        self.assertEqual(tracker.view("A", now_ns=105).residual.state, "TRIGGER")
        self.assertEqual(tracker.view("A", now_ns=106).residual.reason, "completion_history_gap")
