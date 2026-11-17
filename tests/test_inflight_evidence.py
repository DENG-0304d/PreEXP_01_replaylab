from pathlib import Path
import unittest

from replaylab.contracts import LocatedDocument
from replaylab.ledger import GatewayLedger
from test_gateway_order import complete, epoch, event, work
from test_progress import make_tracker


def start(seq, at, *, step="s", version=1, generation=1, event_id=None):
    return event("step_start", seq, at, event_id=event_id or f"start-{step}",
                 replica_id="A", epoch=generation, step_id=step, start_ns=at,
                 boundary_id="gpu", work_version=version)


def snapshot(seq, at, sample, *, step="s", version=1, generation=1, status="PENDING_VERIFIED",
             marker="recorded", reason=None, start_id=None, source_seq=None, event_id=None):
    row = event("inflight_snapshot", seq, at, source=sample, event_id=event_id,
                replica_id="A", epoch=generation, step_id=step, sample_ns=sample,
                start_event_id=start_id or f"start-{step}", work_version=version,
                boundary_id="gpu", status=status, marker_status=marker, validation_reason=reason)
    if source_seq is not None:
        row["payload"]["source_seq"] = source_seq
    return row


class InflightTests(unittest.TestCase):
    def trackers(self, *, age_error=0, duration_error=0):
        from replaylab.inflight import ClockBounds, InflightConfig, InflightTracker

        progress = make_tracker()
        tracker = InflightTracker(InflightConfig(10), progress.health_profile,
                                  clock_bounds=ClockBounds(age_error, duration_error, 0))
        ledger = GatewayLedger("run")

        def apply(row):
            update = ledger.apply(row)
            return tracker.consume(update, progress.consume(update))

        apply(epoch(1, 0))
        return tracker, progress, apply

    def test_t03_visible_work_version_belonging_to_other_iteration_is_unknown(self):
        tracker, _, apply = self.trackers()
        for row in [work(2, 0, version=1), start(3, 0), work(4, 1, version=2, key="heavy"),
                    start(5, 1, step="other", version=2), snapshot(6, 100, 100, version=2)]:
            apply(row)
        [case] = [case for case in tracker.view("A", now_ns=100).iterations if case.step_id == "s"]
        self.assertEqual((case.raw_status, case.state, case.reason),
                         ("PENDING_VERIFIED", "UNKNOWN", "work_version_mismatch"))
        self.assertEqual(case.snapshot.event_id, "e6")

    def test_t03_unverified_reasons_and_done_before_public_completion(self):
        for marker, reason in [("not_recorded", "marker_not_recorded"), ("wrong_epoch", "marker_epoch_mismatch"),
                               ("coverage_unknown", "coverage_unknown"), ("query_error", "query_error")]:
            with self.subTest(marker=marker):
                tracker, progress, apply = self.trackers()
                apply(work(2, 0))
                apply(start(3, 0))
                apply(snapshot(4, 100, 100, status="UNVERIFIED", marker=marker, reason=reason))
                [case] = tracker.view("A", now_ns=100).iterations
                self.assertEqual((case.raw_status, case.state, case.reason, case.snapshot.event_id),
                                 ("UNVERIFIED", "UNKNOWN", reason, "e4"))
                apply(snapshot(5, 101, 101, status="DONE_VERIFIED"))
                self.assertEqual(tracker.view("A", now_ns=101).iterations[0].reason, "device_completed")
                self.assertIsNone(progress.view("A", now_ns=101).ema)
                apply(complete(6, 105, endpoint=101))
                self.assertEqual(progress.view("A", now_ns=105).last_used_end_ns, 101)

    def test_t04_expired_at_receipt_old_pending_and_epoch_reset_are_distinct(self):
        tracker, progress, apply = self.trackers()
        apply(work(2, 0))
        apply(start(3, 0))
        apply(snapshot(4, 120, 100))
        self.assertEqual(tracker.view("A", now_ns=120).iterations[0].reason, "evidence_expired")
        apply(complete(5, 121, endpoint=115))
        late = apply(snapshot(6, 121, 110))
        self.assertEqual(late.reason, "iteration_sealed")
        self.assertEqual(tracker.view("A", now_ns=121).iterations[0].snapshot.event_id, "e4")
        self.assertEqual(progress.view("A", now_ns=121).completion_end_ns, 115)
        apply(epoch(7, 122, value=2))
        self.assertEqual(apply(snapshot(8, 122, 122)).reason, "stale_epoch")
        self.assertEqual(tracker.view("A", now_ns=122).iterations, ())
        self.assertIsNone(progress.view("A", now_ns=122).ema)
        apply(work(9, 123, generation=2))
        apply(start(10, 123, generation=2, event_id="new-start"))
        apply(snapshot(11, 134, 134, generation=2, start_id="new-start"))
        self.assertEqual(tracker.view("A", now_ns=134).state, "TRIGGER")

    def test_same_tick_cursor_is_forward_only_and_old_source_sample_cannot_replace(self):
        tracker, _, apply = self.trackers()
        apply(work(2, 0))
        apply(start(3, 0))
        apply(snapshot(4, 100, 100, source_seq=2))
        before = tracker.view("A", now_ns=100, before_gw_seq=5)
        self.assertEqual(before.state, "TRIGGER")
        self.assertEqual(apply(snapshot(5, 100, 100, source_seq=1)).reason, "older_snapshot")
        self.assertEqual(tracker.view("A", now_ns=100).iterations[0].snapshot.event_id, "e4")
        with self.assertRaises(ValueError):
            tracker.view("A", now_ns=100, before_gw_seq=5)
        apply(snapshot(6, 100, 100, status="DONE_VERIFIED", source_seq=3))
        self.assertEqual(tracker.view("A", now_ns=100).state, "CLEAR")
        self.assertEqual(before.state, "TRIGGER")

    def test_t12_frozen_thresholds_parallel_iterations_ttl_and_clock_uncertainty(self):
        from replaylab.inflight import InflightThresholds

        tracker, _, apply = self.trackers()
        for row in [work(2, 0), start(3, 0), work(4, 0, version=2, key="heavy"),
                    start(5, 0, step="heavy", version=2), snapshot(6, 100, 100),
                    snapshot(7, 100, 100, step="heavy", version=2)]:
            apply(row)
        limits = InflightThresholds({("A", "light"): 99, ("A", "heavy"): 100})
        cases = {case.step_id: case.state for case in tracker.view("A", now_ns=110, thresholds=limits).iterations}
        self.assertEqual(cases, {"s": "TRIGGER", "heavy": "CLEAR"})
        self.assertEqual(tracker.view("A", now_ns=111, thresholds=limits).state, "UNKNOWN")
        tracker, _, apply = self.trackers(age_error=1)
        for row in [work(2, 0), start(3, 0), snapshot(4, 108, 100)]:
            apply(row)
        self.assertEqual(tracker.view("A", now_ns=109).state, "TRIGGER")
        [case] = tracker.view("A", now_ns=110).iterations
        self.assertEqual((case.raw_status, case.state, case.reason),
                         ("PENDING_VERIFIED", "UNKNOWN", "clock_age_uncertain"))

    def test_late_start_and_work_reference_become_visible_without_retroactive_trigger(self):
        tracker, _, apply = self.trackers()
        apply(snapshot(2, 10, 10))
        before = tracker.view("A", now_ns=10)
        self.assertEqual(before.iterations[0].reason, "missing_step_start")
        row = start(3, 0)
        apply({**row, "gw_at_ns": 11})
        self.assertEqual(tracker.view("A", now_ns=11).iterations[0].reason, "missing_work_version")
        apply(work(4, 12, sampled=0))
        self.assertEqual(tracker.view("A", now_ns=12).state, "TRIGGER")
        self.assertEqual(before.state, "UNKNOWN")
