from decimal import Decimal, localcontext
from pathlib import Path
import tempfile
import unittest

from replaylab.artifacts import read_artifact, write_artifact
from replaylab.contracts import ArtifactKind, FieldLocation, FormatError, LocatedDocument
from test_contracts import study
from test_gateway_order import complete, epoch, work


def make_config(**observation):
    from replaylab.progress import ProgressConfig

    data = study()
    data["observation"].update(work_ttl_ns=20, completion_ttl_ns=40, residual_ttl_ns=20)
    data["observation"].update(observation)
    return ProgressConfig.from_study(LocatedDocument(Path("study.json"), data, {}))


def make_tracker():
    from replaylab.progress import FrozenHealthProfile, ProgressTracker

    rows = [
        dict(replica_id="A", work_key="light", healthy_median_ns=10, gap_limit_ns=5,
             inflight_limit_ns=8, calibration_count=7),
        dict(replica_id="A", work_key="heavy", healthy_median_ns=20, gap_limit_ns=16,
             inflight_limit_ns=21, calibration_count=7),
    ]
    health = FrozenHealthProfile.from_document(LocatedDocument(Path("health_profile.csv"), rows, {}, True))
    return ProgressTracker(make_config(), health)


class ProgressTests(unittest.TestCase):
    def setUp(self):
        from replaylab.ledger import GatewayLedger

        self.ledger, self.tracker = GatewayLedger("run"), make_tracker()
        self.consume(epoch(1, 0))

    def consume(self, row, *, location=None):
        return self.tracker.consume(self.ledger.apply(row, location=location))

    def test_first_ema_alpha_update_threshold_and_inclusive_residual_ttl(self):
        self.consume(complete(2, 100, endpoint=100, duration=10))
        self.assertEqual(self.tracker.view("A", now_ns=100).ema, Decimal(1))
        self.consume(complete(3, 105, endpoint=105, duration=30, step="s2"))
        for time, expected in [(105, "CLEAR"), (125, "CLEAR"), (126, "UNKNOWN")]:
            view = self.tracker.view("A", now_ns=time)
            self.assertEqual(view.ema, Decimal(2))
            self.assertEqual(view.last_used_end_ns, 105)
            self.assertEqual(view.residual.state, expected)

    def test_same_iteration_new_event_id_deduplicates_all_public_fields(self):
        row = complete(2, 100, endpoint=100, duration=30)
        self.consume(row)
        update = self.consume(complete(3, 104, endpoint=100, duration=30))
        self.assertEqual(update.reason, "duplicate_completion")
        self.assertEqual(self.tracker.view("A", now_ns=104).residual.evidence[0].event_id, "e2")
        self.assertEqual(self.tracker.view("A", now_ns=104).last_used_end_ns, 100)
        for key, value in [("boundary_id", "wrong"), ("end_ns", 99),
                           ("duration_ns", 20), ("work_key", "heavy")]:
            other = {**row, "gw_seq": 4, "gw_at_ns": 105, "event_id": "other"}
            other["payload"] = {**row["payload"], key: value}
            from replaylab.ledger import GatewayLedger

            ledger, tracker = GatewayLedger("run"), make_tracker()
            tracker.consume(ledger.apply(epoch(1, 0)))
            tracker.consume(ledger.apply(row))
            with self.subTest(field=key), self.assertRaises(FormatError) as caught:
                tracker.consume(ledger.apply(other, location=FieldLocation(Path("events.jsonl"), 11, "$")))
            self.assertEqual((caught.exception.line, caught.exception.field), (11, f"payload.{key}"))
            self.assertEqual(tracker.view("A", now_ns=105).cursor.gw_seq, 2)

    def test_different_iterations_same_end_both_update_at_receive_sequence(self):
        self.consume(complete(2, 100, endpoint=100, duration=10))
        self.consume(complete(3, 100, endpoint=100, duration=30, step="s2"))
        view = self.tracker.view("A", now_ns=100)
        self.assertEqual(view.ema, Decimal(2))
        self.assertEqual(view.completion.event_id, "e3")
        self.assertEqual(view.residual.evidence[0].event_id, "e3")

    def test_invalid_duration_advances_watermark_and_older_completion_can_seal(self):
        self.consume(complete(2, 100, endpoint=100, duration=30))
        invalid = self.consume(complete(3, 105, endpoint=105, duration=0, step="invalid"))
        older = self.consume(complete(4, 106, endpoint=103, duration=10, step="late"))
        self.assertEqual((invalid.reason, older.reason), ("invalid_duration", "older_completion"))
        self.assertEqual(older.completed_step, ("A", 1, "late"))
        repeated = self.consume(complete(5, 107, endpoint=103, duration=10, step="late"))
        self.assertEqual(repeated.reason, "duplicate_completion")
        view = self.tracker.view("A", now_ns=107)
        self.assertEqual((view.completion_end_ns, view.last_used_end_ns, view.ema), (105, 100, Decimal(3)))
        self.assertEqual(view.completion.event_id, "e3")
        self.assertEqual(view.residual.evidence[0].event_id, "e2")

    def test_endpoint_in_future_is_audited_without_sealing_or_moving_watermark(self):
        update = self.consume(complete(2, 100, endpoint=101))
        self.assertEqual(update.reason, "completion_end_in_future")
        self.assertIsNone(update.completed_step)
        view = self.tracker.view("A", now_ns=101)
        self.assertIsNone(view.completion_end_ns)
        self.assertIsNone(view.ema)

    def test_conditioned_gap_busy_backlog_idle_and_threshold(self):
        self.consume(complete(2, 100, endpoint=100))
        self.consume(work(3, 101))
        self.assertEqual(self.tracker.view("A", now_ns=105).gap.state, "CLEAR")
        self.assertEqual(self.tracker.view("A", now_ns=106).gap.state, "TRIGGER")
        self.consume(work(4, 107, busy=False, backlog=True, version=2))
        self.assertEqual(self.tracker.view("A", now_ns=107).gap.state, "TRIGGER")
        self.consume(work(5, 108, busy=False, version=3, key="uncovered"))
        view = self.tracker.view("A", now_ns=108)
        self.assertEqual((view.gap.state, view.gap.reason), ("CLEAR", "idle"))
        self.assertEqual(self.tracker.view("A", now_ns=129).gap.reason, "work_expired")
        self.consume(work(6, 145, version=4))
        self.assertEqual(self.tracker.view("A", now_ns=145).gap.reason, "completion_expired")

    def test_uncovered_current_bucket_keeps_existing_residual_and_alert_threshold_is_separate(self):
        from replaylab.progress import AlertThresholds
        from test_contracts import put_json

        self.consume(complete(2, 100, endpoint=100, duration=30))
        self.consume(complete(3, 105, endpoint=105, duration=15, key="uncovered", step="unknown"))
        self.consume(work(4, 106, key="uncovered"))
        common = self.tracker.view("A", now_ns=107)
        self.assertEqual((common.gap.state, common.gap.reason), ("UNKNOWN", "missing_health_bucket"))
        self.assertEqual((common.ema, common.last_used_end_ns, common.state), (Decimal(3), 100, "TRIGGER"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "study.json"
            data = study()
            data["observation"]["alert_policies"] = {
                "same_budget": {"threshold": 3, "gap_limits": {"A": {"light": 12}}},
            }
            put_json(path, data)
            loaded = read_artifact(path, ArtifactKind.STUDY)
            stricter = AlertThresholds.from_study(loaded, policy_id="same_budget")
        other = self.tracker.view("A", now_ns=107, thresholds=stricter)
        self.assertEqual((other.ema, other.residual.evidence, other.state),
                         (common.ema, common.residual.evidence, "CLEAR"))
        self.assertEqual(self.tracker.speed_factor("A", now_ns=107).value, Decimal(2))
        self.consume(work(5, 112, version=2))
        baseline = self.tracker.view("A", now_ns=112)
        alternate = self.tracker.view("A", now_ns=112, thresholds=stricter)
        self.assertEqual((baseline.gap.state, alternate.gap.state), ("TRIGGER", "CLEAR"))
        self.assertEqual(baseline.gap.evidence, alternate.gap.evidence)

    def test_late_lower_work_version_cannot_replace_current_version(self):
        self.consume(work(2, 100, version=2, key="heavy"))
        update = self.consume(work(3, 105, sampled=95, version=1))
        self.assertEqual(update.reason, "older_work_version")
        self.assertEqual(self.tracker.view("A", now_ns=105).work.payload["work_key"], "heavy")

    def test_ema_ignores_callers_decimal_context(self):
        with localcontext() as context:
            context.prec = 1
            self.consume(complete(2, 100, endpoint=100, duration=17))
            self.consume(complete(3, 105, endpoint=105, duration=24, step="s2"))
            self.assertEqual(self.tracker.view("A", now_ns=105).ema, Decimal("2.05"))


class CalibrationTests(unittest.TestCase):
    def test_frozen_integer_boundaries_place_equality_in_right_bucket(self):
        from replaylab.progress import bucket_workload

        config = make_config(workload_buckets={"prefill": [1, 3], "decode": [2]})
        left = bucket_workload({"prefill": 1, "decode": 1}, config)
        right = bucket_workload({"decode": 2, "prefill": 1}, config)
        self.assertEqual(left, '[["decode",0],["prefill",1]]')
        self.assertEqual(right, '[["decode",1],["prefill",1]]')
        self.assertEqual(bucket_workload({"decode": 3, "prefill": 2}, config), right)
        self.assertIsNone(bucket_workload({"decode": 1}, config))
        with self.assertRaises(FormatError):
            make_config(workload_buckets={"decode": [2, 2]})

    def test_integer_median_uses_ties_to_even_and_ignores_validation(self):
        from replaylab.progress import CalibrationRun, CalibrationSample, FrozenHealthProfile, build_health_profile

        config = make_config(conditional_thresholds={
            "A": {"down": {"gap_limit_ns": 5, "inflight_limit_ns": 10},
                  "up": {"gap_limit_ns": 6, "inflight_limit_ns": 11}},
        })
        base = 2 ** 60
        samples = [CalibrationSample("A", key, duration) for key, duration in [
            ("down", base), ("down", base + 1), ("up", base + 1), ("up", base + 2),
        ]]
        runs = [
            CalibrationRun("valid", "v1", "validation", "synthetic", True, tuple(samples) * 2),
            CalibrationRun("cal", "c1", "calibration", "synthetic", True, tuple(samples)),
            CalibrationRun("degraded", "c2", "calibration", "synthetic", False, tuple(samples)),
        ]
        profile = build_health_profile(runs, config)
        rows = profile.to_rows()
        self.assertEqual([(row["work_key"], row["healthy_median_ns"], row["calibration_count"])
                          for row in rows], [("down", base, 2), ("up", base + 2, 2)])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "health_profile.csv"
            write_artifact(path, ArtifactKind.HEALTH_PROFILE, rows)
            self.assertEqual(FrozenHealthProfile.from_document(read_artifact(path, ArtifactKind.HEALTH_PROFILE)),
                             profile)

    def test_missing_parameter_blocks_and_duplicate_calibration_attempt_is_rejected(self):
        from replaylab.progress import CalibrationRun, CalibrationSample, ProgressBlocked, build_health_profile

        with self.assertRaises(ProgressBlocked) as caught:
            make_config(residual_ttl_ns=None)
        self.assertEqual(caught.exception.missing_fields, ("observation.residual_ttl_ns",))
        runs = [CalibrationRun(name, "same", "calibration", "synthetic", True,
                               (CalibrationSample("A", "light", 5),)) for name in ("a", "b")]
        with self.assertRaisesRegex(ValueError, "attempt_id"):
            build_health_profile(runs, make_config())
        with self.assertRaises(ProgressBlocked) as caught:
            build_health_profile(runs[:1], make_config())
        self.assertIn("gap_limit_ns", caught.exception.missing_fields[0])
