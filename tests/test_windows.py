from dataclasses import replace
import unittest

from replaylab.artifacts import ImportedBundle
from replaylab.contracts import ArtifactKind, WINDOW_KEYS
from test_contracts import plan
from test_pairing import paired_runs
from test_scoring import events, located
from test_statistics import configured


def window_fixture(*, arms=None, planned=None, minimum=1):
    from replaylab.pairing import analyze_pairs
    from replaylab.statistics import StatisticsConfig, summarize_effects

    arms = arms or ((0, 100_000_000),)
    planned = planned or arms
    blocks, runs = [], []
    anchor = 10
    for ordinal, (have, scheduled) in enumerate(zip(arms, planned), 1):
        pair_id = f"p{ordinal}"
        block = plan()["blocks"][0]
        block.update(pair_id=pair_id, study_split="validation")
        block["branches"] = [{"role": "reference", "delay_ns": None},
                             {"role": "immediate", "delay_ns": 0}]
        block["branches"].extend({"role": "delayed", "delay_ns": delay} for delay in scheduled if delay)
        blocks.append(block)
        base, immediate, delayed = paired_runs()
        selected = [(base, None)]
        if 0 in have:
            selected.append((immediate, 0))
        if 100_000_000 in have:
            selected.append((delayed, 100_000_000))
        for position, (run, delay) in enumerate(selected):
            rec = run.record
            run_id = f"{pair_id}-{rec['role']}"
            rec.update(run_id=run_id, pair_id=pair_id, study_split="validation", planned_delay_ns=delay,
                       anchor_definition_id="frozen-anchor")
            rec["independent_start"].update(attempt_id=run_id, started_at_ns=ordinal * 10 + position)
            doc = run.artifacts[ArtifactKind.EVENTS]
            specs = []
            for row in doc.data:
                at, kind = row["gw_at_ns"], row["type"]
                if delay and kind in {"rule_armed", "dispatch_decision", "dispatch_sent"}:
                    at += delay - 4
                if kind == "tick":
                    at = 200_000_000
                specs.append((at, kind, row["payload"]))
            run.artifacts[ArtifactKind.EVENTS] = located("events.jsonl", events(run_id, *specs))
            runs.append(run)
    manifest = dict(study_id="s", config_sha256="a" * 64, plan_sha256="b" * 64, runs=[run.record for run in runs])
    bundle = ImportedBundle(located("manifest.json", manifest), located("plan.json", {"schema_version": 2, "blocks": blocks}),
                            tuple(runs), ())
    study = located("study.json", configured(min_independent_blocks=minimum, primary_contrast="immediate-delayed"))
    pairing = analyze_pairs(bundle, study)
    stats = summarize_effects(pairing, StatisticsConfig.from_study(study))
    return bundle, pairing, stats, anchor


def shadow_row(*, pair="p1", baseline=10, inflight=100_000_010, references=()):
    row = dict(zip(WINDOW_KEYS, ("s", "a" * 64, "b" * 64, "synthetic", "validation", "example", pair, "a")))
    row.update(baseline_intended_arm_ns=baseline, inflight_intended_arm_ns=inflight,
               evidence_refs_json=list(references), baseline_caught_state="caught", coverage_cell="both_caught",
               inflight_goodput_delta=999)
    return row


class WindowTests(unittest.TestCase):
    def base(self, **kwargs):
        from replaylab.statistics import frozen_scope
        from replaylab.windows import ShadowWindowInput, SignalEvidence, build_action_index

        bundle, pairing, stats, anchor = window_fixture(**kwargs)
        scope = frozen_scope(pairing.study_id, bundle.runs[0].record)
        key = tuple(shadow_row()[name] for name in WINDOW_KEYS)
        window = ShadowWindowInput(key, anchor, scope, "same_threshold_or",
                                   SignalEvidence(10, 10, assessment="executable"),
                                   SignalEvidence(100_000_010, 100_000_010, assessment="executable"))
        return bundle, window, build_action_index(pairing, stats, bundle=bundle)

    def test_t09_between_neighbours_and_planned_without_valid_result_are_distinct(self):
        from replaylab.windows import match_delay

        _, window, index = self.base()
        between = match_delay(window, intended_arm_ns=50_000_010, index=index)
        self.assertEqual((between.state, between.neighbors_ns, between.observations),
                         ("UNTESTED_DELAY", (0, 100_000_000), ()))
        exact = match_delay(window, intended_arm_ns=100_000_010, index=index)
        self.assertEqual((exact.state, exact.delay_ns), ("MATCHED", 100_000_000))
        [observed] = exact.observations
        self.assertEqual((observed.run_id, observed.actual_arm_ns, observed.first_redirect_ns, observed.goodput_delta),
                         ("p1-delayed", 100_000_010, 100_000_010, 2))
        _, window, index = self.base(arms=((0, 100_000_000),), planned=((0, 50_000_000, 100_000_000),))
        failed = match_delay(window, intended_arm_ns=50_000_010, index=index)
        self.assertEqual((failed.state, failed.observations), ("PLANNED_NO_VALID_PAIR", ()))
        self.assertIn("branch_missing", failed.reasons)

    def test_t09_planned_failed_attempt_keeps_original_reason(self):
        from replaylab.pairing import analyze_pairs
        from replaylab.statistics import StatisticsConfig, summarize_effects
        from replaylab.windows import build_action_index, match_delay

        bundle, window, _ = self.base(planned=((0, 50_000_000, 100_000_000),))
        blocked = dict(bundle.runs[0].record, run_id="failed-50", role="delayed", planned_delay_ns=50_000_000,
                       run_status="failed")
        bundle.manifest.data["runs"].append(blocked)
        exclusion = dict(run_id="failed-50", branch=blocked, error=None, reason="runner_stopped")
        bundle = replace(bundle, excluded_runs=(exclusion,))
        study = located("study.json", configured(min_independent_blocks=1))
        pairing = analyze_pairs(bundle, study)
        index = build_action_index(pairing, summarize_effects(pairing, StatisticsConfig.from_study(study)), bundle=bundle)
        result = match_delay(window, intended_arm_ns=50_000_010, index=index)
        self.assertEqual(result.state, "PLANNED_NO_VALID_PAIR")
        self.assertEqual(result.attempts, (("failed-50", "failed"),))
        self.assertEqual(result.excluded[0]["issues"][0]["detail"][0]["reason"], "runner_stopped")

    def test_t11_group_difference_uses_only_common_blocks_and_separate_opportunities(self):
        from replaylab.windows import analyze_windows

        _, window, index = self.base(arms=((0, 100_000_000), (0,), (0,), (100_000_000,), (100_000_000,)),
                                     planned=((0, 100_000_000),) * 5, minimum=2)
        [row] = analyze_windows((window,), index=index).windows
        self.assertEqual((row.baseline.match.effect.n, row.inflight.match.effect.n), (3, 3))
        self.assertEqual((row.common_effect.n, row.common_effect.mean, row.coverage_cell), (1, -8, "both_caught"))
        self.assertIsNone(row.common_effect.interval)
        self.assertEqual(row.opportunity_state, "VERIFIED_OPPORTUNITY")
        self.assertEqual(row.report()["closed_loop_goodput_delta"], None)

    def test_t11_empty_overlap_does_not_subtract_disjoint_means(self):
        from replaylab.windows import analyze_windows

        _, window, index = self.base(arms=((0,), (100_000_000,)), planned=((0, 100_000_000),) * 2)
        [row] = analyze_windows((window,), index=index).windows
        self.assertEqual(row.coverage_cell, "both_caught")
        self.assertEqual(row.common_effect.n, 0)
        self.assertIsNone(row.csv_row()["goodput_delta_between_groups"])
        self.assertEqual(row.report()["common_effect"]["reason"], "no_common_independent_blocks")

    def test_t11_unknown_untested_and_missed_have_separate_denominators(self):
        from replaylab.windows import SignalEvidence, analyze_windows

        _, window, index = self.base()
        no_candidate = SignalEvidence(None, None, assessment="no_candidate", reason="no_eligible_candidate")
        unknown = replace(window.inflight, assessment="unknown", reason="timeline_missing")
        windows = (
            window, replace(window, baseline=no_candidate, inflight=no_candidate),
            replace(window, inflight=unknown),
            replace(window, inflight=replace(window.inflight, intended_arm_ns=50_000_010)),
        )
        result = analyze_windows(windows, index=index)
        self.assertEqual([item.coverage_cell for item in result.windows], ["both_caught", "both_missed", None, None])
        self.assertEqual([item.inflight.caught_state for item in result.windows],
                         ["caught", "missed", "unknown", "untested"])
        summary = result.summary()
        self.assertEqual((summary["total_window_n"], summary["verified_opportunity_n"], summary["four_cell_n"]),
                         (4, 4, 2))
        _, window, missing = self.base(arms=((),), planned=((0, 100_000_000),))
        [row] = analyze_windows((window,), index=missing).windows
        self.assertEqual((row.opportunity_state, row.coverage_cell), ("INSUFFICIENT_EVIDENCE", None))

    def test_frozen_scope_and_missing_timeline_do_not_authorize_coverage(self):
        from replaylab.statistics import StatisticsConfig, summarize_effects
        from replaylab.pairing import analyze_pairs
        from replaylab.windows import analyze_windows, build_action_index, match_delay, read_shadow_windows

        bundle, window, index = self.base()
        data = located("windows.csv", [shadow_row()])
        [imported] = read_shadow_windows(data, bundle=bundle)
        [row] = analyze_windows((imported,), index=index).windows
        self.assertEqual((row.inflight.match.state, row.inflight.caught_state, row.coverage_cell),
                         ("MATCHED", "unknown", None))
        self.assertEqual(row.inflight.match.observations[0].goodput_delta, 2)
        self.assertEqual(row.inflight.reason, "timeline_missing")
        for scope in [replace(window.scope, source_kind="real"),
                      replace(window.scope, anchor_definition=("other",)),
                      replace(window.scope, frozen_action=("different-rule",))]:
            result = match_delay(replace(window, scope=scope), intended_arm_ns=100_000_010, index=index)
            self.assertFalse(result.observations)
        data = located("study.json", configured(min_independent_blocks=2))
        pairing = analyze_pairs(bundle, data)
        stats = summarize_effects(pairing, StatisticsConfig.from_study(data))
        result = match_delay(window, intended_arm_ns=10, index=build_action_index(pairing, stats, bundle=bundle))
        self.assertEqual((result.state, result.effect.reason), ("MATCHED", "insufficient_independent_blocks"))

    def test_missing_csv_window_retains_planned_window_and_scope(self):
        from replaylab.windows import analyze_windows, read_shadow_windows

        bundle, _, index = self.base(arms=((0, 100_000_000), (0, 100_000_000)))
        imported = read_shadow_windows(located("windows.csv", [shadow_row()]), bundle=bundle)
        result = analyze_windows(imported, index=index)
        self.assertEqual(result.summary()["total_window_n"], 2)
        self.assertEqual(result.windows[1].window.reason, "missing_shadow_window")
        self.assertIsNone(result.windows[1].coverage_cell)
