from dataclasses import replace
from fractions import Fraction
import unittest

from test_contracts import plan, study
from test_pairing import bundle_fixture, paired_runs
from test_scoring import located, outcome, run_fixture


def configured(**overrides):
    data = study()
    data["pairing"].update(
        bootstrap_samples=256, bootstrap_interpolation="linear", min_independent_blocks=2,
        delta_goodput_requests=2,
    )
    data["pairing"].update(overrides)
    return data


class StatisticsTests(unittest.TestCase):
    def config(self, **overrides):
        from replaylab.statistics import StatisticsConfig

        return StatisticsConfig.from_study(located("study.json", configured(**overrides)))

    def test_bootstrap_samples_independent_blocks_and_is_order_stable(self):
        from replaylab.statistics import BlockEffect, bootstrap_interval, describe_blocks

        observations = [BlockEffect(("p",), 2), BlockEffect(("p",), 6), BlockEffect(("q",), 8)]
        config = self.config()
        described = describe_blocks(observations)
        self.assertEqual((described.n, described.mean, described.minimum, described.maximum), (2, 6, 4, 8))
        interval = bootstrap_interval(observations, config)
        self.assertEqual(interval, bootstrap_interval(list(reversed(observations)), config))
        self.assertEqual((interval.lower, interval.upper), (Fraction(4), Fraction(8)))

    def test_status_boundaries_harm_and_missing_parameters(self):
        from replaylab.statistics import BlockEffect, summarize_blocks

        for values, status, reason in [
            ([2, 2], "supported", None), ([1, 1], "meaningful_gain_excluded", None),
            ([-1, -2], "meaningful_gain_excluded", None), ([1, 4], "inconclusive", "interval_too_wide"),
            ([3], "inconclusive", "insufficient_independent_blocks"), ([], "no_valid_pairs", "no_valid_pairs"),
        ]:
            with self.subTest(values=values):
                report = summarize_blocks([BlockEffect((str(i),), val) for i, val in enumerate(values)],
                                          self.config(), validation=True)
                self.assertEqual((report.status, report.reason), (status, reason))
                self.assertEqual(report.stable_harm, values == [-1, -2])
        effects = [BlockEffect(("p",), 2), BlockEffect(("q",), 4)]
        missing = summarize_blocks(effects, self.config(bootstrap_samples=None), validation=True)
        self.assertEqual((missing.n, missing.mean, missing.interval, missing.reason),
                         (2, 3, None, "missing_inference_config"))

    def test_t11_guardrail_empty_unknown_and_failed_do_not_remove_effect(self):
        from replaylab.pairing import analyze_pairs
        from replaylab.statistics import StatisticsConfig, evaluate_b_guardrail, summarize_effects

        data = configured(min_independent_blocks=1)
        data["slo"]["b_generation_guardrail"] = dict(threshold_ns=2, max_new_violations=1)
        runs = paired_runs()
        original = plan()
        original["blocks"][0]["study_split"] = "validation"
        for run in runs:
            run.record["study_split"] = "validation"
        doc = located("study.json", data)
        paired = analyze_pairs(bundle_fixture(runs, plan_data=original), doc)
        config = StatisticsConfig.from_study(doc)
        main = paired.contrasts[0]
        self.assertEqual(evaluate_b_guardrail([main], config).blocks[0].reason, "empty_b_cohort")
        cohort = frozenset({"r1", "r2"})
        ref = replace(main.reference, score=replace(main.reference.score, b_existing_ids=cohort))
        candidate = replace(main.candidate, score=replace(main.candidate.score, b_existing_ids=cohort))
        for new, unknown, expected in [
            (("r1",), (), "pass"), (("r1", "r2"), (), "fail"), (("r1",), ("r2",), "unknown"),
        ]:
            row = replace(main, reference=ref, candidate=candidate,
                          delta=replace(main.delta, generation_new_violation_ids=new, generation_unknown_ids=unknown))
            result = evaluate_b_guardrail([row], config)
            self.assertEqual(result.status, expected)
            grouped = summarize_effects(replace(paired, contrasts=(row,)), config)
            [group] = grouped.groups
            self.assertEqual((group.effect.status, group.effect.n), ("supported", 1))
            self.assertEqual(group.joint_status, "supported" if expected == "pass" else "inconclusive")
            self.assertEqual(grouped.research_status, None)  # Synthetic algorithm fixture.

    def test_t11_aa_is_listed_with_raw_delta_and_anchor_diagnostics(self):
        from replaylab.pairing import analyze_pairs
        from replaylab.statistics import StatisticsConfig, summarize_effects

        cfg = located("study.json", configured())
        raw = plan()
        raw["blocks"][0].update(kind="aa", branches=[{"role": role, "delay_ns": None}
                                                     for role in ("aa_first", "aa_second")])
        pair = analyze_pairs(bundle_fixture([
            run_fixture("a", role="aa_first"), run_fixture("b", role="aa_second",
                                                          outcomes=[outcome("b", status="timeout")]),
        ], plan_data=raw), cfg)
        result = summarize_effects(pair, StatisticsConfig.from_study(cfg))
        self.assertEqual(result.groups, ())
        self.assertEqual(result.aa[0]["goodput_delta"], -1)
        self.assertEqual(result.aa[0]["anchors"]["reference"]["anchor_at_ns"], 10)
