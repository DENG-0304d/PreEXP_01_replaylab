"""Descriptive estimates, independent-block intervals, and research decisions."""

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from fractions import Fraction
from hashlib import sha256

from .contracts import CONFIDENCE, NONNEG_NUMBER, POS_INT, SHA256, TEXT, UINT, ArtifactKind, LocatedDocument, check, choice
from .pairing import ContrastValidation, PairingResult


@dataclass(frozen=True)
class StatisticsConfig:
    delta_goodput_requests: int | None
    min_independent_blocks: int | None
    interval_algorithm: str | None
    confidence_level: Fraction | None
    seed: int | None
    bootstrap_samples: int | None
    bootstrap_interpolation: str | None
    primary_contrast: str | None
    primary_delay_ns: int | None
    capture_budget: Mapping[str, Fraction] | None
    generation_max_new_violations: int | None
    missing_fields: tuple[str, ...]

    @classmethod
    def from_study(cls, study: LocatedDocument) -> "StatisticsConfig":
        pairing = study.data.get("pairing") or {}
        specs = {
            "delta_goodput_requests": UINT, "min_independent_blocks": POS_INT,
            "interval_algorithm": choice("percentile_bootstrap", "descriptive_only"),
            "confidence_level": CONFIDENCE, "seed": int, "bootstrap_samples": POS_INT,
            "bootstrap_interpolation": choice("linear"),
            "primary_contrast": choice("immediate-reference", "delayed-reference", "immediate-delayed"),
            "primary_delay_ns": UINT,
        }
        values = {name: pairing.get(name) for name in specs}
        for name, spec in specs.items():
            if values[name] is not None:
                check(study, values[name], spec, ("pairing", name))
        missing = tuple(f"pairing.{name}" for name in specs
                        if name != "primary_delay_ns" and values[name] is None)
        if values["confidence_level"] is not None:
            values["confidence_level"] = Fraction(str(values["confidence_level"]))
        budget = pairing.get("capture_perturbation_budget")
        if budget is not None:
            check(study, budget, dict, ("pairing", "capture_perturbation_budget"))
            for metric, value in budget.items():
                check(study, value, NONNEG_NUMBER, ("pairing", "capture_perturbation_budget", metric))
            budget = {metric: Fraction(str(limit)) for metric, limit in budget.items()}
        guardrail = (study.data.get("slo") or {}).get("b_generation_guardrail")
        return cls(**values, capture_budget=budget, missing_fields=missing,
                   generation_max_new_violations=None if guardrail is None else guardrail["max_new_violations"])


@dataclass(frozen=True, order=True)
class FrozenScope:
    study_id: str
    source_kind: str
    condition_id: str
    anchor_id: str
    anchor_at_ns: int
    anchor_definition: tuple[str, ...]
    frozen_action: tuple[str, ...]


def frozen_scope(study_id: str, record: Mapping) -> FrozenScope | None:
    identity = ("source_kind", "condition_id", "anchor_id")
    definition = ("arrival_trace_id", "prehistory_id", "eval_cohort_id")
    action = ("intervention_digest", "router_id", "router_digest", "diversion_rule_id", "slo_id", "clock_id")
    required = (*identity, *definition, *action)
    if any(not TEXT.accepts(record.get(name)) for name in required) or not UINT.accepts(record.get("anchor_at_ns")):
        return None
    anchor_id = record.get("anchor_definition_id")
    if anchor_id is not None and not TEXT.accepts(anchor_id):
        return None
    files = record.get("files")
    if not isinstance(files, Mapping):
        return None
    references = [files.get(name) for name in ("arrivals.csv", "inputs.jsonl")]
    if not all(isinstance(ref, Mapping) and SHA256.accepts(ref.get("sha256")) for ref in references):
        return None
    hashes = tuple(ref["sha256"].lower() for ref in references)
    return FrozenScope(
        study_id, *(record[name] for name in identity), record["anchor_at_ns"],
        (anchor_id or "", *(record[name] for name in definition), *hashes),
        tuple(record[name] for name in action),
    )


def independent_block(contrast: ContrastValidation) -> tuple[str, ...]:
    """Anchors and delays belong to their enclosing randomized replay block."""
    return contrast.planned.block[:3]


@dataclass(frozen=True)
class BlockEffect:
    block: tuple[str, ...]
    value: int | Fraction


@dataclass(frozen=True)
class EffectInterval:
    lower: Fraction
    upper: Fraction


@dataclass(frozen=True)
class Description:
    observations: tuple[BlockEffect, ...]
    n: int
    mean: Fraction | None
    minimum: Fraction | None
    maximum: Fraction | None


def describe_blocks(effects: Sequence[BlockEffect]) -> Description:
    grouped = defaultdict(list)
    for effect in effects:
        grouped[effect.block].append(Fraction(effect.value))
    observations = tuple(BlockEffect(key, sum(values) / len(values)) for key, values in sorted(grouped.items()))
    values = [item.value for item in observations]
    if not values:
        return Description((), 0, None, None, None)
    return Description(observations, len(values), sum(values) / len(values), min(values), max(values))


def _quantile(values: list[Fraction], p: Fraction) -> Fraction:
    rank = p * (len(values) - 1)
    i = rank.numerator // rank.denominator
    return values[i] + (rank - i) * (values[min(i + 1, len(values) - 1)] - values[i])


def bootstrap_interval(effects: Sequence[BlockEffect], config: StatisticsConfig) -> EffectInterval:
    rows = describe_blocks(effects).observations
    if (not rows or config.interval_algorithm != "percentile_bootstrap" or config.seed is None
            or config.bootstrap_samples is None or config.confidence_level is None
            or config.bootstrap_interpolation != "linear"):
        raise ValueError("bootstrap requires independent blocks and frozen percentile settings")
    n = len(rows)
    # A version-independent uniform stream. Rejection avoids modulo bias.
    ceiling = (1 << 64) - ((1 << 64) % n)
    prefix = f"{config.seed}:".encode("ascii")
    counter = 0
    means = []
    for _ in range(config.bootstrap_samples):
        sample = Fraction(0)
        for _ in range(n):
            while True:
                draw = int.from_bytes(sha256(prefix + str(counter).encode("ascii")).digest()[:8], "big")
                counter += 1
                if draw < ceiling:
                    break
            sample += rows[draw % n].value
        means.append(sample / n)
    means.sort()
    p = (1 - config.confidence_level) / 2
    return EffectInterval(_quantile(means, p), _quantile(means, 1 - p))


@dataclass(frozen=True)
class EffectReport:
    description: Description
    interval: EffectInterval | None
    status: str
    reason: str | None = None
    stable_harm: bool = False

    @property
    def n(self) -> int:
        return self.description.n

    @property
    def mean(self) -> Fraction | None:
        return self.description.mean

    def report(self) -> dict:
        return {
            "n": self.n, "mean": report_number(self.mean),
            "minimum": report_number(self.description.minimum), "maximum": report_number(self.description.maximum),
            "blocks": [dict(block=row.block, value=report_number(row.value)) for row in self.description.observations],
            "interval": (None if self.interval is None else
                         dict(lower=report_number(self.interval.lower), upper=report_number(self.interval.upper))),
            "status": self.status, "reason": self.reason, "stable_harm": self.stable_harm,
        }


def report_number(value: int | Fraction | None) -> int | float | None:
    if value is None:
        return None
    return value.numerator if value.denominator == 1 else float(value)


def summarize_blocks(
    effects: Sequence[BlockEffect], config: StatisticsConfig, *, validation: bool,
    validity_reason: str | None = None,
) -> EffectReport:
    description = describe_blocks(effects)
    if not description.n:
        return EffectReport(description, None, "no_valid_pairs", "no_valid_pairs")
    if not validation:
        return EffectReport(description, None, "inconclusive", "validation_required")
    if validity_reason is not None:
        return EffectReport(description, None, "inconclusive", validity_reason)
    if config.interval_algorithm == "descriptive_only":
        return EffectReport(description, None, "inconclusive", "descriptive_only")
    if config.missing_fields:
        return EffectReport(description, None, "inconclusive", "missing_inference_config")
    if description.n < config.min_independent_blocks:
        return EffectReport(description, None, "inconclusive", "insufficient_independent_blocks")
    interval = bootstrap_interval(description.observations, config)
    if interval.lower >= config.delta_goodput_requests:
        return EffectReport(description, interval, "supported")
    if interval.upper < config.delta_goodput_requests:
        return EffectReport(description, interval, "meaningful_gain_excluded", stable_harm=interval.upper < 0)
    return EffectReport(description, interval, "inconclusive", "interval_too_wide")


@dataclass(frozen=True)
class GuardrailBlock:
    block: tuple[str, ...]
    anchor_id: str
    status: str
    reason: str | None
    b_existing_n: int | None
    new_violations: int | None
    unknown_request_ids: tuple[str, ...] | None


@dataclass(frozen=True)
class GuardrailResult:
    status: str
    max_new_violations: int | None
    blocks: tuple[GuardrailBlock, ...]


def evaluate_b_guardrail(
    contrasts: Sequence[ContrastValidation], config: StatisticsConfig,
) -> GuardrailResult:
    limit = config.generation_max_new_violations
    if limit is None:
        return GuardrailResult("not_configured", None, ())
    results = []
    for row in contrasts:
        if (row.delta is None or row.planned.study_split != "validation"
                or row.planned.contrast != config.primary_contrast
                or (config.primary_delay_ns is not None and row.planned.delay_ns != config.primary_delay_ns)):
            continue
        cohort = row.candidate.score.b_existing_ids
        unknown = row.delta.generation_unknown_ids
        known = row.delta.generation_new_violation_ids
        new = None if known is None else len(known)
        if not cohort:
            state, reason = "unknown", "empty_b_cohort"
        elif known is None or unknown is None:
            state, reason = "unknown", "missing_generation_scores"
        elif new > limit:
            state, reason = "fail", "new_violations_above_limit"
        elif unknown:
            state, reason = "unknown", "unscorable_b_requests"
        else:
            state, reason = "pass", None
        results.append(GuardrailBlock(independent_block(row), row.planned.anchor_id, state, reason,
                                      len(cohort), new, unknown))
    states = {row.status for row in results}
    status = "fail" if "fail" in states else "pass" if states == {"pass"} else "unknown"
    return GuardrailResult(status, limit, tuple(results))


def capture_validity(
    contrasts: Sequence[ContrastValidation], config: StatisticsConfig, source_kind: str,
) -> str | None:
    if source_kind != "real":
        return None
    if not config.capture_budget:
        return "capture_budget_unfrozen"
    for row in contrasts:
        for branch in (row.reference, row.candidate):
            evidence = branch.record.get("capture_perturbation")
            if not isinstance(evidence, dict):
                return "capture_perturbation_unverified"
            for name, limit in config.capture_budget.items():
                value = evidence.get(name)
                if not NONNEG_NUMBER.accepts(value):
                    return "capture_perturbation_unverified"
                if Fraction(str(value)) > limit:
                    return "capture_perturbation_budget_exceeded"
    return None


@dataclass(frozen=True)
class GroupEffect:
    scope: FrozenScope
    study_split: str
    contrast: str
    delay_ns: int | None
    effect: EffectReport
    guardrail: GuardrailResult | None
    joint_status: str | None
    joint_reason: str | None

    def report(self) -> dict:
        return dict(scope=asdict(self.scope), study_split=self.study_split, contrast=self.contrast,
                    delay_ns=self.delay_ns, full_pool=self.effect.report(),
                    guardrail=None if self.guardrail is None else asdict(self.guardrail),
                    joint_status=self.joint_status, joint_reason=self.joint_reason)


@dataclass(frozen=True)
class StatisticsResult:
    config: StatisticsConfig
    groups: tuple[GroupEffect, ...]
    aa: tuple[dict, ...]
    research_status: str | None
    research_reason: str | None
    independent_block_n: int
    independent_action_block_n: int
    research_sample_n: int

    def summary(self) -> dict:
        missing = set(self.config.missing_fields)
        if not self.config.capture_budget and any(group.scope.source_kind == "real"
                                                   and group.study_split == "validation" for group in self.groups):
            missing.add("pairing.capture_perturbation_budget")
        return {
            "effect_groups": [item.report() for item in self.groups], "aa_diagnostics": self.aa,
            "inference_missing_fields": sorted(missing),
            "independent_block_n": self.independent_block_n,
            "independent_action_block_n": self.independent_action_block_n,
            "research_sample_n": self.research_sample_n,
            "interval_configuration": {
                "algorithm": self.config.interval_algorithm, "samples": self.config.bootstrap_samples,
                "confidence_level": report_number(self.config.confidence_level),
                "interpolation": self.config.bootstrap_interpolation, "seed": self.config.seed,
            },
        }


def summarize_effects(pairing: PairingResult, config: StatisticsConfig) -> StatisticsResult:
    buckets = defaultdict(list)
    aa = []
    for row in pairing.contrasts:
        if row.planned.kind == "aa":
            aa.append(_aa_diagnostic(row))
        if row.planned.kind != "action":
            continue
        source = row.reference or row.candidate
        scope = None if source is None else frozen_scope(pairing.study_id, source.record)
        if scope is not None:
            buckets[(scope, row.planned.study_split, row.planned.contrast, row.planned.delay_ns)].append(row)
    groups = []
    for (scope, split, contrast, delay), rows in sorted(
        buckets.items(), key=lambda entry: (*entry[0][:3], -1 if entry[0][3] is None else entry[0][3]),
    ):
        usable = [row for row in rows if row.delta is not None]
        effect = summarize_blocks(
            [BlockEffect(independent_block(row), row.delta.goodput_delta) for row in usable],
            config, validation=split == "validation",
            validity_reason=capture_validity(usable, config, scope.source_kind) if usable else None,
        )
        guardrail = joint = reason = None
        if contrast == config.primary_contrast and (config.primary_delay_ns is None or delay == config.primary_delay_ns):
            guardrail = evaluate_b_guardrail(usable, config)
            joint, reason = effect.status, effect.reason
            if effect.status == "supported" and guardrail.status in {"fail", "unknown"}:
                joint = "inconclusive"
                reason = "b_guardrail_failed" if guardrail.status == "fail" else "b_guardrail_unscorable"
        groups.append(GroupEffect(scope, split, contrast, delay, effect, guardrail, joint, reason))
    primary = [row for row in groups if row.joint_status is not None and row.scope.source_kind == "real"
               and row.study_split == "validation"]
    if len(primary) == 1:
        research, why = primary[0].joint_status, primary[0].joint_reason
    elif len(primary) > 1:
        research, why = "inconclusive", "multiple_real_primary_groups"
    elif not any(row.delta is not None for row in pairing.contrasts):
        research, why = "no_valid_pairs", "no_valid_pairs"
    elif any(row.scope.source_kind == "real" and row.study_split == "validation" for row in groups):
        research, why = "inconclusive", ("missing_inference_config" if config.missing_fields
                                         else "no_valid_primary_contrast")
    else:
        research, why = None, "no_real_validation_actions"
    evaluated = [row for row in pairing.contrasts if row.delta is not None]
    action = [row for row in evaluated if row.planned.kind == "action"]
    return StatisticsResult(
        config, tuple(groups), tuple(aa), research, why,
        len({independent_block(row) for row in evaluated}),
        len({independent_block(row) for row in action}),
        len({independent_block(row) for row in action
             if row.planned.study_split == "validation" and row.candidate.record["source_kind"] == "real"}),
    )


def _aa_diagnostic(row: ContrastValidation) -> dict:
    runs = {"reference": row.reference, "candidate": row.candidate}
    return {
        "block": independent_block(row), "anchor_id": row.planned.anchor_id, "study_split": row.planned.study_split,
        "run_ids": {side: run.run_id if run is not None else None for side, run in runs.items()},
        "anchors": {side: run.run.artifacts[ArtifactKind.ANCHOR_STATE].data if run is not None else None
                    for side, run in runs.items()},
        "goodput_delta": row.delta.goodput_delta if row.delta is not None else None,
        "status": row.status, "issues": [asdict(item) for item in row.issues],
    }
