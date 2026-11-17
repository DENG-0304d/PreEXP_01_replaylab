"""Enumerate planned contrasts, verify captured evidence, and report scored differences."""

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from hashlib import sha256

from .artifacts import ImportedBundle, ImportedRun, json_text
from .contracts import SHA256, TEXT, UINT, ArtifactKind, FormatError, LocatedDocument, check, choice
from .progress import _required
from .scoring import RunScore, ScoreDelta, ScoringConfig, compare_scores, score_run


_BLOCK = ("condition_id", "pair_id", "anchor_id")
_STATE_FIELDS = ("cached_tokens_total", "pending_prefill_tokens", "unfinished_requests")
_ID_SETS = ("queued_ids", "active_ids")
_TIMING_INCOMPLETE = {"evaluation_outcome_unknown", "observation_incomplete", "b_cohort_unverified",
                      "branch_missing", "branch_import_failed", "missing_client_send"}
_CHANGE_KINDS = {"endpoint_change", "epoch_change", "observation_gap", "gateway_restart",
                 "ledger_gap", "concurrent_injection", "residual_injection"}


@dataclass(frozen=True)
class PairingConfig:
    scoring: ScoringConfig
    send_jitter_ns: int
    arm_jitter_ns: int
    router_id: str
    router_digest: str
    rule_id: str
    clock_id: str
    prehistory_tolerances: dict[str, int]
    attempt_selection: str | None
    injection: dict

    @classmethod
    def from_study(cls, study: LocatedDocument) -> "PairingConfig":
        values = _required(study, ("pairing", "prehistory_tolerances"), dict)
        for field, value in values.items():
            key = ("pairing", "prehistory_tolerances", field)
            if field not in (*_STATE_FIELDS, *_ID_SETS):
                raise study.error(key, "unknown prehistory tolerance field")
            check(study, value, UINT, key)
        policy = study.data["pairing"].get("attempt_selection")
        if policy is not None:
            check(study, policy, choice("earliest_started_at"), ("pairing", "attempt_selection"))
        return cls(
            ScoringConfig.from_study(study),
            _required(study, ("replay", "send_jitter_ns"), UINT),
            _required(study, ("replay", "arm_jitter_ns"), UINT),
            _required(study, ("router", "router_id"), TEXT),
            _required(study, ("router", "router_digest"), SHA256),
            _required(study, ("diversion", "rule_id"), TEXT),
            _required(study, ("replay", "clock_id"), TEXT),
            dict(values), policy, dict(study.data.get("injection") or {}),
        )


@dataclass(frozen=True)
class PlannedContrast:
    study_id: str
    condition_id: str
    pair_id: str
    anchor_id: str
    anchor_at_ns: int
    study_split: str
    kind: str
    contrast: str
    reference_role: str
    reference_delay_ns: int | None
    candidate_role: str
    candidate_delay_ns: int | None

    @property
    def block(self) -> tuple[str, str, str, str]:
        return self.study_id, self.condition_id, self.pair_id, self.anchor_id

    @property
    def delay_ns(self) -> int | None:
        if self.candidate_role == "delayed":
            return self.candidate_delay_ns
        if self.reference_role == "delayed":
            return self.reference_delay_ns
        return self.candidate_delay_ns


def enumerate_contrasts(plan: LocatedDocument, *, study_id: str) -> tuple[PlannedContrast, ...]:
    planned = []
    for block in plan.data["blocks"]:
        contrasts = []
        kind = block["kind"]
        if kind == "action":
            contrasts.append(("reference", None, "immediate", 0))
            for delay in sorted(branch["delay_ns"] for branch in block["branches"] if branch["role"] == "delayed"):
                contrasts.extend([("reference", None, "delayed", delay), ("delayed", delay, "immediate", 0)])
        elif kind == "aa":
            contrasts.append(("aa_first", None, "aa_second", None))
        elif kind == "live":
            contrasts.append(("baseline_live", None, "inflight_live", None))
        for ref, ref_delay, candidate, delay in contrasts:
            planned.append(PlannedContrast(
                study_id, block["condition_id"], block["pair_id"], block["anchor_id"], block["anchor_at_ns"],
                block["study_split"], kind, f"{candidate}-{ref}", ref, ref_delay, candidate, delay,
            ))
    return tuple(planned)


@dataclass(frozen=True)
class Issue:
    code: str
    run_id: str | None = None
    field: str | None = None
    detail: object = None


@dataclass(frozen=True)
class ScoredRun:
    run: ImportedRun
    score: RunScore | None
    issues: tuple[Issue, ...]
    fingerprint: str

    @property
    def record(self) -> dict:
        return self.run.record

    @property
    def run_id(self) -> str:
        return self.record["run_id"]


def _run_fingerprint(run: ImportedRun) -> str:
    # Identical input traces are expected. Compare the two run-owned streams and startup evidence.
    events = run.artifacts[ArtifactKind.EVENTS].data
    event_ids, reservations = {}, {}
    for row in events:
        event_ids.setdefault(row["event_id"], len(event_ids))
        reservation = row["payload"].get("reservation_id")
        if isinstance(reservation, str):
            reservations.setdefault(reservation, len(reservations))

    def normalize(value, field=None):
        if isinstance(value, dict):
            return {key: normalize(item, key) for key, item in value.items() if key != "run_id"}
        if isinstance(value, list):
            child = "event_id" if field == "evidence_event_ids" else None
            return [normalize(item, child) for item in value]
        if isinstance(value, str):
            if field in {"event_id", "start_event_id"}:
                return event_ids.get(value, value)
            if field == "reservation_id":
                return reservations.get(value, value)
        return value

    outcomes = sorted(run.artifacts[ArtifactKind.OUTCOMES].data, key=lambda row: row["request_id"])
    streams = [normalize(events), normalize(outcomes)]
    startup = {key: value for key, value in run.record["independent_start"].items() if key != "attempt_id"}
    return sha256(json_text([startup, *streams]).encode("utf-8")).hexdigest()


def _platform_issues(run: ImportedRun, score: RunScore, config: PairingConfig) -> list[Issue]:
    record = run.record
    issues = []

    def add(code, field, detail=None):
        issues.append(Issue(code, record["run_id"], field, detail))

    platform = run.artifacts.get(ArtifactKind.PLATFORM_EVIDENCE)
    if platform is None or not platform.data:
        add("platform_evidence_missing", "platform_evidence.jsonl")
        return issues
    evidence = platform.data
    startup = record["independent_start"]
    begin, anchor, end = startup["started_at_ns"], record["anchor_at_ns"], score.required_until_ns
    expected_bindings = startup.get("replica_bindings")
    expected_owner = startup.get("runner_attempt_id")
    if expected_owner is None or any(
        row["attempt_id"] != startup["attempt_id"] or row["runner_attempt_id"] != expected_owner
        or row["run_id"] != record["run_id"] for row in evidence
    ):
        add("platform_attempt_mismatch", "platform_evidence.attempt_id")
    bindings = [row for row in evidence if row["kind"] == "binding_snapshot"]
    before = max((row for row in bindings if row["observed_at_ns"] <= begin),
                 key=lambda row: row["observed_at_ns"], default=None)
    after = min((row for row in bindings if row["observed_at_ns"] >= end),
                key=lambda row: row["observed_at_ns"], default=None)
    internal = [row for row in bindings if begin <= row["observed_at_ns"] <= end]
    anchor_verified = any(row["observed_at_ns"] == anchor or row["payload"].get("phase") == "anchor"
                          for row in internal)
    identity = ("pod_uid", "container_id", "worker_epoch", "gpu_uuid")
    if (not isinstance(expected_bindings, dict)
            or any(not isinstance(expected_bindings.get(replica), dict)
                   or any(expected_bindings[replica].get(field) is None for field in identity)
                   for replica in ("A", "B"))
            or before is None or after is None or not anchor_verified):
        add("platform_binding_unverified", "platform_evidence.binding_snapshot")
    elif any(row["status"] != "pass" or row["payload"].get("replica_bindings") != expected_bindings
             for row in (before, *internal, after)):
        add("platform_binding_changed", "platform_evidence.binding_snapshot")
    if not any(_covers_platform_window(row, begin, end) for row in bindings):
        add("platform_coverage_unverified", "platform_evidence.binding_snapshot.coverage_end_ns",
            dict(required_start_ns=begin, required_end_ns=end))
    for row in evidence:
        if begin <= row["observed_at_ns"] <= end and row["kind"] in _CHANGE_KINDS:
            add("platform_continuity_broken", "platform_evidence.kind", row["kind"])
    leases = [row for row in evidence if row["kind"] == "lease_result" and row["observed_at_ns"] <= begin]
    if not leases or any(row["status"] != "pass" for row in leases):
        add("platform_lease_unconfirmed", "platform_evidence.lease_result")
    cleanup = [row for row in evidence if row["kind"] == "cleanup_check"]
    if (not any(row["observed_at_ns"] <= begin for row in cleanup)
            or not any(row["observed_at_ns"] >= end for row in cleanup)
            or any(row["status"] != "pass" for row in cleanup)):
        add("platform_cleanup_unconfirmed", "platform_evidence.cleanup_check")
    if record.get("action_replay_status") in {"blocked", "failed", "fail"}:
        add("action_replay_unavailable", "action_replay_status")
    issues.extend(_injection_issues(run, evidence, config))
    return issues


def _covers_platform_window(row: dict, begin: int, end: int) -> bool:
    """A binding-watch receipt attests a continuous interval as of its observation time."""
    payload = row["payload"]
    start, stop = payload.get("coverage_start_ns"), payload.get("coverage_end_ns")
    return (row["status"] == "pass" and UINT.accepts(start) and UINT.accepts(stop)
            and start <= begin <= end <= stop <= row["observed_at_ns"])


def _injection_issues(run: ImportedRun, evidence: list, config: PairingConfig) -> list[Issue]:
    record = run.record
    issues = []
    required = ("level", "scope", "implementation_id", "start_offset_ns", "stop_offset_ns",
                "start_jitter_ns", "stop_jitter_ns")
    missing = [name for name in required if config.injection.get(name) is None]
    if missing:
        return [Issue("injection_config_missing", record["run_id"], "injection", missing)]
    captured_times = {}
    for edge in ("start", "stop"):
        records = [row for row in evidence if row["kind"] == f"injector_{edge}"]
        field = f"platform_evidence.injector_{edge}"
        if len(records) != 1 or records[0]["status"] != "pass":
            issues.append(Issue("injection_evidence_missing", record["run_id"], field))
            continue
        payload = records[0]["payload"]
        when = payload.get("actual_at_ns")
        if not UINT.accepts(when):
            issues.append(Issue("injection_time_missing", record["run_id"], f"{field}.actual_at_ns"))
            continue
        captured_times[edge] = when
        for name, expected in (
            ("level", config.injection["level"]), ("scope", config.injection["scope"]),
            ("implementation_id", config.injection["implementation_id"]), ("condition_id", record["condition_id"]),
        ):
            if payload.get(name) != expected:
                issues.append(Issue("injection_condition_mismatch", record["run_id"], f"{field}.{name}"))
        planned_at = record["anchor_at_ns"] + config.injection[f"{edge}_offset_ns"]
        if abs(when - planned_at) > config.injection[f"{edge}_jitter_ns"]:
            issues.append(Issue("injection_jitter_exceeded", record["run_id"], f"{field}.actual_at_ns",
                                dict(actual_at_ns=when, planned_at_ns=planned_at)))
    if len(captured_times) == 2 and captured_times["stop"] <= captured_times["start"]:
        issues.append(Issue("injection_order_invalid", record["run_id"], "platform_evidence.injector_stop"))
    return issues


def _evaluate_run(run: ImportedRun, config: PairingConfig, *, reused_attempt: bool) -> ScoredRun:
    record = run.record
    run_id = record["run_id"]
    try:
        score = score_run(run, config.scoring)
    except FormatError as error:
        return ScoredRun(run, None, (Issue("run_structure_error", run_id, error.field, error.as_dict()),), "")
    issues = [Issue(code, run_id) for code in score.incomplete_reasons]
    if reused_attempt:
        issues.append(Issue("reused_attempt_id", run_id, "independent_start.attempt_id"))
    for name, expected, code in (
        ("slo_id", config.scoring.slo_id, "slo_id_mismatch"),
        ("router_id", config.router_id, "router_id_mismatch"),
        ("router_digest", config.router_digest, "router_digest_mismatch"),
        ("diversion_rule_id", config.rule_id, "rule_id_mismatch"),
        ("clock_id", config.clock_id, "clock_id_mismatch"),
    ):
        if record[name] != expected:
            issues.append(Issue(code, run_id, name))
    anchor = run.artifacts[ArtifactKind.ANCHOR_STATE].data
    for field in ("anchor_id", "anchor_at_ns"):
        if record[field] != anchor[field]:
            issues.append(Issue("anchor_state_mismatch", run_id, field))
    missing, jitter, bad_receipt, clock_order = [], [], [], []
    for key, row in score.requests.items():
        sent = score.client_sends.get(key)
        if sent is None:
            missing.append(key)
        elif abs(sent - row.scheduled_arrival_ns) > config.send_jitter_ns:
            jitter.append(key)
        if row.actual_arrival_ns is not None and key in score.gateway_receipts:
            if row.actual_arrival_ns != score.gateway_receipts[key]:
                bad_receipt.append(key)
        if sent is not None and key in score.gateway_receipts and score.gateway_receipts[key] < sent:
            clock_order.append(key)
    for keys, code, field in (
        (missing, "missing_client_send", "arrival.client_sent_ns"),
        (jitter, "send_jitter_exceeded", "arrival.client_sent_ns"),
        (bad_receipt, "gateway_arrival_mismatch", "outcomes.actual_arrival_ns"),
        (clock_order, "gateway_arrival_before_client_send", "arrival.client_sent_ns"),
    ):
        if keys:
            issues.append(Issue(code, run_id, field, sorted(keys)))
    if record["role"] in {"immediate", "delayed"}:
        if len(score.arms) != 1:
            issues.append(Issue("arm_missing_or_ambiguous", run_id, "rule_armed"))
        else:
            arm = score.arms[0]
            if arm.payload["rule_id"] != config.rule_id:
                issues.append(Issue("rule_id_mismatch", run_id, "rule_armed.rule_id"))
            planned_at = record["anchor_at_ns"] + record["planned_delay_ns"]
            if abs(arm.cursor.gw_at_ns - planned_at) > config.arm_jitter_ns:
                issues.append(Issue("arm_jitter_exceeded", run_id, "rule_armed.gw_at_ns",
                                    dict(actual_at_ns=arm.cursor.gw_at_ns, planned_at_ns=planned_at)))
    elif record["role"] in {"reference", "aa_first", "aa_second"} and (score.arms or score.redirected_n):
        issues.append(Issue("unexpected_diversion", run_id, "rule_armed"))
    if record["source_kind"] == "real":
        issues.extend(_platform_issues(run, score, config))
    return ScoredRun(run, score, tuple(issues), _run_fingerprint(run))


@dataclass(frozen=True)
class ContrastValidation:
    planned: PlannedContrast
    reference: ScoredRun | None
    candidate: ScoredRun | None
    issues: tuple[Issue, ...]
    delta: ScoreDelta | None
    reference_attempt_ids: tuple[str, ...] = ()
    candidate_attempt_ids: tuple[str, ...] = ()

    @property
    def status(self) -> str:
        if self.delta is not None:
            return "valid"
        if any(issue.code == "live_out_of_scope" for issue in self.issues):
            return "not_evaluated"
        if all(issue.code in _TIMING_INCOMPLETE for issue in self.issues):
            return "incomplete"
        return "incomparable"


def validate_contrast(
    planned: PlannedContrast, reference: ScoredRun, candidate: ScoredRun, *, config: PairingConfig,
) -> ContrastValidation:
    issues = [*reference.issues, *candidate.issues]

    def add(code, field, run_id=None):
        issues.append(Issue(code, run_id, field))

    for run, role, delay in [(reference, planned.reference_role, planned.reference_delay_ns),
                             (candidate, planned.candidate_role, planned.candidate_delay_ns)]:
        for name in (*_BLOCK, "anchor_at_ns", "study_split"):
            if run.record[name] != getattr(planned, name):
                add(f"{name}_mismatch", name, run.run_id)
        if run.record["role"] != role or run.record.get("planned_delay_ns") != delay:
            add("planned_branch_mismatch", "role/planned_delay_ns", run.run_id)

    for name, code in (
        ("source_kind", "source_kind_mismatch"), ("arrival_trace_id", "arrival_trace_mismatch"),
        ("prehistory_id", "prehistory_id_mismatch"), ("eval_cohort_id", "eval_cohort_id_mismatch"),
        ("intervention_digest", "intervention_digest_mismatch"),
    ):
        if reference.record[name] != candidate.record[name]:
            add(code, name)
    for name, code in (("arrivals.csv", "arrival_hash_mismatch"), ("inputs.jsonl", "input_hash_mismatch")):
        if reference.record["files"][name]["sha256"].lower() != candidate.record["files"][name]["sha256"].lower():
            add(code, f"files[{name!r}].sha256")
    if reference.run_id == candidate.run_id or (reference.fingerprint and reference.fingerprint == candidate.fingerprint):
        add("relabelled_run", "events/outcomes/independent_start")

    if reference.score is not None and candidate.score is not None:
        if reference.score.eval_ids != candidate.score.eval_ids:
            add("evaluation_cohort_mismatch", "arrivals.in_eval")
        if reference.score.b_existing_ids != candidate.score.b_existing_ids:
            add("b_cohort_mismatch", "b_existing_request_ids")
    left = reference.run.artifacts[ArtifactKind.ANCHOR_STATE].data
    right = candidate.run.artifacts[ArtifactKind.ANCHOR_STATE].data
    for replica in ("A", "B"):
        before, after = left[replica], right[replica]
        if before["warm_prefix_set_id"] != after["warm_prefix_set_id"]:
            add("warm_prefix_mismatch", f"{replica}.warm_prefix_set_id")
        for name in _STATE_FIELDS:
            if abs(before[name] - after[name]) > config.prehistory_tolerances.get(name, 0):
                add("prehistory_state_mismatch", f"{replica}.{name}")
        for name in _ID_SETS:
            if len(set(before[name]) ^ set(after[name])) > config.prehistory_tolerances.get(name, 0):
                add("prehistory_state_mismatch", f"{replica}.{name}")
    delta = compare_scores(reference.score, candidate.score) if not issues else None
    return ContrastValidation(planned, reference, candidate, tuple(issues), delta)


def _slot(record: dict) -> tuple:
    return (*(record.get(key) for key in _BLOCK), record.get("role"), record.get("planned_delay_ns"))


def _choose_attempt(available: list[ScoredRun], excluded: list[dict], policy: str | None):
    ids = tuple(sorted(run.record["independent_start"]["attempt_id"] for run in available))
    if len(available) == 1:
        return available[0], ids, None
    if not available:
        if excluded:
            return None, (), Issue("branch_import_failed", detail=excluded)
        return None, (), Issue("branch_missing")
    if policy == "earliest_started_at":
        earliest = min(run.record["independent_start"]["started_at_ns"] for run in available)
        selected = [run for run in available if run.record["independent_start"]["started_at_ns"] == earliest]
        if len(selected) == 1:
            return selected[0], ids, None
    return None, ids, Issue("ambiguous_attempts", detail=[run.run_id for run in available])


@dataclass(frozen=True)
class PairingResult:
    study_id: str
    runs: tuple[ScoredRun, ...]
    contrasts: tuple[ContrastValidation, ...]
    generation_threshold_ns: int | None
    generation_max_new_violations: int | None

    def metrics_rows(self) -> list[dict]:
        output = []
        for run in self.runs:
            record, score = run.record, run.score
            base = {key: record[key] for key in (*_BLOCK, "source_kind", "study_split", "run_id", "role")}
            output.append(dict(
                **base, study_id=self.study_id, row_type="run", planned_delay_ns=record.get("planned_delay_ns"),
                status="valid" if not run.issues else "excluded",
                reason=";".join(sorted({issue.code for issue in run.issues})) or None,
                **_score_counts(score),
            ))
        for row in self.contrasts:
            planned, delta = row.planned, row.delta
            kinds = {run.record["source_kind"] for run in (row.reference, row.candidate) if run is not None}
            record = dict(
                **{name: getattr(planned, name) for name in (*_BLOCK, "study_id", "study_split", "contrast")},
                row_type="contrast", source_kind=next(iter(kinds)) if len(kinds) == 1 else None,
                left_run_id=row.reference.run_id if row.reference is not None else None,
                right_run_id=row.candidate.run_id if row.candidate is not None else None,
                status=row.status, reason=";".join(sorted({item.code for item in row.issues})) or None,
                planned_delay_ns=planned.delay_ns,
            )
            if delta is not None:
                record.update(_score_counts(row.candidate.score))
                record.update(goodput_delta=delta.goodput_delta, b_new_violations=len(delta.b_new_violation_ids),
                              b_recovered=len(delta.b_recovered_ids), b_violation_delta=delta.b_violation_delta)
            output.append(record)
        return output

    def summary(self) -> dict:
        evaluated = [row for row in self.contrasts if row.delta is not None]
        action = [row for row in evaluated if row.planned.kind == "action"]
        aa = [row for row in evaluated if row.planned.kind == "aa"]
        action_blocks = {row.planned.block for row in action}
        research_blocks = {row.planned.block for row in action if row.planned.study_split == "validation"
                           and row.candidate.record["source_kind"] == "real"}
        return dict(
            pairing_status="valid_contrasts" if evaluated else "no_valid_pairs",
            planned_contrast_n=len(self.contrasts), valid_contrast_n=len(evaluated),
            bad_contrast_n=sum(row.status in {"incomplete", "incomparable"} for row in self.contrasts),
            contrast_status_counts=dict(Counter(row.status for row in self.contrasts)),
            independent_block_n=len({row.planned.block for row in evaluated}),
            independent_action_block_n=len(action_blocks), research_sample_n=len(research_blocks),
            aa=dict(valid_contrast_n=len(aa), independent_block_n=len({row.planned.block for row in aa}),
                    deltas=[row.delta.goodput_delta for row in aa]),
            generation_threshold_ns=self.generation_threshold_ns,
            generation_max_new_violations=self.generation_max_new_violations,
            run_scores=[_run_report(run) for run in self.runs],
            contrasts=[_contrast_report(row) for row in self.contrasts],
        )


def _score_counts(score: RunScore | None) -> dict:
    if score is None:
        return {}
    return dict(eval_n=len(score.eval_ids), goodput=score.goodput, unmet_n=score.unmet_n,
                rejected_n=score.rejected_n, timeout_n=score.timeout_n,
                b_existing_n=len(score.b_existing_ids) if score.b_cohort_verified else None,
                b_violation_n=score.b_violation_n, redirected_n=score.redirected_n, blocked_n=score.blocked_n)


def _run_report(run: ScoredRun) -> dict:
    record = run.record
    score = run.score
    report = {name: record[name] for name in
              (*_BLOCK, "run_id", "source_kind", "study_split", "role", "run_status")}
    report.update(attempt_id=record["independent_start"]["attempt_id"],
                  planned_delay_ns=record.get("planned_delay_ns"),
                  issues=[asdict(issue) for issue in run.issues], **_score_counts(score))
    if score is not None:
        unknown_generation = sorted(key for key, item in score.generation.items() if item.state == "unknown")
        known_generation_failures = (sum(item.state == "fail" for item in score.generation.values())
                                     if score.generation_configured and score.b_cohort_verified else None)
        report.update(
            known_goodput=score.known_goodput, unknown_request_ids=score.unknown_request_ids,
            observed_until_ns=score.observed_until_ns, required_until_ns=score.required_until_ns,
            b_cohort_verified=score.b_cohort_verified, b_existing_ids=sorted(score.b_existing_ids),
            b_unknown_ids=score.b_unknown_ids,
            b_generation={key: asdict(item) for key, item in score.generation.items()},
            b_generation_violation_n=None if unknown_generation else known_generation_failures,
            b_generation_known_violation_n=known_generation_failures, b_generation_unknown_ids=unknown_generation,
            requests={key: dict(**asdict(item), client_sent_ns=score.client_sends.get(key),
                                gateway_received_ns=score.gateway_receipts.get(key))
                      for key, item in score.requests.items()},
        )
    return report


def _contrast_report(row: ContrastValidation) -> dict:
    delta = row.delta
    known_generation_new = (len(delta.generation_new_violation_ids)
                            if delta is not None and delta.generation_new_violation_ids is not None else None)
    return dict(
        **asdict(row.planned), status=row.status,
        reference_run_id=row.reference.run_id if row.reference is not None else None,
        candidate_run_id=row.candidate.run_id if row.candidate is not None else None,
        reference_attempt_ids=row.reference_attempt_ids, candidate_attempt_ids=row.candidate_attempt_ids,
        issues=[asdict(issue) for issue in row.issues],
        delta=None if delta is None else asdict(delta),
        b_generation_new_violations=(None if delta is None or delta.generation_unknown_ids else known_generation_new),
        b_generation_known_new_violations=known_generation_new,
    )


def analyze_pairs(bundle: ImportedBundle, study: LocatedDocument) -> PairingResult:
    config = PairingConfig.from_study(study)
    study_id = bundle.manifest.data["study_id"]
    counts = Counter(run.record["independent_start"]["attempt_id"] for run in bundle.runs)
    scored = [_evaluate_run(run, config, reused_attempt=counts[run.record["independent_start"]["attempt_id"]] > 1)
              for run in bundle.runs]
    run_evidence = defaultdict(set)
    for run in scored:
        if run.fingerprint:
            run_evidence[run.fingerprint].add(tuple(run.record[key] for key in (*_BLOCK, "study_split")))
    planned_slots = {
        (*(block[key] for key in _BLOCK), branch["role"], branch["delay_ns"])
        for block in bundle.plan.data["blocks"] for branch in block["branches"]
    }
    for index, run in enumerate(scored):
        extra = []
        if _slot(run.record) not in planned_slots:
            extra.append(Issue("unplanned_branch", run.run_id, "condition_id/pair_id/anchor_id/role/planned_delay_ns"))
        if len(run_evidence[run.fingerprint]) > 1:
            extra.append(Issue("reused_run_evidence", run.run_id, "events/outcomes/independent_start"))
        if extra:
            scored[index] = replace(run, issues=(*run.issues, *extra))
    attempts, excluded = defaultdict(list), defaultdict(list)
    for run in scored:
        attempts[_slot(run.record)].append(run)
    for item in bundle.excluded_runs:
        excluded[_slot(item["branch"])].append(item)
    results = []
    for planned in enumerate_contrasts(bundle.plan, study_id=study_id):
        if planned.kind == "live":
            results.append(ContrastValidation(planned, None, None, (Issue("live_out_of_scope"),), None))
            continue
        keys = [(*planned.block[1:], planned.reference_role, planned.reference_delay_ns),
                (*planned.block[1:], planned.candidate_role, planned.candidate_delay_ns)]
        (ref, left_ids, left_issue), (candidate, right_ids, right_issue) = [
            _choose_attempt(attempts[key], excluded[key], config.attempt_selection) for key in keys
        ]
        if ref is not None and candidate is not None:
            checked = validate_contrast(planned, ref, candidate, config=config)
            issues, delta = checked.issues, checked.delta
        else:
            issues = tuple(Issue(issue.code, field=field, detail=issue.detail)
                           for field, issue in (("reference", left_issue), ("candidate", right_issue))
                           if issue is not None)
            issues += tuple(issue for run in (ref, candidate) if run is not None for issue in run.issues)
            delta = None
        results.append(ContrastValidation(planned, ref, candidate, issues, delta, left_ids, right_ids))
    return PairingResult(study_id, tuple(scored), tuple(results), config.scoring.generation_threshold_ns,
                         config.scoring.generation_max_new_violations)
