"""Schema v2 for replay inputs and the canonical file types.

Run entries name their file references under files and their launch metadata
under independent_start. Diagnostics use record_id and optional observed_at_ns.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
import math
from pathlib import Path
import re


SCHEMA_VERSION = 2
Key = tuple[str | int, ...]


class ArtifactKind(StrEnum):
    STUDY = "study.json"
    PLAN = "plan.json"
    MANIFEST = "manifest.json"
    ARRIVALS = "arrivals.csv"
    INPUTS = "inputs.jsonl"
    HEALTH_PROFILE = "health_profile.csv"
    EVENTS = "events.jsonl"
    ANCHOR_STATE = "anchor_state.json"
    OUTCOMES = "outcomes.csv"
    DIAGNOSTICS = "diagnostics.jsonl"
    PLATFORM_EVIDENCE = "platform_evidence.jsonl"
    METRICS = "metrics.csv"
    WINDOWS = "windows.csv"
    TIMELINE = "timeline.jsonl"
    STATUS = "status.json"
    SUMMARY = "summary.json"
    MOCK_SCENARIO = "mock_scenario.json"
    ATTEMPTS = "attempts.json"
    PREFLIGHT = "preflight.json"
    MOCK_PREFLIGHT = "mock_preflight.json"
    KUBERNETES_TEMPLATE = "kubernetes_template.json"


def field_name(key: Key) -> str:
    field = ""
    for part in key:
        if isinstance(part, int) or not part.isidentifier():
            field += f"[{part!r}]"
        else:
            field += f".{part}" if field else part
    return field or "$"


@dataclass(frozen=True)
class FieldLocation:
    file: Path
    line: int
    field: str

    def error(self, reason: str) -> "FormatError":
        return FormatError(self.file, self.line, self.field, reason)


class FormatError(ValueError):
    def __init__(self, file: Path, line: int, field: str, reason: str) -> None:
        super().__init__(f"{file}:{line}: {field}: {reason}")
        self.file, self.line, self.field, self.reason = file, line, field, reason

    def as_dict(self) -> dict:
        return dict(file=str(self.file), line=self.line, field=self.field, reason=self.reason)


@dataclass(frozen=True)
class LocatedDocument:
    file: Path
    data: object
    lines: Mapping[Key, int]
    row_oriented: bool = False

    def location(self, key: Key = ()) -> FieldLocation:
        source_key = key
        while source_key and source_key not in self.lines:
            source_key = source_key[:-1]
        field_key = key[1:] if self.row_oriented and key and type(key[0]) is int else key
        return FieldLocation(self.file, self.lines.get(source_key, 1), field_name(field_key))

    def error(self, key: Key, reason: str) -> FormatError:
        return self.location(key).error(reason)


@dataclass(frozen=True)
class Rule:
    accepts: Callable[[object], bool]
    expected: str


@dataclass(frozen=True)
class OptionalField:
    spec: "Spec"


@dataclass(frozen=True)
class Nullable:
    spec: "Spec"


Spec = type | Rule | OptionalField | Nullable | dict[str, "Spec"] | list["Spec"]
_TYPE_NAMES = {int: "an integer", bool: "a boolean", str: "a string", dict: "an object", list: "an array"}


def choice(*values: str) -> Rule:
    return Rule(lambda value: type(value) is str and value in values, f"expected one of {', '.join(values)}")


TEXT = Rule(lambda v: type(v) is str and bool(v), "expected a nonempty string")
UINT = Rule(lambda v: type(v) is int and v >= 0, "expected a nonnegative integer")
POS_INT = Rule(lambda v: type(v) is int and v > 0, "expected a positive integer")
SHA256 = Rule(lambda v: type(v) is str and re.fullmatch(r"[0-9a-fA-F]{64}", v) is not None,
              "expected a 64-digit SHA-256 hex digest")
VERSION = Rule(lambda v: type(v) is int and v == SCHEMA_VERSION, "expected integer schema_version 2")


def number(value: object) -> bool:
    return type(value) is int or (type(value) is float and math.isfinite(value))


POS_NUMBER = Rule(lambda v: number(v) and v > 0, "expected a positive number")
NUMBER = Rule(number, "expected a number")
NONNEG_NUMBER = Rule(lambda v: number(v) and v >= 0, "expected a nonnegative number")
ALPHA = Rule(lambda v: number(v) and 0 < v <= 1, "expected a number in (0, 1]")
CONFIDENCE = Rule(lambda v: number(v) and 0 < v < 1, "expected a number in (0, 1)")
PPM = Rule(lambda v: type(v) is int and 0 <= v <= 1_000_000, "expected an integer in [0, 1000000]")
REPLICA = choice("A", "B")
SOURCE = choice("real", "synthetic")
SPLIT = choice("calibration", "discovery", "validation")
BLOCK_ROLES = {
    "action": ("reference", "immediate", "delayed"),
    "aa": ("aa_first", "aa_second"),
    "shadow_only": ("shadow",),
    "live": ("baseline_live", "inflight_live"),
}
ROLE = choice(*(role for roles in BLOCK_ROLES.values() for role in roles))


def check(doc: LocatedDocument, value: object, spec: Spec, key: Key = (), *, allow_null=False) -> None:
    if isinstance(spec, OptionalField):
        spec = spec.spec
    if value is None and (allow_null or isinstance(spec, Nullable)):
        return
    if isinstance(spec, Nullable):
        spec = spec.spec
    if isinstance(spec, dict):
        if type(value) is not dict:
            raise doc.error(key, "expected an object")
        for name, child in spec.items():
            if name in value:
                check(doc, value[name], child, (*key, name), allow_null=allow_null)
            elif not isinstance(child, OptionalField):
                raise doc.error((*key, name), "required field is missing")
    elif isinstance(spec, list):
        if type(value) is not list:
            raise doc.error(key, "expected an array")
        for index, item in enumerate(value):
            check(doc, item, spec[0], (*key, index), allow_null=allow_null)
    elif isinstance(spec, Rule):
        if not spec.accepts(value):
            raise doc.error(key, spec.expected)
    elif type(value) is not spec:
        raise doc.error(key, f"expected {_TYPE_NAMES[spec]}")


def _check_ns_fields(doc: LocatedDocument, value: object, key: Key = ()) -> None:
    """Check nanosecond fields in extensible objects as well as fixed schemas."""
    if isinstance(value, dict):
        for name, child in value.items():
            child_key = (*key, name)
            if name.endswith("_ns") and child is not None and type(child) is not int:
                raise doc.error(child_key, "expected an integer in nanoseconds")
            _check_ns_fields(doc, child, child_key)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _check_ns_fields(doc, child, (*key, index))


_MODEL = {"name": TEXT, "revision": TEXT}
_REPLICA_CONFIG = {"endpoint": OptionalField(Nullable(TEXT)), "startup_args": OptionalField(Nullable([TEXT]))}
_KUBE_REPLICA = dict(deployment_name=TEXT, service_name=TEXT, container_name=TEXT, expected_gpu_uuid=TEXT)
_KUBE = {
    "namespace": TEXT, "node_name": TEXT, "bench_lease_name": TEXT,
    "replicas": {"A": _KUBE_REPLICA, "B": _KUBE_REPLICA},
    "gateway_image_digest": TEXT, "runner_image_digest": TEXT, "backend_image_digest": TEXT,
    "output_mount_path": TEXT, "max_clock_alignment_error_ns": UINT,
    "locality_acceptance": {
        "a_min_slowdown": POS_NUMBER, "a_recovery_tolerance": NONNEG_NUMBER,
        "b_tolerance": NONNEG_NUMBER, "gateway_tolerance": NONNEG_NUMBER,
        "metrics": dict, "min_samples": POS_INT,
    },
    "injector_max_runtime_ns": POS_INT,
}
_STUDY_SECTIONS = {
    "environment": {
        "model": _MODEL, "tokenizer": _MODEL, "vllm": {"version": TEXT, "build_id": TEXT},
        "gpu": {"model": TEXT, "device_ids": [TEXT]},
        "replicas": {"A": _REPLICA_CONFIG, "B": _REPLICA_CONFIG},
        "gateway_version": TEXT, "collector_version": TEXT,
        "kubernetes": OptionalField(_KUBE),
    },
    "replay": {
        "input_trace_path": TEXT, "warmup_trace_path": TEXT, "prehistory_trace_path": TEXT,
        "clock_id": TEXT, "rotation_seed": int, "send_jitter_ns": UINT,
        "arm_jitter_ns": UINT, "observation_end_ns": UINT,
    },
    "observation": {
        "tick_ns": POS_INT, "work_ttl_ns": POS_INT, "completion_ttl_ns": POS_INT,
        "residual_ttl_ns": POS_INT, "evidence_ttl_ns": POS_INT, "alpha": ALPHA,
        "threshold": POS_NUMBER, "health_profile_path": TEXT, "workload_buckets": dict,
        "conditional_thresholds": dict, "healthy_alert_budget": NONNEG_NUMBER,
        "healthy_redirect_budget": NONNEG_NUMBER,
    },
    "router": {
        "router_id": TEXT, "router_digest": SHA256, "state_ttl_ns": POS_INT,
        "prefill_cost_ns_per_token": POS_INT, "decode_load_cost_ns": POS_INT,
        "speed_factor_min": POS_NUMBER, "speed_factor_max": POS_NUMBER,
        "reservation_ack_policy": TEXT, "reservation_timeout_ns": POS_INT,
    },
    "diversion": {
        "rule_id": TEXT, "rule_seed": int, "ratio_ppm": PPM, "duration_ns": POS_INT,
        "max_redirects": UINT, "max_b_unfinished": UINT, "max_b_prefill_tokens": UINT,
        "b_state_ttl_ns": POS_INT,
    },
    "slo": {
        "slo_id": TEXT, "threshold_source": TEXT, "deadline_basis": choice("scheduled_arrival_ns"),
        "b_generation_guardrail": OptionalField(Nullable({
            "threshold_ns": POS_INT, "max_new_violations": UINT,
        })),
    },
    "pairing": {
        "prehistory_tolerances": dict, "delta_goodput_requests": OptionalField(Nullable(UINT)),
        "min_independent_blocks": OptionalField(Nullable(POS_INT)),
        "interval_algorithm": OptionalField(Nullable(TEXT)),
        "confidence_level": OptionalField(Nullable(CONFIDENCE)), "seed": OptionalField(Nullable(int)),
        "capture_perturbation_budget": OptionalField(Nullable(dict)), "primary_contrast": OptionalField(Nullable(TEXT)),
        "bootstrap_samples": OptionalField(Nullable(POS_INT)),
        "bootstrap_interpolation": OptionalField(Nullable(TEXT)),
        "primary_delay_ns": OptionalField(Nullable(UINT)),
    },
    "injection": {
        "scope": TEXT, "implementation_id": TEXT, "level": TEXT,
        "start_offset_ns": int, "stop_offset_ns": int, "cleanup_timeout_ns": POS_INT,
        "condition_id": TEXT, "start_jitter_ns": OptionalField(UINT), "stop_jitter_ns": OptionalField(UINT),
    },
}


def is_hardware_study(data: dict) -> bool:
    environment = data.get("environment")
    return data.get("source_kind") == "real" or (isinstance(environment, dict) and "kubernetes" in environment)


def validate_study(document: LocatedDocument) -> None:
    data = document.data
    check(document, data, {"schema_version": VERSION, "source_kind": OptionalField(SOURCE)})
    hardware = is_hardware_study(data)
    for name, spec in _STUDY_SECTIONS.items():
        if name not in data:
            raise document.error((name,), "required field is missing")
        check(document, data[name], spec, (name,), allow_null=hardware)

    environment = data["environment"] or {}
    for replica_id in ("A", "B"):
        replica = (environment.get("replicas") or {}).get(replica_id)
        if not hardware and not (replica.get("endpoint") or replica.get("startup_args")):
            raise document.error(("environment", "replicas", replica_id), "expected endpoint or startup_args")
    router = data["router"] or {}
    if router.get("speed_factor_min") is not None and router.get("speed_factor_max") is not None:
        if router["speed_factor_min"] > router["speed_factor_max"]:
            raise document.error(("router", "speed_factor_max"), "must be at least speed_factor_min")
    injection = data["injection"] or {}
    if injection.get("start_offset_ns") is not None and injection.get("stop_offset_ns") is not None:
        if injection["start_offset_ns"] >= injection["stop_offset_ns"]:
            raise document.error(("injection", "stop_offset_ns"), "must be greater than start_offset_ns")
    kubernetes = environment.get("kubernetes") or {}
    runtime = kubernetes.get("injector_max_runtime_ns")
    start, stop, jitter = (injection.get(key) for key in ("start_offset_ns", "stop_offset_ns", "stop_jitter_ns"))
    if None not in (runtime, start, stop, jitter) and runtime < stop - start + jitter:
        raise document.error(("environment", "kubernetes", "injector_max_runtime_ns"),
                             "must cover planned injection duration and stop jitter")
    _check_ns_fields(document, data)


def missing_hardware_fields(data: dict, command: str) -> list[str]:
    """Return unresolved declared values needed by an execution-side command."""
    if not is_hardware_study(data) or command not in {"preflight", "run"}:
        return []
    sections = _STUDY_SECTIONS if command == "run" else {
        "environment": _STUDY_SECTIONS["environment"],
        "replay": {"clock_id": TEXT},
        "injection": _STUDY_SECTIONS["injection"],
    }
    missing = []

    def visit(value: object, spec: Spec, key: Key) -> None:
        if isinstance(spec, OptionalField):
            spec = spec.spec
        if isinstance(spec, Nullable):
            if value is None:
                return
            spec = spec.spec
        if value is None:
            missing.append(field_name(key))
        elif isinstance(spec, dict):
            for name, child in spec.items():
                if name in value:
                    visit(value[name], child, (*key, name))
        elif isinstance(spec, list):
            for index, item in enumerate(value):
                visit(item, spec[0], (*key, index))

    visit(data, sections, ())
    environment = data["environment"] or {}
    for replica_id in ("A", "B"):
        replica = (environment.get("replicas") or {}).get(replica_id)
        if replica is not None and not (replica.get("endpoint") or replica.get("startup_args")):
            missing.append(f"environment.replicas.{replica_id}.endpoint/startup_args")
    injection = data["injection"]
    if environment.get("kubernetes") is not None and injection is not None:
        missing.extend(f"injection.{key}" for key in ("start_jitter_ns", "stop_jitter_ns") if key not in injection)
    return sorted(missing)


_BRANCH_KEY = dict(condition_id=TEXT, pair_id=TEXT, anchor_id=TEXT, anchor_at_ns=UINT, study_split=SPLIT)
_PLAN_BLOCK = {
    **_BRANCH_KEY, "kind": choice(*BLOCK_ROLES),
    "branches": [{"role": ROLE, "delay_ns": Nullable(UINT)}],
}


def check_delay(doc: LocatedDocument, role: str, delay: int | None, key: Key) -> None:
    if role == "immediate" and delay != 0:
        raise doc.error(key, "immediate delay must be 0")
    if role == "delayed" and (delay is None or delay <= 0):
        raise doc.error(key, "delayed delay must be a positive integer")
    if role not in {"immediate", "delayed"} and delay is not None:
        raise doc.error(key, "delay must be null for this role")


def validate_plan(document: LocatedDocument) -> None:
    check(document, document.data, {"schema_version": VERSION, "blocks": [_PLAN_BLOCK]})
    _check_ns_fields(document, document.data)
    seen = set()
    for index, block in enumerate(document.data["blocks"]):
        key = ("blocks", index)
        identity = tuple(block[field] for field in ("condition_id", "pair_id", "anchor_id"))
        if identity in seen:
            raise document.error((*key, "pair_id"), "duplicate plan block identity")
        seen.add(identity)
        roles = BLOCK_ROLES[block["kind"]]
        branches = set()
        for offset, branch in enumerate(block["branches"]):
            role, delay = branch["role"], branch["delay_ns"]
            branch_key = (*key, "branches", offset)
            if role not in roles:
                raise document.error((*branch_key, "role"), "role is invalid for this block kind")
            check_delay(document, role, delay, (*branch_key, "delay_ns"))
            if (role, delay) in branches:
                raise document.error((*branch_key, "role"), "duplicate role and delay")
            branches.add((role, delay))
        required_roles = {"reference", "immediate"} if block["kind"] == "action" else set(roles)
        absent = required_roles - {role for role, _ in branches}
        if absent:
            raise document.error((*key, "branches"), "missing roles: " + ", ".join(sorted(absent)))


RUN_FILE_KINDS = (
    ArtifactKind.ARRIVALS, ArtifactKind.INPUTS, ArtifactKind.HEALTH_PROFILE, ArtifactKind.EVENTS,
    ArtifactKind.ANCHOR_STATE, ArtifactKind.OUTCOMES, ArtifactKind.DIAGNOSTICS,
)
_FILE_REF = {"path": TEXT, "sha256": SHA256}
_PLATFORM_START = dict(
    runner_attempt_id=TEXT, gateway_pod_uid=TEXT, runner_pod_uid=TEXT,
    replica_bindings={"A": dict, "B": dict}, preflight_ref=TEXT, timing_capability_ref=TEXT,
)
_RUN_IDENTITY = {
    **_BRANCH_KEY, "run_id": TEXT, "source_kind": SOURCE, "role": ROLE,
    "planned_delay_ns": OptionalField(Nullable(UINT)), "run_status": choice("complete", "failed", "blocked"),
}
_RUN_EVIDENCE = {
    "arrival_trace_id": TEXT, "prehistory_id": TEXT, "eval_cohort_id": TEXT, "slo_id": TEXT,
    "router_id": TEXT, "router_digest": SHA256, "diversion_rule_id": TEXT,
    "intervention_digest": SHA256, "clock_id": TEXT,
    "independent_start": {
        "attempt_id": TEXT, "replica_epochs": {"A": UINT, "B": UINT}, "started_at_ns": UINT,
    },
    "files": {kind.value: _FILE_REF for kind in RUN_FILE_KINDS},
}


def validate_manifest_header(doc: LocatedDocument) -> None:
    check(doc, doc.data, {
        "study_id": TEXT, "schema_version": VERSION, "config_sha256": SHA256,
        "plan_path": TEXT, "plan_sha256": SHA256, "runs": list,
    })
    seen = set()
    for index, run in enumerate(doc.data["runs"]):
        identity = run.get("run_id") if isinstance(run, dict) else None
        if type(identity) is str:
            if identity in seen:
                raise doc.error(("runs", index, "run_id"), "duplicate run_id")
            seen.add(identity)


def validate_manifest_run(doc: LocatedDocument, index: int, *, kubernetes: bool = False) -> None:
    key = ("runs", index)
    run = doc.data["runs"][index]
    check(doc, run, _RUN_IDENTITY, key)
    _check_ns_fields(doc, run, key)
    check_delay(doc, run["role"], run.get("planned_delay_ns"), (*key, "planned_delay_ns"))
    if run["run_status"] != "complete":
        check(doc, run, {"reason": TEXT}, key)
        return
    check(doc, run, _RUN_EVIDENCE, key)
    platform = ArtifactKind.PLATFORM_EVIDENCE.value
    if run["source_kind"] == "real" and (
        kubernetes or platform in run["files"] or "runner_attempt_id" in run["independent_start"]
    ):
        check(doc, run["files"], {platform: _FILE_REF}, (*key, "files"))
        check(doc, run["independent_start"], _PLATFORM_START, (*key, "independent_start"))
    for filename, reference in run["files"].items():
        check(doc, reference, _FILE_REF, (*key, "files", filename))


_ANCHOR_REPLICA = dict(queued_ids=[TEXT], active_ids=[TEXT], warm_prefix_set_id=TEXT,
                       cached_tokens_total=UINT, pending_prefill_tokens=UINT, unfinished_requests=UINT)
_ANCHOR = {
    "anchor_id": TEXT, "anchor_at_ns": UINT, "phase": choice("before_intervention"),
    "b_existing_request_ids": [TEXT], "A": _ANCHOR_REPLICA, "B": _ANCHOR_REPLICA,
}
_EPOCH = dict(replica_id=REPLICA, epoch=UINT)
_STEP = {**_EPOCH, "step_id": TEXT, "boundary_id": TEXT}
_RESERVATION = dict(request_id=TEXT, replica_id=REPLICA, reservation_id=TEXT, reason=TEXT)
_RULE = dict(rule_id=TEXT, activation_id=TEXT, reason=TEXT)
_EVENT_PAYLOADS = {
    "tick": {},
    "replica_epoch": _EPOCH,
    "work_state": dict(**_EPOCH, work_version=UINT, work_key=TEXT, busy=bool, backlog=bool, sampled_ns=UINT),
    "step_start": dict(**_STEP, start_ns=UINT, work_version=OptionalField(Nullable(UINT))),
    "step_complete": dict(**_STEP, end_ns=UINT, duration_ns=int, work_key=TEXT),
    "inflight_snapshot": dict(**_STEP, sample_ns=UINT, start_event_id=Nullable(TEXT), work_version=Nullable(UINT),
                              status=choice("PENDING_VERIFIED", "DONE_VERIFIED", "UNVERIFIED"),
                              marker_status=TEXT, validation_reason=Nullable(TEXT), source_seq=OptionalField(UINT)),
    "route_state": dict(**_EPOCH, state_version=UINT, sampled_ns=UINT, ready=bool,
                       pending_prefill_tokens=UINT, unfinished_requests=UINT),
    "arrival": dict(request_id=TEXT, client_sent_ns=UINT),
    "dispatch_decision": dict(
        request_id=TEXT, default_target=Nullable(REPLICA), actual_target=Nullable(REPLICA),
        eligible=bool, selected=bool, b_gate=choice("pass", "deny", "unknown", "not_applicable"),
        reason=TEXT, rule_active=bool, route_state_refs=dict,
        cache_hint_tokens=Nullable(Rule(lambda v: type(v) is dict or UINT.accepts(v), "expected token count or mapping")),
    ),
    "dispatch_sent": dict(request_id=TEXT, target=REPLICA, reservation_id=TEXT),
    "rule_armed": _RULE,
    "rule_disarmed": _RULE,
    "dispatch_ack": dict(**_RESERVATION, state_version=UINT),
    "reservation_release": dict(**_RESERVATION, state_version=Nullable(UINT)),
    "candidate_alert": dict(signal_group=TEXT, replica_id=REPLICA, evidence_event_ids=[TEXT],
                            transition=TEXT, policy_id=TEXT),
}

CSV_SCHEMAS = {
    ArtifactKind.ARRIVALS: dict(
        request_id=TEXT, scheduled_arrival_ns=UINT, in_eval=bool, ttft_deadline_ns=UINT,
        terminal_deadline_ns=UINT, input_id=TEXT, prompt_tokens=UINT, max_output_tokens=POS_INT,
    ),
    ArtifactKind.HEALTH_PROFILE: dict(
        replica_id=REPLICA, work_key=TEXT, healthy_median_ns=POS_INT,
        gap_limit_ns=POS_INT, inflight_limit_ns=POS_INT, calibration_count=POS_INT,
    ),
    ArtifactKind.OUTCOMES: dict(
        run_id=TEXT, request_id=TEXT,
        terminal_status=choice("ok", "rejected", "timeout", "cancelled", "backend_error", "unresolved"),
        actual_arrival_ns=Nullable(UINT), first_token_ns=Nullable(UINT), last_token_ns=Nullable(UINT),
        output_tokens=Nullable(UINT), terminal_ns=Nullable(UINT),
    ),
}

WINDOW_KEYS = ("study_id", "config_sha256", "plan_sha256", "source_kind", "study_split",
               "condition_id", "pair_id", "anchor_id")
_SIGNAL_WINDOW_FIELDS = {
    "alert_usable_ns": UINT, "intended_arm_ns": UINT, "matched_delay_ns": UINT,
    "matched_run_ids_json": [TEXT], "actual_arm_times_json": [Nullable(UINT)],
    "first_redirect_times_json": [Nullable(UINT)], "eligible_n": UINT, "blocked_n": UINT,
    "goodput_delta": NUMBER, "caught_state": TEXT,
}
CSV_SCHEMAS[ArtifactKind.WINDOWS] = {
    **dict(zip(WINDOW_KEYS, (TEXT, SHA256, SHA256, SOURCE, SPLIT, TEXT, TEXT, TEXT))),
    **{f"{group}_{field}": Nullable(spec) for group in ("baseline", "inflight")
       for field, spec in _SIGNAL_WINDOW_FIELDS.items()},
    "opportunity_state": Nullable(TEXT), "coverage_cell": Nullable(TEXT),
    "goodput_delta_between_groups": Nullable(NUMBER),
    "reason": Nullable(TEXT), "evidence_refs_json": Nullable(list),
}
_METRICS_KEYS = (
    "study_id", "source_kind", "study_split", "condition_id", "pair_id", "anchor_id",
    "row_type", "role", "contrast", "planned_delay_ns", "run_id", "left_run_id", "right_run_id",
)
CSV_SCHEMAS[ArtifactKind.METRICS] = {
    **dict.fromkeys(_METRICS_KEYS, Nullable(TEXT)),
    "planned_delay_ns": Nullable(UINT), "status": Nullable(TEXT), "reason": Nullable(TEXT),
    **dict.fromkeys(("eval_n", "goodput", "unmet_n", "rejected_n", "timeout_n", "b_existing_n",
                     "b_violation_n", "b_new_violations", "b_recovered", "redirected_n", "blocked_n"),
                    Nullable(UINT)),
    "b_violation_delta": Nullable(int), "goodput_delta": Nullable(int),
}

_ROW_SCHEMAS = {
    **CSV_SCHEMAS,
    ArtifactKind.INPUTS: dict(input_id=TEXT, prompt=str, sampling=dict),
    ArtifactKind.EVENTS: dict(
        run_id=TEXT, event_id=TEXT, type=choice(*_EVENT_PAYLOADS),
        event_at_ns=UINT, gw_at_ns=UINT, gw_seq=UINT, payload=dict,
    ),
    ArtifactKind.DIAGNOSTICS: dict(
        run_id=TEXT, kind=TEXT, record_id=TEXT,
        observed_at_ns=OptionalField(Nullable(UINT)), payload=OptionalField(dict),
    ),
    ArtifactKind.PLATFORM_EVIDENCE: dict(
        run_id=TEXT, attempt_id=TEXT, runner_attempt_id=TEXT, kind=TEXT, observed_at_ns=UINT,
        status=TEXT, payload=dict, record_seq=OptionalField(UINT),
    ),
}
_UNIQUE_ROWS = {
    ArtifactKind.ARRIVALS: ("request_id",), ArtifactKind.INPUTS: ("input_id",),
    ArtifactKind.HEALTH_PROFILE: ("replica_id", "work_key"),
    ArtifactKind.OUTCOMES: ("run_id", "request_id"), ArtifactKind.WINDOWS: WINDOW_KEYS,
}


def validate_canonical(kind: ArtifactKind, document: LocatedDocument) -> None:
    if kind == ArtifactKind.MANIFEST:
        validate_manifest_header(document)
        for index in range(len(document.data["runs"])):
            validate_manifest_run(document, index)
        return
    if kind == ArtifactKind.ANCHOR_STATE:
        check(document, document.data, _ANCHOR)
        _check_ns_fields(document, document.data)
        return
    if kind not in _ROW_SCHEMAS:
        return
    check(document, document.data, list)
    unique_keys = _UNIQUE_ROWS.get(kind)
    seen = set()
    prior = {}
    for index, row in enumerate(document.data):
        key = (index,)
        check(document, row, _ROW_SCHEMAS[kind], key)
        _check_ns_fields(document, row, key)
        if unique_keys:
            identity = tuple(row[name] for name in unique_keys)
            if identity in seen:
                raise document.error((*key, unique_keys[-1]), f"duplicate key: {', '.join(unique_keys)}")
            seen.add(identity)
        if kind == ArtifactKind.ARRIVALS:
            for name in ("ttft_deadline_ns", "terminal_deadline_ns"):
                if row[name] < row["scheduled_arrival_ns"]:
                    raise document.error((*key, name), "must be at or after scheduled_arrival_ns")
        if kind == ArtifactKind.EVENTS:
            payload_spec = _EVENT_PAYLOADS[row["type"]]
            check(document, row["payload"], payload_spec, (*key, "payload"))
            if row["type"] == "tick" and row["payload"]:
                raise document.error((*key, "payload"), "tick payload must be empty")
            sequence, timestamp = row["gw_seq"], row["gw_at_ns"]
            previous = prior.get(row["run_id"])
            if previous is not None:
                if sequence <= previous[0]:
                    raise document.error((*key, "gw_seq"), "must be strictly increasing per run in file order")
                if timestamp < previous[1]:
                    raise document.error((*key, "gw_at_ns"), "must be nondecreasing per run in file order")
            prior[row["run_id"]] = sequence, timestamp
