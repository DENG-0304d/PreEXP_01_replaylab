"""Assess declared prerequisites and separately report observed platform checks."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from .artifacts import read_artifact
from .contracts import ArtifactKind, LocatedDocument, Nullable, Rule, TEXT, VERSION, check, choice, missing_hardware_fields

if TYPE_CHECKING:
    from .hardware import LeaseToken


BLOCKED_UNTIL_HARDWARE = "BLOCKED_UNTIL_HARDWARE"
BACKEND_NOT_CONNECTED = "真实执行后端尚未接入"
_STATUSES = ("fail", "blocked", "not_tested", "pass")
_GPU = ("environment.gpu", "environment.kubernetes.node_name",
        "environment.kubernetes.replicas.A.expected_gpu_uuid",
        "environment.kubernetes.replicas.B.expected_gpu_uuid")
_MODEL = ("environment.model", "environment.tokenizer", "environment.vllm",
          "environment.kubernetes.backend_image_digest")
_AUTH = ("environment.authorization",)
_BINDING = ("environment.replicas", "environment.kubernetes.namespace", "environment.kubernetes.node_name",
            "environment.kubernetes.replicas", "environment.kubernetes.gateway_image_digest",
            "environment.gateway_version")
_CLOCK = ("replay.clock_id", "environment.kubernetes.max_clock_alignment_error_ns")
_LEASE = ("environment.kubernetes.namespace", "environment.kubernetes.bench_lease_name",
          "environment.kubernetes.runner_image_digest", "environment.kubernetes.output_mount_path",
          "environment.kubernetes.injector_max_runtime_ns", "environment.kubernetes.output_pvc_name",
          "environment.kubernetes.recovery_subject")
_INJECTION = ("injection", "environment.kubernetes.injector_max_runtime_ns")
_COLLECTOR = ("environment.collector_version",)
_CHECK_FIELDS = {
    "authorization": _AUTH,
    "model_ready": _MODEL + _AUTH,
    "clock_alignment": _CLOCK + _AUTH,
    "gpu_allocation": _GPU + _AUTH,
    "cuda_concurrency": _GPU + _MODEL + _AUTH + _LEASE,
    "injection_locality": _GPU + _MODEL + _AUTH + _BINDING + _CLOCK + _LEASE + _INJECTION
                          + ("environment.kubernetes.locality_acceptance",),
    "replica_binding": _BINDING + _GPU + _MODEL + _AUTH,
    "attempt_isolation": _LEASE + _AUTH,
    "engine_progress": _GPU + _MODEL + _AUTH + _BINDING + _COLLECTOR,
    "device_verification": _GPU + _MODEL + _AUTH + _BINDING + _COLLECTOR + _LEASE,
    "evidence_freshness": _GPU + _MODEL + _AUTH + _BINDING + _CLOCK + _COLLECTOR + _LEASE
                          + ("observation.evidence_ttl_ns", "pairing.capture_perturbation_budget"),
}
_ACTION_CHECKS = ("authorization", "model_ready", "clock_alignment", "gpu_allocation", "cuda_concurrency",
                  "injection_locality", "replica_binding", "attempt_isolation")
_TIMING_CHECKS = ("gpu_allocation", "model_ready", "authorization", "clock_alignment", "replica_binding",
                  "attempt_isolation", "device_verification", "evidence_freshness")
_MUTATING_CHECKS = {"cuda_concurrency", "injection_locality", "device_verification", "evidence_freshness"}


@dataclass(frozen=True)
class EvidenceRef:
    path: str
    sha256: str
    field: str


@dataclass(frozen=True)
class CheckObservation:
    status: str
    reason: str
    evidence_refs: tuple[EvidenceRef, ...] = ()

    def __post_init__(self) -> None:
        if self.status not in _STATUSES or not self.reason:
            raise ValueError("expected a preflight status and reason")
        if self.status in {"pass", "fail"} and not self.evidence_refs:
            raise ValueError("an executed check needs evidence")


class PlatformProbe(Protocol):
    """Return observed results. Execute mutating checks only with a current owner."""

    source_kind: str

    def check(self, name: str, *, owner: LeaseToken | None = None) -> CheckObservation | None: ...


@dataclass(frozen=True)
class _MockProbe:
    observations: dict[str, CheckObservation]
    source_kind = "synthetic"

    def check(self, name: str, *, owner: LeaseToken | None = None) -> CheckObservation | None:
        return self.observations.get(name)


def read_mock_probe(path: Path) -> PlatformProbe:
    digest = sha256(path.read_bytes()).hexdigest()
    document = read_artifact(path, ArtifactKind.MOCK_PREFLIGHT, expected_sha256=digest)
    check(document, document.data, {"schema_version": VERSION, "source_kind": choice("synthetic"), "checks": dict})
    observations = {}
    for name, row in document.data["checks"].items():
        key = ("checks", name)
        if name not in _CHECK_FIELDS:
            raise document.error(key, "unknown preflight check")
        check(document, row, {"status": choice(*_STATUSES), "reason": TEXT}, key)
        refs = (EvidenceRef(str(path), digest, f"checks.{name}"),) if row["status"] in {"pass", "fail"} else ()
        observations[name] = CheckObservation(row["status"], row["reason"], refs)
    return _MockProbe(observations)


def _missing_fields(study: LocatedDocument, command: str) -> list[str]:
    missing = set(missing_hardware_fields(study.data, command))
    environment = study.data["environment"] or {}
    key = ("environment", "authorization")
    if "authorization" in environment:
        authorization = environment["authorization"]
        check(study, authorization, Nullable(Rule(lambda value: type(value) is bool or value in ("authorized", "denied"),
                                                   "expected authorized, denied, or a boolean")), key)
        if authorization in (None, False, "denied"):
            missing.add("environment.authorization")
    gpu = environment.get("gpu") or {}
    devices = gpu.get("device_ids")
    if devices is not None and len({device for device in devices if device is not None}) < 2:
        missing.add("environment.gpu.device_ids")
    kubernetes = environment.get("kubernetes")
    if isinstance(kubernetes, dict):
        if "authorization" not in environment:
            missing.add("environment.authorization")
        for field in ("recovery_subject", "output_pvc_name"):
            check(study, kubernetes.get(field), Nullable(TEXT), ("environment", "kubernetes", field))
            if kubernetes.get(field) is None:
                missing.add(f"environment.kubernetes.{field}")
        for section, field in (("observation", "evidence_ttl_ns"), ("pairing", "capture_perturbation_budget")):
            if (study.data[section] or {}).get(field) is None:
                missing.add(f"{section}.{field}")
        if command == "run" and study.data["replay"] is not None:
            inputs = study.data["replay"].get("inputs_path")
            check(study, inputs, Nullable(TEXT), ("replay", "inputs_path"))
            if inputs is None:
                missing.add("replay.inputs_path")
    return sorted(missing)


def _affects(field: str, requirement: str) -> bool:
    return (field == requirement or field.startswith(requirement + ".") or field.startswith(requirement + "[")
            or requirement.startswith(field + "."))


def _check_row(name: str, source_kind: str, observation: CheckObservation, missing: list[str]) -> dict:
    return {"name": name, "status": observation.status, "source_kind": source_kind, "reason": observation.reason,
            "missing_fields": missing, "evidence_refs": [vars(ref) for ref in observation.evidence_refs]}


@dataclass(frozen=True)
class PreflightReport:
    source_kind: str
    checks: tuple[dict, ...]
    missing_fields: tuple[str, ...]
    action_replay_status: str
    inflight_timing_status: str
    device_capability: str

    def as_dict(self) -> dict:
        simulated = self.source_kind == "synthetic"
        lead_reason = "device_timing_unverified"
        if self.device_capability == "device_verified":
            freshness = next(row for row in self.checks if row["name"] == "evidence_freshness")
            if freshness["status"] != "pass":
                lead_reason = "evidence_freshness_unverified"
            elif self.inflight_timing_status == "pass":
                lead_reason = "device_lead_requires_comparable_run_evidence"
        return {
            "schema_version": 2, "source_kind": self.source_kind, "checks": list(self.checks),
            "missing_fields": list(self.missing_fields), "action_replay_status": self.action_replay_status,
            "inflight_timing_status": self.inflight_timing_status, "device_capability": self.device_capability,
            "device_lead_status": "unverified", "device_lead_ns": None,
            "device_lead_reason": lead_reason,
            "backend_status": "mock_platform" if simulated else "not_connected",
            "backend_reason": None if simulated else BACKEND_NOT_CONNECTED,
            "reason": None if simulated else BLOCKED_UNTIL_HARDWARE,
        }


def evaluate_preflight(study: LocatedDocument, *, command: str, probe: PlatformProbe | None = None,
                       owner: LeaseToken | None = None) -> PreflightReport:
    """Evaluate capability checks without treating declarations as observations."""
    if command not in {"preflight", "run"}:
        raise ValueError("expected an execution-side command")
    source_kind = probe.source_kind if probe is not None else "real"
    simulated = source_kind == "synthetic"
    if simulated and command != "preflight":
        raise ValueError("mock platform observations are only for preflight")
    declared_missing = _missing_fields(study, command)
    missing_fields = [] if simulated else declared_missing
    rows = []
    for name, requirements in _CHECK_FIELDS.items():
        missing = [field for field in missing_fields if any(_affects(field, required) for required in requirements)]
        if missing:
            observation = CheckObservation("blocked", BLOCKED_UNTIL_HARDWARE)
        elif probe is None:
            observation = CheckObservation("not_tested", "real_probe_not_executed")
        elif not simulated and name in _MUTATING_CHECKS and owner is None:
            observation = CheckObservation("blocked", "bench_exclusivity_required")
        else:
            observation = probe.check(name, owner=owner) or CheckObservation("not_tested", "probe_result_missing")
        rows.append(_check_row(name, source_kind, observation, missing))
    statuses = {row["name"]: row["status"] for row in rows}
    action = min((statuses[name] for name in _ACTION_CHECKS), key=_STATUSES.index) if simulated else "blocked"
    timing = min((statuses[name] for name in _TIMING_CHECKS), key=_STATUSES.index)
    capability = "unavailable"
    if statuses["device_verification"] == "pass":
        capability = "device_verified"
    elif statuses["engine_progress"] == "pass":
        capability = "engine_progress_only"
        if timing == "not_tested":
            timing = "blocked"
    return PreflightReport(source_kind, tuple(rows), tuple(missing_fields), action, timing, capability)
