"""Deployment contracts and a synthetic bench control lifecycle."""

from collections.abc import Iterator
import argparse
from dataclasses import dataclass, replace
from pathlib import Path
import re
import sys
from threading import Lock, RLock
from typing import Protocol, TypedDict

from .artifacts import read_artifact, write_artifact
from .contracts import ArtifactKind, FormatError, LocatedDocument, Rule, TEXT, check, field_name, is_hardware_study
from .preflight import BLOCKED_UNTIL_HARDWARE, EvidenceRef


@dataclass(frozen=True)
class RunnerIdentity:
    runner_attempt_id: str
    pod_uid: str


@dataclass(frozen=True)
class AttemptIdentity:
    runner: RunnerIdentity
    run_id: str
    attempt_id: str
    output_dir: Path


@dataclass(frozen=True)
class LeaseSnapshot:
    resource_version: str
    holder: RunnerIdentity | None


@dataclass(frozen=True)
class LeaseToken:
    owner: RunnerIdentity
    acquired_resource_version: str


@dataclass(frozen=True)
class LeaseResult:
    status: str
    reason: str
    snapshot: LeaseSnapshot | None
    token: LeaseToken | None = None


class LeaseStore(Protocol):
    """A conditional update compares both resourceVersion and holder."""

    def read(self) -> LeaseSnapshot | None: ...

    def compare_and_set(self, expected: LeaseSnapshot, holder: RunnerIdentity | None) -> LeaseSnapshot | None:
        """Return None for a conflict. The caller rereads state before retrying."""
        ...


class MemoryLeaseStore:
    """An in-memory Kubernetes Lease model for synthetic orchestration."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._snapshot: LeaseSnapshot | None = None

    def initialize(self) -> LeaseSnapshot:
        with self._lock:
            if self._snapshot is None:
                self._snapshot = LeaseSnapshot("1", None)
            return self._snapshot

    def read(self) -> LeaseSnapshot | None:
        with self._lock:
            return self._snapshot

    def compare_and_set(self, expected: LeaseSnapshot, holder: RunnerIdentity | None) -> LeaseSnapshot | None:
        with self._lock:
            if self._snapshot != expected:
                return None
            self._snapshot = LeaseSnapshot(str(int(expected.resource_version) + 1), holder)
            return self._snapshot


@dataclass(frozen=True)
class OperationResult:
    status: str
    reason: str


@dataclass(frozen=True)
class InjectorRecord:
    attempt: AttemptIdentity
    deadline_ns: int
    started_at_ns: int
    stopped_at_ns: int | None = None
    stop_reason: str | None = None


class SyntheticInjector:
    """Virtual A-side supervisor whose clock advances independently of the runner."""

    def __init__(self, *, max_runtime_ns: int) -> None:
        if type(max_runtime_ns) is not int or max_runtime_ns <= 0:
            raise ValueError("expected a positive integer injector runtime in ns")
        self.max_runtime_ns = max_runtime_ns
        self._now_ns = 0
        self._active: AttemptIdentity | None = None
        self._records: dict[AttemptIdentity, InjectorRecord] = {}
        self._lock = Lock()

    def record(self, attempt: AttemptIdentity) -> InjectorRecord | None:
        with self._lock:
            return self._records.get(attempt)

    def is_running(self) -> bool:
        with self._lock:
            return self._active is not None

    def start(self, attempt: AttemptIdentity, *, deadline_ns: int) -> OperationResult:
        with self._lock:
            existing = self._records.get(attempt)
            if existing is not None:
                return OperationResult("pass", "already_started" if existing.stopped_at_ns is None else "already_stopped")
            if self._active is not None:
                return OperationResult("blocked", "another_injector_is_active")
            if type(deadline_ns) is not int or not self._now_ns < deadline_ns <= self._now_ns + self.max_runtime_ns:
                return OperationResult("blocked", "injector_deadline_outside_hard_runtime_limit")
            self._records[attempt] = InjectorRecord(attempt, deadline_ns, self._now_ns)
            self._active = attempt
            return OperationResult("pass", "injector_started")

    def stop(self, attempt: AttemptIdentity) -> None:
        with self._lock:
            if self._active == attempt:
                self._stop(attempt, self._now_ns, "cleanup")

    def advance(self, now_ns: int) -> None:
        with self._lock:
            if type(now_ns) is not int or now_ns < self._now_ns:
                raise ValueError("expected nondecreasing integer injector time in ns")
            self._now_ns = now_ns
            if self._active is not None:
                record = self._records[self._active]
                if record.deadline_ns <= now_ns:
                    self._stop(record.attempt, record.deadline_ns, "hard_deadline")

    def _stop(self, attempt: AttemptIdentity, stop_at_ns: int, reason: str) -> None:
        self._records[attempt] = replace(self._records[attempt], stopped_at_ns=stop_at_ns, stop_reason=reason)
        self._active = None


@dataclass(frozen=True)
class CleanupState:
    requests_drained: bool
    reservations_cleared: bool
    reset_ready: bool

    @property
    def confirmed(self) -> bool:
        return all(value is True for value in (self.requests_drained, self.reservations_cleared, self.reset_ready))


@dataclass(frozen=True)
class CleanupReport:
    attempt: AttemptIdentity
    status: str
    reason: str
    state: CleanupState
    injector_stopped: bool


@dataclass(frozen=True)
class RecoveryEvidence:
    subject: str
    owner: RunnerIdentity
    old_runner_stopped: bool
    injector_stopped: bool
    cleanup: CleanupState
    evidence_refs: tuple[EvidenceRef, ...]


class BenchControl:
    """Synthetic K04/K05 lifecycle. The real platform must supply its own probes and supervisor."""

    source_kind = "synthetic"

    def __init__(self, lease: LeaseStore, injector: SyntheticInjector, *, recovery_subject: str) -> None:
        self.lease = lease
        self.injector = injector
        self.recovery_subject = recovery_subject
        self._lock = RLock()
        self._token: LeaseToken | None = None
        self._active: AttemptIdentity | None = None
        self._last_cleanup: CleanupReport | None = None
        self._attempts: dict[tuple[str, str], AttemptIdentity] = {}
        self._paths: set[Path] = set()
        self._closed: set[AttemptIdentity] = set()
        self._runner_attempt_ids: set[str] = set()

    def register_attempt(self, attempt: AttemptIdentity) -> OperationResult:
        with self._lock:
            key = (attempt.run_id, attempt.attempt_id)
            if key in self._attempts:
                return OperationResult("blocked", "attempt_identity_reused")
            path = attempt.output_dir.resolve()
            if any(path == used or path in used.parents or used in path.parents for used in self._paths):
                return OperationResult("blocked", "attempt_output_not_isolated")
            try:
                path.mkdir(parents=True, exist_ok=False)
            except FileExistsError:
                return OperationResult("blocked", "attempt_output_already_exists")
            except OSError:
                return OperationResult("blocked", "attempt_output_unwritable")
            self._attempts[key] = attempt
            self._paths.add(path)
            return OperationResult("pass", "attempt_registered")

    def acquire(self, owner: RunnerIdentity) -> LeaseResult:
        with self._lock:
            current = self.lease.read()
            if current is not None and current.holder is None and owner.runner_attempt_id in self._runner_attempt_ids:
                return LeaseResult("blocked", "runner_attempt_reused", current)
            result = self._update_lease(None, owner)
            if result.status == "pass":
                self._token = LeaseToken(owner, result.snapshot.resource_version)
                self._runner_attempt_ids.add(owner.runner_attempt_id)
                return replace(result, reason="lease_acquired", token=self._token)
            return result

    def _update_lease(self, expected: RunnerIdentity | None, target: RunnerIdentity | None) -> LeaseResult:
        # Recheck owner on each read after a conflict. Resource versions are opaque.
        for _ in range(3):
            current = self.lease.read()
            if current is None:
                return LeaseResult("blocked", "lease_not_initialized", None)
            if current.holder != expected:
                reason = "lease_held" if expected is None else "lease_holder_changed"
                return LeaseResult("blocked", reason, current)
            updated = self.lease.compare_and_set(current, target)
            if updated is not None:
                return LeaseResult("pass", "lease_updated", updated)
        return LeaseResult("blocked", "lease_conflict", self.lease.read())

    def _is_current(self, token: LeaseToken | None) -> bool:
        current = self.lease.read()
        return token is not None and token == self._token and current is not None and current.holder == token.owner

    def reset(self, token: LeaseToken, attempt: AttemptIdentity) -> OperationResult:
        with self._lock:
            if not self._is_current(token) or attempt.runner != token.owner:
                return OperationResult("blocked", "attempt_owner_mismatch")
            if self._attempts.get((attempt.run_id, attempt.attempt_id)) != attempt or attempt in self._closed:
                return OperationResult("blocked", "attempt_not_available")
            if self._active is not None:
                if self._active == attempt:
                    return OperationResult("pass", "already_bound")
                return OperationResult("blocked", "previous_attempt_cleanup_unconfirmed")
            if self.injector.is_running():
                return OperationResult("blocked", "injector_still_active")
            self._active = attempt
            self._last_cleanup = None
            return OperationResult("pass", "synthetic_reset_confirmed")

    def start_injector(self, token: LeaseToken, attempt: AttemptIdentity, *, deadline_ns: int) -> OperationResult:
        with self._lock:
            if not self._is_current(token) or attempt.runner != token.owner:
                return OperationResult("blocked", "attempt_owner_mismatch")
            if self._active != attempt or self._last_cleanup is not None:
                record = self.injector.record(attempt)
                if record is not None and record.stopped_at_ns is not None:
                    return OperationResult("pass", "already_stopped")
                return OperationResult("blocked", "attempt_not_active")
            return self.injector.start(attempt, deadline_ns=deadline_ns)

    def cleanup(self, token: LeaseToken, attempt: AttemptIdentity, *, state: CleanupState) -> CleanupReport:
        with self._lock:
            if not self._is_current(token) or self._active != attempt:
                return CleanupReport(attempt, "blocked", "attempt_owner_mismatch", state, False)
            self.injector.stop(attempt)
            stopped = not self.injector.is_running()
            clean = state.confirmed and stopped
            report = CleanupReport(attempt, "pass" if clean else "blocked",
                                   "cleanup_confirmed" if clean else "cleanup_unconfirmed", state, stopped)
            self._last_cleanup = report
            if clean:
                self._closed.add(attempt)
                self._active = None
            return report

    def release(self, token: LeaseToken, *, cleanup: CleanupReport | None = None) -> LeaseResult:
        with self._lock:
            if not self._is_current(token):
                return LeaseResult("blocked", "lease_holder_changed", self.lease.read())
            if (self._active is not None or self.injector.is_running() or cleanup != self._last_cleanup
                    or (cleanup is not None and cleanup.status != "pass")):
                return LeaseResult("blocked", "cleanup_unconfirmed", self.lease.read())
            result = self._update_lease(token.owner, None)
            if result.status == "pass":
                self._clear_holder()
                return replace(result, reason="lease_released")
            return result

    def recover(self, stale: LeaseSnapshot, *, evidence: RecoveryEvidence) -> LeaseResult:
        with self._lock:
            if not self.recovery_subject or evidence.subject != self.recovery_subject:
                return LeaseResult("blocked", "recovery_subject_not_authorized", self.lease.read())
            if (stale.holder is None or stale.holder != evidence.owner or evidence.old_runner_stopped is not True
                    or evidence.injector_stopped is not True or not evidence.cleanup.confirmed or not evidence.evidence_refs):
                return LeaseResult("blocked", "recovery_evidence_incomplete", self.lease.read())
            if self.injector.is_running():
                return LeaseResult("blocked", "injector_still_active", self.lease.read())
            result = self._update_lease(stale.holder, None)
            if result.status == "pass":
                self._clear_holder()
                return replace(result, reason="stale_lease_recovered")
            return result

    def _clear_holder(self) -> None:
        if self._active is not None:
            self._closed.add(self._active)
        self._active = None
        self._token = None
        self._last_cleanup = None


class PlatformEvidence(TypedDict):
    run_id: str
    attempt_id: str
    runner_attempt_id: str
    kind: str
    observed_at_ns: int
    status: str
    payload: dict
    record_seq: int


class PlatformCollector(Protocol):
    """Use per-run monotonic ns. Archive coverage gaps and binding changes as offline evidence."""

    def capture_binding(self, attempt: AttemptIdentity, *, phase: str) -> PlatformEvidence:
        """At reset, anchor and end, record one ready A/B endpoint each with Pod, node, container, worker and GPU identity."""
        ...

    def collect(self, attempt: AttemptIdentity) -> Iterator[PlatformEvidence]:
        """Archive binding, endpoint, epoch, gap, lease, injector and cleanup records with attempt ownership."""
        ...


@dataclass(frozen=True)
class FrozenActionBranch:
    """Fields from the immutable study and original plan branch, before any real attempt."""

    condition_id: str
    pair_id: str
    anchor_id: str
    anchor_at_ns: int
    study_split: str
    role: str
    delay_ns: int | None
    input_trace_path: Path
    inputs_path: Path
    warmup_trace_path: Path
    prehistory_trace_path: Path
    study_path: Path
    plan_path: Path
    config_sha256: str
    plan_sha256: str


@dataclass(frozen=True)
class IterationIdentity:
    attempt: AttemptIdentity
    replica_id: str
    replica_epoch: int
    step_id: str
    work_version: int


@dataclass(frozen=True)
class DeviceObservation:
    """Raw producer-side facts. Gateway receipt and dispatch must be recorded separately."""

    iteration: IterationIdentity
    sample_ns: int
    published_ns: int
    clock_mapping_id: str
    marker_recorded: bool | None
    marker_iteration: IterationIdentity | None
    covers_target_work: bool | None
    query_result: str
    evidence_refs: tuple[EvidenceRef, ...]


class ActionReplay(Protocol):
    """M1b consumes the frozen branch and emits canonical run artifacts on real hardware."""

    def run(self, attempt: AttemptIdentity, branch: FrozenActionBranch) -> Path:
        """Return the attempt manifest after independent reset, open-loop input and cleanup."""
        ...


class DeviceCollector(Protocol):
    """M1c must archive capture provenance, marker/query results and clock error separately from freshness."""

    def sample(self, iteration: IterationIdentity) -> DeviceObservation:
        """Capture one raw marker/work binding, query outcome and sampling-time record."""
        ...


class HardwareBlocked(RuntimeError):
    """This adapter is waiting for hardware integration and measured evidence."""


class BlockedHardware:
    def run(self, attempt: AttemptIdentity, branch: FrozenActionBranch) -> Path:
        raise HardwareBlocked(f"{BLOCKED_UNTIL_HARDWARE}: M1b action replay backend not connected")

    def sample(self, iteration: IterationIdentity) -> DeviceObservation:
        raise HardwareBlocked(f"{BLOCKED_UNTIL_HARDWARE}: M1c device collector not connected")


class DeploymentBlocked(HardwareBlocked):
    def __init__(self, missing_fields: list[str]) -> None:
        self.missing_fields = tuple(sorted(missing_fields))
        super().__init__(f"{BLOCKED_UNTIL_HARDWARE}: missing {', '.join(self.missing_fields)}")


_DNS_NAME = Rule(lambda value: type(value) is str and len(value) <= 63
                 and re.fullmatch(r"[a-z0-9](?:[-a-z0-9]*[a-z0-9])?", value) is not None,
                 "expected a Kubernetes DNS label")
_HOST_LABEL = Rule(lambda value: type(value) is str and len(value) <= 63
                   and re.fullmatch(r"[A-Za-z0-9](?:[-_.A-Za-z0-9]*[A-Za-z0-9])?", value) is not None,
                   "expected a nonempty Kubernetes node label value")
_IMAGE = Rule(lambda value: type(value) is str and re.fullmatch(r"[^\s@]+@sha256:[0-9a-f]{64}", value) is not None,
               "expected an image pinned with @sha256:<digest>")
_PORT = Rule(lambda value: type(value) is int and 1 <= value <= 65535, "expected an integer port from 1 to 65535")
_ABSOLUTE_PATH = Rule(lambda value: type(value) is str and value.startswith("/") and not any(c.isspace() for c in value),
                      "expected an absolute container or HTTP path")
_STORAGE_SIZE = Rule(lambda value: type(value) is str and re.fullmatch(r"[1-9][0-9]*[KMGTPE]i", value) is not None,
                     "expected a positive binary Kubernetes storage size, such as 5Gi")
_KUBE = ("environment", "kubernetes")
_RENDER_FIELDS = {
    "namespace": _DNS_NAME, "node_hostname_label": _HOST_LABEL, "bench_lease_name": _DNS_NAME,
    "gateway_deployment_name": _DNS_NAME, "gateway_service_name": _DNS_NAME, "job_name": _DNS_NAME,
    "gateway_image_digest": _IMAGE, "runner_image_digest": _IMAGE, "backend_image_digest": _IMAGE,
    "backend_args": [TEXT], "backend_port": _PORT, "gateway_port": _PORT,
    "backend_startup_probe_path": _ABSOLUTE_PATH, "backend_readiness_probe_path": _ABSOLUTE_PATH,
    "gateway_startup_probe_path": _ABSOLUTE_PATH, "gateway_readiness_probe_path": _ABSOLUTE_PATH,
    "output_mount_path": _ABSOLUTE_PATH, "output_pvc_name": _DNS_NAME, "output_storage_size": _STORAGE_SIZE,
    "output_storage_class": TEXT, "recovery_subject": TEXT,
    **{f"replicas.{replica}.{field}": _DNS_NAME for replica in "AB"
       for field in ("deployment_name", "service_name", "container_name")},
}


def _render_values(study: LocatedDocument) -> dict:
    if not is_hardware_study(study.data) or study.data.get("source_kind") == "synthetic":
        raise study.error(("source_kind",), "expected a real Kubernetes study")
    values, missing = {}, []
    for reference, spec in _RENDER_FIELDS.items():
        value = (study.data["environment"] or {}).get("kubernetes")
        parts = tuple(reference.split("."))
        for part in parts:
            value = value.get(part) if isinstance(value, dict) else None
        if value is None:
            missing.append(field_name((*_KUBE, *parts)))
        else:
            check(study, value, spec, (*_KUBE, *parts))
            values[reference] = value
    if missing:
        raise DeploymentBlocked(missing)
    for suffix in ("deployment_name", "service_name"):
        fields = [f"replicas.A.{suffix}", f"replicas.B.{suffix}", f"gateway_{suffix}"]
        names = set()
        for reference in fields:
            name = values[reference]
            if name in names:
                raise study.error((*_KUBE, *reference.split(".")), "A, B and gateway resource names must be distinct")
            names.add(name)
    return values


def _substitute(template: LocatedDocument, node: object, values: dict, key: tuple = ()) -> object:
    if isinstance(node, dict):
        if "$config" in node:
            ref = node["$config"]
            if len(node) != 1 or not isinstance(ref, str) or ref not in values:
                raise template.error((*key, "$config"), "unknown or malformed deployment parameter")
            return values[ref]
        return {name: _substitute(template, value, values, (*key, name)) for name, value in node.items()}
    if isinstance(node, list):
        return [_substitute(template, value, values, (*key, index)) for index, value in enumerate(node)]
    return node


def render_kubernetes(study: LocatedDocument, *, template_dir: Path, out_dir: Path) -> tuple[Path, ...]:
    """Render staged manifests locally. Lease bootstrap remains a separate, create-only artifact."""
    values = _render_values(study)
    rendered = {}
    for name in ("bench", "lease-bootstrap"):
        template = read_artifact(template_dir / f"{name}.template.json", ArtifactKind.KUBERNETES_TEMPLATE)
        check(template, template.data, dict)
        rendered[out_dir / f"{name}.json"] = _substitute(template, template.data, values)
    try:
        out_dir.mkdir(parents=True, exist_ok=False)
    except OSError as exc:
        raise FormatError(out_dir, 1, "--out", f"cannot create a new output directory: {exc}") from exc
    for path, data in rendered.items():
        write_artifact(path, ArtifactKind.KUBERNETES_TEMPLATE, data)
    return tuple(rendered)


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m replaylab.hardware", allow_abbrev=False)
    commands = parser.add_subparsers(dest="command", required=True)
    renderer = commands.add_parser("render", allow_abbrev=False)
    renderer.add_argument("--config", type=Path, required=True)
    renderer.add_argument("--template-dir", type=Path, required=True)
    renderer.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        study = read_artifact(args.config, ArtifactKind.STUDY)
        render_kubernetes(study, template_dir=args.template_dir, out_dir=args.out)
    except DeploymentBlocked as exc:
        print(exc, file=sys.stderr)
        return 3
    except FormatError as exc:
        print(exc, file=sys.stderr)
        return 2
    except OSError as exc:
        print(FormatError(Path(exc.filename or args.out), 1, "$", f"cannot write manifest: {exc}"), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
