"""Synthetic open-loop replay with independent attempts and raw-byte artifacts."""

from dataclasses import dataclass
from hashlib import sha256
from heapq import heappop, heappush
from pathlib import Path

from .artifacts import json_text, read_artifact, write_artifact
from .contracts import POS_INT, TEXT, UINT, ArtifactKind, FormatError, LocatedDocument
from .diversion import DiversionConfig, DiversionController
from .ledger import GatewayCursor, GatewayLedger
from .mock_backend import Failure, MockBackend, MockScenario
from .progress import FrozenHealthProfile, ProgressConfig, ProgressTracker, _required
from .router import GatewayRouter, RouterConfig, RoutingRequest


_ORDER = {
    "injection": 0, "terminal": 1, "feedback": 1, "failure": 2, "anchor": 3,
    "arm": 4, "timer": 5, "tick": 6, "arrival": 7, "decision": 8, "handoff": 9,
}


def _digest(data: bytes) -> str:
    return sha256(data).hexdigest()


def frozen_branch_order(block: dict, rotation_seed: int) -> tuple[dict, ...]:
    key = [rotation_seed, block["condition_id"], block["pair_id"], block["anchor_id"], block["study_split"]]

    def order(branch):
        wire = json_text([*key, branch["role"], branch["delay_ns"]])
        return _digest(wire.encode()), wire

    return tuple(sorted(block["branches"], key=order))


def _input(study: LocatedDocument, key: tuple[str, ...], kind: ArtifactKind) -> tuple[LocatedDocument, bytes]:
    value = _required(study, key, TEXT)
    path = study.file.parent / value
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise study.error(key, f"cannot read configured input: {exc}") from exc
    return read_artifact(path, kind, expected_sha256=_digest(raw)), raw


@dataclass(frozen=True)
class SyntheticInputs:
    study_id: str
    arrivals: LocatedDocument
    arrivals_bytes: bytes
    inputs_bytes: bytes
    prompts: dict[str, str]
    warmup: list[dict]
    prehistory: list[dict]
    health: FrozenHealthProfile
    scenario: MockScenario
    scenario_sha256: str
    prehistory_id: str
    observation_end_ns: int
    tick_ns: int
    rotation_seed: int
    injection_offsets: tuple[int, int]

    @classmethod
    def from_study(cls, study: LocatedDocument) -> "SyntheticInputs":
        arrivals, arrival_bytes = _input(study, ("replay", "input_trace_path"), ArtifactKind.ARRIVALS)
        inputs, input_bytes = _input(study, ("replay", "inputs_path"), ArtifactKind.INPUTS)
        warmup, warm_bytes = _input(study, ("replay", "warmup_trace_path"), ArtifactKind.INPUTS)
        prehistory, earlier_bytes = _input(study, ("replay", "prehistory_trace_path"), ArtifactKind.EVENTS)
        profile, health_bytes = _input(study, ("observation", "health_profile_path"), ArtifactKind.HEALTH_PROFILE)
        scenario_doc, scenario_bytes = _input(study, ("replay", "mock_scenario_path"), ArtifactKind.MOCK_SCENARIO)
        scenario = MockScenario.from_document(scenario_doc)
        prompts = {row["input_id"]: row["prompt"] for row in inputs.data}
        observation_end = _required(study, ("replay", "observation_end_ns"), UINT)
        for index, row in enumerate(arrivals.data):
            prompt = prompts.get(row["input_id"])
            if prompt is None:
                raise arrivals.error((index, "input_id"), "request input is missing")
            if len(prompt.split()) != row["prompt_tokens"]:
                raise arrivals.error((index, "prompt_tokens"), "synthetic whitespace token count differs from prompt")
            if row["scheduled_arrival_ns"] >= observation_end:
                raise arrivals.error((index, "scheduled_arrival_ns"), "arrivals must precede the observation end")
        for index, row in enumerate(prehistory.data):
            if row["type"] not in {"work_state", "step_start", "step_complete", "inflight_snapshot"}:
                raise prehistory.error((index, "type"), "prehistory contains a non-observation event")
        # Check scripted feedback before starting any attempt so a malformed shared input is exit 2.
        feedback = sorted(scenario.observations, key=lambda row: row["at_ns"])
        validator = GatewayLedger("validation")
        for index, replica in enumerate(("A", "B"), 1):
            validator.apply(dict(run_id="validation", event_id=f"initial-{replica}", type="replica_epoch",
                                 gw_at_ns=0, gw_seq=index, event_at_ns=0, payload=dict(replica_id=replica, epoch=1)))
        for index, row in enumerate(feedback, 3):
            validator.apply(dict(run_id="validation", event_id=row["event_id"], type=row["type"], gw_at_ns=row["at_ns"],
                                 gw_seq=index, event_at_ns=row.get("event_at_ns", row["at_ns"]), payload=row["payload"]),
                            location=scenario_doc.location(("observations", scenario.observations.index(row))))
        return cls(
            _required(study, ("study_id",), TEXT), arrivals, arrival_bytes, input_bytes, prompts, warmup.data,
            prehistory.data, FrozenHealthProfile.from_document(profile), scenario, _digest(scenario_bytes),
            _digest(warm_bytes + earlier_bytes + health_bytes), observation_end,
            _required(study, ("observation", "tick_ns"), POS_INT),
            _required(study, ("replay", "rotation_seed"), int),
            tuple(_required(study, ("injection", field), int) for field in ("start_offset_ns", "stop_offset_ns")),
        )


@dataclass(frozen=True)
class RunResult:
    manifest: dict
    cleanup_blocked: bool
    complete_n: int


@dataclass(frozen=True)
class _RuntimeConfig:
    progress: ProgressConfig
    router: RouterConfig
    action: DiversionConfig

    @classmethod
    def from_study(cls, study: LocatedDocument) -> "_RuntimeConfig":
        return cls(ProgressConfig.from_study(study), RouterConfig.from_study(study), DiversionConfig.from_study(study))


class _Attempt:
    def __init__(self, rec: dict, inputs: SyntheticInputs, config: _RuntimeConfig, failure: Failure | None):
        self.rec, self.inputs, self.config, self.failure = rec, inputs, config, failure
        self.clock = self.seq = self.scheduled = 0
        self.queue = []
        self.events, self.diagnostics = [], []
        self.outcomes, self.arrivals = {}, {}
        self.reservation_ids = []
        self.request_data = {row["request_id"]: row for row in inputs.arrivals.data}
        self.anchor_state = None
        self.releases_n = self.transitions_n = 0
        self.gateway = GatewayLedger(rec["run_id"])
        self.progress = ProgressTracker(config.progress, inputs.health)
        self.action = DiversionController(config.action, GatewayRouter(config.router, self.progress),
                                          anchor_id=rec["anchor_id"], anchor_at_ns=rec["anchor_at_ns"])
        self.inject_at = tuple(rec["anchor_at_ns"] + offset for offset in inputs.injection_offsets)
        self.backend = MockBackend(inputs.scenario, attempt_id=rec["independent_start"]["attempt_id"],
                                   condition_id=rec["condition_id"], injection_window=self.inject_at)

    def _cursor(self) -> GatewayCursor:
        self.seq += 1
        return GatewayCursor(self.rec["run_id"], self.seq, self.clock)

    def _emit(self, kind: str, payload: dict, *, cursor=None, event_id=None, event_at_ns=None, local=False) -> None:
        cursor = cursor or self._cursor()
        row = dict(run_id=cursor.run_id, event_id=event_id or f"gateway-{cursor.gw_seq:06}", type=kind,
                   gw_seq=cursor.gw_seq, gw_at_ns=cursor.gw_at_ns,
                   event_at_ns=cursor.gw_at_ns if event_at_ns is None else event_at_ns, payload=payload)
        update = self.gateway.apply(row)
        self.progress.consume(update)
        if not local:
            self.action.consume(update)
        self.events.append(row)

    def _diag(self, kind: str, **details) -> None:
        self.diagnostics.append(dict(run_id=self.rec["run_id"], record_id=f"diag-{len(self.diagnostics) + 1:04}",
                                     observed_at_ns=self.clock, kind=kind,
                                     payload=dict(source_kind="synthetic", **details)))

    def _schedule(self, clock: int, kind: str, value=None) -> None:
        self.scheduled += 1
        heappush(self.queue, (clock, _ORDER[kind], self.scheduled, kind, value))

    def _initialize(self) -> None:
        self.backend.reset()
        self._diag("reset", attempt_id=self.backend.owner, unfinished_n=len(self.backend.jobs))
        self.backend.warmup(self.inputs.warmup)
        self._diag("warmup", prefix_set_id=self.backend.prefix_id, input_n=len(self.inputs.warmup))
        for replica in ("A", "B"):
            self._emit("replica_epoch", dict(replica_id=replica, epoch=1))
            self._emit("route_state", self.backend.snapshot(replica, self.clock))
        anchor, end = self.rec["anchor_at_ns"], self.inputs.observation_end_ns
        for index, at in enumerate(self.inject_at):
            self._schedule(at, "injection", index)
        for row in self.inputs.prehistory:
            self._schedule(row["gw_at_ns"], "feedback", {**row, "at_ns": row["gw_at_ns"]})
        for row in self.inputs.scenario.observations:
            self._schedule(row["at_ns"], "feedback", row)
        self._schedule(anchor, "anchor")
        if self.rec["planned_delay_ns"] is not None:
            self._schedule(anchor + self.rec["planned_delay_ns"], "arm")
        for at in sorted(set(range(0, end, self.inputs.tick_ns)) | {anchor, end}):
            self._schedule(at, "tick")
        for row in self.inputs.arrivals.data:
            self._schedule(row["scheduled_arrival_ns"], "arrival", row)
        if self.failure is not None:
            self._schedule(self.failure.at_ns, "failure")

    def _outcome(self, key, status, *, job=None, at=None) -> None:
        known_completed = job is not None and status == "ok"
        self.outcomes[key] = dict(
            run_id=self.rec["run_id"], request_id=key, terminal_status=status, actual_arrival_ns=self.arrivals.get(key),
            first_token_ns=job.first_token_ns if known_completed else None,
            last_token_ns=(job.terminal_ns if job.output_tokens > 1 else job.first_token_ns) if known_completed else None,
            output_tokens=job.output_tokens if known_completed else None if status == "unresolved" else 0,
            terminal_ns=job.terminal_ns if known_completed else at,
        )

    def _flush_local_facts(self) -> None:
        for item in self.action.transitions[self.transitions_n:]:
            if item.kind != "rule_armed":
                self._emit(item.kind, item.payload(), local=True)
        self.transitions_n = len(self.action.transitions)
        for item in self.action.releases[self.releases_n:]:
            held = item.reservation
            self._emit("reservation_release", dict(request_id=held.request_id, replica_id=held.target,
                                                   reservation_id=held.reservation_id, state_version=None,
                                                   reason=item.reason), local=True)
            terminal = self.action.terminal_outcome(held.request_id)
            if held.sent_at is None and terminal is not None:
                self._outcome(held.request_id, terminal.status, at=terminal.cursor.gw_at_ns)
        self.releases_n = len(self.action.releases)

    def _handle(self, kind: str, value) -> None:
        if kind == "injection":
            if value == 0:
                self.backend.start_injection()
            else:
                self.backend.stop_injection()
            edge = "start" if value == 0 else "stop"
            self._diag(f"injector_{edge}", actual_at_ns=self.clock, planned_at_ns=self.inject_at[value])
        elif kind == "feedback":
            self._emit(value["type"], value["payload"], event_id=value["event_id"], event_at_ns=value.get("event_at_ns"))
        elif kind == "anchor":
            self.anchor_state = self.backend.anchor(self.rec["anchor_id"], self.clock)
        elif kind == "arm":
            cursor = self._cursor()
            event = self.action.arm(cursor=cursor, activation_id=f"activation-{self.rec['anchor_id']}")
            self._emit(event.kind, event.payload(), cursor=cursor, local=True)
        elif kind == "timer":
            self.action.on_timer(cursor=self._cursor())
        elif kind == "tick":
            self._emit("tick", {})
        elif kind == "arrival":
            key = value["request_id"]
            self.arrivals[key] = self.clock
            self._emit("arrival", dict(request_id=key, client_sent_ns=self.clock))
            self._schedule(self.clock, "decision", value)
        elif kind == "decision":
            key = value["request_id"]
            cursor = self._cursor()
            prompt = self.inputs.prompts[value["input_id"]]
            hint = self.backend.cache_hint(prompt, value["prompt_tokens"])
            decision = self.action.decide(RoutingRequest(key, self.arrivals[key], value["prompt_tokens"]),
                                          cursor=cursor, cache_hints=dict(A=hint, B=hint))
            self._emit("dispatch_decision", decision.payload(), cursor=cursor, local=True)
            if decision.actual_target is None:
                terminal = self.action.terminal_outcome(key)
                self._outcome(key, terminal.status, at=self.clock)
            else:
                self.reservation_ids.append(decision.reservation_id)
                self._schedule(self.clock, "handoff", decision)
        elif kind == "handoff":
            key = value.request_id
            cursor = self._cursor()
            proposed = self.request_data[key]

            def accept(decision):
                job = self.backend.submit(proposed, target=decision.actual_target, at_ns=self.clock,
                                          reservation_id=decision.reservation_id)
                self._schedule(job.terminal_ns, "terminal", job)
                return True

            sent = self.action.handoff(key, cursor=cursor, accept=accept)
            if sent.sent:
                self._emit("dispatch_sent", sent.dispatch_sent_payload(), cursor=cursor, local=True)
                target = value.actual_target
                snapshot = self.backend.snapshot(target, self.clock)
                self._emit("route_state", snapshot)
                self._emit("dispatch_ack", dict(request_id=key, replica_id=target, reservation_id=value.reservation_id,
                                                state_version=snapshot["state_version"], reason="accepted"))
            elif sent.terminal_status:
                self._outcome(key, sent.terminal_status, at=sent.cursor.gw_at_ns)
        elif kind == "terminal":
            self.action.record_terminal(value.request_id, cursor=self._cursor(), terminal_status="ok")
            self._outcome(value.request_id, "ok", job=value)
            self._emit("route_state", self.backend.snapshot(value.target, self.clock))
        self._flush_local_facts()

    def execute(self) -> tuple[str | None, bool]:
        reason = None
        try:
            self._initialize()
            while self.queue or self.action.next_deadline_ns() is not None:
                earliest = self.queue[0][:2] if self.queue else (self.inputs.observation_end_ns + 1, 0)
                deadline = self.action.next_deadline_ns()
                is_timer = deadline is not None and (deadline, _ORDER["timer"]) < earliest
                if is_timer:
                    at, kind, value = deadline, "timer", None
                else:
                    if not self.queue:
                        break
                    at, _, _, kind, value = heappop(self.queue)
                if at > self.inputs.observation_end_ns:
                    break
                self.clock = at
                if kind == "failure":
                    reason = self.failure.reason
                    self._diag("attempt_failed", reason=reason)
                    break
                self._handle(kind, value)
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            self._diag("attempt_failed", reason=reason)
        finally:
            for key in sorted(self.arrivals.keys() - self.outcomes.keys()):
                self._outcome(key, "unresolved")
            released = 0
            for res_id in self.reservation_ids:
                if self.action.router.reservation(res_id) is not None:
                    self.action.router.release(res_id, cursor=self._cursor(), reason="attempt_cleanup")
                    released += 1
            confirm = self.failure is None or self.failure.cleanup_confirmed
            cleaned = self.backend.cleanup(confirm=confirm)
            self._diag("cleanup_check", confirmed=cleaned, attempt_id=self.backend.owner,
                       released_reservation_n=released,
                       process_owned=self.backend.process_owner == self.backend.owner,
                       injection_owned=self.backend.injection_owner == self.backend.owner)
        return reason, cleaned

    def save(self, root: Path) -> dict:
        path = root / "runs" / self.rec["run_id"]
        path.mkdir(parents=True, exist_ok=False)
        (path / ArtifactKind.ARRIVALS).write_bytes(self.inputs.arrivals_bytes)
        (path / ArtifactKind.INPUTS).write_bytes(self.inputs.inputs_bytes)
        write_artifact(path / ArtifactKind.HEALTH_PROFILE, ArtifactKind.HEALTH_PROFILE, self.inputs.health.to_rows())
        write_artifact(path / ArtifactKind.EVENTS, ArtifactKind.EVENTS, self.events)
        write_artifact(path / ArtifactKind.DIAGNOSTICS, ArtifactKind.DIAGNOSTICS, self.diagnostics)
        write_artifact(path / ArtifactKind.OUTCOMES, ArtifactKind.OUTCOMES, self.outcomes.values())
        if self.anchor_state is not None:
            write_artifact(path / ArtifactKind.ANCHOR_STATE, ArtifactKind.ANCHOR_STATE, self.anchor_state)
        return {file.name: dict(path=file.relative_to(root).as_posix(), sha256=_digest(file.read_bytes()))
                for file in sorted(path.iterdir())}


def _check_schedule(plan: LocatedDocument, inputs: SyntheticInputs) -> None:
    for index, block in enumerate(plan.data["blocks"]):
        anchor = block["anchor_at_ns"]
        if not 0 <= anchor + inputs.injection_offsets[0] < anchor + inputs.injection_offsets[1] <= inputs.observation_end_ns:
            raise plan.error(("blocks", index, "anchor_at_ns"), "synthetic injection window is outside the observation")
        if any(row["gw_at_ns"] >= anchor for row in inputs.prehistory):
            raise plan.error(("blocks", index, "anchor_at_ns"), "common prehistory must precede the anchor")
        for offset, branch in enumerate(block["branches"]):
            delay = branch["delay_ns"]
            if delay is not None and anchor + delay >= inputs.observation_end_ns:
                raise plan.error(("blocks", index, "branches", offset, "delay_ns"), "planned arm must precede observation end")


def run_study(study: LocatedDocument, plan: LocatedDocument, *, out_dir: Path) -> RunResult:
    inputs = SyntheticInputs.from_study(study)
    config = _RuntimeConfig.from_study(study)
    _check_schedule(plan, inputs)
    study_hash = _digest(study.file.read_bytes())
    plan_bytes = plan.file.read_bytes()
    (out_dir / "plan.json").write_bytes(plan_bytes)
    failures = {}
    for failure in inputs.scenario.failures:
        key = failure.pair_id, failure.role
        if key in failures:
            raise FormatError(study.file, 1, "replay.mock_scenario_path", "multiple failures target the same branch role")
        failures[key] = failure
    manifest = dict(study_id=inputs.study_id, schema_version=2, source_kind="synthetic",
                    config_sha256=study_hash, plan_path="plan.json", plan_sha256=_digest(plan_bytes), runs=[])
    retry_records = []
    cleanup_blocked = False

    def attempt(rec: dict, failure: Failure | None):
        run = _Attempt(rec, inputs, config, failure)
        reason, cleaned = run.execute()
        rec.update(run_status="complete" if reason is None and cleaned else "failed",
                   cleanup_confirmed=cleaned, observed_until_ns=run.clock, files=run.save(out_dir))
        if reason is not None or not cleaned:
            rec["reason"] = reason or "cleanup_unconfirmed"
        return cleaned

    for block in plan.data["blocks"]:
        for branch in frozen_branch_order(block, inputs.rotation_seed):
            numbered = len(manifest["runs"]) + 1
            run_id = f"synthetic-{numbered:04}"
            record = {key: block[key] for key in ("condition_id", "pair_id", "anchor_id", "anchor_at_ns", "study_split")}
            record.update(
                run_id=run_id, source_kind="synthetic", role=branch["role"], planned_delay_ns=branch["delay_ns"],
                arrival_trace_id=_digest(inputs.arrivals_bytes), prehistory_id=inputs.prehistory_id,
                eval_cohort_id=_digest(json_text(sorted(row["request_id"] for row in inputs.arrivals.data if row["in_eval"])).encode()),
                slo_id=study.data["slo"]["slo_id"], router_id=study.data["router"]["router_id"],
                router_digest=study.data["router"]["router_digest"], diversion_rule_id=config.action.rule_id,
                intervention_digest=_digest(json_text([study_hash, inputs.scenario_sha256, block["condition_id"]]).encode()),
                clock_id=study.data["replay"]["clock_id"],
                independent_start=dict(attempt_id=f"attempt-{numbered:04}", replica_epochs=dict(A=1, B=1),
                                       started_at_ns=0, reset_receipt_id=f"{run_id}/diag-0001"),
                healthy_intervals=[[0, block["anchor_at_ns"] + inputs.injection_offsets[0]]],
                device_capability="unavailable",
            )
            manifest["runs"].append(record)
            if cleanup_blocked or block["kind"] == "live":
                record.update(run_status="blocked", reason="previous_cleanup_unconfirmed" if cleanup_blocked else "live_out_of_scope")
                record.pop("independent_start")
                continue
            failure = failures.get((block["pair_id"], branch["role"]))
            cleanup_blocked = not attempt(record, failure)
            if (failure is not None and failure.diagnostic_retry and record["run_status"] == "failed" and not cleanup_blocked):
                retry_id = f"{run_id}-retry-1"
                retry = {**record, "run_id": retry_id, "retry_of_run_id": run_id}
                retry["independent_start"] = dict(record["independent_start"], attempt_id=f"attempt-{numbered:04}-retry-1",
                                                  reset_receipt_id=f"{retry_id}/diag-0001")
                retry.pop("reason", None)
                cleanup_blocked = not attempt(retry, None)
                retry_records.append(retry)
    attempts_path = out_dir / ArtifactKind.ATTEMPTS
    write_artifact(attempts_path, ArtifactKind.ATTEMPTS,
                   dict(schema_version=2, source_kind="synthetic", diagnostic_retries=retry_records))
    manifest.update(attempts_path=attempts_path.name, attempts_sha256=_digest(attempts_path.read_bytes()))
    write_artifact(out_dir / ArtifactKind.MANIFEST, ArtifactKind.MANIFEST, manifest)
    total = sum(record["run_status"] == "complete" for record in manifest["runs"])
    return RunResult(manifest, cleanup_blocked, total)
