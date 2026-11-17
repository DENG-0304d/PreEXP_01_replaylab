"""An owned, deterministic CPU queue for synthetic replay."""

from dataclasses import dataclass
from hashlib import sha256

from .artifacts import json_text
from .contracts import POS_INT, TEXT, UINT, VERSION, LocatedDocument, check, choice


@dataclass(frozen=True)
class Failure:
    pair_id: str
    role: str
    at_ns: int
    cleanup_confirmed: bool
    diagnostic_retry: bool
    reason: str = "simulated_runner_abort"


@dataclass(frozen=True)
class MockScenario:
    service_ns: dict[str, int]
    degraded_a_ns: int
    noise_seed: int
    noise_by_condition: dict[str, int]
    observations: tuple[dict, ...]
    failures: tuple[Failure, ...]

    @classmethod
    def from_document(cls, document: LocatedDocument) -> "MockScenario":
        data = document.data
        check(document, data, {
            "schema_version": VERSION, "service_ns": {"A": POS_INT, "B": POS_INT},
            "degraded_a_ns": POS_INT, "noise_seed": int, "noise_by_condition": dict,
            "observations": [dict(at_ns=UINT, event_id=TEXT, type=choice(
                "work_state", "step_start", "step_complete", "inflight_snapshot"), payload=dict)],
            "failures": [dict(pair_id=TEXT, role=TEXT, at_ns=UINT, cleanup_confirmed=bool, diagnostic_retry=bool)],
        })
        for condition, bound in data["noise_by_condition"].items():
            check(document, bound, UINT, ("noise_by_condition", condition))
        failures = []
        for index, row in enumerate(data["failures"]):
            reason = row.get("reason", "simulated_runner_abort")
            check(document, reason, choice("simulated_runner_abort", "simulated_runner_timeout"),
                  ("failures", index, "reason"))
            failures.append(Failure(*(row[name] for name in
                                      ("pair_id", "role", "at_ns", "cleanup_confirmed", "diagnostic_retry")), reason))
        return cls(dict(data["service_ns"]), data["degraded_a_ns"], data["noise_seed"],
                   dict(data["noise_by_condition"]), tuple(data["observations"]), tuple(failures))


@dataclass(frozen=True)
class MockJob:
    request_id: str
    target: str
    reservation_id: str
    prompt_tokens: int
    start_ns: int
    first_token_ns: int
    terminal_ns: int
    output_tokens: int


class MockBackend:
    def __init__(
        self, scenario: MockScenario, *, attempt_id: str, condition_id: str, injection_window: tuple[int, int],
    ) -> None:
        self.scenario = scenario
        self.owner = attempt_id
        self.condition_id = condition_id
        self.injection_window = injection_window
        self.jobs: dict[str, MockJob] = {}
        self.versions = {"A": 0, "B": 0}
        self.prefixes = frozenset()
        self.prefix_id = ""
        self.process_owner: str | None = None
        self.injection_owner: str | None = None

    def reset(self) -> None:
        if self.process_owner is not None or self.injection_owner is not None:
            raise ValueError("mock resources still have an owner")
        self.jobs.clear()
        self.versions = {"A": 0, "B": 0}
        self.prefixes = frozenset()
        self.prefix_id = ""
        self.process_owner = self.owner

    def warmup(self, inputs: list[dict]) -> None:
        self.prefixes = frozenset(row["prompt"] for row in inputs)
        self.prefix_id = sha256(json_text(sorted(self.prefixes)).encode()).hexdigest()

    def cache_hint(self, prompt: str, tokens: int) -> int:
        return tokens if prompt in self.prefixes else 0

    def start_injection(self) -> None:
        self.injection_owner = self.owner

    def stop_injection(self) -> None:
        if self.injection_owner == self.owner:
            self.injection_owner = None

    def submit(self, request: dict, *, target: str, at_ns: int, reservation_id: str) -> MockJob:
        if self.process_owner != self.owner or request["request_id"] in self.jobs:
            raise ValueError("mock submission needs an owned backend and a fresh request")
        start = max((job.terminal_ns for job in self.jobs.values() if job.target == target), default=at_ns)
        start = max(start, at_ns)
        begin, end = self.injection_window
        latency = self.scenario.service_ns[target]
        if target == "A" and begin <= start < end:
            latency = self.scenario.degraded_a_ns
        spread = self.scenario.noise_by_condition.get(self.condition_id, 0)
        if spread:
            wire = json_text([self.scenario.noise_seed, self.owner, request["request_id"], target]).encode()
            latency += int.from_bytes(sha256(wire).digest()[:8], "big") % (spread + 1)
        output_tokens = min(2, request["max_output_tokens"])
        first = start + max(1, latency // 2)
        terminal = start + latency
        job = MockJob(request["request_id"], target, reservation_id, request["prompt_tokens"],
                      start, first, terminal, output_tokens)
        self.jobs[job.request_id] = job
        return job

    def snapshot(self, replica: str, now_ns: int) -> dict:
        self.versions[replica] += 1
        jobs = [job for job in self.jobs.values() if job.target == replica and job.terminal_ns > now_ns]
        return dict(replica_id=replica, epoch=1, state_version=self.versions[replica], sampled_ns=now_ns,
                    ready=self.process_owner == self.owner,
                    pending_prefill_tokens=sum(job.prompt_tokens for job in jobs if job.start_ns > now_ns),
                    unfinished_requests=len(jobs))

    def anchor(self, anchor_id: str, now_ns: int) -> dict:
        state = dict(anchor_id=anchor_id, anchor_at_ns=now_ns, phase="before_intervention",
                     b_existing_request_ids=sorted(job.request_id for job in self.jobs.values()
                                                   if job.target == "B" and job.terminal_ns > now_ns))
        for replica in ("A", "B"):
            jobs = [job for job in self.jobs.values() if job.target == replica and job.terminal_ns > now_ns]
            queued = [job for job in jobs if job.start_ns > now_ns]
            state[replica] = dict(
                queued_ids=sorted(job.request_id for job in queued),
                active_ids=sorted(job.request_id for job in jobs if job.start_ns <= now_ns),
                warm_prefix_set_id=self.prefix_id, cached_tokens_total=sum(len(p.split()) for p in self.prefixes),
                pending_prefill_tokens=sum(job.prompt_tokens for job in queued), unfinished_requests=len(jobs),
            )
        return state

    def cleanup(self, *, confirm: bool) -> bool:
        self.stop_injection()
        if self.process_owner == self.owner:
            self.jobs.clear()
            self.process_owner = None
        return confirm and self.process_owner is None and self.injection_owner is None
