import json
import unittest

from replaylab.artifacts import ImportedBundle, write_artifact
from replaylab.contracts import ArtifactKind
from test_artifacts import digest, make_bundle
from test_contracts import plan, put_json, study
from test_scoring import arrival, dispatch, events, located, outcome, run_fixture


def paired_runs():
    result = []
    requests = [arrival(f"q{i:02}") for i in range(50)]
    for number, (role, delay, goodput, redirects) in enumerate([
        ("reference", None, 40, 0), ("immediate", 0, 50, 3), ("delayed", 4, 42, 1),
    ], 1):
        run_id = f"case-{role}"
        outcomes = [outcome(run_id, a["request_id"], status="ok" if i < goodput else "timeout")
                    for i, a in enumerate(requests)]
        run = run_fixture(run_id, role=role, delay=delay, arrivals=requests, outcomes=outcomes, started=number)
        specs = [(row["gw_at_ns"], row["type"], row["payload"]) for row in run.artifacts[ArtifactKind.EVENTS].data]
        for index in range(redirects):
            key = requests[index]["request_id"]
            at = 10 + delay + index
            specs.append((at, "dispatch_decision", dict(request_id=key, default_target="A",
                actual_target="B", eligible=True, selected=True, b_gate="pass", reason="redirected",
                rule_active=True, route_state_refs={}, cache_hint_tokens=0)))
            specs.append((at, *dispatch("B", key)))
        run.artifacts[ArtifactKind.EVENTS] = located("events.jsonl", events(run_id, *specs))
        result.append(run)
    return result


def paired_package(root, roles=("reference", "immediate")):
    config, plan_path, manifest = make_bundle(root, roles=roles)
    index = json.loads(manifest.read_bytes())
    for number, record in enumerate(index["runs"], 1):
        run_id, role = record["run_id"], record["role"]
        specs = [(1, "arrival", dict(request_id="q", client_sent_ns=1)), (100, "tick", {})]
        if role != "reference":
            specs.append((10 + record["planned_delay_ns"], "rule_armed",
                          dict(rule_id="fixture", activation_id="a", reason="planned")))
        path = manifest.parent / run_id / "events.jsonl"
        write_artifact(path, ArtifactKind.EVENTS, events(run_id, *specs))
        record["files"]["events.jsonl"]["sha256"] = digest(path)
        record["independent_start"]["started_at_ns"] = number
    put_json(manifest, index)
    return config, plan_path, manifest


def bundle_fixture(runs, *, plan_data=None, excluded=()):
    return ImportedBundle(located("manifest.json", {"study_id": "s"}),
                          located("plan.json", plan_data or plan()), tuple(runs), tuple(excluded))


class PairingTests(unittest.TestCase):
    def analyze(self, runs, *, data=None, plan_data=None, excluded=()):
        from replaylab.pairing import analyze_pairs

        return analyze_pairs(bundle_fixture(runs, plan_data=plan_data, excluded=excluded),
                             located("study.json", data or study()))

    def test_t07_three_deltas_one_action_block_and_separate_aa(self):
        data = plan()
        data["blocks"].append(dict(
            condition_id="example", pair_id="aa", anchor_id="a", anchor_at_ns=10, study_split="discovery",
            kind="aa", branches=[{"role": role, "delay_ns": None} for role in ("aa_first", "aa_second")],
        ))
        runs = paired_runs()
        runs.extend([run_fixture("aa-1", role="aa_first", pair="aa"),
                     run_fixture("aa-2", role="aa_second", pair="aa",
                                 outcomes=[outcome("aa-2", status="timeout")])])
        result = self.analyze(runs, plan_data=data)
        self.assertEqual([run.score.goodput for run in result.runs if run.record["pair_id"] == "p"], [40, 50, 42])
        self.assertEqual([run.score.redirected_n for run in result.runs if run.record["pair_id"] == "p"], [0, 3, 1])
        action = {row.planned.contrast: row.delta.goodput_delta for row in result.contrasts if row.planned.kind == "action"}
        self.assertEqual(action, {"immediate-reference": 10, "delayed-reference": 2, "immediate-delayed": 8})
        self.assertEqual(result.summary()["independent_action_block_n"], 1)
        self.assertEqual(result.summary()["research_sample_n"], 0)
        self.assertEqual(result.summary()["aa"]["deltas"], [-1])

    def test_t10_reports_each_mismatch_and_isolates_the_bad_delayed(self):
        changes = [
            ("arrival_hash_mismatch", lambda run: run.record["files"]["arrivals.csv"].update(sha256="d" * 64)),
            ("b_cohort_mismatch", lambda run: run.artifacts[ArtifactKind.ANCHOR_STATE].data.update(b_existing_request_ids=["q00"])),
            ("slo_id_mismatch", lambda run: run.record.update(slo_id="other")),
            ("source_kind_mismatch", lambda run: run.record.update(source_kind="real")),
            ("rule_id_mismatch", lambda run: run.record.update(diversion_rule_id="other")),
            ("intervention_digest_mismatch", lambda run: run.record.update(intervention_digest="d" * 64)),
            ("prehistory_state_mismatch", lambda run: run.artifacts[ArtifactKind.ANCHOR_STATE].data["B"].update(unfinished_requests=1)),
            ("evaluation_outcome_unknown", lambda run: run.artifacts[ArtifactKind.OUTCOMES].data.pop()),
        ]
        for reason, modify in changes:
            with self.subTest(reason=reason):
                runs = paired_runs()
                modify(runs[2])
                result = self.analyze(runs)
                self.assertEqual([p.planned.contrast for p in result.contrasts if p.delta is not None],
                                 ["immediate-reference"])
                bad = [p for p in result.contrasts if p.delta is None]
                self.assertEqual(len(bad), 2)
                self.assertTrue(all(reason in {issue.code for issue in p.issues} for p in bad))

    def test_t10_relabelled_run_cannot_become_a_second_aa_attempt(self):
        data = plan()
        data["blocks"][0].update(kind="aa", branches=[{"role": role, "delay_ns": None}
                                                       for role in ("aa_first", "aa_second")])
        runs = [run_fixture("r1", role="aa_first"), run_fixture("new-id", role="aa_second")]
        for row in runs[1].artifacts[ArtifactKind.EVENTS].data:
            row["event_id"] = f"new-{row['gw_seq']}"
        result = self.analyze(runs, plan_data=data)
        [contrast] = result.contrasts
        self.assertIsNone(contrast.delta)
        self.assertIn("relabelled_run", {item.code for item in contrast.issues})

    def test_t10_two_complete_attempts_require_frozen_selection(self):
        runs = paired_runs()
        runs.append(run_fixture("extra", role="immediate", delay=0, started=9))
        ambiguous = self.analyze(runs)
        bad = [row for row in ambiguous.contrasts if row.planned.contrast == "immediate-reference"][0]
        self.assertIn("ambiguous_attempts", {issue.code for issue in bad.issues})
        self.assertEqual(bad.candidate_attempt_ids, ("case-immediate", "extra"))
        data = study()
        data["pairing"]["attempt_selection"] = "earliest_started_at"
        chosen = self.analyze(runs, data=data)
        row = next(row for row in chosen.contrasts if row.planned.contrast == "immediate-reference")
        self.assertEqual(row.delta.goodput_delta, 10)
        self.assertEqual(row.candidate_attempt_ids, ("case-immediate", "extra"))

    def test_client_send_jitter_and_nonzero_gateway_delay_are_distinct(self):
        baseline = paired_runs()
        delayed = baseline[2]
        receipt = delayed.artifacts[ArtifactKind.EVENTS].data[0]
        receipt["gw_at_ns"] = 2
        receipt["event_at_ns"] = 2
        delayed.artifacts[ArtifactKind.OUTCOMES].data[0]["actual_arrival_ns"] = 2
        rows = delayed.artifacts[ArtifactKind.EVENTS].data
        rows.sort(key=lambda row: row["gw_at_ns"])
        for i, row in enumerate(rows, 1):
            row["gw_seq"] = i
        result = self.analyze(baseline)
        self.assertEqual(result.summary()["valid_contrast_n"], 3)
        receipt["payload"]["client_sent_ns"] = 1
        invalid = self.analyze(baseline)
        row = next(row for row in invalid.contrasts if row.planned.contrast == "delayed-reference")
        self.assertIn("send_jitter_exceeded", {issue.code for issue in row.issues})

    def test_real_record_contract_engine_only_preserves_action_and_detects_measured_drift(self):
        # In-memory evidence for exercising the reader. No evidence is collected or exported here.
        data = study()
        data["injection"].update(start_jitter_ns=1, stop_jitter_ns=1)
        runs = paired_runs()
        for run in runs:
            run.record.update(source_kind="real", device_capability="engine_progress_only")
            bindings = {replica: dict(pod_uid=f"pod-{replica}", container_id=f"c-{replica}", worker_epoch=1,
                                      gpu_uuid=f"uuid-{replica}") for replica in ("A", "B")}
            startup = run.record["independent_start"]
            startup.update(runner_attempt_id="runner", replica_bindings=bindings)
            specs = [
                (0, "binding_snapshot", dict(replica_bindings=bindings)),
                (11, "binding_snapshot", dict(replica_bindings=bindings, phase="anchor")),
                (100, "binding_snapshot", dict(replica_bindings=bindings, coverage_start_ns=0, coverage_end_ns=100)),
                (0, "lease_result", {}), (0, "cleanup_check", {}), (100, "cleanup_check", {}),
            ]
            for edge, at in [("start", 10), ("stop", 18)]:
                specs.append((at, f"injector_{edge}", dict(actual_at_ns=at, level="low", scope="A",
                                                           implementation_id="fixture", condition_id="example")))
            rows = [dict(run_id=run.record["run_id"], attempt_id=startup["attempt_id"],
                         runner_attempt_id="runner", observed_at_ns=at, kind=kind, status="pass",
                         payload=payload) for at, kind, payload in specs]
            run.artifacts[ArtifactKind.PLATFORM_EVIDENCE] = located("platform_evidence.jsonl", rows)
        result = self.analyze(runs, data=data)
        self.assertEqual(result.summary()["valid_contrast_n"], 3)
        records = runs[2].artifacts[ArtifactKind.PLATFORM_EVIDENCE].data
        coverage = records[2]["payload"]
        del coverage["coverage_end_ns"]
        missing = self.analyze(runs, data=data)
        self.assertEqual(missing.summary()["valid_contrast_n"], 1)
        row = next(row for row in missing.contrasts if row.planned.contrast == "delayed-reference")
        self.assertIn("platform_coverage_unverified", {issue.code for issue in row.issues})
        coverage["coverage_end_ns"] = 100
        records[-1]["payload"].update(actual_at_ns=20, level="other")
        records.append(dict(records[0], kind="endpoint_change", observed_at_ns=50))
        result = self.analyze(runs, data=data)
        self.assertEqual(result.summary()["valid_contrast_n"], 1)
        failures = next(row for row in result.contrasts if row.planned.contrast == "delayed-reference")
        self.assertTrue({"platform_continuity_broken", "injection_condition_mismatch", "injection_jitter_exceeded"}
                        <= {issue.code for issue in failures.issues})
