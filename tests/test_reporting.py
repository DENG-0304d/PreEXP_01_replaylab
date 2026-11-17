import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class ReportingTests(unittest.TestCase):
    def test_writer_rejects_boolean_time_before_creating_csv(self):
        from replaylab.artifacts import write_artifact
        from replaylab.contracts import ArtifactKind, FormatError

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "arrivals.csv"
            row = dict(request_id="a", scheduled_arrival_ns=True, in_eval=False,
                       ttft_deadline_ns=2, terminal_deadline_ns=3, input_id="i",
                       prompt_tokens=1, max_output_tokens=1)
            with self.assertRaises(FormatError) as caught:
                write_artifact(path, ArtifactKind.ARRIVALS, [row])
            self.assertEqual(caught.exception.field, "scheduled_arrival_ns")
            self.assertFalse(path.exists())

    def test_csv_can_round_trip_a_quoted_carriage_return_with_lf_record_ends(self):
        from replaylab.artifacts import read_artifact, write_artifact
        from replaylab.contracts import ArtifactKind

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "arrivals.csv"
            row = dict(request_id="a\rb", scheduled_arrival_ns=1, in_eval=False,
                       ttft_deadline_ns=2, terminal_deadline_ns=3, input_id="i",
                       prompt_tokens=1, max_output_tokens=1)
            write_artifact(path, ArtifactKind.ARRIVALS, [row])
            self.assertNotIn(b"\r\n", path.read_bytes())
            self.assertEqual(read_artifact(path, ArtifactKind.ARRIVALS).data, [row])

    def test_canonical_csv_and_jsonl_type_order(self):
        from replaylab.artifacts import write_artifact
        from replaylab.contracts import ArtifactKind

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            arrivals = [
                dict(request_id=rid, scheduled_arrival_ns=t, in_eval=True, ttft_deadline_ns=20,
                     terminal_deadline_ns=21, input_id="i", prompt_tokens=1, max_output_tokens=1)
                for rid, t in [("z", 10), ("b", 2), ("a", 2)]
            ]
            path = root / "arrivals.csv"
            write_artifact(path, ArtifactKind.ARRIVALS, arrivals)
            self.assertEqual(path.read_bytes().splitlines(), [
                b"request_id,scheduled_arrival_ns,in_eval,ttft_deadline_ns,terminal_deadline_ns,input_id,prompt_tokens,max_output_tokens",
                b"a,2,1,20,21,i,1,1", b"b,2,1,20,21,i,1,1", b"z,10,1,20,21,i,1,1",
            ])
            self.assertNotIn(b"\r", path.read_bytes())
            path = root / "events.jsonl"
            events = [
                dict(run_id=run, gw_seq=seq, gw_at_ns=30, event_at_ns=30, event_id=str(seq),
                     type="tick", payload={})
                for run, seq in [("b", 1), ("a", 10), ("a", 2)]
            ]
            write_artifact(path, ArtifactKind.EVENTS, events)
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual([(row["run_id"], row["gw_seq"]) for row in rows], [("a", 2), ("a", 10), ("b", 1)])

    def test_metrics_windows_and_timeline_sort_keys(self):
        from replaylab.artifacts import read_artifact
        from replaylab.contracts import ArtifactKind
        from replaylab.reporting import write_report

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = write_report(root, "metrics.csv", [
                dict(study_id="s", row_type="contrast", run_id="z", planned_delay_ns=None, reason="unknown"),
                dict(study_id="s", row_type="contrast", run_id="b", planned_delay_ns=10),
                dict(study_id="s", row_type="contrast", run_id="a", planned_delay_ns=2),
            ])
            with path.open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([row["planned_delay_ns"] for row in rows], ["2", "10", ""])
            self.assertEqual(rows[-1]["reason"], "unknown")
            window = dict(study_id="s", config_sha256="0" * 64, source_kind="synthetic",
                          study_split="discovery", condition_id="c", pair_id="p", anchor_id="a")
            path = write_report(root, "windows.csv", [
                dict(**window, plan_sha256="b" * 64, reason="untested"),
                dict(**window, plan_sha256="a" * 64, baseline_matched_run_ids_json=["r1", "r2"]),
            ])
            self.assertEqual(len(read_artifact(path, ArtifactKind.WINDOWS).data), 2)
            with path.open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([row["plan_sha256"] for row in rows], ["a" * 64, "b" * 64])
            self.assertEqual(json.loads(rows[0]["baseline_matched_run_ids_json"]), ["r1", "r2"])
            self.assertEqual(rows[1]["baseline_goodput_delta"], "")
            path = write_report(root, "timeline.jsonl", [
                dict(run_id="r", origin="diagnostic", source_ref="z", type="last"),
                dict(run_id="r", origin="shadow", gw_at_ns=2, gw_seq=10, audit_seq=10, type="second"),
                dict(run_id="r", origin="diagnostic", source_ref="a", type="third"),
                dict(run_id="r", origin="shadow", gw_at_ns=2, gw_seq=10, audit_seq=2, type="first"),
            ])
            self.assertEqual([json.loads(line)["type"] for line in path.read_text().splitlines()],
                             ["first", "second", "third", "last"])

    def test_diagnostics_and_platform_tie_breakers(self):
        from replaylab.artifacts import write_artifact
        from replaylab.contracts import ArtifactKind

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "records.jsonl"
            write_artifact(path, ArtifactKind.DIAGNOSTICS, [
                dict(run_id="r", kind="note", record_id=key, observed_at_ns=ns)
                for key, ns in [("z", None), ("b", 2), ("a", 2)]
            ])
            self.assertEqual([json.loads(line)["record_id"] for line in path.read_text().splitlines()], ["a", "b", "z"])
            write_artifact(path, ArtifactKind.PLATFORM_EVIDENCE, [
                dict(run_id="r", attempt_id="a", runner_attempt_id="runner", kind="binding_snapshot",
                     observed_at_ns=1, status="pass", payload={}, record_seq=sequence)
                for sequence in [10, 2]
            ])
            self.assertEqual([json.loads(line)["record_seq"] for line in path.read_text().splitlines()], [2, 10])

    def test_csv_and_jsonl_bytes_ignore_hash_iteration_order(self):
        script = """
from pathlib import Path
import sys
from replaylab.artifacts import write_artifact
from replaylab.contracts import ArtifactKind
from replaylab.reporting import write_report

target = Path(sys.argv[1])
ids = {"beta", "alpha", "gamma", "delta"}
write_artifact(target / "inputs.jsonl", ArtifactKind.INPUTS, (
    dict(input_id=item, prompt="text", sampling={key: 1 for key in ids}) for item in ids
))
write_report(target, "metrics.csv", (dict(study_id="s", run_id=item, goodput=1) for item in ids))
write_report(target, "timeline.jsonl", (
    dict(run_id="r", origin="diagnostic", source_ref=item) for item in ids
))
"""
        with tempfile.TemporaryDirectory() as directory:
            outputs = []
            for seed in ("1", "23"):
                out = Path(directory) / seed
                out.mkdir()
                subprocess.run(
                    [sys.executable, "-B", "-c", script, str(out)],
                    cwd=Path(__file__).resolve().parents[1],
                    env={**os.environ, "PYTHONHASHSEED": seed}, check=True,
                )
                outputs.append({file.name: file.read_bytes() for file in out.iterdir()})
            self.assertEqual(outputs[0], outputs[1])
