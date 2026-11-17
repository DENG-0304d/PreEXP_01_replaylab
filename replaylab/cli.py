"""Entry points for the four replaylab commands."""

import argparse
from collections.abc import Sequence
from pathlib import Path
import sys

from .artifacts import load_manifest_bundle, read_artifact, write_artifact
from .contracts import ArtifactKind, FormatError, LocatedDocument, is_hardware_study
from .diversion import DiversionConfig
from .pairing import analyze_pairs
from .progress import ProgressBlocked
from .preflight import BLOCKED_UNTIL_HARDWARE, evaluate_preflight, read_mock_probe
from .reporting import write_attempt_reports, write_report
from .runner import SyntheticInputs, run_study
from .shadow import ShadowConfig, write_shadow_reports
from .statistics import StatisticsConfig, summarize_effects
from .windows import COMPARISONS, analyze_windows, build_action_index, read_shadow_windows


_COMMANDS = {
    "preflight": (),
    "run": ("--plan",),
    "shadow": ("--manifest",),
    "paired": ("--manifest",),
}
_ARGV = Path("<argv>")


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise FormatError(_ARGV, 1, "arguments", message)


def _parser() -> _Parser:
    parser = _Parser(prog="python -m replaylab", allow_abbrev=False)
    commands = parser.add_subparsers(dest="command", required=True)
    for name, inputs in _COMMANDS.items():
        command = commands.add_parser(name, allow_abbrev=False)
        command.add_argument("--out", type=Path, required=True)
        command.add_argument("--config", type=Path, required=True)
        for flag in inputs:
            command.add_argument(flag, type=Path, required=True)
        if name == "preflight":
            command.add_argument("--mock-platform", type=Path)
        if name == "paired":
            command.add_argument("--shadow", type=Path)
            command.add_argument("--shadow-comparison", choices=COMPARISONS, default="same_threshold_or")
        if name == "shadow":
            command.add_argument("--comparison", choices=COMPARISONS, default="same_threshold_or")
    return parser


def _create_output(args: Sequence[str]) -> Path | None:
    """Locate an unambiguous --out even if another argument fails parsing."""
    paths = []
    for index, argument in enumerate(args):
        if argument == "--":
            break
        if argument == "--out" or argument.startswith("--out="):
            if argument == "--out":
                value = args[index + 1] if index + 1 < len(args) else ""
                if value.startswith("-"):
                    value = ""
            else:
                value = argument.partition("=")[2]
            if not value:
                raise FormatError(_ARGV, 1, "--out", "expected an output directory")
            paths.append(Path(value))
    if not paths:
        return None
    if len(paths) != 1:
        raise FormatError(_ARGV, 1, "--out", "provide one output directory")
    path = paths[0]
    try:
        path.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise FormatError(path, 1, "--out", "output directory must be new") from exc
    except (OSError, ValueError) as exc:
        raise FormatError(path, 1, "--out", f"cannot create output directory: {exc}") from exc
    return path


def _execute_replay_command(args: argparse.Namespace) -> dict:
    config = read_artifact(args.config, ArtifactKind.STUDY)
    plan = read_artifact(args.plan, ArtifactKind.PLAN) if args.command == "run" else None
    mock_path = getattr(args, "mock_platform", None)
    if is_hardware_study(config.data) or mock_path is not None:
        probe = read_mock_probe(mock_path) if mock_path is not None else None
        preflight = evaluate_preflight(config, command=args.command, probe=probe)
        detail = preflight.as_dict()
        write_artifact(args.out / ArtifactKind.PREFLIGHT, ArtifactKind.PREFLIGHT, detail)
        if probe is not None:
            return dict(exit_code=0, execution_status="synthetic_preflight_complete", scope="m1a",
                        summary_fields=detail, research_reason="synthetic_only")
        return dict(exit_code=3, execution_status="blocked", reason=BLOCKED_UNTIL_HARDWARE,
                    missing_fields=preflight.missing_fields, scope="m1a", summary_fields=detail)
    try:
        if plan is None:
            SyntheticInputs.from_study(config)
            ShadowConfig.from_study(config)
            detail = dict(source_kind="synthetic", backend_status="cpu_mock", device_capability="unavailable",
                          action_replay_status="synthetic_only", inflight_timing_status="synthetic_only",
                          checks=[dict(name="synthetic_inputs", status="pass", source_kind="synthetic")])
            write_artifact(args.out / ArtifactKind.PREFLIGHT, ArtifactKind.PREFLIGHT, dict(schema_version=2, **detail))
            return dict(exit_code=0, execution_status="synthetic_preflight_complete", scope="m0",
                        summary_fields=detail, research_reason="synthetic_only")
        result = run_study(config, plan, out_dir=args.out)
    except ProgressBlocked as exc:
        return dict(exit_code=3, execution_status="blocked", reason="synthetic_config_missing", scope="m0",
                    missing_fields=exc.missing_fields)
    reasons = [row for row in result.manifest["runs"] if row["run_status"] != "complete"]
    summary = dict(source_kind="synthetic", complete_run_n=result.complete_n,
                   planned_run_n=len(result.manifest["runs"]), attempts_without_complete_results=reasons)
    if result.cleanup_blocked or result.complete_n == 0:
        reason = "cleanup_unconfirmed" if result.cleanup_blocked else "no_importable_runs"
        return dict(exit_code=3, execution_status="blocked", reason=reason, scope="m0",
                    summary_fields=summary, research_reason="synthetic_only")
    return dict(exit_code=0, execution_status="synthetic_run_complete", scope="m0",
                summary_fields=summary, research_reason="synthetic_only")


def _check_inputs(args: argparse.Namespace) -> dict:
    if args.command in {"preflight", "run"}:
        return _execute_replay_command(args)

    bundle = load_manifest_bundle(args.manifest, args.config)
    shadow = getattr(args, "shadow", None)
    count = len(bundle.runs)
    summary = {
        "imported_run_count": count,
        "imported_run_ids": sorted(run.record["run_id"] for run in bundle.runs),
        "excluded_runs": list(bundle.excluded_runs),
        "planned_branches": [
            {**{key: value for key, value in block.items() if key != "branches"}, **branch}
            for block in bundle.plan.data["blocks"] for branch in block["branches"]
        ],
    }
    if args.command == "paired":
        study = read_artifact(args.config, ArtifactKind.STUDY,
                              expected_sha256=bundle.manifest.data["config_sha256"],
                              digest_location=bundle.manifest.location(("config_sha256",)))
        try:
            pairing = analyze_pairs(bundle, study)
        except ProgressBlocked as exc:
            return dict(exit_code=3, execution_status="blocked", reason="missing_pairing_config",
                        scope="pairing", missing_fields=exc.missing_fields, summary_fields=summary)
        summary.update(pairing.summary())
        statistics = summarize_effects(pairing, StatisticsConfig.from_study(study))
        summary.update(statistics.summary())
        window_report = None
        if shadow is not None:
            shadow_doc = read_artifact(shadow, ArtifactKind.WINDOWS)
            timeline_path = shadow.with_name(ArtifactKind.TIMELINE.value)
            timeline = None
            if timeline_path.exists():
                try:
                    timeline = read_artifact(timeline_path, ArtifactKind.TIMELINE)
                except FormatError as exc:
                    summary["shadow_timeline_error"] = exc.as_dict()
                    timeline = LocatedDocument(timeline_path, [], {}, row_oriented=True)
            try:
                diversion = DiversionConfig.from_study(study)
            except ProgressBlocked as exc:
                summary["shadow_evidence_missing_fields"] = exc.missing_fields
                diversion = None
            source = study.data.get("source_kind") or ("real" if is_hardware_study(study.data) else "synthetic")
            inputs = read_shadow_windows(
                shadow_doc, bundle=bundle, timeline=timeline, comparison=args.shadow_comparison,
                observation_end_ns=study.data["replay"]["observation_end_ns"],
                diversion=diversion, expected_source=source,
            )
            window_report = analyze_windows(inputs, index=build_action_index(pairing, statistics, bundle=bundle))
            summary["shadow_comparison"] = args.shadow_comparison
            summary["shadow_windows"] = window_report.summary()
        try:
            write_report(args.out, ArtifactKind.METRICS.value, pairing.metrics_rows())
            if window_report is not None:
                write_report(args.out, ArtifactKind.WINDOWS.value, [row.csv_row() for row in window_report.windows])
        except OSError as exc:
            raise FormatError(Path(exc.filename or args.out), 1, "$", f"cannot write report: {exc}") from exc
        research = dict(research_status=statistics.research_status, research_reason=statistics.research_reason)
        if count == 0:
            return dict(exit_code=3, execution_status="blocked", reason="no_importable_runs",
                        scope="pairing", summary_fields=summary, **research)
        if summary["valid_contrast_n"] == 0:
            return dict(exit_code=4, execution_status="no_valid_contrast", reason="no_valid_pairs",
                        scope="pairing", summary_fields=summary, **research)
        return dict(exit_code=0, execution_status="pairing_complete", scope="pairing", summary_fields=summary, **research)
    study = read_artifact(args.config, ArtifactKind.STUDY, expected_sha256=bundle.manifest.data["config_sha256"],
                          digest_location=bundle.manifest.location(("config_sha256",)))
    try:
        summary.update(shadow=write_shadow_reports(bundle, study, out_dir=args.out, comparison=args.comparison))
    except ProgressBlocked as exc:
        return dict(exit_code=3, execution_status="blocked", reason="shadow_config_missing",
                    scope="shadow", missing_fields=exc.missing_fields, summary_fields=summary)
    if count == 0:
        return dict(exit_code=3, execution_status="blocked", reason="no_importable_runs",
                    scope="shadow", summary_fields=summary)
    return dict(exit_code=0, execution_status="shadow_complete", scope="shadow", summary_fields=summary,
                research_reason="shadow_timing_only")


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    command = args[0] if args and args[0] in _COMMANDS else None
    parse_error = None
    try:
        parsed = _parser().parse_args(args)
    except SystemExit as exc:
        return exc.code
    except FormatError as exc:
        parse_error = exc
    out_dir = None
    try:
        out_dir = _create_output(args)
        if parse_error is not None:
            raise parse_error
        command = parsed.command
        result = _check_inputs(parsed)
    except FormatError as exc:
        print(exc, file=sys.stderr)
        result = dict(exit_code=2, execution_status="invalid_input", reason=exc.reason, error=exc)
    except OSError as exc:
        error = FormatError(Path(exc.filename or args[0]), 1, "$", f"cannot read or write an artifact: {exc}")
        print(error, file=sys.stderr)
        result = dict(exit_code=2, execution_status="io_error", reason=error.reason, error=error)
    if out_dir is not None:
        try:
            write_attempt_reports(out_dir, command=command, **result)
        except OSError as exc:
            print(FormatError(out_dir, 1, "--out", f"cannot write attempt reports: {exc}"), file=sys.stderr)
            return 2
    return result["exit_code"]
