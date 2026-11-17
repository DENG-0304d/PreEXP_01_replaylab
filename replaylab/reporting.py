"""Deterministic artifact ordering and execution reports."""

from collections.abc import Iterable, Mapping
from pathlib import Path

from .artifacts import json_text, sort_fields, write_artifact
from .contracts import ArtifactKind, FormatError, WINDOW_KEYS


_METRICS_PREFIX = ("study_id", "source_kind", "study_split", "condition_id", "pair_id", "anchor_id", "row_type")
_METRICS_TAIL = ("planned_delay_ns", "run_id", "left_run_id", "right_run_id")
_TIMELINE_ONLINE = ("gw_at_ns", "gw_seq", "origin", "signal_group", "audit_seq")
_TIMELINE_OFFLINE = ("source_ref", "evidence_event_ids", "origin", "audit_seq")


def _metrics_key(row: Mapping[str, object]) -> tuple:
    role_or_contrast = row.get("role") or row.get("contrast")
    return (sort_fields(row, _METRICS_PREFIX), sort_fields({"role": role_or_contrast}, ("role",)),
            sort_fields(row, _METRICS_TAIL))


def _timeline_key(row: Mapping[str, object]) -> tuple:
    online = (
        row.get("gw_at_ns") is not None and row.get("gw_seq") is not None
        and row.get("origin") not in {"diagnostic", "diagnostics"}
    )
    keys = _TIMELINE_ONLINE if online else _TIMELINE_OFFLINE
    return sort_fields(row, ("run_id",)), not online, sort_fields(row, keys)


_REPORT_ORDER = {
    ArtifactKind.METRICS: _metrics_key,
    ArtifactKind.WINDOWS: lambda row: sort_fields(row, WINDOW_KEYS),
    ArtifactKind.TIMELINE: _timeline_key,
}


def write_report(out_dir: Path, name: str, rows: Iterable[Mapping[str, object]]) -> Path:
    kind = ArtifactKind(name)
    if kind not in _REPORT_ORDER:
        raise ValueError(f"unsupported row report: {name}")
    key = _REPORT_ORDER[kind]
    path = out_dir / kind.value
    write_artifact(path, kind, sorted(rows, key=lambda row: (key(row), json_text(row))))
    return path


def write_attempt_reports(
    out_dir: Path, *, command: str | None, exit_code: int, execution_status: str,
    reason: str | None = None, error: FormatError | None = None,
    missing_fields: Iterable[str] = (), summary_fields: Mapping[str, object] | None = None,
    scope: str = "contracts",
    research_status: str | None = None, research_reason: str | None = "not_evaluated",
) -> None:
    report = {
        "command": command, "exit_code": exit_code, "execution_status": execution_status,
        "scope": scope, "reason": reason,
    }
    if error is not None:
        report["error"] = error.as_dict()
    fields = sorted(missing_fields)
    if fields:
        report["missing_fields"] = fields
    write_artifact(out_dir / ArtifactKind.STATUS, ArtifactKind.STATUS, report)
    summary = {
        **(summary_fields or {}), **report, "research_status": research_status, "research_reason": research_reason,
    }
    write_artifact(out_dir / ArtifactKind.SUMMARY, ArtifactKind.SUMMARY, summary)
