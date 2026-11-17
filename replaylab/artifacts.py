"""Read and write schema v2 artifacts with source positions and raw-byte integrity checks."""

from bisect import bisect_right
from collections.abc import Iterable, Mapping
import csv
from dataclasses import dataclass
import hashlib
import io
import json
import math
from pathlib import Path
import re

from .contracts import (
    ArtifactKind, CSV_SCHEMAS, FieldLocation, FormatError, Key, LocatedDocument,
    Nullable, NUMBER, POS_INT, RUN_FILE_KINDS, UINT, validate_canonical, validate_manifest_header,
    validate_manifest_run, validate_plan, validate_study,
)


ArtifactData = Mapping[str, object] | Iterable[Mapping[str, object]]
_SPACE = re.compile(r"[ \t\r\n]*")
_LINE_BREAK = re.compile(r"\r\n|\r|\n")
_INTEGER = re.compile(r"-?[0-9]+")
_ORDER = {
    ArtifactKind.ARRIVALS: ("scheduled_arrival_ns", "request_id"),
    ArtifactKind.HEALTH_PROFILE: ("replica_id", "work_key"),
    ArtifactKind.OUTCOMES: ("run_id", "request_id"),
    ArtifactKind.INPUTS: ("input_id",),
    ArtifactKind.EVENTS: ("run_id", "gw_seq"),
    ArtifactKind.DIAGNOSTICS: ("run_id", "observed_at_ns", "kind", "record_id"),
    ArtifactKind.PLATFORM_EVIDENCE: ("run_id", "attempt_id", "observed_at_ns", "kind", "record_seq"),
}


def json_text(value: object) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False, separators=(",", ":"))


def _read_bytes(path: Path, expected_sha256: str | None, digest_location: FieldLocation | None) -> bytes:
    root = FieldLocation(path, 1, "$")
    try:
        raw = path.read_bytes()
    except (OSError, ValueError) as exc:
        raise root.error(f"cannot read file: {exc}") from exc
    if expected_sha256 is not None:
        actual = hashlib.sha256(raw).hexdigest()
        if actual != expected_sha256.lower():
            location = digest_location or FieldLocation(path, 1, "sha256")
            raise location.error(f"SHA-256 mismatch for {path}: expected {expected_sha256}, got {actual}")
    return raw


def _read_text(path: Path, expected_sha256: str | None, digest_location: FieldLocation | None) -> str:
    raw = _read_bytes(path, expected_sha256, digest_location)
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        line = len(re.findall(rb"\r\n|\r|\n", raw[:exc.start])) + 1
        raise FormatError(path, line, "$", "invalid UTF-8") from exc


def _decode_json(
    text: str, path: Path, lines: dict[Key, int], root: Key = (), *, first_line: int = 1
) -> object:
    doc = LocatedDocument(path, None, lines, row_oriented=bool(root))
    lines[root] = first_line
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise FormatError(path, first_line + exc.lineno - 1, "$", f"invalid JSON: {exc.msg}") from exc
    except (ValueError, RecursionError) as exc:
        raise doc.error(root, f"invalid JSON: {exc}") from exc

    breaks = [match.start() for match in _LINE_BREAK.finditer(text)]
    raw_decode = json.JSONDecoder().raw_decode

    def skip(offset: int) -> int:
        return _SPACE.match(text, offset).end()

    def scan(offset: int, key: Key, *, field_offset: int | None = None) -> int:
        offset = skip(offset)
        position = offset if field_offset is None else field_offset
        lines[key] = first_line + bisect_right(breaks, position)
        if text[offset] == "{":
            offset = skip(offset + 1)
            keys = set()
            while text[offset] != "}":
                field_start = offset
                name, end = raw_decode(text, offset)
                child = (*key, name)
                if name in keys:
                    lines[child] = first_line + bisect_right(breaks, field_start)
                    raise doc.error(child, "duplicate JSON field")
                keys.add(name)
                offset = skip(scan(skip(end) + 1, child, field_offset=field_start))
                if text[offset] != ",":
                    break
                offset = skip(offset + 1)
            return offset + 1
        if text[offset] == "[":
            offset = skip(offset + 1)
            index = 0
            while text[offset] != "]":
                offset = skip(scan(offset, (*key, index)))
                index += 1
                if text[offset] != ",":
                    break
                offset = skip(offset + 1)
            return offset + 1
        scalar, end = raw_decode(text, offset)
        if type(scalar) is float and not math.isfinite(scalar):
            raise doc.error(key, "JSON number must be finite")
        return end

    try:
        scan(0, root)
    except RecursionError as exc:
        raise doc.error(root, "JSON nesting is too deep") from exc
    return value


def _read_csv_cell(kind: ArtifactKind, field: str, raw: str, location: FieldLocation) -> object:
    spec = CSV_SCHEMAS[kind].get(field)
    if isinstance(spec, Nullable):
        if not raw:
            return None
        spec = spec.spec
    if spec is bool:
        if raw not in {"0", "1"}:
            raise location.error("expected CSV boolean 0 or 1")
        return raw == "1"
    if spec is int or spec is UINT or spec is POS_INT or field.endswith("_ns"):
        if _INTEGER.fullmatch(raw):
            try:
                return int(raw)
            except ValueError as exc:
                raise location.error("integer is too large to parse") from exc
        raise location.error("expected an integer")
    if isinstance(spec, list) or spec is list or spec is NUMBER:
        try:
            value = json.loads(raw)
        except (ValueError, RecursionError) as exc:
            raise location.error("invalid JSON in CSV cell") from exc
        if value is None:
            raise location.error("use an empty CSV cell for an unknown value")
        return value
    return raw


def _read_csv(path: Path, kind: ArtifactKind, text: str) -> LocatedDocument:
    reader = csv.reader(io.StringIO(text, newline=""), strict=True)
    lines: dict[Key, int] = {}
    rows = []
    start = 1
    try:
        header = next(reader, None)
        if not header:
            raise FormatError(path, 1, "$header", "CSV header is required")
        seen = set()
        for field in header:
            if not field or field in seen:
                raise FormatError(path, 1, field or "$header", "empty or duplicate CSV column")
            seen.add(field)
        for field in CSV_SCHEMAS[kind]:
            if field not in seen:
                raise FormatError(path, 1, field, "required CSV column is missing")
        while True:
            start = reader.line_num + 1
            cells = next(reader, None)
            if cells is None:
                break
            if len(cells) != len(header):
                field = header[len(cells)] if len(cells) < len(header) else "$row"
                raise FormatError(path, start, field, f"expected {len(header)} CSV cells, got {len(cells)}")
            lines[(len(rows),)] = start
            rows.append({field: _read_csv_cell(kind, field, raw, FieldLocation(path, start, field))
                         for field, raw in zip(header, cells)})
    except csv.Error as exc:
        raise FormatError(path, start, "$row", f"invalid CSV: {exc}") from exc
    return LocatedDocument(path, rows, lines, row_oriented=True)


def _read_document(
    path: Path, kind: ArtifactKind, expected_sha256: str | None = None,
    digest_location: FieldLocation | None = None,
) -> LocatedDocument:
    text = _read_text(path, expected_sha256, digest_location)
    if kind in CSV_SCHEMAS:
        return _read_csv(path, kind, text)
    lines: dict[Key, int] = {}
    if kind.value.endswith(".jsonl"):
        rows = []
        for line_number, line in enumerate(io.StringIO(text, newline=""), 1):
            if not line.strip():
                raise FormatError(path, line_number, "$", "blank JSONL line; expected one object")
            rows.append(_decode_json(line, path, lines, (len(rows),), first_line=line_number))
        return LocatedDocument(path, rows, lines, row_oriented=True)
    value = _decode_json(text, path, lines)
    return LocatedDocument(path, value, lines)


def read_artifact(
    path: Path, kind: ArtifactKind, *, expected_sha256: str | None = None,
    digest_location: FieldLocation | None = None,
) -> LocatedDocument:
    document = _read_document(path, kind, expected_sha256, digest_location)
    _validate_document(kind, document)
    return document


def _validate_document(kind: ArtifactKind, document: LocatedDocument) -> None:
    if kind == ArtifactKind.STUDY:
        validate_study(document)
    elif kind == ArtifactKind.PLAN:
        validate_plan(document)
    else:
        validate_canonical(kind, document)


def resolve_manifest_path(manifest_path: Path, declared_path: str) -> Path:
    return manifest_path.parent / declared_path


@dataclass(frozen=True)
class ImportedRun:
    record: dict
    artifacts: Mapping[ArtifactKind, LocatedDocument]


@dataclass(frozen=True)
class ImportedBundle:
    manifest: LocatedDocument
    plan: LocatedDocument
    runs: tuple[ImportedRun, ...]
    excluded_runs: tuple[dict, ...]


_RUN_ARTIFACTS = {kind.value: kind for kind in (*RUN_FILE_KINDS, ArtifactKind.PLATFORM_EVIDENCE)}


def _load_run_files(manifest: LocatedDocument, index: int, record: dict) -> dict:
    documents = {}
    for filename, reference in sorted(record["files"].items()):
        path = resolve_manifest_path(manifest.file, reference["path"])
        digest_location = manifest.location(("runs", index, "files", filename, "sha256"))
        kind = _RUN_ARTIFACTS.get(filename)
        if kind is None:
            _read_bytes(path, reference["sha256"], digest_location)
            continue
        documents[kind] = read_artifact(
            path, kind, expected_sha256=reference["sha256"],
            digest_location=digest_location,
        )
    for kind in (ArtifactKind.OUTCOMES, ArtifactKind.EVENTS, ArtifactKind.DIAGNOSTICS, ArtifactKind.PLATFORM_EVIDENCE):
        if kind not in documents:
            continue
        document = documents[kind]
        for row_index, row in enumerate(document.data):
            if row["run_id"] != record["run_id"]:
                raise document.error((row_index, "run_id"), "does not match manifest run_id")
            if kind == ArtifactKind.PLATFORM_EVIDENCE:
                start = record["independent_start"]
                for key in ("attempt_id", "runner_attempt_id"):
                    if key in start and row[key] != start[key]:
                        raise document.error((row_index, key), f"does not match manifest {key}")
    return documents


def load_manifest_bundle(manifest_path: Path, config_path: Path) -> ImportedBundle:
    manifest = _read_document(manifest_path, ArtifactKind.MANIFEST)
    validate_manifest_header(manifest)
    data = manifest.data
    config = read_artifact(config_path, ArtifactKind.STUDY, expected_sha256=data["config_sha256"],
                           digest_location=manifest.location(("config_sha256",)))
    plan_path = resolve_manifest_path(manifest_path, data["plan_path"])
    plan = read_artifact(plan_path, ArtifactKind.PLAN, expected_sha256=data["plan_sha256"],
                         digest_location=manifest.location(("plan_sha256",)))
    kubernetes = "kubernetes" in (config.data["environment"] or {})
    runs, excluded = [], []
    for index, record in enumerate(data["runs"]):
        error = None
        try:
            validate_manifest_run(manifest, index, kubernetes=kubernetes)
            if record["run_status"] == "complete":
                runs.append(ImportedRun(record, _load_run_files(manifest, index, record)))
                continue
            reason = record["reason"]
        except FormatError as exc:
            error = exc.as_dict()
            reason = exc.reason
        fields = record if isinstance(record, dict) else {}
        excluded.append({
            "run_id": fields.get("run_id"), "reason": reason, "error": error,
            "branch": {key: fields.get(key) for key in (
                "condition_id", "pair_id", "anchor_id", "study_split", "role", "planned_delay_ns"
            )},
        })
    return ImportedBundle(manifest, plan, tuple(runs), tuple(excluded))


def sort_fields(row: Mapping[str, object], names: Iterable[str]) -> tuple:
    def typed(value: object) -> tuple:
        if type(value) in {int, float}:
            return 0, value
        if type(value) is str:
            return 1, value
        if value is None:
            return 3, ""
        return 2, json_text(value)

    return tuple(typed(row.get(name)) for name in names)


def _csv_value(value: object) -> object:
    if type(value) is bool:
        return int(value)
    if type(value) in {dict, list, tuple, float}:
        return json_text(value)
    return value


def write_artifact(path: Path, kind: ArtifactKind, data: ArtifactData) -> None:
    row_oriented = kind in CSV_SCHEMAS or kind.value.endswith(".jsonl")
    lines = {}
    if row_oriented:
        data = [dict(row) for row in data]
        if kind in {ArtifactKind.METRICS, ArtifactKind.WINDOWS}:
            data = [{column: row.get(column) for column in CSV_SCHEMAS[kind]} for row in data]
        if kind in _ORDER:
            data.sort(key=lambda row: (sort_fields(row, _ORDER[kind]), json_text(row)))
        first_line = 2 if kind in CSV_SCHEMAS else 1
        lines = {(index,): index + first_line for index in range(len(data))}
    else:
        data = dict(data)
    _validate_document(kind, LocatedDocument(path, data, lines, row_oriented=row_oriented))
    if kind in CSV_SCHEMAS:
        fields = CSV_SCHEMAS[kind]
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, lineterminator="\n", quoting=csv.QUOTE_MINIMAL)
            writer.writerow(fields)
            writer.writerows([_csv_value(row.get(field)) for field in fields] for row in data)
    elif kind.value.endswith(".jsonl"):
        with path.open("w", encoding="utf-8", newline="\n") as stream:
            for row in data:
                stream.write(json_text(row) + "\n")
    else:
        path.write_text(json_text(data) + "\n", encoding="utf-8", newline="\n")
