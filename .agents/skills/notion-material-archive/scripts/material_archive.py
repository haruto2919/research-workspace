#!/usr/bin/env python3
"""Prepare and render Company Markdown for Notion archival.

This helper never writes to source files. It creates temporary manifests,
staged image copies, and rendered Markdown under /tmp.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse


SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".svg"}
REMOTE_SCHEMES = {"http", "https", "data", "file-upload"}
TEMP_PREFIX = "notion-material-archive-"

INLINE_IMAGE_RE = re.compile(
    r"!\[(?P<alt>[^\]]*)\]\("
    r"(?P<target><[^>\n]+>|[^\s)\n]+)"
    r"(?:\s+(?P<title>\"[^\"]*\"|'[^']*'|\([^)]*\)))?"
    r"\)"
)
HTML_IMAGE_RE = re.compile(
    r"<img\b[^>]*?\bsrc=(?P<quote>[\"'])(?P<target>.*?)(?P=quote)[^>]*>",
    re.IGNORECASE,
)
REFERENCE_IMAGE_RE = re.compile(
    r"!\[(?P<alt>[^\]]*)\]\[(?P<label>[^\]]*)\]"
)
REFERENCE_DEF_RE = re.compile(
    r"(?m)^[ \t]{0,3}\[(?P<label>[^\]]+)\]:[ \t]*"
    r"(?P<target><[^>\n]+>|\S+)"
)
H1_RE = re.compile(r"(?m)^#\s+(?P<title>.+?)\s*$")
FENCE_OPEN_RE = re.compile(r"^[ \t]{0,3}(?P<fence>`{3,}|~{3,})(?P<info>.*)$")
LEGACY_INLINE_MATH_RE = re.compile(r"\\\((?P<body>[\s\S]*?)\\\)")
LEGACY_DISPLAY_MATH_RE = re.compile(r"\\\[(?P<body>[\s\S]*?)\\\]")
NOTION_DISPLAY_MATH_RE = re.compile(
    r"(?ms)^\$\$\s*\n(?P<body>[\s\S]*?)\n\$\$\s*$"
)
NOTION_INLINE_MATH_RE = re.compile(
    r"(?<!\$)\$(?!\$)(?P<body>[^\n$]+?)(?<!\$)\$(?!\$)"
)
NOTION_FETCH_INLINE_MATH_RE = re.compile(
    r"(?<!\$)\$`(?P<body>[^`\n]+?)`\$(?!\$)"
)
NOTION_TABLE_RE = re.compile(
    r"<table(?P<attrs>[^>]*)>(?P<body>[\s\S]*?)</table>",
    re.IGNORECASE,
)
NOTION_TABLE_ROW_RE = re.compile(
    r"<tr[^>]*>(?P<body>[\s\S]*?)</tr>",
    re.IGNORECASE,
)
NOTION_TABLE_CELL_RE = re.compile(
    r"<td[^>]*>(?P<body>[\s\S]*?)</td>",
    re.IGNORECASE,
)
TABLE_SEPARATOR_CELL_RE = re.compile(r"^:?(?P<dashes>-{3,}):?$")


class ArchiveError(RuntimeError):
    """A preflight or rendering error."""


@dataclass(frozen=True)
class ImageMatch:
    start: int
    end: int
    markup: str
    target: str
    alt: str
    kind: str


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _unique_token(content: str, kind: str, index: int) -> str:
    token = f"\ue000NOTION_ARCHIVE_{kind}_{index}\ue001"
    if token in content:
        raise ArchiveError(f"Reserved helper token occurs in source: {token!r}")
    return token


def protect_fenced_code(
    content: str,
) -> tuple[str, dict[str, str], int, int]:
    """Replace fenced code blocks with inert tokens.

    The returned mapping restores the exact original bytes after Notion-specific
    Markdown normalization. Mermaid blocks are counted separately because they
    are an important structural artifact in meeting materials.
    """

    lines = content.splitlines(keepends=True)
    output: list[str] = []
    replacements: dict[str, str] = {}
    fenced_count = 0
    mermaid_count = 0
    index = 0

    while index < len(lines):
        opening = FENCE_OPEN_RE.match(lines[index].rstrip("\r\n"))
        if opening is None:
            output.append(lines[index])
            index += 1
            continue

        fence = opening.group("fence")
        fence_char = fence[0]
        minimum_length = len(fence)
        closing_re = re.compile(
            rf"^[ \t]{{0,3}}{re.escape(fence_char)}"
            rf"{{{minimum_length},}}[ \t]*$"
        )
        block_lines = [lines[index]]
        info = opening.group("info").strip().casefold()
        index += 1
        while index < len(lines):
            block_lines.append(lines[index])
            candidate = lines[index].rstrip("\r\n")
            index += 1
            if closing_re.match(candidate):
                break

        block = "".join(block_lines)
        token = _unique_token(content, "FENCED_CODE", fenced_count)
        if block.endswith("\r\n"):
            placeholder = token + "\r\n"
        elif block.endswith("\n"):
            placeholder = token + "\n"
        elif block.endswith("\r"):
            placeholder = token + "\r"
        else:
            placeholder = token
        replacements[placeholder] = block
        output.append(placeholder)
        fenced_count += 1
        if info and info.split(maxsplit=1)[0] == "mermaid":
            mermaid_count += 1

    return "".join(output), replacements, fenced_count, mermaid_count


def protect_inline_code(
    content: str,
) -> tuple[str, dict[str, str]]:
    """Protect complete backtick code spans outside fenced code blocks."""

    output: list[str] = []
    replacements: dict[str, str] = {}
    index = 0
    token_index = 0
    while index < len(content):
        if content[index] != "`":
            output.append(content[index])
            index += 1
            continue

        run_end = index
        while run_end < len(content) and content[run_end] == "`":
            run_end += 1
        fence = content[index:run_end]
        closing = content.find(fence, run_end)
        if closing < 0:
            output.append(fence)
            index = run_end
            continue

        span_end = closing + len(fence)
        span = content[index:span_end]
        token = _unique_token(content, "INLINE_CODE", token_index)
        replacements[token] = span
        output.append(token)
        token_index += 1
        index = span_end
    return "".join(output), replacements


def restore_protected(content: str, replacements: dict[str, str]) -> str:
    for token, original in replacements.items():
        content = content.replace(token, original)
    return content


def split_pipe_row(line: str) -> list[str] | None:
    stripped = line.strip()
    if not (stripped.startswith("|") and stripped.endswith("|")):
        return None

    body = stripped[1:-1]
    cells: list[str] = []
    current: list[str] = []
    index = 0
    while index < len(body):
        character = body[index]
        if character == "\\" and index + 1 < len(body):
            current.append(character)
            current.append(body[index + 1])
            index += 2
            continue
        if character == "|":
            cells.append("".join(current).strip())
            current = []
        else:
            current.append(character)
        index += 1
    cells.append("".join(current).strip())
    return cells


def is_table_separator(cells: list[str] | None) -> bool:
    return bool(cells) and all(
        TABLE_SEPARATOR_CELL_RE.fullmatch(cell) is not None for cell in cells
    )


def canonicalize_notion_rich_text(value: str) -> str:
    return NOTION_FETCH_INLINE_MATH_RE.sub(
        lambda match: f"${match.group('body')}$",
        value.strip(),
    )


def table_content_sha256(rows: list[list[str]]) -> str:
    return text_sha256(
        "\u241e".join(
            "\u241f".join(canonicalize_notion_rich_text(cell) for cell in row)
            for row in rows
        )
    )


def normalize_table_separators(content: str) -> tuple[str, int, int]:
    """Remove unsupported alignment colons from Markdown table separators."""

    output: list[str] = []
    separator_count = 0
    aligned_separator_count = 0
    for line in content.splitlines(keepends=True):
        bare = line.rstrip("\r\n")
        newline = line[len(bare) :]
        cells = split_pipe_row(bare)
        if not is_table_separator(cells):
            output.append(line)
            continue

        assert cells is not None
        separator_count += 1
        if any(":" in cell for cell in cells):
            aligned_separator_count += 1
        indentation = bare[: len(bare) - len(bare.lstrip())]
        normalized_cells = [
            TABLE_SEPARATOR_CELL_RE.fullmatch(cell).group("dashes")
            for cell in cells
        ]
        output.append(
            f"{indentation}| "
            + " | ".join(normalized_cells)
            + f" |{newline}"
        )
    return "".join(output), separator_count, aligned_separator_count


def _pipe_table_profiles(content: str) -> list[dict[str, Any]]:
    lines = content.splitlines()
    profiles: list[dict[str, Any]] = []
    index = 0
    while index < len(lines):
        first_cells = split_pipe_row(lines[index])
        if first_cells is None:
            index += 1
            continue

        rows: list[list[str]] = []
        start = index
        while index < len(lines):
            cells = split_pipe_row(lines[index])
            if cells is None:
                break
            rows.append(cells)
            index += 1

        if len(rows) < 2 or not is_table_separator(rows[1]):
            continue
        column_count = len(rows[0])
        visible_rows = [rows[0], *rows[2:]]
        if any(len(row) != column_count for row in visible_rows):
            raise ArchiveError(
                "Markdown table has inconsistent column counts near "
                f"line {start + 1}"
            )
        profiles.append(
            {
                "start": start,
                "row_count": len(visible_rows),
                "column_count": column_count,
                "header_row": True,
                "separator_leak": False,
                "header_sha256": text_sha256("\u241f".join(rows[0])),
                "content_sha256": table_content_sha256(visible_rows),
            }
        )
    return profiles


def _notion_xml_table_profiles(content: str) -> list[dict[str, Any]]:
    profiles: list[dict[str, Any]] = []
    for match in NOTION_TABLE_RE.finditer(content):
        rows = [
            [
                canonicalize_notion_rich_text(cell.group("body"))
                for cell in NOTION_TABLE_CELL_RE.finditer(row.group("body"))
            ]
            for row in NOTION_TABLE_ROW_RE.finditer(match.group("body"))
        ]
        column_count = len(rows[0]) if rows else 0
        separator_leak = any(is_table_separator(row) for row in rows)
        profiles.append(
            {
                "start": match.start(),
                "row_count": len(rows),
                "column_count": column_count,
                "header_row": (
                    re.search(
                        r'\bheader-row\s*=\s*["\']true["\']',
                        match.group("attrs"),
                        re.IGNORECASE,
                    )
                    is not None
                ),
                "separator_leak": separator_leak,
                "header_sha256": (
                    text_sha256("\u241f".join(rows[0])) if rows else None
                ),
                "content_sha256": table_content_sha256(rows),
            }
        )
    return profiles


def notion_markdown_profile(content: str) -> dict[str, Any]:
    """Return deterministic structural expectations for Notion round trips."""

    masked, fenced, fenced_count, mermaid_count = protect_fenced_code(content)
    masked = NOTION_FETCH_INLINE_MATH_RE.sub(
        lambda match: f"${match.group('body')}$",
        masked,
    )
    masked, inline_code = protect_inline_code(masked)

    display_bodies: list[str] = []

    def hold_display(match: re.Match[str]) -> str:
        display_bodies.append(match.group("body"))
        return _unique_token(masked, "DISPLAY_PROFILE", len(display_bodies) - 1)

    without_display = NOTION_DISPLAY_MATH_RE.sub(hold_display, masked)
    inline_bodies = [
        match.group("body")
        for match in NOTION_INLINE_MATH_RE.finditer(without_display)
    ]

    table_profiles = [
        *_pipe_table_profiles(without_display),
        *_notion_xml_table_profiles(without_display),
    ]
    table_profiles.sort(key=lambda item: item["start"])
    for table in table_profiles:
        table.pop("start", None)

    return {
        "inline_math_count": len(inline_bodies),
        "inline_math_sha256": [text_sha256(body) for body in inline_bodies],
        "display_math_count": len(display_bodies),
        "display_math_sha256": [text_sha256(body) for body in display_bodies],
        "table_count": len(table_profiles),
        "tables": table_profiles,
        "fenced_code_block_count": fenced_count,
        "mermaid_code_block_count": mermaid_count,
    }


def compile_notion_markdown(content: str) -> tuple[str, dict[str, Any]]:
    """Compile common Markdown math and tables to Notion-safe Markdown.

    Source files stay untouched. Only generated content is normalized, and code
    spans/blocks are restored byte-for-byte after compilation.
    """

    masked, fenced, fenced_count, mermaid_count = protect_fenced_code(content)
    masked, inline_code = protect_inline_code(masked)

    inline_open_count = len(re.findall(r"\\\(", masked))
    inline_close_count = len(re.findall(r"\\\)", masked))
    if inline_open_count != inline_close_count:
        raise ArchiveError(
            "Unbalanced legacy inline math delimiters outside code: "
            f"{inline_open_count} opening, {inline_close_count} closing"
        )

    held_inline: dict[str, str] = {}

    def convert_inline(match: re.Match[str]) -> str:
        body = match.group("body")
        if "\n" in body or "\r" in body:
            raise ArchiveError("Legacy inline math cannot span multiple lines")
        # Nested \[...\] is sometimes used to mean visible square brackets.
        # It is invalid inside inline math, so preserve the visible brackets.
        body = body.replace(r"\[", "[").replace(r"\]", "]")
        token = _unique_token(masked, "INLINE_MATH", len(held_inline))
        held_inline[token] = f"${body}$"
        return token

    masked, inline_converted = LEGACY_INLINE_MATH_RE.subn(
        convert_inline, masked
    )
    if inline_converted != inline_open_count:
        raise ArchiveError("Could not pair all legacy inline math delimiters")

    display_open_count = len(re.findall(r"\\\[", masked))
    display_close_count = len(re.findall(r"\\\]", masked))
    if display_open_count != display_close_count:
        raise ArchiveError(
            "Unbalanced legacy display math delimiters outside code: "
            f"{display_open_count} opening, {display_close_count} closing"
        )

    def convert_display(match: re.Match[str]) -> str:
        body = match.group("body")
        if body.startswith("\r\n"):
            body = body[2:]
        elif body.startswith(("\n", "\r")):
            body = body[1:]
        if body.endswith("\r\n"):
            body = body[:-2]
        elif body.endswith(("\n", "\r")):
            body = body[:-1]
        return f"$$\n{body}\n$$"

    masked, display_converted = LEGACY_DISPLAY_MATH_RE.subn(
        convert_display, masked
    )
    if display_converted != display_open_count:
        raise ArchiveError("Could not pair all legacy display math delimiters")

    masked = restore_protected(masked, held_inline)
    masked, table_separators, aligned_separators = normalize_table_separators(
        masked
    )
    masked = restore_protected(masked, inline_code)
    compiled = restore_protected(masked, fenced)
    profile = notion_markdown_profile(compiled)
    return compiled, {
        "legacy_inline_math_converted": inline_converted,
        "legacy_display_math_converted": display_converted,
        "table_separators_normalized": table_separators,
        "aligned_table_separators_normalized": aligned_separators,
        "fenced_code_blocks_preserved": fenced_count,
        "mermaid_code_blocks_preserved": mermaid_count,
        "profile": profile,
    }


def audit_notion_round_trip(
    expected_profile: dict[str, Any],
    fetched_content: str,
) -> list[str]:
    actual = notion_markdown_profile(fetched_content)
    errors: list[str] = []
    for key in (
        "inline_math_count",
        "inline_math_sha256",
        "display_math_count",
        "display_math_sha256",
        "table_count",
        "fenced_code_block_count",
        "mermaid_code_block_count",
    ):
        if actual.get(key) != expected_profile.get(key):
            errors.append(
                f"{key} mismatch: expected {expected_profile.get(key)!r}, "
                f"got {actual.get(key)!r}"
            )

    expected_tables = expected_profile.get("tables", [])
    actual_tables = actual.get("tables", [])
    if len(expected_tables) == len(actual_tables):
        for index, (expected, observed) in enumerate(
            zip(expected_tables, actual_tables), start=1
        ):
            for key in (
                "row_count",
                "column_count",
                "header_row",
                "header_sha256",
                "content_sha256",
            ):
                if expected.get(key) != observed.get(key):
                    errors.append(
                        f"table {index} {key} mismatch: "
                        f"expected {expected.get(key)!r}, "
                        f"got {observed.get(key)!r}"
                    )
            if observed.get("separator_leak"):
                errors.append(f"table {index} contains a separator data row")
    return errors


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def normalize_target(raw_target: str) -> str:
    target = raw_target.strip()
    if target.startswith("<") and target.endswith(">"):
        target = target[1:-1]
    return unquote(target)


def reference_definitions(content: str) -> dict[str, str]:
    definitions: dict[str, str] = {}
    for match in REFERENCE_DEF_RE.finditer(content):
        definitions[match.group("label").strip().casefold()] = match.group("target")
    return definitions


def find_image_matches(content: str) -> list[ImageMatch]:
    matches: list[ImageMatch] = []

    for match in INLINE_IMAGE_RE.finditer(content):
        matches.append(
            ImageMatch(
                start=match.start(),
                end=match.end(),
                markup=match.group(0),
                target=match.group("target"),
                alt=match.group("alt"),
                kind="markdown-inline",
            )
        )

    for match in HTML_IMAGE_RE.finditer(content):
        matches.append(
            ImageMatch(
                start=match.start(),
                end=match.end(),
                markup=match.group(0),
                target=match.group("target"),
                alt="",
                kind="html",
            )
        )

    definitions = reference_definitions(content)
    for match in REFERENCE_IMAGE_RE.finditer(content):
        label = (match.group("label") or match.group("alt")).strip().casefold()
        target = definitions.get(label)
        if target:
            matches.append(
                ImageMatch(
                    start=match.start(),
                    end=match.end(),
                    markup=match.group(0),
                    target=target,
                    alt=match.group("alt"),
                    kind="markdown-reference",
                )
            )

    matches.sort(key=lambda item: (item.start, item.end))
    for previous, current in zip(matches, matches[1:]):
        if current.start < previous.end:
            raise ArchiveError(
                f"Overlapping image markup is unsupported near: {current.markup!r}"
            )
    return matches


def title_from_content(content: str, source: Path) -> str:
    match = H1_RE.search(content)
    if not match:
        return source.stem
    title = match.group("title").strip()
    return title or source.stem


def collect_markdown_sources(source: Path, recursive: bool) -> list[Path]:
    if source.is_file():
        if source.suffix.casefold() != ".md":
            raise ArchiveError(f"Only Markdown files are supported: {source}")
        return [source]
    if not source.is_dir():
        raise ArchiveError(f"Source does not exist: {source}")
    iterator = source.rglob("*.md") if recursive else source.glob("*.md")
    return sorted((path.resolve() for path in iterator if path.is_file()), key=str)


def resolve_local_image(
    raw_target: str,
    markdown_path: Path,
    company_root: Path,
) -> tuple[str, Path | None]:
    normalized = normalize_target(raw_target)
    parsed = urlparse(normalized)
    if parsed.scheme.casefold() in REMOTE_SCHEMES or normalized.startswith("//"):
        return "remote", None
    if parsed.scheme and len(parsed.scheme) > 1:
        return "remote", None

    candidate = Path(normalized)
    if not candidate.is_absolute():
        candidate = markdown_path.parent / candidate
    resolved = candidate.resolve()
    if not is_relative_to(resolved, company_root):
        raise ArchiveError(
            f"Referenced image is outside .company: {normalized} -> {resolved}"
        )
    if resolved.suffix.casefold() not in SUPPORTED_IMAGE_SUFFIXES:
        raise ArchiveError(f"Unsupported local image type: {resolved}")
    if not resolved.is_file():
        raise ArchiveError(f"Referenced image does not exist: {resolved}")
    return "local", resolved


def prepare(args: argparse.Namespace) -> int:
    workspace_root = Path(args.workspace_root).expanduser().resolve()
    company_root = (workspace_root / ".company").resolve()
    if not company_root.is_dir():
        raise ArchiveError(f"Company root not found: {company_root}")

    source = Path(args.source).expanduser()
    if not source.is_absolute():
        source = workspace_root / source
    source = source.resolve()
    if not is_relative_to(source, company_root):
        raise ArchiveError(f"Source must be under {company_root}: {source}")

    markdown_paths = collect_markdown_sources(source, args.recursive)
    if not markdown_paths:
        raise ArchiveError(f"No Markdown files found: {source}")

    stage_dir = Path(
        tempfile.mkdtemp(prefix=TEMP_PREFIX, dir=args.temp_root)
    ).resolve()
    rendered_dir = (stage_dir / "rendered").resolve()
    assets_dir = (stage_dir / "assets").resolve()
    rendered_dir.mkdir()
    assets_dir.mkdir()

    documents: list[dict[str, Any]] = []
    global_asset_index = 0
    try:
        for markdown_path in markdown_paths:
            content = markdown_path.read_text(encoding="utf-8")
            _, compilation_preview = compile_notion_markdown(content)
            image_entries: list[dict[str, Any]] = []
            seen_local_paths: dict[Path, dict[str, Any]] = {}

            for image_match in find_image_matches(content):
                disposition, resolved_path = resolve_local_image(
                    image_match.target, markdown_path, company_root
                )
                entry: dict[str, Any] = {
                    "kind": image_match.kind,
                    "start": image_match.start,
                    "end": image_match.end,
                    "original_markup": image_match.markup,
                    "target": normalize_target(image_match.target),
                    "alt": image_match.alt,
                    "disposition": disposition,
                }
                if disposition == "local" and resolved_path is not None:
                    existing = seen_local_paths.get(resolved_path)
                    if existing is None:
                        global_asset_index += 1
                        asset_id = f"asset-{global_asset_index:04d}"
                        digest = sha256_file(resolved_path)
                        staged_name = (
                            f"{global_asset_index:04d}-{digest[:12]}-"
                            f"{resolved_path.name}"
                        )
                        staged_path = assets_dir / staged_name
                        shutil.copy2(resolved_path, staged_path)
                        if sha256_file(staged_path) != digest:
                            raise ArchiveError(
                                f"Staged image checksum mismatch: {resolved_path}"
                            )
                        existing = {
                            "asset_id": asset_id,
                            "source_path": str(resolved_path),
                            "staged_name": staged_name,
                            "staged_path": str(staged_path),
                            "sha256": digest,
                            "size_bytes": resolved_path.stat().st_size,
                            "content_type": (
                                mimetypes.guess_type(resolved_path.name)[0]
                                or "application/octet-stream"
                            ),
                        }
                        seen_local_paths[resolved_path] = existing
                    entry.update(existing)
                image_entries.append(entry)

            documents.append(
                {
                    "source_path": str(markdown_path),
                    "source_sha256": sha256_file(markdown_path),
                    "source_size_bytes": markdown_path.stat().st_size,
                    "title": title_from_content(content, markdown_path),
                    "images": image_entries,
                    "local_image_count": sum(
                        item["disposition"] == "local" for item in image_entries
                    ),
                    "remote_image_count": sum(
                        item["disposition"] == "remote" for item in image_entries
                    ),
                    "notion_compilation_preview": compilation_preview,
                }
            )

        manifest = {
            "schema_version": 2,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "workspace_root": str(workspace_root),
            "company_root": str(company_root),
            "requested_source": str(source),
            "recursive": bool(args.recursive),
            "stage_dir": str(stage_dir),
            "assets_dir": str(assets_dir),
            "rendered_dir": str(rendered_dir),
            "documents": documents,
        }
        output = (
            Path(args.output).expanduser().resolve()
            if args.output
            else stage_dir / "manifest.json"
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "manifest": str(output),
                    "documents": len(documents),
                    "local_images": sum(
                        doc["local_image_count"] for doc in documents
                    ),
                    "remote_images": sum(
                        doc["remote_image_count"] for doc in documents
                    ),
                    "stage_dir": str(stage_dir),
                    "assets_dir": str(assets_dir),
                },
                ensure_ascii=False,
            )
        )
        return 0
    except Exception:
        shutil.rmtree(stage_dir, ignore_errors=True)
        raise


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ArchiveError(f"Expected JSON object: {path}")
    return value


def validate_manifest_temp_path(manifest: dict[str, Any]) -> Path:
    stage_dir = Path(manifest["stage_dir"]).resolve()
    temp_root = Path(tempfile.gettempdir()).resolve()
    if not is_relative_to(stage_dir, temp_root) or not stage_dir.name.startswith(
        TEMP_PREFIX
    ):
        raise ArchiveError(f"Refusing unsafe temporary path: {stage_dir}")
    return stage_dir


def render(args: argparse.Namespace) -> int:
    manifest_path = Path(args.manifest).expanduser().resolve()
    manifest = load_json(manifest_path)
    validate_manifest_temp_path(manifest)
    if args.upload_map:
        upload_map = load_json(Path(args.upload_map).expanduser().resolve())
        assets = upload_map.get("assets")
        if not isinstance(assets, dict):
            raise ArchiveError("Upload map must contain an 'assets' object")
    else:
        assets = {}

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rendered_documents: list[dict[str, Any]] = []

    for index, document in enumerate(manifest["documents"], start=1):
        source_path = Path(document["source_path"])
        if sha256_file(source_path) != document["source_sha256"]:
            raise ArchiveError(f"Source changed after prepare: {source_path}")
        content = source_path.read_text(encoding="utf-8")
        replacements: list[tuple[int, int, str]] = []
        for image in document["images"]:
            if image["disposition"] != "local":
                continue
            asset_id = image["asset_id"]
            replacement = assets.get(asset_id)
            if not isinstance(replacement, str) or not replacement.strip():
                raise ArchiveError(f"Missing upload map value for {asset_id}")
            if content[image["start"] : image["end"]] != image["original_markup"]:
                raise ArchiveError(
                    f"Image markup changed after prepare: {source_path}"
                )
            replacements.append((image["start"], image["end"], replacement))

        for start, end, replacement in sorted(replacements, reverse=True):
            content = content[:start] + replacement + content[end:]

        content, notion_compilation = compile_notion_markdown(content)
        rendered_name = (
            f"{index:04d}-{document['source_sha256'][:12]}-{source_path.name}"
        )
        rendered_path = output_dir / rendered_name
        rendered_path.write_text(content, encoding="utf-8")
        rendered_documents.append(
            {
                "source_path": str(source_path),
                "title": document["title"],
                "rendered_path": str(rendered_path),
                "local_image_count": document["local_image_count"],
                "rendered_sha256": sha256_file(rendered_path),
                "source_sha256": document["source_sha256"],
                "notion_compilation": notion_compilation,
            }
        )

    render_manifest = {
        "schema_version": 2,
        "source_manifest": str(manifest_path),
        "documents": rendered_documents,
    }
    render_manifest_path = output_dir / "render-manifest.json"
    render_manifest_path.write_text(
        json.dumps(render_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "render_manifest": str(render_manifest_path),
                "documents": len(rendered_documents),
            },
            ensure_ascii=False,
        )
    )
    return 0


def audit(args: argparse.Namespace) -> int:
    render_manifest = load_json(
        Path(args.render_manifest).expanduser().resolve()
    )
    documents = render_manifest.get("documents")
    if not isinstance(documents, list) or not documents:
        raise ArchiveError("Render manifest has no documents")
    if args.document_index < 1 or args.document_index > len(documents):
        raise ArchiveError(
            f"Document index out of range: {args.document_index}"
        )
    expected = documents[args.document_index - 1].get(
        "notion_compilation", {}
    ).get("profile")
    if not isinstance(expected, dict):
        raise ArchiveError("Render manifest is missing the Notion profile")
    fetched_content = Path(args.page_content).expanduser().read_text(
        encoding="utf-8"
    )
    errors = audit_notion_round_trip(expected, fetched_content)
    result = {
        "document_index": args.document_index,
        "verified": not errors,
        "errors": errors,
    }
    print(json.dumps(result, ensure_ascii=False))
    return 0 if not errors else 3


def cleanup(args: argparse.Namespace) -> int:
    manifest_path = Path(args.manifest).expanduser().resolve()
    manifest = load_json(manifest_path)
    stage_dir = validate_manifest_temp_path(manifest)
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    print(json.dumps({"removed": str(stage_dir), "exists": stage_dir.exists()}))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("source")
    prepare_parser.add_argument("--workspace-root", required=True)
    prepare_parser.add_argument("--output")
    prepare_parser.add_argument("--recursive", action="store_true")
    prepare_parser.add_argument("--temp-root", default=None)
    prepare_parser.set_defaults(func=prepare)

    render_parser = subparsers.add_parser("render")
    render_parser.add_argument("--manifest", required=True)
    render_parser.add_argument("--upload-map")
    render_parser.add_argument("--output-dir", required=True)
    render_parser.set_defaults(func=render)

    audit_parser = subparsers.add_parser("audit")
    audit_parser.add_argument("--render-manifest", required=True)
    audit_parser.add_argument("--page-content", required=True)
    audit_parser.add_argument("--document-index", type=int, default=1)
    audit_parser.set_defaults(func=audit)

    cleanup_parser = subparsers.add_parser("cleanup")
    cleanup_parser.add_argument("--manifest", required=True)
    cleanup_parser.set_defaults(func=cleanup)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return args.func(args)
    except (ArchiveError, OSError, UnicodeError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
