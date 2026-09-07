#!/usr/bin/env python3
"""Validate and normalize a brainstorm-to-notion destination config."""

from __future__ import annotations

import argparse
import json
import re
import sys
import uuid
from pathlib import Path
from urllib.parse import urlparse


NOTION_ID_RE = re.compile(
    r"(?i)(?<![0-9a-f])([0-9a-f]{32}|"
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})"
    r"(?![0-9a-f])"
)


class ConfigError(ValueError):
    pass


def nonempty_string(config: dict[str, object], key: str) -> str:
    value = config.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{key} must be a non-empty string")
    return value.strip()


def normalize_notion_id(raw_value: str, field_name: str) -> str:
    try:
        return str(uuid.UUID(raw_value.strip()))
    except ValueError as error:
        raise ConfigError(f"{field_name} is not a valid Notion ID") from error


def page_id_from_url(raw_url: str) -> str:
    parsed = urlparse(raw_url.strip())
    if parsed.scheme.casefold() != "https" or not parsed.hostname:
        raise ConfigError("parent_page_url must be an HTTPS Notion URL")

    host = parsed.hostname.casefold()
    allowed = (
        host == "notion.so"
        or host.endswith(".notion.so")
        or host == "notion.site"
        or host.endswith(".notion.site")
        or host == "notion.com"
        or host.endswith(".notion.com")
    )
    if not allowed:
        raise ConfigError("parent_page_url host is not a recognized Notion host")

    matches = NOTION_ID_RE.findall(raw_url)
    normalized = {str(uuid.UUID(match)) for match in matches}
    if len(normalized) != 1:
        raise ConfigError("parent_page_url must contain exactly one Notion page ID")
    return normalized.pop()


def validate_config(config: object) -> dict[str, object]:
    if not isinstance(config, dict):
        raise ConfigError("config root must be a JSON object")
    if config.get("schema_version") != 2:
        raise ConfigError("schema_version must be 2")
    if config.get("enabled") is not True:
        raise ConfigError("enabled must be true before this destination can be used")

    workspace_name = nonempty_string(config, "workspace_name")
    destination_type = config.get("destination_type")
    if destination_type == "page":
        return validate_page_destination(config, workspace_name)
    if destination_type == "data_source":
        return validate_data_source_destination(config, workspace_name)
    raise ConfigError("destination_type must be 'page' or 'data_source'")


def validate_page_destination(
    config: dict[str, object], workspace_name: str
) -> dict[str, object]:
    parent_page_title = nonempty_string(config, "parent_page_title")
    raw_url = config.get("parent_page_url", "")
    raw_id = config.get("parent_page_id", "")
    if not isinstance(raw_url, str) or not isinstance(raw_id, str):
        raise ConfigError("parent_page_url and parent_page_id must be strings")

    url_id = page_id_from_url(raw_url) if raw_url.strip() else None
    explicit_id = (
        normalize_notion_id(raw_id, "parent_page_id") if raw_id.strip() else None
    )
    if url_id is None and explicit_id is None:
        raise ConfigError("set parent_page_url or parent_page_id")
    if url_id is not None and explicit_id is not None and url_id != explicit_id:
        raise ConfigError("parent_page_url and parent_page_id refer to different pages")

    return {
        "valid": True,
        "schema_version": 2,
        "destination_type": "page",
        "workspace_name": workspace_name,
        "parent_page_title": parent_page_title,
        "parent_page_url": raw_url.strip() or None,
        "parent_page_id": explicit_id or url_id,
    }


def validate_data_source_destination(
    config: dict[str, object], workspace_name: str
) -> dict[str, object]:
    data_source_title = nonempty_string(config, "data_source_title")
    raw_id = nonempty_string(config, "data_source_id")
    data_source_id = normalize_notion_id(raw_id, "data_source_id")

    raw_mapping = config.get("property_mapping")
    if not isinstance(raw_mapping, dict):
        raise ConfigError("property_mapping must be a JSON object")
    property_mapping = {
        logical_name: nonempty_string(raw_mapping, logical_name)
        for logical_name in ("title", "created_date", "keywords")
    }
    if len(set(property_mapping.values())) != len(property_mapping):
        raise ConfigError("property_mapping values must name different properties")

    return {
        "valid": True,
        "schema_version": 2,
        "destination_type": "data_source",
        "workspace_name": workspace_name,
        "data_source_title": data_source_title,
        "data_source_id": data_source_id,
        "property_mapping": property_mapping,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()

    try:
        path = Path(args.config).expanduser().resolve()
        config = json.loads(path.read_text(encoding="utf-8"))
        result = validate_config(config)
    except (OSError, json.JSONDecodeError, ConfigError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
