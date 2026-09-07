from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "validate_destination.py"
)
SPEC = importlib.util.spec_from_file_location("validate_destination", SCRIPT_PATH)
assert SPEC and SPEC.loader
validate_destination = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = validate_destination
SPEC.loader.exec_module(validate_destination)


class DestinationConfigTest(unittest.TestCase):
    def base_config(self) -> dict[str, object]:
        return {
            "schema_version": 2,
            "enabled": True,
            "destination_type": "page",
            "workspace_name": "Research Workspace",
            "parent_page_title": "壁打ちアーカイブ",
            "parent_page_url": "",
            "parent_page_id": "",
        }

    def data_source_config(self) -> dict[str, object]:
        return {
            "schema_version": 2,
            "enabled": True,
            "destination_type": "data_source",
            "workspace_name": "nitech_notion",
            "data_source_title": "研究ノート_DB",
            "data_source_id": "e8cad5f0-b4be-83a1-9c58-0724c9251ad1",
            "property_mapping": {
                "title": "名前",
                "created_date": "作成日",
                "keywords": "キーワード",
            },
        }

    def test_accepts_page_url(self) -> None:
        config = self.base_config()
        config["parent_page_url"] = (
            "https://www.notion.so/Brainstorm-"
            "3a2842497e4a81ef8944e5f6efb048a5"
        )
        result = validate_destination.validate_config(config)
        self.assertEqual(
            result["parent_page_id"],
            "3a284249-7e4a-81ef-8944-e5f6efb048a5",
        )

    def test_accepts_page_id(self) -> None:
        config = self.base_config()
        config["parent_page_id"] = "3a2842497e4a81ef8944e5f6efb048a5"
        result = validate_destination.validate_config(config)
        self.assertEqual(
            result["parent_page_id"],
            "3a284249-7e4a-81ef-8944-e5f6efb048a5",
        )

    def test_rejects_disabled_config(self) -> None:
        config = self.base_config()
        config["enabled"] = False
        config["parent_page_id"] = "3a2842497e4a81ef8944e5f6efb048a5"
        with self.assertRaises(validate_destination.ConfigError):
            validate_destination.validate_config(config)

    def test_rejects_url_and_id_mismatch(self) -> None:
        config = self.base_config()
        config["parent_page_url"] = (
            "https://app.notion.com/p/3a2842497e4a81ef8944e5f6efb048a5"
        )
        config["parent_page_id"] = "11111111111111111111111111111111"
        with self.assertRaises(validate_destination.ConfigError):
            validate_destination.validate_config(config)

    def test_requires_workspace_and_title(self) -> None:
        config = self.base_config()
        config["workspace_name"] = ""
        config["parent_page_id"] = "3a2842497e4a81ef8944e5f6efb048a5"
        with self.assertRaises(validate_destination.ConfigError):
            validate_destination.validate_config(config)

    def test_accepts_data_source(self) -> None:
        result = validate_destination.validate_config(self.data_source_config())
        self.assertEqual(
            result["data_source_id"],
            "e8cad5f0-b4be-83a1-9c58-0724c9251ad1",
        )
        self.assertEqual(result["property_mapping"]["title"], "名前")

    def test_rejects_incomplete_property_mapping(self) -> None:
        config = self.data_source_config()
        config["property_mapping"] = {
            "title": "名前",
            "created_date": "作成日",
            "keywords": "",
        }
        with self.assertRaises(validate_destination.ConfigError):
            validate_destination.validate_config(config)

    def test_rejects_duplicate_property_mapping(self) -> None:
        config = self.data_source_config()
        config["property_mapping"] = {
            "title": "名前",
            "created_date": "作成日",
            "keywords": "作成日",
        }
        with self.assertRaises(validate_destination.ConfigError):
            validate_destination.validate_config(config)

    def test_rejects_invalid_data_source_id(self) -> None:
        config = self.data_source_config()
        config["data_source_id"] = "not-a-notion-id"
        with self.assertRaises(validate_destination.ConfigError):
            validate_destination.validate_config(config)

    def test_rejects_unknown_destination_type(self) -> None:
        config = self.base_config()
        config["destination_type"] = "database"
        with self.assertRaises(validate_destination.ConfigError):
            validate_destination.validate_config(config)


if __name__ == "__main__":
    unittest.main()
