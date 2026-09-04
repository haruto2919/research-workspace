from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "material_archive.py"
)
SPEC = importlib.util.spec_from_file_location("material_archive", SCRIPT_PATH)
assert SPEC and SPEC.loader
material_archive = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = material_archive
SPEC.loader.exec_module(material_archive)


class MaterialArchiveTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.company = self.root / ".company"
        self.materials = self.company / "lab" / "materials"
        self.materials.mkdir(parents=True)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def run_prepare(
        self, source: Path, recursive: bool = False
    ) -> tuple[int, Path]:
        output = self.root / "manifest.json"
        args = type(
            "Args",
            (),
            {
                "workspace_root": str(self.root),
                "source": str(source),
                "output": str(output),
                "recursive": recursive,
                "temp_root": str(self.root),
            },
        )()
        result = material_archive.prepare(args)
        return result, output

    def run_render(
        self,
        manifest_path: Path,
        upload_map_path: Path | None = None,
    ) -> tuple[int, Path]:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        output_dir = Path(manifest["rendered_dir"])
        args = type(
            "Args",
            (),
            {
                "manifest": str(manifest_path),
                "upload_map": (
                    str(upload_map_path) if upload_map_path is not None else None
                ),
                "output_dir": str(output_dir),
            },
        )()
        result = material_archive.render(args)
        return result, output_dir / "render-manifest.json"

    def test_prepare_and_render_preserve_text_and_replace_only_local_image(self):
        image = self.materials / "figure.png"
        image.write_bytes(b"\x89PNG\r\n\x1a\noriginal-image-bytes")
        source = self.materials / "report.md"
        original = (
            "---\n"
            "date: 2026-07-19\n"
            "---\n\n"
            "# Exact title\n\n"
            "Text before.\n\n"
            "![local alt](<figure.png>)\n\n"
            "![remote](https://example.com/remote.png)\n\n"
            "Text after.\n"
        )
        source.write_text(original, encoding="utf-8")

        _, manifest_path = self.run_prepare(source)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        document = manifest["documents"][0]
        self.assertEqual(document["title"], "Exact title")
        self.assertEqual(document["local_image_count"], 1)
        self.assertEqual(document["remote_image_count"], 1)
        self.assertEqual(source.read_text(encoding="utf-8"), original)

        asset_id = next(
            image_entry["asset_id"]
            for image_entry in document["images"]
            if image_entry["disposition"] == "local"
        )
        upload_map_path = self.root / "upload-map.json"
        upload_map_path.write_text(
            json.dumps(
                {
                    "assets": {
                        asset_id: (
                            '<image src="file-upload://example"></image>'
                        )
                    }
                }
            ),
            encoding="utf-8",
        )
        output_dir = Path(manifest["rendered_dir"])
        render_args = type(
            "Args",
            (),
            {
                "manifest": str(manifest_path),
                "upload_map": str(upload_map_path),
                "output_dir": str(output_dir),
            },
        )()
        material_archive.render(render_args)
        render_manifest = json.loads(
            (output_dir / "render-manifest.json").read_text(encoding="utf-8")
        )
        rendered_path = Path(
            render_manifest["documents"][0]["rendered_path"]
        )
        rendered = rendered_path.read_text(encoding="utf-8")
        expected = original.replace(
            "![local alt](<figure.png>)",
            '<image src="file-upload://example"></image>',
        )
        self.assertEqual(rendered, expected)
        self.assertIn(
            "![remote](https://example.com/remote.png)", rendered
        )
        self.assertEqual(source.read_text(encoding="utf-8"), original)

    def test_compile_normalizes_math_and_table_without_touching_code(self):
        original = (
            "# Math and tables\n\n"
            r"Inline \(\theta_i\) and `literal \(\theta_i\)`." "\n\n"
            "\\[\n"
            "A\\in\\mathbb{R}^{r\\times d}\n"
            "\\]\n\n"
            "| Left | Right | Center |\n"
            "| :--- | ---: | :---: |\n"
            "| a | b | c |\n\n"
            "```mermaid\n"
            "flowchart TD\n"
            r"    A[\"keep \(\theta_i\) and | ---: |\"]" "\n"
            "```\n"
        )

        compiled, report = material_archive.compile_notion_markdown(original)

        self.assertIn(r"Inline $\theta_i$", compiled)
        self.assertIn(r"`literal \(\theta_i\)`", compiled)
        self.assertIn(
            "$$\nA\\in\\mathbb{R}^{r\\times d}\n$$",
            compiled,
        )
        self.assertIn("| --- | --- | --- |", compiled)
        self.assertIn(r"keep \(\theta_i\) and | ---: |", compiled)
        self.assertEqual(report["legacy_inline_math_converted"], 1)
        self.assertEqual(report["legacy_display_math_converted"], 1)
        self.assertEqual(report["table_separators_normalized"], 1)
        self.assertEqual(
            report["aligned_table_separators_normalized"], 1
        )
        self.assertEqual(report["fenced_code_blocks_preserved"], 1)
        self.assertEqual(report["mermaid_code_blocks_preserved"], 1)
        profile = report["profile"]
        self.assertEqual(profile["inline_math_count"], 1)
        self.assertEqual(profile["display_math_count"], 1)
        self.assertEqual(profile["table_count"], 1)
        self.assertEqual(
            profile["tables"][0],
            {
                "row_count": 2,
                "column_count": 3,
                "header_row": True,
                "separator_leak": False,
                "header_sha256": material_archive.text_sha256(
                    "Left\u241fRight\u241fCenter"
                ),
                "content_sha256": material_archive.table_content_sha256(
                    [
                        ["Left", "Right", "Center"],
                        ["a", "b", "c"],
                    ]
                ),
            },
        )

    def test_prepare_and_render_compile_image_free_markdown(self):
        source = self.materials / "math-table.md"
        original = (
            "# Compile preview\n\n"
            r"Value: \(x+1\)." "\n\n"
            "| Metric | Value |\n"
            "| --- | ---: |\n"
            "| score | 1 |\n"
        )
        source.write_text(original, encoding="utf-8")

        _, manifest_path = self.run_prepare(source)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        preview = manifest["documents"][0]["notion_compilation_preview"]
        self.assertEqual(preview["legacy_inline_math_converted"], 1)
        self.assertEqual(
            preview["aligned_table_separators_normalized"], 1
        )
        self.assertEqual(preview["profile"]["table_count"], 1)

        _, render_manifest_path = self.run_render(manifest_path)
        render_manifest = json.loads(
            render_manifest_path.read_text(encoding="utf-8")
        )
        rendered_document = render_manifest["documents"][0]
        rendered = Path(rendered_document["rendered_path"]).read_text(
            encoding="utf-8"
        )
        self.assertIn("Value: $x+1$.", rendered)
        self.assertIn("| --- | --- |", rendered)
        self.assertNotIn("---:", rendered)
        self.assertEqual(
            rendered_document["notion_compilation"]["profile"]["table_count"],
            1,
        )
        self.assertEqual(source.read_text(encoding="utf-8"), original)

    def test_round_trip_audit_checks_math_and_table_structure(self):
        source = (
            "# Audit\n\n"
            r"Inline \(x+1\)." "\n\n"
            "\\[\n"
            "A=B\n"
            "\\]\n\n"
            "| Metric | Value |\n"
            "| --- | ---: |\n"
            "| score | 1 |\n"
        )
        _, report = material_archive.compile_notion_markdown(source)
        expected = report["profile"]
        fetched = (
            "$`x+1`$\n"
            "$$\n"
            "A=B\n"
            "$$\n"
            '<table header-row="true">\n'
            "<tr><td>Metric</td><td>Value</td></tr>\n"
            "<tr><td>score</td><td>1</td></tr>\n"
            "</table>\n"
        )
        self.assertEqual(
            material_archive.audit_notion_round_trip(expected, fetched),
            [],
        )

        broken = (
            "$`x+1`$\n"
            "$$\n"
            "A=B\n"
            "$$\n"
            "<table>\n"
            "<tr><td>Metric</td><td>Value</td></tr>\n"
            "<tr><td>---</td><td>---:</td></tr>\n"
            "<tr><td>score</td><td>1</td></tr>\n"
            "</table>\n"
        )
        errors = material_archive.audit_notion_round_trip(expected, broken)
        self.assertTrue(any("row_count mismatch" in error for error in errors))
        self.assertTrue(any("header_row mismatch" in error for error in errors))
        self.assertTrue(
            any("separator data row" in error for error in errors)
        )

    def test_directory_is_non_recursive_by_default(self):
        (self.materials / "top.md").write_text("# Top\n", encoding="utf-8")
        nested = self.materials / "nested"
        nested.mkdir()
        (nested / "nested.md").write_text("# Nested\n", encoding="utf-8")

        _, manifest_path = self.run_prepare(self.materials)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(
            [document["title"] for document in manifest["documents"]],
            ["Top"],
        )

    def test_directory_can_be_recursive_when_explicit(self):
        (self.materials / "top.md").write_text("# Top\n", encoding="utf-8")
        nested = self.materials / "nested"
        nested.mkdir()
        (nested / "nested.md").write_text("# Nested\n", encoding="utf-8")

        _, manifest_path = self.run_prepare(self.materials, recursive=True)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(
            sorted(
                document["title"] for document in manifest["documents"]
            ),
            ["Nested", "Top"],
        )

    def test_rejects_source_outside_company(self):
        source = self.root / "outside.md"
        source.write_text("# Outside\n", encoding="utf-8")
        with self.assertRaises(material_archive.ArchiveError):
            self.run_prepare(source)

    def test_rejects_missing_local_image(self):
        source = self.materials / "missing.md"
        source.write_text("# Missing\n\n![x](missing.png)\n", encoding="utf-8")
        with self.assertRaises(material_archive.ArchiveError):
            self.run_prepare(source)


if __name__ == "__main__":
    unittest.main()
