"""Contract tests for the bounded, inert JSON UI renderer."""

import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from jevlocal.integration.json_render import catalog_fingerprint, render_html, validate_ui


def valid_ui():
    return {
        "schema_version": 1,
        "title": "Local recommendation",
        "layout": {"kind": "grid", "columns": 2, "gap": "normal"},
        "components": [
            {"id": "head", "type": "heading", "props": {"text": "Overview", "level": 1}},
            {"id": "body", "type": "text", "props": {"text": "你好 🌍", "tone": "default"}},
            {"id": "metric", "type": "metric", "props": {"label": "Count", "value": "3", "unit": "items"}},
            {"id": "list", "type": "list", "props": {"title": "Items", "items": ["one", "two"], "ordered": False}},
            {"id": "notice", "type": "notice", "props": {"text": "Review before acting", "tone": "warning"}},
            {"id": "button", "type": "button", "props": {"label": "Continue", "action_id": "handoff_1"}},
        ],
    }


class JsonRenderTests(unittest.TestCase):
    def assert_invalid(self, mutate):
        value = valid_ui()
        mutate(value)
        with self.assertRaises(ValueError):
            validate_ui(value)

    def test_unicode_copy_and_determinism(self):
        value = valid_ui()
        before = copy.deepcopy(value)
        normalized = validate_ui(value)
        self.assertEqual(normalized, before)
        self.assertIsNot(normalized, value)
        self.assertIsNot(normalized["components"], value["components"])
        normalized["components"][3]["props"]["items"][0] = "changed"
        self.assertEqual(value, before)
        self.assertEqual(render_html(value), render_html(value))
        self.assertIn("你好 🌍", render_html(value))

    def test_all_components_and_static_safety(self):
        html = render_html(valid_ui())
        for token in ("<main", "<h1", "<h2", "<ul", "<button", "disabled", "aria-disabled", "Content-Security-Policy"):
            self.assertIn(token, html)
        self.assertNotIn("<script", html)
        self.assertNotIn("<form", html)
        self.assertNotIn("https://", html)
        self.assertIn("default-src 'none'", html)
        self.assertIn("script-src 'none'", html)
        self.assertIn("form-action 'none'", html)
        self.assertIn("frame-src 'none'", html)
        self.assertIn("本地模型建议", html)
        self.assertIn("非授权", html)

    def test_injection_escaped_and_button_inert(self):
        value = valid_ui()
        value["title"] = '<script>alert("x")</script>'
        value["components"][1]["props"]["text"] = '<img src="https://evil.example/x" onerror="alert(1)">'
        value["components"][5]["props"]["label"] = '<svg onload="x">'
        html = render_html(value)
        self.assertIn("&lt;script&gt;", html)
        self.assertIn("&lt;img", html)
        self.assertIn("&lt;svg", html)
        self.assertNotIn("<img", html)
        self.assertNotIn("<svg", html)
        self.assertNotIn("onclick=", html)
        self.assertNotIn("href=", html)
        self.assertNotIn("<img src=", html)
        self.assertEqual(html.count("<button"), 1)
        self.assertIn('type="button" disabled', html)

    def test_top_level_layout_and_component_exactness(self):
        self.assert_invalid(lambda v: v.update(extra=1))
        self.assert_invalid(lambda v: v.update(schema_version=True))
        self.assert_invalid(lambda v: v.update(schema_version=2))
        self.assert_invalid(lambda v: v.update(title=""))
        self.assert_invalid(lambda v: v.update(title="x" * 121))
        self.assert_invalid(lambda v: v["layout"].update(style="color:red"))
        self.assert_invalid(lambda v: v["layout"].update(kind="flex"))
        self.assert_invalid(lambda v: v["layout"].update(columns=True))
        self.assert_invalid(lambda v: v["layout"].update(columns=4))
        self.assert_invalid(lambda v: v["layout"].update(gap="huge"))
        self.assert_invalid(lambda v: v["components"][0].update(url="https://example.org"))
        self.assert_invalid(lambda v: v["components"][0].update(type="iframe"))
        self.assert_invalid(lambda v: v["components"][0]["props"].update(css="x"))
        self.assert_invalid(lambda v: v["components"].append(copy.deepcopy(v["components"][0])))
        self.assert_invalid(lambda v: v.update(components=[]))
        self.assert_invalid(lambda v: v.update(components=v["components"] * 5))
        self.assert_invalid(lambda v: v["components"][0].update(id="9bad"))
        self.assert_invalid(lambda v: v["components"][0].update(id="é"))
        self.assert_invalid(lambda v: v["components"][0].update(id="a" * 65))

    def test_layout_stack_and_list_ordered(self):
        value = valid_ui()
        value["layout"] = {"kind": "stack", "columns": 1, "gap": "compact"}
        value["components"][3]["props"]["ordered"] = True
        self.assertIn("<ol", render_html(value))
        self.assert_invalid(lambda v: v.update(layout={"kind": "stack", "columns": 2, "gap": "normal"}))

    def test_each_props_type_and_bounds(self):
        self.assert_invalid(lambda v: v["components"][0]["props"].update(level=True))
        self.assert_invalid(lambda v: v["components"][0]["props"].update(level=4))
        self.assert_invalid(lambda v: v["components"][1]["props"].update(tone="danger"))
        self.assert_invalid(lambda v: v["components"][1]["props"].update(text="x" * 2001))
        self.assert_invalid(lambda v: v["components"][2]["props"].update(value="x" * 121))
        self.assert_invalid(lambda v: v["components"][3]["props"].update(ordered=1))
        self.assert_invalid(lambda v: v["components"][3]["props"].update(items=[]))
        self.assert_invalid(lambda v: v["components"][3]["props"].update(items=["x"] * 13))
        self.assert_invalid(lambda v: v["components"][3]["props"].update(items=[1]))
        self.assert_invalid(lambda v: v["components"][4]["props"].update(tone="critical"))
        self.assert_invalid(lambda v: v["components"][5]["props"].update(action_id="https://x"))
        self.assert_invalid(lambda v: v["components"][5]["props"].update(action_id="javascript:alert(1)"))
        self.assert_invalid(lambda v: v["components"][5]["props"].update(label="x\n y"))

    def test_invalid_json_shape_and_size(self):
        self.assert_invalid(lambda v: v.update(title="bad\x00title"))
        self.assert_invalid(lambda v: v.update(title="\ud800"))
        self.assert_invalid(lambda v: v["components"][0]["props"].pop("level"))
        self.assert_invalid(lambda v: v.update(layout=[]))
        self.assert_invalid(lambda v: v.update(components="not-list"))
        value = valid_ui()
        value["components"] = [
            {"id": f"c{i}", "type": "text", "props": {"text": "界" * 1900, "tone": "default"}}
            for i in range(8)
        ]
        self.assertGreater(len(json.dumps(value, ensure_ascii=False).encode("utf-8")), 32768)
        with self.assertRaises(ValueError):
            validate_ui(value)
        with self.assertRaises(ValueError):
            render_html(value)

    def test_fingerprint_tracks_source_bytes(self):
        current = catalog_fingerprint()
        self.assertRegex(current, r"^[0-9a-f]{64}$")
        self.assertEqual(current, catalog_fingerprint())
        source = Path(importlib.util.find_spec("jevlocal.integration.json_render").origin)
        with tempfile.TemporaryDirectory() as folder:
            changed = Path(folder) / "json_render_changed.py"
            changed.write_bytes(source.read_bytes() + b"\n# renderer or schema change\n")
            spec = importlib.util.spec_from_file_location("json_render_changed", changed)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self.assertNotEqual(current, module.catalog_fingerprint())


if __name__ == "__main__":
    unittest.main()
