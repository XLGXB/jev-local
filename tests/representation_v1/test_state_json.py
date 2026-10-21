"""Focused, dependency-free tests for the opt-in public state representation."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from decision_ft.state_json import (  # noqa: E402
    POLICY_ID,
    policy_manifest,
    prepare_public_request,
    render_state,
)
import decision_ft.state_json as module  # noqa: E402


def request(state: object) -> dict:
    return {
        "state": state,
        "questions": {
            "q": {
                "type": "choice",
                "instructions": "Pick one",
                "criteria": {"second": "B", "first": "A"},
            },
        },
        "model": "local",
    }


class StateJsonTests(unittest.TestCase):
    def test_distinct_legacy_collision_and_scalar_types(self) -> None:
        self.assertEqual(str(None or ""), str("" or ""))  # legacy-style collapse
        cases = [
            (None, "null"), ("", '""'), ("null", '"null"'),
            ({}, "{}"), ([], "[]"), (True, "true"),
            (False, "false"), (0, "0"), (1.0, "1.0"),
        ]
        for value, expected in cases:
            with self.subTest(value=value):
                self.assertEqual(render_state(value), expected)
                self.assertEqual(json.loads(render_state(value)), value)

    def test_nested_unicode_order_and_no_normalization(self) -> None:
        state = {"后": [True, {"前": None, "é": "e\u0301"}], "先": [0, 1.0]}
        rendered = render_state(state)
        self.assertEqual(rendered, '{"后":[true,{"前":null,"é":"é"}],"先":[0,1.0]}')
        self.assertEqual(json.loads(rendered), state)
        self.assertNotIn("\\u", rendered)

    def test_reject_invalid_state(self) -> None:
        cycle = []
        cycle.append(cycle)
        invalid = [cycle, math.nan, math.inf, -math.inf, {1: "x"},
                   {"x": (1, 2)}, "\ud800", {"\udfff": "x"}, {"x": object()}]
        for value in invalid:
            with self.subTest(kind=type(value).__name__):
                with self.assertRaises((TypeError, ValueError, UnicodeError)):
                    render_state(value)

    def test_manifest_binds_current_source_and_format(self) -> None:
        manifest = policy_manifest()
        self.assertEqual(manifest["schema_version"], 1)
        self.assertEqual(manifest["id"], POLICY_ID)
        self.assertEqual(manifest["serializer_sha256"],
                         hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest())
        self.assertEqual(manifest["format"]["json_dumps"]["separators"], [",", ":"])

    def test_prepare_preserves_public_fields_order_and_input(self) -> None:
        raw = request({"z": True, "a": [None, "null", 1.0]})
        baseline = copy.deepcopy(raw)
        prepared = prepare_public_request(raw, expected_policy=policy_manifest())
        self.assertEqual(raw, baseline)
        self.assertIsNot(prepared, raw)
        self.assertIsNot(prepared["questions"], raw["questions"])
        self.assertEqual(list(prepared), list(raw))
        self.assertEqual(prepared["questions"], raw["questions"])
        self.assertEqual(list(prepared["questions"]["q"]["criteria"]), ["second", "first"])
        self.assertEqual(prepared["model"], "local")
        self.assertEqual(prepared["state"], '{"z":true,"a":[null,"null",1.0]}')
        self.assertEqual(prepare_public_request(request("null"), expected_policy=policy_manifest())["state"], '"null"')

    def test_explicit_binding_required_and_exact(self) -> None:
        raw = request(None)
        with self.assertRaises(TypeError):
            prepare_public_request(raw)
        for changed in [
            {**policy_manifest(), "id": "wrong"},
            {**policy_manifest(), "serializer_sha256": "0" * 64},
            {**policy_manifest(), "schema_version": True},
            {**policy_manifest(), "extra": 1},
            {**policy_manifest(), "format": {"encoding": "UTF-8"}},
            None,
        ]:
            with self.subTest(changed=changed):
                with self.assertRaises(ValueError):
                    prepare_public_request(raw, expected_policy=changed)

    def test_reject_nonpublic_and_unknown_fields(self) -> None:
        for field in ("label", "target", "metadata", "_meta", "unknown"):
            raw = request(None)
            raw[field] = "private"
            with self.subTest(level="request", field=field):
                with self.assertRaises(ValueError):
                    prepare_public_request(raw, expected_policy=policy_manifest())
            raw = request(None)
            raw["questions"]["q"][field] = "private"
            with self.subTest(level="question", field=field):
                with self.assertRaises(ValueError):
                    prepare_public_request(raw, expected_policy=policy_manifest())

    def test_question_type_and_presence(self) -> None:
        for kind in ("noul", "choice", "score"):
            raw = request([])
            raw["questions"]["q"]["type"] = kind
            self.assertEqual(prepare_public_request(raw, expected_policy=policy_manifest())["state"], "[]")
        for kind in (None, "boolean", True, 1):
            raw = request([])
            raw["questions"]["q"]["type"] = kind
            with self.assertRaises(ValueError):
                prepare_public_request(raw, expected_policy=policy_manifest())
        for missing in ("state", "questions"):
            raw = request(None)
            del raw[missing]
            with self.assertRaises(ValueError):
                prepare_public_request(raw, expected_policy=policy_manifest())


if __name__ == "__main__":
    unittest.main()
