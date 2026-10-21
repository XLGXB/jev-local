"""Pure fixture tests for frozen state representation and checkpoint identity."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from decision_ft.data import Atom  # noqa: E402
from decision_ft import state_json_binding as binding  # noqa: E402
from decision_ft import state_json  # noqa: E402


def make_atom(state: object = None) -> Atom:
    return Atom(
        "row::q", "group", state,
        {"type": "choice", "instructions": "Pick", "criteria": {"b": "B", "a": "A"}},
        ["b", "a"], [0.25, 0.75], 1, "synthetic", "program_verified",
    )


def make_request(state: object = None) -> dict:
    atom = make_atom(state)
    return {"state": state, "questions": {"q": copy.deepcopy(atom.question)}}


class BindingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frozen = binding.binding_manifest()  # pure fixture; real callers pin externally

    def test_manifest_hash_and_strict_binding(self) -> None:
        self.assertEqual(self.frozen["schema_version"], 1)
        self.assertEqual(
            self.frozen["binding_sha256"],
            hashlib.sha256(Path(binding.__file__).read_bytes()).hexdigest(),
        )
        binding.validate_binding(self.frozen)
        for changed in (
            {**self.frozen, "binding_sha256": "0" * 64},
            {**self.frozen, "schema_version": True},
            {**self.frozen, "extra": 1},
            {**self.frozen, "policy": {**self.frozen["policy"], "id": "wrong"}},
            None,
        ):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                binding.validate_binding(changed)

    def test_training_preserves_supervision_and_copies(self) -> None:
        original = make_atom({"z": True, "a": [None, 1.0]})
        before = copy.deepcopy(original)
        seen = []
        real_prepare = binding.state_json.prepare_public_request

        def recording_prepare(public, *, expected_policy):
            seen.append(copy.deepcopy(public))
            return real_prepare(public, expected_policy=expected_policy)

        with patch.object(binding.state_json, "prepare_public_request", side_effect=recording_prepare):
            represented = binding.prepare_training_atoms([original], expected_binding=self.frozen)[0]
        self.assertEqual(original, before)
        self.assertEqual(set(seen[0]), {"state", "questions"})
        self.assertEqual(set(seen[0]["questions"]["question"]), {"type", "instructions", "criteria"})
        self.assertEqual(represented.state, '{"z":true,"a":[null,1.0]}')
        self.assertEqual(represented.keys, ["b", "a"])
        self.assertIsNot(represented.keys, original.keys)
        self.assertEqual(represented.target, original.target)
        self.assertIsNot(represented.target, original.target)
        self.assertEqual((represented.uid, represented.group_id, represented.label,
                          represented.task, represented.label_source),
                         (original.uid, original.group_id, original.label,
                          original.task, original.label_source))
        self.assertEqual(represented.kind, original.kind)
        self.assertEqual(represented.source_fingerprint, original.fingerprint)
        expected_digest = hashlib.sha256(json.dumps(
            self.frozen, ensure_ascii=False, allow_nan=False,
            sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")).hexdigest()
        self.assertEqual(represented.representation_digest, expected_digest)
        self.assertNotEqual(represented.fingerprint, original.fingerprint)

    def test_policy_only_change_changes_digest_and_fingerprint(self) -> None:
        atom = make_atom(None)
        first = binding.prepare_training_atoms([atom], expected_binding=self.frozen)[0]
        second_policy = copy.deepcopy(self.frozen["policy"])
        second_policy["format"]["encoding"] = "controlled fixture policy change"
        second_binding = {**self.frozen, "policy": second_policy}
        # Simulate a separately pinned newer policy while preserving this source file.
        with patch.object(state_json, "policy_manifest", return_value=second_policy):
            second = binding.prepare_training_atoms([atom], expected_binding=second_binding)[0]
        self.assertEqual(first.state, second.state)
        self.assertEqual(first.source_fingerprint, second.source_fingerprint)
        self.assertEqual(first.question, second.question)
        self.assertNotEqual(first.representation_digest, second.representation_digest)
        self.assertNotEqual(first.fingerprint, second.fingerprint)

    def test_inference_uses_same_transform_without_labels(self) -> None:
        raw = make_request({"x": [False, "null"]})
        before = copy.deepcopy(raw)
        pairs = binding.prepare_inference_atoms(raw, expected_binding=self.frozen)
        self.assertEqual(raw, before)
        self.assertEqual([qid for qid, _ in pairs], ["q"])
        inferred = pairs[0][1]
        trained = binding.prepare_training_atoms([make_atom(raw["state"])], expected_binding=self.frozen)[0]
        self.assertEqual(inferred.state, trained.state)
        self.assertEqual(inferred.question, trained.question)
        self.assertEqual(inferred.keys, trained.keys)
        self.assertEqual(inferred.fingerprint, trained.fingerprint)
        self.assertEqual(inferred.target, [])
        self.assertEqual(inferred.label_source, "not_a_label")

    def test_reject_double_transform_subclass_and_option_mismatch(self) -> None:
        first = binding.prepare_training_atoms([make_atom(None)], expected_binding=self.frozen)[0]
        with self.assertRaises(TypeError):
            binding.prepare_training_atoms([first], expected_binding=self.frozen)
        wrong = make_atom(None)
        wrong.keys.reverse()
        with self.assertRaises(ValueError):
            binding.prepare_training_atoms([wrong], expected_binding=self.frozen)
        with self.assertRaises(ValueError):
            binding.prepare_inference_atoms(make_request(None), expected_binding={})

    def test_marker_exclusive_round_trip_and_strict_fields(self) -> None:
        freeze_hash = "a" * 64
        with tempfile.TemporaryDirectory() as temporary:
            path = binding.write_checkpoint_marker(
                temporary, expected_binding=self.frozen,
                training_freeze_sha256=freeze_hash, epoch=2,
            )
            self.assertEqual(path.name, "state_representation.json")
            self.assertEqual([file.name for file in Path(temporary).iterdir()], [path.name])
            marker = binding.read_checkpoint_marker(temporary, expected_binding=self.frozen)
            self.assertEqual(marker["binding"], self.frozen)
            self.assertEqual((marker["training_freeze_sha256"], marker["epoch"]), (freeze_hash, 2))
            with self.assertRaises(FileExistsError):
                binding.write_checkpoint_marker(
                    temporary, expected_binding=self.frozen,
                    training_freeze_sha256=freeze_hash, epoch=2,
                )
            for key, bad in (
                ("schema_version", True), ("status", "native"),
                ("epoch", True), ("epoch", 0),
                ("training_freeze_sha256", "A" * 64),
                ("binding", {**self.frozen, "binding_sha256": "0" * 64}),
            ):
                changed = {**marker, key: bad}
                path.write_text(json.dumps(changed), encoding="utf-8")
                with self.subTest(key=key, bad=bad), self.assertRaises(ValueError):
                    binding.read_checkpoint_marker(temporary, expected_binding=self.frozen)
            path.write_text(json.dumps({**marker, "extra": 1}), encoding="utf-8")
            with self.assertRaises(ValueError):
                binding.read_checkpoint_marker(temporary, expected_binding=self.frozen)
            path.write_text('{"schema_version":1,"schema_version":1}', encoding="utf-8")
            with self.assertRaises(ValueError):
                binding.read_checkpoint_marker(temporary, expected_binding=self.frozen)

    def test_marker_write_invalid_inputs_do_not_create_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            for digest, epoch in (("A" * 64, 1), ("a" * 64, 0), ("a" * 64, True)):
                with self.subTest(digest=digest, epoch=epoch), self.assertRaises(ValueError):
                    binding.write_checkpoint_marker(
                        temporary, expected_binding=self.frozen,
                        training_freeze_sha256=digest, epoch=epoch,
                    )
                self.assertEqual(list(Path(temporary).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
