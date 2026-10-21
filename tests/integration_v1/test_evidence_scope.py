"""Deterministic evidence applicability and snapshot verification."""
import copy
import hashlib
import json
import unittest
from unittest.mock import patch

from jevlocal.integration.evidence_scope import (
    candidate_digest,
    scope_evidence,
    verify_scope_evidence,
)


class EvidenceScopeTests(unittest.TestCase):
    def setUp(self):
        self.candidates = [
            {"id": "a", "source": "doc-a", "text": "alpha\nsecond"},
            {"id": "b", "source": "doc-b", "start_line": 4, "text": "beta"},
            {"id": "c", "source": "doc-c", "text": "gamma"},
        ]
        self.target = {"object_id": "Object-A", "revision": "v1"}
        self.bindings = {
            "a": self.binding(0, "Object-A", "v1"),
            "b": self.binding(1, "Object-A", None),
            "c": self.binding(2, "Object-B", "v1"),
        }

    def binding(self, index, object_id, revision):
        return {"object_id": object_id, "revision": revision,
                "candidate_sha256": candidate_digest(self.candidates[index])}

    def arguments(self, **changes):
        result = {"query": "alpha", "candidates": copy.deepcopy(self.candidates),
                  "target": copy.deepcopy(self.target),
                  "bindings": copy.deepcopy(self.bindings), "max_chars": 100000}
        result.update(changes)
        return result

    def prepare(self, **changes):
        return scope_evidence(**self.arguments(**changes))

    def test_match_unknown_mismatch_and_derived_order(self):
        self.bindings["b"]["revision"] = "v1"
        result = self.prepare()
        self.assertEqual(result["status"], "prepared")
        self.assertEqual(result["counts"], {"total": 3, "match": 2, "mismatch": 1, "unknown": 0})
        self.assertEqual([x["status"] for x in result["applicability"]], ["match", "match", "mismatch"])
        self.assertEqual([x["id"] for x in result["rank_input"]["candidates"]], ["a", "b"])
        self.assertEqual(result["rank_input"]["candidates"][1]["start_line"], 4)
        self.assertEqual([x["id"] for x in result["snapshot"]["candidates"]], ["a", "b", "c"])
        self.assertEqual(result["rank_input"]["max_chars"], 12000)
        self.assertEqual(verify_scope_evidence(self.arguments(), result, result["rank_input"])["counts"], result["counts"])

    def test_known_conflict_overrides_unknown_and_no_matches(self):
        self.bindings["a"]["revision"] = None
        self.bindings["a"]["object_id"] = "wrong"
        self.bindings["b"]["object_id"] = None
        result = self.prepare()
        self.assertEqual(result["status"], "no_matches")
        self.assertEqual(result["counts"], {"total": 3, "match": 0, "mismatch": 2, "unknown": 1})
        self.assertEqual(result["rank_input"], None)
        self.assertEqual(result["rank_input_sha256"], None)
        self.assertIn("object_id_mismatch", result["applicability"][0]["reasons"])

    def test_all_unknown_no_matches(self):
        for value in self.bindings.values():
            value["object_id"] = None
            value["revision"] = None
        result = self.prepare()
        self.assertEqual(result["counts"]["unknown"], 3)
        self.assertEqual(result["status"], "no_matches")

    def test_matching_does_not_inspect_proposition_or_success(self):
        texts = ["The requested artifact was saved.",
                 "The requested artifact could not be saved.",
                 "Unrelated formatting preference: use sentence case."]
        for candidate, text in zip(self.candidates, texts):
            candidate["text"] = text
        self.bindings = {c["id"]: self.binding(i, "Object-A", "v1")
                         for i, c in enumerate(self.candidates)}
        result = self.prepare()
        self.assertEqual(result["counts"]["match"], 3)
        self.assertEqual([c["text"] for c in result["rank_input"]["candidates"]], texts)

    def test_excluded_change_invalidates_old_manifest_not_derived_input(self):
        old = self.prepare()
        changed = self.arguments()
        changed["candidates"][2]["text"] = "different excluded content"
        changed["bindings"]["c"]["candidate_sha256"] = candidate_digest(changed["candidates"][2])
        new = scope_evidence(**changed)
        self.assertEqual(new["rank_input"], old["rank_input"])
        self.assertNotEqual(new["snapshot_sha256"], old["snapshot_sha256"])
        with self.assertRaises(ValueError):
            verify_scope_evidence(changed, old, old["rank_input"])
        # A newly consistent manifest is not proof of the historical parent
        # of this same projection. Invocation provenance is a separate claim.
        report = verify_scope_evidence(changed, new, old["rank_input"])
        self.assertEqual(report["status"], "consistent")
        self.assertFalse(report["origin_authenticated"])

    def test_digest_normalization_and_stale_bindings(self):
        normalized = {"id": "a", "source": "doc-a", "start_line": 1,
                      "text": "alpha\nsecond"}
        expected = hashlib.sha256(json.dumps(normalized, sort_keys=True,
            ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()
        self.assertEqual(candidate_digest(self.candidates[0]), expected)
        self.assertEqual(candidate_digest(self.candidates[0]),
                         candidate_digest({**self.candidates[0], "start_line": 1}))
        for change in ({"text": "changed"}, {"source": "changed"}, {"start_line": 2}):
            with self.subTest(change=change):
                altered = copy.deepcopy(self.candidates)
                altered[0].update(change)
                with self.assertRaises(ValueError):
                    self.prepare(candidates=altered)
        with self.assertRaises(ValueError):
            candidate_digest({**self.candidates[0], "extra": True})

    def test_identity_is_exact_typed_and_unmodified(self):
        for invalid in (1, True, "", " ", "x\n", "x\u200d", "\ud800", "x" * 257,
                        "token: hidden"):
            with self.subTest(invalid=repr(invalid)):
                target = copy.deepcopy(self.target)
                target["revision"] = invalid
                with self.assertRaises(ValueError):
                    self.prepare(target=target)
        self.bindings["a"]["revision"] = "V1"
        self.assertEqual(self.prepare()["applicability"][0]["status"], "mismatch")

    def test_missing_duplicate_and_extra_fields_rejected(self):
        bindings = copy.deepcopy(self.bindings)
        del bindings["c"]
        with self.assertRaises(ValueError):
            self.prepare(bindings=bindings)
        duplicates = self.candidates + [copy.deepcopy(self.candidates[0])]
        with self.assertRaises(ValueError):
            self.prepare(candidates=duplicates)
        with self.assertRaises(ValueError):
            self.prepare(target={**self.target, "extra": 1})
        with self.assertRaises(ValueError):
            self.prepare(bindings={**self.bindings, "extra": self.binding(0, None, None)})
        with self.assertRaises(ValueError):
            self.prepare(rank_max_chars=True)

    def test_sanitized_snapshot_and_separate_hash_layers(self):
        self.candidates[0]["text"] = "alpha token=private-value"
        self.candidates[0]["source"] = "Authorization: secret-value"
        self.bindings["a"] = self.binding(0, "Object-A", "v1")
        result = self.prepare()
        wire = json.dumps(result)
        self.assertNotIn("private-value", wire)
        self.assertNotIn("secret-value", wire)
        self.assertNotEqual(self.bindings["a"]["candidate_sha256"],
                            result["snapshot"]["bindings"]["a"]["candidate_sha256"])
        self.assertEqual(result["snapshot"]["bindings"]["a"]["candidate_sha256"],
                         candidate_digest(result["snapshot"]["candidates"][0]))

    def test_budget_is_nonpartial_and_verifier_rejects_it(self):
        result = self.prepare(max_chars=35)
        self.assertEqual(result["status"], "requires_larger_budget")
        self.assertFalse("snapshot" in result)
        with self.assertRaises(ValueError):
            verify_scope_evidence(self.arguments(), result)

    def test_explicit_tuple_known_secrets_can_be_verified_without_mutation(self):
        args = self.arguments(known_secrets=("unrelated-secret",))
        result = scope_evidence(**args)
        self.assertEqual(verify_scope_evidence(args, result, result['rank_input'])['status'], 'consistent')
        self.assertIs(type(args['known_secrets']), tuple)

    def test_verification_rejects_all_tamper_shapes(self):
        self.bindings["b"]["revision"] = "v1"
        args = self.arguments()
        result = self.prepare()
        for path, value in (("version", True), ("counts", {"total": 3, "match": True, "mismatch": 1, "unknown": 0}),
                            ("rank_input_sha256", "0" * 64), ("applicability", []),
                            ("snapshot", {})):
            with self.subTest(path=path):
                tampered = copy.deepcopy(result)
                tampered[path] = value
                with self.assertRaises(ValueError):
                    verify_scope_evidence(args, tampered)
        with self.assertRaises(ValueError):
            verify_scope_evidence(args, result, {**result["rank_input"], "max_chars": True})
        with self.assertRaises(ValueError):
            verify_scope_evidence({**args, "extra": 1}, result)

    def test_no_mutation_file_or_model_route(self):
        args = self.arguments()
        before = copy.deepcopy(args)
        with patch("builtins.open", side_effect=AssertionError("file IO forbidden")), \
             patch("jevlocal.integration.context_tools._store", side_effect=AssertionError("store forbidden")), \
             patch("jevlocal.integration.context_tools._scores", side_effect=AssertionError("model forbidden")):
            result = scope_evidence(**args)
            self.assertEqual(verify_scope_evidence(args, result, result["rank_input"])["status"], "consistent")
        self.assertEqual(args, before)

    def test_recursive_and_unserializable_input_fails_closed(self):
        candidate = copy.deepcopy(self.candidates[0])
        candidate["text"] = []
        candidate["text"].append(candidate["text"])
        with self.assertRaises(ValueError):
            candidate_digest(candidate)
        result = self.prepare()
        recursive = copy.deepcopy(result)
        recursive["unexpected"] = recursive
        with self.assertRaises(ValueError):
            verify_scope_evidence(self.arguments(), recursive, result["rank_input"])
        with self.assertRaises(ValueError):
            candidate_digest({**self.candidates[0], "start_line": True})


if __name__ == "__main__":
    unittest.main()
