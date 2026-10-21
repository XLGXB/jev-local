"""Frozen, standard-library acceptance for the canonical/training data bridge."""
from __future__ import annotations

import copy
import math
import unittest

from decision_ft.data import atomize, permutation, public_request, validate_record
from jevlocal.integration.data_bridge import BridgeError, from_training_record, to_training_record
from jevlocal.schema import Label, Question, Sample


def sample(split="train"):
    questions = [
        Question("yn", "noul", "Is it useful?", {"yes": "keep", "no": "drop"}),
        Question("pick", "choice", "Choose action", {"a": "first", "b": "second", "c": "third"}),
        Question("level", "score", "Rate", ["low", "mid", "high"]),
    ]
    return Sample("real-17", "routing", {"kind": "synthetic", "ref": "unit:17"},
                  "Public state", questions,
                  {"yn": Label(1, 0.8),
                   "pick": Label("b", {"a": .2, "b": .6, "c": .2}),
                   "level": Label(2, [.1, .3, .6])}, split,
                  {"extractor": "unit", "extractor_version": "1"})


def bridge(s, **kw):
    return to_training_record(s, group_id="group-17", label_source="human_verified",
                              task="routing", training_allowed=True, **kw)


class BridgeTests(unittest.TestCase):
    def test_roundtrip_three_soft_types_metadata_and_no_mutation(self):
        s = sample()
        before = copy.deepcopy(s.to_json())
        row = bridge(s)
        validate_record(row)
        self.assertEqual(row["_meta"]["id"], "real-17")
        self.assertEqual(row["questions"]["yn"]["criteria"], {"true": "keep", "false": "drop"})
        self.assertEqual(row["questions"]["yn"]["target"], {"false": .2, "true": .8})
        self.assertIs(row["questions"]["yn"]["label"], True)
        self.assertEqual(row["questions"]["level"]["target"], {"0": .1, "1": .3, "2": .6})
        self.assertEqual(row["questions"]["pick"]["target"], {"a": .2, "b": .6, "c": .2})
        self.assertEqual(s.to_json(), before)
        self.assertEqual(from_training_record(row).to_json(), before)
        self.assertEqual(row["_meta"]["bridge_version"], 1)
        self.assertEqual(row["_meta"]["split"], "train")
        for q in public_request(row)["questions"].values():
            self.assertNotIn("label", q)
            self.assertNotIn("target", q)
        self.assertNotIn("_meta", public_request(row))
        for a in atomize([row]):
            self.assertNotIn("label", a.question)
            self.assertNotIn("target", a.question)

    def test_bool_and_noul_probabilities(self):
        for p in (0.0, .2, .8, 1.0):
            s = sample()
            s.labels["yn"] = Label(0, p)
            row = bridge(s)
            self.assertIs(row["questions"]["yn"]["label"], False)
            self.assertAlmostEqual(row["questions"]["yn"]["target"]["false"], 1-p)
            self.assertAlmostEqual(row["questions"]["yn"]["target"]["true"], p)
            self.assertEqual(from_training_record(row).labels["yn"].soft, p)

    def test_permutation_uses_current_choice_content_and_order(self):
        row = permutation(bridge(sample()), 12)
        back = from_training_record(row)
        self.assertEqual(list(back.questions[1].criteria), list(row["questions"]["pick"]["criteria"]))
        self.assertEqual(back.labels["pick"].soft, {"a": .2, "b": .6, "c": .2})
        row["questions"]["pick"]["criteria"]["b"] = "updated description"
        row["questions"]["pick"]["label"] = "a"
        row["questions"]["pick"]["target"] = {"a": .7, "b": .2, "c": .1}
        back = from_training_record(row)
        self.assertEqual(back.questions[1].criteria["b"], "updated description")
        self.assertEqual(back.labels["pick"].hard, "a")
        self.assertEqual(back.labels["pick"].soft["a"], .7)

    def test_calib_mapping_and_source_gate(self):
        s = sample("calib")
        row = bridge(s)
        self.assertEqual(row["_meta"]["split"], "calibration")
        self.assertEqual(from_training_record(row).split, "calib")
        with self.assertRaises(BridgeError):
            to_training_record(s, group_id="g", label_source="teacher", task="t", training_allowed=True)
        s.split = "dev"
        row = to_training_record(s, group_id="g", label_source="teacher", task="t", training_allowed=True)
        self.assertEqual(from_training_record(row).split, "dev")
        s.split = "test"
        with self.assertRaises(BridgeError):
            to_training_record(s, group_id="g", label_source="teacher", task="t", training_allowed=True)

    def test_new_noul_keys_and_teacher_sidecar(self):
        s = sample()
        s.questions[0].criteria = {"false": "drop", "true": "keep"}
        s.labels["yn"].teacher = {"kind": "votes", "count": 3}
        row = bridge(s)
        self.assertEqual(list(row["questions"]["yn"]["criteria"]), ["false", "true"])
        back = from_training_record(row)
        self.assertEqual(back.to_json(), s.to_json())
        row["questions"]["yn"]["instructions"] = "Changed public instructions"
        self.assertEqual(from_training_record(row).questions[0].instructions,
                         "Changed public instructions")

    def test_reject_unknown_single_missing_authorization_ood_nan(self):
        s = sample()
        s.labels["yn"] = Label(None, .5, unanswerable=True)
        with self.assertRaises(BridgeError): bridge(s)
        s = sample(); s.questions[1].criteria = {"a": "only"}; s.labels["pick"] = Label("a", {"a": 1.0})
        with self.assertRaises(BridgeError): bridge(s)
        s = sample()
        with self.assertRaises(BridgeError):
            to_training_record(s, group_id="g", label_source="human_verified", task="t", training_allowed=False)
        with self.assertRaises(BridgeError):
            to_training_record(s, group_id="g", label_source="human_verified", task="t", training_allowed=None)
        with self.assertRaises(BridgeError):
            to_training_record(s, group_id="", label_source="human_verified", task="t", training_allowed=True)
        with self.assertRaises(BridgeError): bridge(sample("ood"))
        s.labels["yn"] = Label(1, math.nan)
        with self.assertRaises(BridgeError): bridge(s)

    def test_reject_duplicate_qid_candidate_conflict_and_malformed_reverse(self):
        s = sample(); s.questions.append(copy.deepcopy(s.questions[0]))
        with self.assertRaises(BridgeError): bridge(s)
        row = bridge(sample())
        row["questions"]["pick"]["target"] = {"a": .1, "b": .2, "alien": .7}
        with self.assertRaises(BridgeError): from_training_record(row)
        row = bridge(sample()); del row["_meta"]["canonical"]
        with self.assertRaises(BridgeError): from_training_record(row)
        row = bridge(sample()); row["_meta"]["training_allowed"] = False
        with self.assertRaises(BridgeError): from_training_record(row)


if __name__ == "__main__":
    unittest.main()
