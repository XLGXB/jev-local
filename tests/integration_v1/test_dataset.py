"""Presplit dataset preparation must never silently repartition bridge records."""
from __future__ import annotations

import copy
import hashlib
import tempfile
import unittest
from pathlib import Path

from decision_ft.data import PARTITIONS, json_bytes, verify_dataset
from jevlocal.integration.data_bridge import to_training_record
from jevlocal.integration.dataset import prepare_presplit_dataset
from jevlocal.schema import Label, Question, Sample


def rows():
    result = []
    for i, split in enumerate(("train", "dev", "calib", "test")):
        q = Question("decision", "noul", f"Decide {i}")
        s = Sample(f"id-{i}", "routing", {"kind": "synthetic", "ref": f"ref-{i}"},
                   f"distinct state {i}", [q], {"decision": Label(i % 2, .2 + .2 * i)},
                   split, {"extractor": "unit", "extractor_version": "1"})
        result.append(to_training_record(s, group_id=f"group-{i}",
                                         label_source="human_verified", task="routing",
                                         training_allowed=True))
    return result


class PresplitTests(unittest.TestCase):
    def test_prepare_preserves_splits_and_verify_dataset_reads(self):
        source = rows()
        before = copy.deepcopy(source)
        with tempfile.TemporaryDirectory() as td:
            output = Path(td) / "prepared"
            manifest = prepare_presplit_dataset(source, output)
            self.assertEqual(set(manifest["files"]), set(PARTITIONS))
            self.assertEqual(manifest["input_hash_kind"], "ordered_records_json_v1")
            self.assertEqual(manifest["input_sha256"], hashlib.sha256(json_bytes(before)).hexdigest())
            self.assertEqual(sum(x["records"] for x in manifest["files"].values()), 4)
            verified = verify_dataset(output)
            for row in source:
                split = row["_meta"]["split"]
                self.assertEqual([r["_meta"]["id"] for r in verified[split]],
                                 [row["_meta"]["id"]])
                self.assertEqual(verified[split][0], row)
            self.assertEqual(source, before)

    def test_same_group_inside_one_partition_allowed(self):
        source = rows()
        extra = copy.deepcopy(source[0])
        extra["_meta"]["id"] = "extra-train"
        extra["state"] = "different train state"
        source.append(extra)
        with tempfile.TemporaryDirectory() as td:
            manifest = prepare_presplit_dataset(source, Path(td) / "prepared")
            self.assertEqual(manifest["files"]["train"]["records"], 2)

    def test_reject_cross_split_group_content_template_and_trajectory(self):
        for mutation in ("group_id", "content", "template_id", "trajectory_id",
                         "source_artifact_sha256"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as td:
                source = rows()
                if mutation == "content":
                    source[1]["state"] = source[0]["state"]
                    source[1]["questions"] = copy.deepcopy(source[0]["questions"])
                elif mutation == "group_id":
                    source[1]["_meta"]["group_id"] = source[0]["_meta"]["group_id"]
                else:
                    source[0]["_meta"][mutation] = "shared"
                    source[1]["_meta"][mutation] = "shared"
                with self.assertRaises(ValueError):
                    prepare_presplit_dataset(source, Path(td) / "prepared")
                self.assertFalse((Path(td) / "prepared").exists())

    def test_reject_duplicate_id_bad_metadata_unknown_split_and_unverified_calibration(self):
        for mutation in ("duplicate", "missing_group", "missing_target", "unknown_split", "ood",
                         "calibration_teacher", "test_teacher"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as td:
                source = rows()
                if mutation == "duplicate":
                    source[1]["_meta"]["id"] = source[0]["_meta"]["id"]
                elif mutation == "missing_group":
                    del source[0]["_meta"]["group_id"]
                elif mutation == "missing_target":
                    del source[0]["questions"]["decision"]["target"]
                elif mutation == "unknown_split":
                    source[0]["_meta"]["split"] = "unknown"
                elif mutation == "ood":
                    source[0]["_meta"]["split"] = "ood"
                elif mutation == "calibration_teacher":
                    source[2]["_meta"]["label_source"] = "teacher"
                else:
                    source[3]["_meta"]["label_source"] = "teacher"
                with self.assertRaises(ValueError):
                    prepare_presplit_dataset(source, Path(td) / "prepared")

    def test_reject_missing_partition_and_target_collision_without_overwrite(self):
        with tempfile.TemporaryDirectory() as td:
            output = Path(td) / "prepared"
            with self.assertRaises(ValueError):
                prepare_presplit_dataset(rows()[:-1], output)
            self.assertFalse(output.exists())
            output.mkdir()
            sentinel = output / "KEEP.txt"
            sentinel.write_text("original", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                prepare_presplit_dataset(rows(), output)
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "original")


if __name__ == "__main__":
    unittest.main()
