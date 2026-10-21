"""CPU-only contracts for the opt-in lossless Laya option encoder."""
import json
from pathlib import Path
import re
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import laya.common as common

from decision_ft.backends import Backend, LayaBackend
from decision_ft.data import Atom
from decision_deploy.common import sha
from decision_deploy.encoding import Frontend


class Tokenizer:
    mask_token = "<MASK>"
    mask_token_id = 3
    cls_token_id = 1
    sep_token_id = 2
    pad_token_id = 0

    def __init__(self):
        self.vocab = {}

    def __call__(self, text, add_special_tokens=False):
        assert not add_special_tokens
        ids = []
        for word in re.findall(r"\S+", text):
            ids.append(self.vocab.setdefault(word, len(self.vocab) + 4))
        return {"input_ids": ids}

    def save_pretrained(self, path):
        Path(path).mkdir()


def atom(kind="choice", criteria=None, instructions="choose", state="evidence"):
    if criteria is None:
        criteria = {"z": "last", "a": "first"} if kind == "choice" else (
            [{"rubric": ["low", 0]}, "high"] if kind == "score" else
            {"false": {"reason": ["no", 0]}, "true": "yes"})
    keys = list(criteria) if kind == "choice" else (["0", "1"] if kind == "score" else ["false", "true"])
    return Atom("test::q", "g", state,
                {"type": kind, "instructions": instructions, "criteria": criteria},
                keys, [0.5, 0.5], 0, "fixture", "fixture")


def backend(cfg, tok=None):
    obj = LayaBackend.__new__(LayaBackend)
    Backend.__init__(obj)
    obj.cfg, obj.tok, obj.common = cfg, tok or Tokenizer(), common
    return obj


class LayaOptionCapacityTests(unittest.TestCase):
    def setUp(self):
        self.old = {"max_len": 512, "head_max_len": 192}
        self.new = {"option_max_len": 64, "head_max_len": 320, "max_len": 1536}

    def test_default_uses_vendor_bytes_and_original_limit(self):
        b = backend(self.old)
        a = atom()
        q = {"t": "choice", "ins": a.question["instructions"], "crit": a.question["criteria"]}
        self.assertEqual(b.encode(a), common.build_sequence(b.tok, a.state, q, 512, 192))
        long = atom(criteria={"a": " ".join(["word"] * 49), "b": "short"})
        with self.assertRaisesRegex(ValueError, "native Laya would truncate an option to 48 tokens"):
            b.encode(long)

    def test_all_forms_match_vendor_when_short_and_preserve_order(self):
        for kind in ("choice", "score", "noul"):
            with self.subTest(kind=kind):
                a = atom(kind)
                tok = Tokenizer()
                old, new = backend(self.old, tok), backend(self.new, tok)
                self.assertEqual(new.encode(a), old.encode(a))
                ids, marks = new.encode(a)
                self.assertEqual(len(marks), len(a.keys))
                self.assertTrue(all(ids[m] == tok.mask_token_id for m in marks))
                options = common.render_options({"t": kind, "ins": a.question["instructions"], "crit": a.question["criteria"]})
                self.assertEqual([tok(" " + o, add_special_tokens=False)["input_ids"][0]
                                  for o in options], [ids[m + 1] for m in marks])

    def test_full_54_token_option_is_not_sliced(self):
        a = atom(criteria={"first": " ".join(f"w{i}" for i in range(53)), "second": "short"})
        b = backend(self.new)
        q = {"t": "choice", "ins": "choose", "crit": a.question["criteria"]}
        self.assertEqual(len(b.tok(" " + common.render_options(q)[0], add_special_tokens=False)["input_ids"]), 54)
        ids, marks = b.encode(a)
        self.assertEqual(marks[1] - marks[0], 55)
        self.assertEqual(ids[marks[0] + 1:marks[1]],
                         b.tok(" " + common.render_options(q)[0], add_special_tokens=False)["input_ids"])

    def test_overflows_are_explicit_and_do_not_cache(self):
        a = atom(criteria={"a": " ".join(["word"] * 64), "b": "short"})
        b = backend(self.new)
        with self.assertRaisesRegex(ValueError, "option_max_len 64"):
            b.encode(a)
        self.assertFalse(b.cache)
        for cfg, message in (({**self.new, "head_max_len": 16}, "head_max_len 16"),
                             ({**self.new, "max_len": 8}, "exceed max_len 8")):
            with self.subTest(message=message):
                b = backend(cfg)
                with self.assertRaisesRegex(ValueError, message):
                    b.encode(atom())
                self.assertFalse(b.cache)
        b = backend({**self.new, "head_max_len": 24})
        with self.assertRaisesRegex(ValueError, "head_max_len 24"):
            b.encode(atom(instructions=" ".join(["instruction"] * 30)))
        self.assertFalse(b.cache)

    def test_invalid_opt_in_capacities_rejected_without_coercion(self):
        for key in self.new:
            for value in (None, False, True, 0, -1, 1.0, "64"):
                with self.subTest(key=key, value=value):
                    b = backend({**self.new, key: value})
                    with self.assertRaisesRegex(ValueError, key):
                        b.encode(atom())

    def test_reserved_mask_sanitized_in_every_segment(self):
        a = atom(criteria={"a": "x <MASK> y", "b": "z"},
                 instructions="choose <MASK> now", state="evidence <MASK> here")
        tok = Tokenizer()
        old, new = backend(self.old, tok), backend(self.new, tok)
        ids, marks = new.encode(a)
        self.assertEqual((ids, marks), old.encode(a))
        self.assertEqual([i for i, value in enumerate(ids) if value == tok.mask_token_id], marks)

    def test_exported_opt_in_config_reloads_same_frontend_encoding(self):
        b = backend(self.new)
        b.model = SimpleNamespace(
            state_dict=lambda: {},
            encoder=SimpleNamespace(config=SimpleNamespace(
                save_pretrained=lambda path: Path(path).mkdir())))
        a = atom(criteria={"first": " ".join(f"w{i}" for i in range(53)), "second": "short"})
        expected = b.encode(a)
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "export"
            with patch("safetensors.torch.save_file"):
                b.export(destination)
            saved = json.loads((destination / "rl_agent_config.json").read_text(encoding="utf-8"))
            self.assertEqual([saved[key] for key in self.new], [self.new[key] for key in self.new])
            with patch("transformers.AutoTokenizer.from_pretrained", return_value=b.tok):
                frontend = Frontend(destination, "laya")
            self.assertEqual(frontend.encode(a), expected)
            self.assertEqual(frontend.sources["training_backends"], sha(Path(__file__).parents[1] / "src/decision_ft/backends.py"))


if __name__ == "__main__":
    unittest.main()
