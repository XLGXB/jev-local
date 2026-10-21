import json
import tempfile
import unittest
from pathlib import Path

from jevlocal.integration.context_tools import (
    ModelRouteError,
    filter_tool_output,
    rank_evidence,
    restore_lines,
    suggest_skills,
)


class ContextToolsTests(unittest.TestCase):
    def test_rank_and_restore_safe_original_by_hash_and_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = rank_evidence(
                "login timeout 登录超时",
                [
                    {"id": "a", "source": "allowed.txt", "start_line": 7,
                     "text": "first line\n登录超时 login timeout\nthird line\n"},
                    {"id": "b", "source": "other", "start_line": 1,
                     "text": "unrelated data\n"},
                ], max_chars=3000, store_dir=tmp)
            self.assertEqual(result["status"], "ok")
            self.assertEqual(result["method"], "lexical")
            self.assertEqual(result["results"][0]["id"], "a")
            item = result["results"][0]
            self.assertEqual(item["quote"], "登录超时 login timeout\n")
            self.assertEqual((item["start_line"], item["end_line"]), (8, 8))
            self.assertEqual(restore_lines(tmp, item["sha256"], 1, 3),
                             "first line\n登录超时 login timeout\nthird line\n")
            self.assertLessEqual(len(json.dumps(result, ensure_ascii=False)), 3000)

    def test_filter_preserves_mandatory_error_context_and_recovery(self):
        output = "prelude\nopening\nmissing module detail\nERROR: no module\nnext clue\ntrailer\n"
        with tempfile.TemporaryDirectory() as tmp:
            result = filter_tool_output("python app.py", 2, output, "start app",
                                        ["do not hide first error"], max_chars=2000,
                                        store_dir=tmp)
            self.assertEqual(result["status"], "ok")
            self.assertEqual(result["command"], "python app.py")
            self.assertEqual(result["exit_code"], 2)
            self.assertIn("ERROR: no module", result["first_error"]["text"])
            self.assertIn("missing module detail", result["excerpt"])
            self.assertIn("next clue", result["excerpt"])
            self.assertEqual(restore_lines(tmp, result["sha256"], 1, 6), output)

    def test_budget_shortage_is_explicit_and_not_partial_success(self):
        result = filter_tool_output("python huge.py", 1, "ERROR: root cause\n",
                                    "mandatory goal", ["mandatory constraint"],
                                    max_chars=60)
        self.assertEqual(result["status"], "requires_larger_budget")
        self.assertNotIn("excerpt", result)
        tiny = filter_tool_output("python huge.py", 1, "ERROR: root cause\n",
                                  "mandatory goal", ["mandatory constraint"], max_chars=35)
        self.assertEqual(tiny["status"], "requires_larger_budget")
        self.assertLessEqual(len(json.dumps(tiny, ensure_ascii=False, separators=(",", ":"))), 35)

    def test_failure_retains_full_chained_traceback_and_restores_crlf(self):
        output = "Traceback (most recent call last):\r\n" + "  frame\r\n" * 8 + \
                 "ValueError: outer\r\nDuring handling of the above exception, another exception occurred:\r\n" + \
                 "  chained frame\r\n" * 8 + "RuntimeError: root cause\r\n"
        with tempfile.TemporaryDirectory() as tmp:
            result = filter_tool_output("cmd", 1, output, "goal", [],
                                        max_chars=10000, store_dir=tmp)
            self.assertEqual(result["method"], "full_failure_output")
            self.assertEqual(result["excerpt"], output)
            self.assertEqual(result["first_error"]["line"], 1)
            self.assertEqual(restore_lines(tmp, result["sha256"], 1, len(output.splitlines())), output)

    def test_zero_exit_error_marker_retains_entire_output(self):
        output = "before\n" + "ordinary diagnostic detail\n" * 7 + "ERROR: root cause\nafter\n"
        result = filter_tool_output("cmd", 0, output, "goal", [], max_chars=10000)
        self.assertEqual(result["method"], "full_failure_output")
        self.assertEqual(result["excerpt"], output)
        self.assertEqual(result["first_error"]["text"], "ERROR: root cause\n")

    def test_nonzero_without_error_keyword_retains_all_and_budget_fails_closed(self):
        output = "ordinary stderr detail\n" * 12
        result = filter_tool_output("cmd", 3, output, "goal", [], max_chars=10000)
        self.assertEqual(result["method"], "full_failure_output")
        self.assertEqual(result["excerpt"], output)
        small = filter_tool_output("cmd", 3, output, "goal", [], max_chars=100)
        self.assertEqual(small["status"], "requires_larger_budget")
        self.assertNotIn("excerpt", small)

    def test_secrets_never_persist_or_reach_callback(self):
        seen = []

        def scorer(query, candidates):
            seen.append((query, candidates))
            return {"a": 0.9}

        with tempfile.TemporaryDirectory() as tmp:
            result = rank_evidence(
                "Bearer querysecret", [{"id": "a", "source": "allowed",
                                        "text": "Authorization: Bearer abcd1234\npassword=hunter2\nvalue secretliteral\n"}],
                max_chars=2000, store_dir=tmp, known_secrets=["secretliteral"],
                score_callback=scorer)
            self.assertEqual(result["status"], "ok")
            payload = json.dumps(seen) + json.dumps(result)
            disk = "".join(p.read_text(encoding="utf-8") for p in Path(tmp).rglob("*"))
            for secret in ["abcd1234", "hunter2", "secretliteral", "querysecret"]:
                self.assertNotIn(secret, payload)
                self.assertNotIn(secret, disk)
            self.assertIn("[REDACTED", disk)

    def test_private_key_block_cannot_be_persisted(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                rank_evidence("key", [{"id": "a", "source": "x", "text":
                    "-----BEGIN PRIVATE KEY-----\nabc\n-----END PRIVATE KEY-----\n"}],
                    max_chars=2000, store_dir=tmp)
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_hash_tamper_and_path_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = rank_evidence("x", [{"id": "a", "source": "x", "text": "x\n"}],
                                   max_chars=2000, store_dir=tmp)
            digest = result["results"][0]["sha256"]
            with self.assertRaises(ValueError):
                restore_lines(tmp, "../" + digest, 1, 1)
            Path(tmp, digest + ".txt").write_text("tampered", encoding="utf-8")
            with self.assertRaises(ValueError):
                restore_lines(tmp, digest, 1, 1)

    def test_crlf_and_repeated_store_preserve_exact_bytes(self):
        text = "first\r\nsecond\r\n"
        with tempfile.TemporaryDirectory() as tmp:
            for _ in range(2):
                result = rank_evidence("second", [{"id": "a", "source": "x", "text": text}],
                                       max_chars=1500, store_dir=tmp)
                digest = result["results"][0]["sha256"]
                self.assertEqual(restore_lines(tmp, digest, 1, 2), text)

    def test_sensitive_identifier_rejected_before_callback(self):
        called = []
        with self.assertRaises(ValueError):
            suggest_skills("task", [{"id": "knownsecret", "description": "x"}],
                           max_chars=1500, known_secrets=["knownsecret"],
                           decision_callback=lambda *args: called.append(args))
        self.assertEqual(called, [])

    def test_quoted_key_dictionary_secrets_do_not_reach_disk_or_callback(self):
        text = '{"api_key": "secret-123", "password": "hunter2"}\n' \
               "{'token': 'python-secret'}\nuseful diagnosis\n"
        seen = []

        def scorer(query, candidates):
            seen.append((query, candidates))
            return {"evidence": 0.8}

        with tempfile.TemporaryDirectory() as tmp:
            result = rank_evidence("useful", [{"id": "evidence", "source": "log", "text": text}],
                                   max_chars=2000, store_dir=tmp, score_callback=scorer)
            digest = result["results"][0]["sha256"]
            stored = restore_lines(tmp, digest, 1, 3)
            visible = json.dumps(result) + json.dumps(seen) + stored
            for secret in ("secret-123", "hunter2", "python-secret"):
                self.assertNotIn(secret, visible)
            self.assertIn("[REDACTED]", stored)

    def test_pattern_sensitive_ids_rejected_without_rewriting(self):
        called = []
        with self.assertRaises(ValueError):
            rank_evidence("task", [{"id": "token:secret-123", "source": "s", "text": "x"}],
                          max_chars=2000, score_callback=lambda *args: called.append(args))
        with self.assertRaises(ValueError):
            suggest_skills("task", [{"id": "token:secret-123", "description": "x"}],
                           max_chars=2000, decision_callback=lambda *args: called.append(args))
        self.assertEqual(called, [])

    def test_redaction_preserves_original_line_numbers(self):
        text = "first\n-----BEGIN PRIVATE KEY-----\nkeyvalue\n-----END PRIVATE KEY-----\n" \
               "secret\nchunk\nTARGET\n"
        result = rank_evidence("TARGET", [{"id": "evidence", "source": "file",
                                           "start_line": 20, "text": text}],
                               max_chars=2000, known_secrets=["secret\nchunk"])
        self.assertEqual(result["results"][0]["start_line"], 26)
        self.assertEqual(result["results"][0]["quote"], "TARGET\n")

    def test_no_error_does_not_invent_first_error(self):
        output = "step one\nstep two\nstep three\nstep four\n"
        result = filter_tool_output("cmd", 0, output, "check", [], max_chars=2000)
        self.assertIsNone(result["first_error"])
        self.assertEqual(result["excerpt"], "step one\nstep two\nstep three\n")

    def test_closed_set_and_callback_validity(self):
        skills = [{"id": "pdf", "description": "Read PDF documents"},
                  {"id": "code", "description": "Review code"}]
        result = suggest_skills("review code", skills, max_chars=1500)
        self.assertEqual(result["method"], "lexical")
        self.assertEqual(result["suggestions"][0]["id"], "code")
        self.assertFalse(result["execution_authorized"])
        for bad in [{"pdf": 0.5, "unknown": 0.5}, {"pdf": float("nan"), "code": 0.1},
                    {"pdf": True, "code": 0.1}]:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                suggest_skills("x", skills, max_chars=1500,
                               decision_callback=lambda task, candidates, b=bad: b)
        with self.assertRaises(ValueError):
            rank_evidence("x", [{"id": "a", "source": "s", "text": "x"}],
                          max_chars=1500, score_callback=lambda q, c: {"unknown": 1.0})

    def test_callback_failure_is_reported_without_hidden_fallback(self):
        def broken(query, candidates):
            raise RuntimeError("route down")
        with self.assertRaises(ModelRouteError):
            rank_evidence("x", [{"id": "a", "source": "s", "text": "x"}],
                          max_chars=1500, score_callback=broken)


if __name__ == "__main__":
    unittest.main()
