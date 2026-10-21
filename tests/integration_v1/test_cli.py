from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

from jevlocal.integration.cli import main


class CLITests(unittest.TestCase):
    def call(self, args):
        stream = io.StringIO()
        with redirect_stdout(stream):
            status = main(args)
        return status, json.loads(stream.getvalue())

    def test_doctor_does_not_claim_model_or_host_acceptance(self):
        status, result = self.call(["doctor"])
        self.assertEqual(status, 0)
        self.assertFalse(result["native_model_validated"])
        self.assertFalse(result["execution_authorized"])
        self.assertEqual(result["active_browser_core"], "fastloop")
        self.assertEqual(result["codex_host_browser_adapter"], "not_connected")

    def test_cli_context_is_explicit_and_does_not_execute_command(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)/"input.json"
            result_path = Path(directory)/"output.json"
            source.write_text(json.dumps({"command": "DO_NOT_EXECUTE", "exit_code": 3,
                "output": "warning\nError: expected fixture\nafter\n", "goal": "check fixture",
                "constraints": ["no command execution"], "max_chars": 2000}), encoding="utf-8")
            status, result = self.call(["context", "filter_tool_output", "--input", str(source), "--output", str(result_path)])
            self.assertEqual(status, 0)
            self.assertEqual(result["exit_code"], 3)
            self.assertEqual(result["first_error"]["line"], 2)
            self.assertEqual(json.loads(result_path.read_text(encoding="utf-8")), result)
            status, _ = self.call(["context", "filter_tool_output", "--input", str(source), "--output", str(result_path)])
            self.assertEqual(status, 2)

    def test_invalid_wire_input_fails_without_echoing_private_content(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"input.json"
            path.write_text('{"private-secret-fixture": 1}', encoding="utf-8")
            status, result = self.call(["validate-envelope", "--input", str(path)])
            self.assertEqual(status, 2)
            self.assertNotIn("private-secret-fixture", json.dumps(result))

    def test_restore_cli_accepts_string_result(self):
        from jevlocal.integration.context_tools import filter_tool_output
        with tempfile.TemporaryDirectory() as directory:
            result = filter_tool_output("fixture", 0, "line one\nline two\n", "check", [],
                                        max_chars=2000, store_dir=directory)
            path = Path(directory)/"request.json"
            path.write_text(json.dumps({"store_dir": directory, "sha256": result["sha256"],
                                        "start_line": 1, "end_line": 2}), encoding="utf-8")
            status, output = self.call(["context", "restore_lines", "--input", str(path)])
            self.assertEqual(status, 0)
            self.assertEqual(output, "line one\nline two\n")


if __name__ == "__main__":
    unittest.main()
