"""Focused, public-fixture checks for the Windows handle navigation boundary."""
from __future__ import annotations

import os
import ctypes
from ctypes import wintypes
from pathlib import Path
import shutil
import struct
import tempfile
import unittest
from contextlib import contextmanager
from unittest.mock import patch

from jevlocal.integration.windows_repository_handles import (
    WindowsDirectoryLease, _parse_file_id_both_dir_info,
)


_STAGE = Path(__file__).resolve().parents[2] / "runtime" / "f06-windows-handle-navigation-01"


@contextmanager
def _public_fixture():
    root = Path(tempfile.mkdtemp(prefix="public-handle-", dir=_STAGE))
    try:
        yield root
    finally:
        # This deletes only the newly made test fixture, never a candidate tree.
        if root.resolve() != root.absolute() or root.parent.resolve() != _STAGE.resolve():
            raise AssertionError("fixture cleanup target escaped stage")
        shutil.rmtree(root)


def _assert_inside(root: Path, path: Path):
    if not path.resolve(strict=False).is_relative_to(root.resolve()):
        raise AssertionError("fixture operation escaped its root")


def _record(name: str, next_offset: int = 0) -> bytes:
    encoded = name.encode("utf-16-le")
    data = bytearray(max(104 + len(encoded), next_offset))
    struct.pack_into("<II", data, 0, next_offset, 7)
    struct.pack_into("<I", data, 60, len(encoded))
    data[104:104 + len(encoded)] = encoded
    return bytes(data)


class DirectoryRecordTests(unittest.TestCase):
    def test_valid_linked_records(self):
        first = _record("one", 112)
        self.assertEqual(_parse_file_id_both_dir_info(first + _record("二")), ["one", "二"])

    def test_rejects_bad_offsets_and_utf16(self):
        for data in (_record("one", 106), _record("one", 65544),
                     _record("one")[:105], _record("one")[:104] + b"\x00\xd8"):
            with self.subTest(data=data[:12]), self.assertRaises(ValueError):
                _parse_file_id_both_dir_info(data)


@unittest.skipUnless(os.name == "nt", "actual Windows handles required")
class WindowsLeaseTests(unittest.TestCase):
    def test_access_and_sharing_failures_reject_instead_of_disappearing(self):
        with _public_fixture() as root:
            with WindowsDirectoryLease(root) as lease:
                for status in (0xC0000022, 0xC0000043):
                    with self.subTest(status=hex(status)):
                        with patch.object(lease._ntdll, "NtOpenFile", return_value=status):
                            with self.assertRaises(OSError):
                                lease.child_metadata(root, "inaccessible.txt")

    def test_process_handle_count_returns_to_baseline(self):
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        kernel.GetProcessHandleCount.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel.GetProcessHandleCount.restype = wintypes.BOOL
        process = kernel.GetCurrentProcess()

        def count():
            value = wintypes.DWORD()
            self.assertTrue(kernel.GetProcessHandleCount(process, ctypes.byref(value)))
            return value.value

        with _public_fixture() as root:
            (root / "branch").mkdir()
            baseline = count()
            with WindowsDirectoryLease(root) as lease:
                lease.chain_identities(root / "branch")
                self.assertGreater(count(), baseline)
            self.assertEqual(count(), baseline)

    def test_candidate_directory_pinned_before_descent(self):
        with _public_fixture() as root:
            branch, target = root / "branch", root / "renamed"
            branch.mkdir()
            with WindowsDirectoryLease(root) as lease:
                self.assertEqual(lease.child_metadata(root, "branch")["kind"], "directory")
                _assert_inside(root, branch)
                _assert_inside(root, target)
                with self.assertRaises(OSError):
                    os.rename(branch, target)

    def test_metadata_only_file_replacement_is_not_a_cached_name_binding(self):
        with _public_fixture() as root:
            original, moved = root / "public.txt", root / "renamed.txt"
            original.write_text("public original", encoding="utf-8")
            with WindowsDirectoryLease(root) as lease:
                first = lease.child_metadata(root, "public.txt")
                _assert_inside(root, original)
                _assert_inside(root, moved)
                os.rename(original, moved)
                original.write_text("public replacement", encoding="utf-8")
                second = lease.child_metadata(root, "public.txt")
                self.assertNotEqual(first["identity"]["inode"], second["identity"]["inode"])

    def test_nested_public_fixture_and_closure(self):
        with _public_fixture() as root:
            (root / "branch").mkdir()
            (root / "branch" / "plain.txt").write_text("public fixture", encoding="utf-8")
            lease = WindowsDirectoryLease(root)
            with lease:
                root_chain = lease.chain_identities(root)
                chain = lease.chain_identities(root / "branch")
                self.assertEqual(chain[0], root_chain[0])
                self.assertEqual(len(chain), 2)
                with patch("os.scandir", side_effect=AssertionError("path enumeration fallback")):
                    self.assertIn("plain.txt", list(lease.iter_names(root / "branch")))
                result = lease.child_metadata(root / "branch", "plain.txt")
                self.assertEqual(result["kind"], "file")
                self.assertEqual(set(result["identity"]),
                                 {"device", "inode", "size", "mtime_ns", "ctime_ns"})
                self.assertGreater(lease.active_handle_count, 0)
            self.assertEqual(lease.opened_handle_count, lease.closed_handle_count)
            self.assertEqual(lease.active_handle_count, 0)

    def test_pinned_identity_refresh_and_rename_conflict(self):
        with _public_fixture() as root:
            branch = root / "branch"
            target = root / "renamed"
            branch.mkdir()
            file_path = branch / "public.txt"
            file_path.write_text("a", encoding="utf-8")
            with WindowsDirectoryLease(root) as lease:
                first = lease.child_metadata(branch, "public.txt")
                opened = lease.opened_handle_count
                file_path.write_text("longer public text", encoding="utf-8")
                second = lease.child_metadata(branch, "public.txt")
                self.assertGreater(lease.opened_handle_count, opened)
                self.assertNotEqual(first["identity"]["size"], second["identity"]["size"])
                directory_identity = lease.chain_identities(branch)[-1]
                _assert_inside(root, branch)
                _assert_inside(root, target)
                with self.assertRaises(OSError) as conflict:
                    os.rename(branch, target)
                self.assertIn(conflict.exception.winerror, (32, 33))
                self.assertEqual(lease.chain_identities(branch)[-1]["inode"],
                                 directory_identity["inode"])
            self.assertTrue(branch.is_dir())

    def test_reparse_child_rejected_when_symlink_available(self):
        with _public_fixture() as root:
            destination = root / "target"
            destination.mkdir()
            link = root / "link"
            try:
                os.symlink(destination, link, target_is_directory=True)
            except OSError as exc:
                self.skipTest(f"local symlink creation unavailable: {exc}")
            with WindowsDirectoryLease(root) as lease:
                self.assertIsNone(lease.child_metadata(root, "link"))
                with self.assertRaises(ValueError):
                    lease.chain_identities(link)

    def test_invalid_child_component_and_closed_lease(self):
        with _public_fixture() as root:
            lease = WindowsDirectoryLease(root)
            with lease:
                for name in ("..", "bad/name", "CON", "name.", "name:stream"):
                    with self.subTest(name=name), self.assertRaises(ValueError):
                        lease.child_metadata(root, name)
            with self.assertRaises(RuntimeError):
                lease.iter_names(root).__next__()

    def test_partial_initialization_closes_drive_and_ancestors(self):
        with _public_fixture() as root:
            missing = root / "not-created"
            lease = WindowsDirectoryLease(missing)
            with self.assertRaises(ValueError):
                with lease:
                    pass
            self.assertGreater(lease.opened_handle_count, 0)
            self.assertEqual(lease.opened_handle_count, lease.closed_handle_count)
            self.assertEqual(lease.active_handle_count, 0)


if __name__ == "__main__":
    unittest.main()
