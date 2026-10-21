"""Fail-closed Windows directory lease for metadata-only repository navigation.

The module is importable off Windows, but lease creation requires native Win32 and
ntdll APIs. It never opens candidate file data or substitutes pathname traversal.
"""
from __future__ import annotations

import ctypes
from ctypes import wintypes
import os
from pathlib import Path
import re
import struct


_MAX_HANDLES = 512
_MAX_COMPONENTS = 256
_DIR_BUFFER_SIZE = 65536
_FILETIME_EPOCH = 116444736000000000
_INVALID_HANDLE = ctypes.c_void_p(-1).value
_FILE_ATTRIBUTE_DIRECTORY = 0x10
_FILE_ATTRIBUTE_DEVICE = 0x40
_FILE_ATTRIBUTE_REPARSE_POINT = 0x400
_READ_ATTRIBUTES = 0x80
_LIST_DIRECTORY = 0x1
_SYNCHRONIZE = 0x00100000
_SHARE_READ_WRITE = 0x1 | 0x2
_OPEN_EXISTING = 3
_BACKUP_SEMANTICS = 0x02000000
_OPEN_REPARSE_POINT = 0x00200000
_OBJ_CASE_INSENSITIVE = 0x40
_OBJ_DONT_REPARSE = 0x1000
_FILE_DIRECTORY_FILE = 0x1
_FILE_OPEN_REPARSE_POINT = 0x200000
_FILE_SYNCHRONOUS_IO_NONALERT = 0x20
_ERROR_NO_MORE_FILES = 18
_FILE_TYPE_DISK = 1
# Absent entries and blocked reparse lookups may be excluded. Access/sharing
# failures instead reject the walk; they must not silently shrink its scope.
_UNAVAILABLE_NTSTATUS = {0xC0000034, 0xC000003A, 0xC000050B, 0xC0000275}
_RESERVED = re.compile(r"(?i)(?:con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\..*)?\Z")


class _UnicodeString(ctypes.Structure):
    _fields_ = [("Length", wintypes.USHORT), ("MaximumLength", wintypes.USHORT),
                ("Buffer", ctypes.c_wchar_p)]


class _ObjectAttributes(ctypes.Structure):
    _fields_ = [("Length", wintypes.ULONG), ("RootDirectory", wintypes.HANDLE),
                ("ObjectName", ctypes.POINTER(_UnicodeString)),
                ("Attributes", wintypes.ULONG),
                ("SecurityDescriptor", wintypes.LPVOID),
                ("SecurityQualityOfService", wintypes.LPVOID)]


class _FileBasicInfo(ctypes.Structure):
    _fields_ = [("CreationTime", ctypes.c_int64), ("LastAccessTime", ctypes.c_int64),
                ("LastWriteTime", ctypes.c_int64), ("ChangeTime", ctypes.c_int64),
                ("FileAttributes", wintypes.DWORD)]


class _FileStandardInfo(ctypes.Structure):
    _fields_ = [("AllocationSize", ctypes.c_int64), ("EndOfFile", ctypes.c_int64),
                ("NumberOfLinks", wintypes.DWORD), ("DeletePending", wintypes.BOOLEAN),
                ("Directory", wintypes.BOOLEAN)]


class _FileIdInfo(ctypes.Structure):
    _fields_ = [("VolumeSerialNumber", ctypes.c_uint64),
                ("FileId", ctypes.c_ubyte * 16)]


class _IoStatusBlock(ctypes.Structure):
    _fields_ = [("Status", ctypes.c_void_p), ("Information", ctypes.c_void_p)]


def _parse_file_id_both_dir_info(data: bytes) -> list[str]:
    """Parse bounded FILE_ID_BOTH_DIR_INFO chain; never trust native offsets."""
    if not data or len(data) > _DIR_BUFFER_SIZE:
        raise ValueError("invalid directory information buffer")
    names = []
    offset = 0
    while True:
        if len(data) - offset < 104:
            raise ValueError("short FILE_ID_BOTH_DIR_INFO record")
        next_offset, _index = struct.unpack_from("<II", data, offset)
        name_length = struct.unpack_from("<I", data, offset + 60)[0]
        short_length = data[offset + 68]
        if (short_length > 24 or short_length % 2 or name_length == 0
                or name_length > 510 or name_length % 2):
            raise ValueError("invalid directory name length")
        record_length = next_offset or len(data) - offset
        if (record_length < 104 + name_length or record_length > len(data) - offset
                or (next_offset and (next_offset < 104 or next_offset % 8))):
            raise ValueError("invalid directory record offset")
        raw = data[offset + 104:offset + 104 + name_length]
        try:
            name = raw.decode("utf-16-le", errors="strict")
        except UnicodeError as exc:
            raise ValueError("invalid directory name UTF-16") from exc
        if "\x00" in name:
            raise ValueError("embedded NUL in directory name")
        if name not in (".", ".."):
            names.append(name)
        if not next_offset:
            return names
        offset += next_offset


def _component(name: str) -> str:
    if (type(name) is not str or not name or name in (".", "..")
            or len(name.encode("utf-16-le")) > 510 or name[-1] in ". "
            or any(c in name for c in "\\/:\x00") or _RESERVED.fullmatch(name)):
        raise ValueError("noncanonical Windows path component")
    return name


class WindowsDirectoryLease:
    """Retain drive/root/ancestor/child handles until the advisory walk ends."""

    def __init__(self, repository_root: Path):
        if os.name != "nt":
            raise OSError("Windows native handle APIs unavailable")
        self.root = Path(repository_root)
        raw = str(self.root)
        if (not re.match(r"^[A-Za-z]:\\", raw) or raw.startswith("\\\\")
                or ":" in raw[2:] or any(part in (".", "..") for part in self.root.parts[1:])):
            raise ValueError("canonical local drive root required")
        self._drive = raw[:3]
        self._root_parts = tuple(_component(p) for p in self.root.parts[1:])
        if len(self._root_parts) > _MAX_COMPONENTS:
            raise ValueError("path component budget exceeded")
        self._kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self._ntdll = ctypes.WinDLL("ntdll", use_last_error=True)
        self._configure_apis()
        self._handles: list[int] = []
        self._directories: dict[tuple[str, ...], tuple[int, dict]] = {}
        self._children: dict[tuple[tuple[str, ...], str], tuple[int, dict, str]] = {}
        self.opened_handle_count = 0
        self.closed_handle_count = 0
        self.active_handle_count = 0
        self._entered = False
        self._closed = False

    def _configure_apis(self) -> None:
        k = self._kernel
        n = self._ntdll
        k.GetDriveTypeW.argtypes = [wintypes.LPCWSTR]
        k.GetDriveTypeW.restype = wintypes.UINT
        k.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                                  wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD,
                                  wintypes.HANDLE]
        k.CreateFileW.restype = wintypes.HANDLE
        k.GetFileType.argtypes = [wintypes.HANDLE]
        k.GetFileType.restype = wintypes.DWORD
        k.GetFileInformationByHandleEx.argtypes = [wintypes.HANDLE, ctypes.c_int,
                                                   wintypes.LPVOID, wintypes.DWORD]
        k.GetFileInformationByHandleEx.restype = wintypes.BOOL
        k.CloseHandle.argtypes = [wintypes.HANDLE]
        k.CloseHandle.restype = wintypes.BOOL
        n.NtOpenFile.argtypes = [ctypes.POINTER(wintypes.HANDLE), wintypes.DWORD,
                                 ctypes.POINTER(_ObjectAttributes), ctypes.POINTER(_IoStatusBlock),
                                 wintypes.DWORD, wintypes.DWORD]
        n.NtOpenFile.restype = ctypes.c_long

    def _retain(self, handle: int) -> int:
        if not handle or handle == _INVALID_HANDLE:
            raise OSError("invalid native handle")
        self._handles.append(handle)
        self.opened_handle_count += 1
        self.active_handle_count += 1
        return handle

    def _room(self) -> None:
        if self.active_handle_count >= _MAX_HANDLES:
            raise ValueError("active handle budget exceeded")

    def _release_probe(self, handle: int) -> None:
        """Close a redundant probe, retaining the original candidate anchor."""
        if not self._kernel.CloseHandle(handle):
            raise OSError(ctypes.get_last_error(), "CloseHandle probe failed")
        self._handles.remove(handle)
        self.closed_handle_count += 1
        self.active_handle_count -= 1

    def _info(self, handle: int) -> tuple[dict, str]:
        if self._kernel.GetFileType(handle) != _FILE_TYPE_DISK:
            raise ValueError("non-disk native handle")
        basic, standard, file_id = _FileBasicInfo(), _FileStandardInfo(), _FileIdInfo()
        for cls, obj in ((0, basic), (1, standard), (18, file_id)):
            if not self._kernel.GetFileInformationByHandleEx(handle, cls, ctypes.byref(obj), ctypes.sizeof(obj)):
                raise OSError(ctypes.get_last_error(), "handle metadata unavailable")
        if basic.FileAttributes & (_FILE_ATTRIBUTE_REPARSE_POINT | _FILE_ATTRIBUTE_DEVICE) or standard.DeletePending:
            raise ValueError("reparse or deletion-pending object forbidden")
        kind = "directory" if standard.Directory else "file"
        if bool(basic.FileAttributes & _FILE_ATTRIBUTE_DIRECTORY) != bool(standard.Directory):
            raise ValueError("inconsistent native directory metadata")
        if standard.EndOfFile < 0:
            raise ValueError("invalid native file size")
        identity = {"device": int(file_id.VolumeSerialNumber),
                    "inode": int.from_bytes(bytes(file_id.FileId), "little"),
                    "size": int(standard.EndOfFile),
                    "mtime_ns": (int(basic.LastWriteTime) - _FILETIME_EPOCH) * 100,
                    "ctime_ns": (int(basic.ChangeTime) - _FILETIME_EPOCH) * 100}
        return identity, kind

    def _open_relative(self, parent: int, name: str, directory: bool) -> int | None:
        _component(name)
        self._room()
        buffer = ctypes.create_unicode_buffer(name)
        length = len(name.encode("utf-16-le"))
        unicode_name = _UnicodeString(length, length, ctypes.cast(buffer, ctypes.c_wchar_p))
        attrs = _ObjectAttributes(ctypes.sizeof(_ObjectAttributes), parent,
                                  ctypes.pointer(unicode_name),
                                  _OBJ_CASE_INSENSITIVE | _OBJ_DONT_REPARSE, None, None)
        iosb = _IoStatusBlock()
        handle = wintypes.HANDLE()
        access = _READ_ATTRIBUTES | _SYNCHRONIZE | (_LIST_DIRECTORY if directory else 0)
        options = _FILE_SYNCHRONOUS_IO_NONALERT | _FILE_OPEN_REPARSE_POINT
        if directory:
            options |= _FILE_DIRECTORY_FILE
        status = int(self._ntdll.NtOpenFile(ctypes.byref(handle), access,
                                           ctypes.byref(attrs), ctypes.byref(iosb),
                                           _SHARE_READ_WRITE, options)) & 0xFFFFFFFF
        if status:
            if status in _UNAVAILABLE_NTSTATUS:
                return None
            raise OSError(f"NtOpenFile failed: NTSTATUS 0x{status:08X}")
        return self._retain(handle.value)

    def __enter__(self) -> "WindowsDirectoryLease":
        if self._entered or self._closed:
            raise RuntimeError("lease cannot be re-entered")
        self._entered = True
        try:
            if self._kernel.GetDriveTypeW(self._drive) not in (2, 3):
                raise ValueError("fixed or removable local drive required")
            self._room()
            handle = self._kernel.CreateFileW(
                self._drive, _LIST_DIRECTORY | _READ_ATTRIBUTES | _SYNCHRONIZE,
                _SHARE_READ_WRITE, None, _OPEN_EXISTING,
                _BACKUP_SEMANTICS | _OPEN_REPARSE_POINT, None)
            if not handle or handle == _INVALID_HANDLE:
                raise OSError(ctypes.get_last_error(), "drive anchor unavailable")
            anchor = self._retain(handle)
            identity, kind = self._info(anchor)
            if kind != "directory":
                raise ValueError("drive anchor not directory")
            self._directories[()] = (anchor, identity)
            self._directory(self._root_parts)
            return self
        except BaseException:
            self.close()
            raise

    def _directory(self, parts: tuple[str, ...]) -> tuple[int, dict]:
        if not self._entered or self._closed:
            raise RuntimeError("lease not active")
        if len(parts) > _MAX_COMPONENTS:
            raise ValueError("path component budget exceeded")
        for count in range(1, len(parts) + 1):
            prefix = parts[:count]
            if prefix in self._directories:
                continue
            parent = self._directories[prefix[:-1]][0]
            handle = self._open_relative(parent, prefix[-1], directory=True)
            if handle is None:
                raise ValueError("navigation directory unavailable")
            identity, kind = self._info(handle)
            if kind != "directory":
                raise ValueError("navigation object not directory")
            pinned = self._children.get((prefix[:-1], prefix[-1].casefold()))
            if pinned and pinned[1] != identity:
                raise ValueError("selected directory identity changed")
            self._directories[prefix] = (handle, identity)
        return self._directories[parts]

    def _parts(self, current_directory: Path) -> tuple[str, ...]:
        current = Path(current_directory)
        try:
            relative = current.relative_to(self.root)
        except ValueError as exc:
            raise ValueError("directory outside repository root") from exc
        parts = self._root_parts + tuple(_component(p) for p in relative.parts)
        if len(parts) > _MAX_COMPONENTS:
            raise ValueError("path component budget exceeded")
        return parts

    def chain_identities(self, current_directory: Path) -> list[dict]:
        parts = self._parts(current_directory)
        chain = []
        for count in range(len(self._root_parts), len(parts) + 1):
            handle, _original = self._directory(parts[:count])
            identity, kind = self._info(handle)
            if kind != "directory":
                raise ValueError("navigation object no longer directory")
            chain.append(identity)
        return chain

    def iter_names(self, current_directory: Path):
        parts = self._parts(current_directory)
        handle, _identity = self._directory(parts)
        info_class = 11
        while True:
            buffer = ctypes.create_string_buffer(_DIR_BUFFER_SIZE)
            if not self._kernel.GetFileInformationByHandleEx(
                    handle, info_class, buffer, _DIR_BUFFER_SIZE):
                error = ctypes.get_last_error()
                if error == _ERROR_NO_MORE_FILES:
                    return
                raise OSError(error, "directory enumeration failed")
            for name in _parse_file_id_both_dir_info(buffer.raw):
                yield name
            info_class = 10

    def child_metadata(self, current_directory: Path, name: str) -> dict | None:
        parts = self._parts(current_directory)
        _component(name)
        parent, _identity = self._directory(parts)
        key = (parts, name.casefold())
        pinned = self._children.get(key)
        if pinned and pinned[2] == "directory":
            identity, kind = self._info(pinned[0])
            return {"kind": kind, "identity": identity}
        # Attribute-only file opens do not establish a sharing lock. Refresh
        # the name relative to its held parent, not just the old object's info.
        handle = self._open_relative(parent, name, directory=False)
        if handle is None:
            return None
        try:
            identity, kind = self._info(handle)
        except ValueError as exc:
            if "reparse or deletion-pending" in str(exc) or "non-disk" in str(exc):
                self._release_probe(handle)
                return None
            raise
        if kind == "directory":
            # Acquire LIST_DIRECTORY access before inference, not only when
            # descending: an attribute-only candidate handle does not prevent
            # a concurrent rename. Compare both opens before publishing it.
            directory_handle, directory_identity = self._directory(parts + (name,))
            if identity != directory_identity:
                raise ValueError("candidate directory identity changed while pinning")
            self._release_probe(handle)
            handle = directory_handle
        elif pinned:
            # A fresh relative name probe is needed on every recheck, but its
            # duplicate handle need not consume the entire walk's budget.
            self._release_probe(handle)
            return {"kind": kind, "identity": dict(identity)}
        self._children[key] = (handle, identity, kind)
        return {"kind": kind, "identity": dict(identity)}

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        first_error = None
        for handle in reversed(self._handles):
            if self._kernel.CloseHandle(handle):
                self.closed_handle_count += 1
                self.active_handle_count -= 1
            elif first_error is None:
                first_error = ctypes.get_last_error()
        self._handles.clear()
        self._directories.clear()
        self._children.clear()
        if first_error is not None:
            raise OSError(first_error, "CloseHandle failed")

    def __exit__(self, _type, _value, _traceback) -> None:
        self.close()
