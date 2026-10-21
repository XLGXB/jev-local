"""Optional explicit context tools. Does not configure Codex or control a browser."""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from . import context_tools


def create_server(store_dir: str | Path, *, native_runtime: dict | None = None,
                  passive_runtime: dict | None = None,
                  passive_records: str | Path | None = None):
    from mcp.server.fastmcp import FastMCP
    from mcp.types import ToolAnnotations

    # Preserve the configured path for each operation's symlink check; resolving
    # it here would hide a symlink root from the storage boundary.
    root = Path(store_dir)
    if not root.is_absolute() or root.is_symlink():
        raise ValueError("store must explicitly name an absolute non-symlink owned directory")
    if (passive_runtime is None) != (passive_records is None):
        raise ValueError("passive runtime and records must be configured together")
    if native_runtime is not None and passive_runtime is not None:
        raise ValueError("active-native and passive-native modes are mutually exclusive")
    scorer = None
    if native_runtime is not None:
        from .native_context import build_native_scorer
        scorer = build_native_scorer(native_runtime)
    passive = None
    if passive_runtime is not None:
        from .passive_session import PassiveSession
        passive = PassiveSession(passive_runtime, passive_records)

    @asynccontextmanager
    async def lifespan(_server):
        try:
            yield {}
        finally:
            if passive is not None:
                passive.close()

    server = FastMCP("jevlocal-context-tools", lifespan=lifespan)
    local_write = ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=True, openWorldHint=False)
    local_read = ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False)

    @server.tool(annotations=local_read)
    def scope_evidence(query: str, candidates: list[dict], target: dict, bindings: dict,
                       max_chars: int = 65536, rank_max_chars: int = 12000) -> dict:
        """Partition caller-declared object/revision applicability before ranking.
        Bindings must include each original candidate_digest; never infer them from text.
        Keeps unknowns and mismatches in a full snapshot. Matching is not relevance,
        authenticated provenance, freshness or authorization. No files/models invoked.
        """
        from .evidence_scope import scope_evidence as prepare
        return prepare(query, candidates, target, bindings, max_chars=max_chars,
                       rank_max_chars=rank_max_chars)

    @server.tool(annotations=local_write)
    def rank_evidence(query: str, candidates: list[dict], max_chars: int = 12000) -> dict:
        """Rank explicitly supplied authorized excerpts, preserving cited lines.
        Uses the operator-selected route (lexical by default, optional active native scoring).
        Active native scoring can change ranking; cited-line selection remains lexical.
        Passive observation is available only through explicit passive_start.
        Never supply credentials or private account data. Source labels are not paths to read.
        """
        arguments = dict(query=query, candidates=candidates, max_chars=max_chars, store_dir=root)
        if scorer is not None:
            from .native_context import invoke_context
            return invoke_context("rank_evidence", arguments, scorer)
        return context_tools.rank_evidence(**arguments)

    @server.tool(annotations=local_write)
    def filter_tool_output(command: str, exit_code: int, output: str, goal: str,
                           constraints: list[str], max_chars: int = 12000) -> dict:
        """Keep mandatory context and a first diagnostic window; save only sanitized text.
        This receives existing output; it never executes command. Do not pass secrets.
        """
        return context_tools.filter_tool_output(command, exit_code, output, goal, constraints,
                                                max_chars=max_chars, store_dir=root)

    @server.tool(annotations=local_read)
    def suggest_skills(task: str, enabled_skills: list[dict], max_chars: int = 8000) -> dict:
        """Suggest only caller-supplied enabled IDs. Never install, run, or approve a skill."""
        arguments = dict(task=task, enabled_skills=enabled_skills, max_chars=max_chars)
        if scorer is not None:
            from .native_context import invoke_context
            return invoke_context("suggest_skills", arguments, scorer)
        return context_tools.suggest_skills(**arguments)

    @server.tool(annotations=local_read)
    def restore_evidence(sha256: str, start_line: int, end_line: int) -> str:
        """Recover exact sanitized lines from the configured owned store and verify their hash."""
        return context_tools.restore_lines(root, sha256, start_line, end_line)

    if passive is not None:
        local_start = ToolAnnotations(readOnlyHint=False, destructiveHint=False,
                                      idempotentHint=False, openWorldHint=False)
        local_cancel = ToolAnnotations(readOnlyHint=False, destructiveHint=False,
                                       idempotentHint=True, openWorldHint=False)

        @server.tool(annotations=local_start)
        def passive_start(query: str, candidates: list[dict], target: dict,
                          bindings: dict, max_chars: int = 65536,
                          rank_max_chars: int = 12000) -> dict:
            """Publish a scoped lexical primary and start one advisory observer.
            Returns a fresh invocation ID; never waits for native inference.
            Supply only caller-selected evidence, not files or secrets.
            """
            return passive.start(query, candidates, target, bindings,
                                 max_chars=max_chars,
                                 rank_max_chars=rank_max_chars)

        @server.tool(annotations=local_read)
        def passive_collect(invocation_id: str) -> dict:
            """Read one explicitly named saved invocation, without inference."""
            return passive.collect(invocation_id)

        @server.tool(annotations=local_cancel)
        def passive_cancel(invocation_id: str) -> dict:
            """Signal cancellation; the sole finalizer owns process cleanup."""
            return passive.cancel(invocation_id)

        @server.tool(annotations=local_read)
        def passive_replay(invocation_id: str) -> dict:
            """Pure consistency replay of one complete saved invocation."""
            return passive.replay(invocation_id)

    return server


def main():
    path = os.environ.get("JEVLOCAL_CONTEXT_STORE")
    if not path or not Path(path).is_absolute():
        raise SystemExit("JEVLOCAL_CONTEXT_STORE must explicitly name an absolute owned directory")
    config = os.environ.get("JEVLOCAL_CONTEXT_NATIVE_RUNTIME")
    passive_config = os.environ.get("JEVLOCAL_CONTEXT_PASSIVE_RUNTIME")
    records = os.environ.get("JEVLOCAL_CONTEXT_PASSIVE_RECORDS")
    if bool(passive_config) != bool(records):
        raise SystemExit("JEVLOCAL_CONTEXT_PASSIVE_RUNTIME and JEVLOCAL_CONTEXT_PASSIVE_RECORDS must both be set")
    if config and passive_config:
        raise SystemExit("active-native and passive-native modes are mutually exclusive")
    native_runtime = None
    if config:
        if not Path(config).is_absolute():
            raise SystemExit("JEVLOCAL_CONTEXT_NATIVE_RUNTIME must name an absolute configuration file")
        from .cli import _read
        native_runtime = _read(config)
    passive_runtime = None
    if passive_config:
        if not Path(passive_config).is_absolute() or not Path(records).is_absolute():
            raise SystemExit("passive runtime file and records directory must be absolute")
        from .cli import _read
        passive_runtime = _read(passive_config)
    create_server(path, native_runtime=native_runtime,
                  passive_runtime=passive_runtime,
                  passive_records=records).run(transport="stdio")


if __name__ == "__main__":
    main()
