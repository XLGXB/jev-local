"""MCP v1.x bridge. No approval issuer/private key is exposed as a tool."""
import os
from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations
from .service import config, invoke, targets

mcp = FastMCP("fast-browser-state-machine-v2")
READ = ToolAnnotations(readOnlyHint=True)
WRITE = ToolAnnotations(readOnlyHint=False, destructiveHint=True, idempotentHint=False, openWorldHint=True)


def registry():
    path = os.environ.get("FASTLOOP_REGISTRY")
    if not path:
        raise RuntimeError("FASTLOOP_REGISTRY required")
    return path


@mcp.tool(annotations=READ)
async def browser_targets() -> list[dict]:
    """List explicit target IDs; not default pages[0]."""
    return await targets(config(registry()))


@mcp.tool(annotations=READ)
async def browser_status(run_id: str) -> dict:
    """Read the persisted bounded-run status. Does not resume or approve anything."""
    return await invoke(registry(), "status", run_id)


@mcp.tool(annotations=WRITE)
async def browser_run(task_id: str, target_id: str, run_id: str, dry_run: bool = True) -> dict:
    """Run a preapproved subgoal. Keep run_id stable on retry. awaiting_form transfers
    control, NOT authorization. needs_verification is NOT final acceptance."""
    return await invoke(registry(), "start", run_id, task_id=task_id, target_id=target_id, dry_run=dry_run)


@mcp.tool(annotations=WRITE)
async def browser_form_fill(run_id: str, grant: dict, values: dict) -> dict:
    """Fill only exact ordinary fields/values covered by a separately signed, single-use
    approval. No passwords/OTP/payment fields, no submit. Returns ready_to_resume.
    Values may appear in the host's tool transcript; never pass secrets."""
    return await invoke(registry(), "fill", run_id, token=grant, values=values)


@mcp.tool(annotations=WRITE)
async def browser_refresh_form(run_id: str) -> dict:
    """Reobserve a paused form and replace its nonce/scope. Invalidates prior grants.
    A new independent approval is required; budgets are NOT reset."""
    return await invoke(registry(), "refresh", run_id)


@mcp.tool(annotations=WRITE)
async def browser_resume(run_id: str) -> dict:
    """Resume only READY, with the same target/task/budgets and revalidated DOM.
    Never use a fresh run ID to replay an unknown effect."""
    return await invoke(registry(), "resume", run_id)


@mcp.tool(annotations=WRITE)
async def browser_cancel(run_id: str) -> dict:
    """Cancel a paused run. Unknown effects stay quarantined pending manual review."""
    return await invoke(registry(), "cancel", run_id)


if __name__ == "__main__":
    mcp.run(transport="stdio")
