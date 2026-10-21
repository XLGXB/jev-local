"""MCP v1.x adapter; deliberately pins mcp<2 (v2 has a different server API)."""
import os
from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations
from .common import read
from .cdp import run,targets

mcp=FastMCP("local-decision-browser")

def registry():
    p=os.environ.get("LOCAL_DECISION_REGISTRY")
    if not p:raise RuntimeError("LOCAL_DECISION_REGISTRY must be configured by the host")
    return p

@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
async def browser_targets() -> list[dict]:
    """List explicit CDP target IDs in the dedicated authorized browser."""
    return await targets(read(registry())["cdp"])

@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False,destructiveHint=True,idempotentHint=False,openWorldHint=True))
async def browser_subgoal(task_id:str,target_id:str,run_id:str,dry_run:bool=True)->dict:
    """Run one preapproved bounded task. Keep run_id stable on retries.
    No arbitrary URLs, selectors, JavaScript, passwords or permission changes.
    needs_verification is NOT final Codex/strict acceptance.
    """
    return await run(registry(),task_id,target_id,run_id,execute=not dry_run)

if __name__=="__main__":
    raise SystemExit("legacy browser MCP retired; no alternate controller is started")
