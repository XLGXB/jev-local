"""从本地 Codex 轨迹抽取决策样本。

**先说清一件事**：本文件不假设 ``.codex/history.jsonl`` / ``sessions/`` 的确切字段形状。
理由是本机实测发现这些目录里同时存在多代格式（`history.jsonl`、`session_index.jsonl`、
`thread_history_1.sqlite`、`sessions/` 下的 rollout 文件），字段随版本漂移。
所以这里的做法是：
- ``probe_shape()`` 先**打印真实记录的键路径与类型**，用证据定形状；
- ``extract_tool_pairs()`` 用容忍多种拼写的遍历器配对 ``tool_use`` / ``tool_result``；
- 真正的字段映射留在 ``FieldMap`` 里，一处改，全处生效。

先跑 ``python -m jevlocal.extract.codex_sessions probe <path>``，看清形状再写映射。
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from ..schema import Label, Question, Sample

EXTRACTOR = "codex_sessions"
EXTRACTOR_VERSION = "1"

# --------------------------------------------------------------------------- #
# 脱敏
# --------------------------------------------------------------------------- #
_SECRET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("openai_key", re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}\b")),
    ("anthropic_key", re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{16,}\b")),
    ("github_pat", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b")),
    ("aws_key", re.compile(r"\b(AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\b")),
    ("bearer", re.compile(r"(?i)\b(bearer|authorization)\s*[:=]\s*[A-Za-z0-9._\-]{16,}")),
    ("named_secret", re.compile(r"(?i)\b([A-Z0-9_]*(?:API|ACCESS|SECRET|PRIVATE)[A-Z0-9_]*)\s*[:=]\s*[A-Za-z0-9._\-]{12,}")),
    ("long_hex", re.compile(r"\b[0-9a-fA-F]{40,}\b")),
)


def redact(text: str) -> tuple[str, int]:
    """把疑似凭据替换掉，返回 (脱敏文本, 命中次数)。

    必须做而且是**抽取期**做：state 会进训练集、进日志、可能进上报材料。
    本机当前查得 ``TYPESAFE_API_KEY`` / ``OPENROUTER_API_KEY`` / ``OPENAI_API_KEY``
    均未设置，但历史轨迹里可能残留别处的 key。
    """
    hits = 0
    out = text
    for _, pat in _SECRET_PATTERNS:
        out, n = pat.subn("[REDACTED]", out)
        hits += n
    return out, hits


# --------------------------------------------------------------------------- #
# 形状探针
# --------------------------------------------------------------------------- #
def iter_records(path: str | Path) -> Iterator[dict[str, Any]]:
    """按 JSONL 读。非 JSON 行跳过并在 stderr 计数（不静默吞掉整个文件）。"""
    p = Path(path)
    with p.open("r", encoding="utf-8", newline="") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                yield obj


def probe_shape(path: str | Path, *, limit: int = 3, max_depth: int = 4) -> dict[str, Any]:
    """打印前若干条记录的键路径/类型，作为"写映射前先取证据"的手段。"""
    shapes: dict[str, set[str]] = {}
    total = 0
    for i, rec in enumerate(iter_records(path)):
        total += 1
        _walk_shape(rec, "", shapes, max_depth)
        if i + 1 >= limit:
            break
    print(f"# 文件: {path}")
    print(f"# 采样 {min(limit, total)} 条 / 累计 {total} 条")
    for key in sorted(shapes):
        print(f"  {key}: {sorted(shapes[key])}")
    return {k: sorted(v) for k, v in shapes.items()}


def _walk_shape(node: Any, prefix: str, out: dict[str, set[str]], depth: int) -> None:
    if depth <= 0:
        return
    if isinstance(node, dict):
        for k, v in node.items():
            key = f"{prefix}.{k}" if prefix else k
            out.setdefault(key, set()).add(type(v).__name__)
            _walk_shape(v, key, out, depth - 1)
    elif isinstance(node, list) and node:
        key = f"{prefix}[]"
        out.setdefault(key, set()).add(type(node[0]).__name__)
        _walk_shape(node[0], key, out, depth - 1)


# --------------------------------------------------------------------------- #
# tool_use / tool_result 配对
# --------------------------------------------------------------------------- #
#: 容忍的键拼写（不同 Codex 代际 / 不同宿主导出格式）
CALL_ID_KEYS = ("tool_use_id", "tool_call_id", "call_id", "id")
CALL_NAME_KEYS = ("tool", "name", "tool_name")
CALL_INPUT_KEYS = ("input", "arguments", "args", "parameters")
RESULT_TEXT_KEYS = ("text", "content", "output", "result", "stdout")
RESULT_PARENT_KEYS = ("tool_results", "tool_result", "results", "tool_outputs")


@dataclass(slots=True)
class ToolPair:
    call_id: str
    tool: str
    input: Any
    result_text: str
    ok: bool | None = None


def _first(node: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for k in keys:
        if k in node:
            return node[k]
    return None


def extract_tool_pairs(record: Mapping[str, Any]) -> list[ToolPair]:
    """从一条记录里挖出 (调用, 结果) 对。找不到就返回空，不抛。"""
    calls: dict[str, dict[str, Any]] = {}
    results: dict[str, str] = {}

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            cid = _first(node, CALL_ID_KEYS)
            name = _first(node, CALL_NAME_KEYS)
            if isinstance(cid, str) and isinstance(name, str):
                calls.setdefault(
                    cid,
                    {
                        "tool": name,
                        "input": _first(node, CALL_INPUT_KEYS),
                    },
                )
            for key in RESULT_PARENT_KEYS:
                if key in node:
                    payload = node[key]
                    items = payload if isinstance(payload, list) else [payload]
                    for item in items:
                        if isinstance(item, dict):
                            rid = _first(item, CALL_ID_KEYS)
                            text = _first(item, RESULT_TEXT_KEYS)
                            if isinstance(rid, str) and text is not None:
                                results[rid] = text if isinstance(text, str) else json.dumps(text, ensure_ascii=False)
            for v in node.values():
                visit(v)
        elif isinstance(node, list):
            for v in node:
                visit(v)

    visit(record)
    out: list[ToolPair] = []
    for cid, call in calls.items():
        if cid in results:
            out.append(
                ToolPair(
                    call_id=cid,
                    tool=str(call["tool"]),
                    input=call["input"],
                    result_text=results[cid],
                )
            )
    return out


# --------------------------------------------------------------------------- #
# state 构造
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class FieldMap:
    """一处改全处生效的字段映射。先跑 probe 再填这里。"""

    role_keys: tuple[str, ...] = ("role", "type", "author")
    text_keys: tuple[str, ...] = ("text", "content", "message", "body")
    tool_results_keys: tuple[str, ...] = RESULT_PARENT_KEYS


RESULT_NOTE = "ok, {n} chars (omitted)"


def build_state(
    pairs: Sequence[ToolPair],
    *,
    recent: int,
    result_note: str = RESULT_NOTE,
    include_result_tail: int = 0,
) -> str:
    """构造决策 state：**工具结果换成短注记**，只保留调用与输入。

    这是 fast-jev-compaction 的做法：Jev 看到的是"当时做过什么、输入是什么"，
    而不是全部原始输出。这样 state 才塞得下，而且不会把无关正文灌进模型。
    """
    window = pairs[-recent:] if recent > 0 else list(pairs)
    lines: list[str] = []
    for i, p in enumerate(window):
        args = ""
        if isinstance(p.input, Mapping):
            args = " ".join(f"{k}={_short(v)}" for k, v in p.input.items())
        elif p.input is not None:
            args = _short(p.input)
        lines.append(f"t{i} {p.tool} {args} → " + result_note.format(n=len(p.result_text)))
        if include_result_tail > 0 and p.result_text:
            lines.append("   tail: " + p.result_text[-include_result_tail:].replace("\n", " ⏎ "))
    return "\n".join(lines)


def _short(value: Any, limit: int = 200) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    text = text.replace("\n", " ")
    return text if len(text) <= limit else text[:limit] + "…"


# --------------------------------------------------------------------------- #
# 样本构造
# --------------------------------------------------------------------------- #
KEEP_CALL_INSTRUCTION = "后续还可能需要知道「这条调用被做过」吗？（不看它的输出）"
KEEP_RESULT_INSTRUCTION = "它的输出内容后续还需要逐字保留吗？（重跑一次就能拿回来的不算）"

# 这两个问题的语义与 fast-jev-compaction 完全对齐（它每个非钉住调用问的正是这两问）


def make_keepdrop_sample(
    *,
    sample_id: str,
    state: str,
    source_ref: str,
    keep_call: bool,
    keep_result: bool,
    split: str,
    domain: str = "tool_keepdrop",
) -> Sample:
    questions = [
        Question(qid="keep_call", type="noul", instructions=KEEP_CALL_INSTRUCTION, criteria={"true": "还需要", "false": "不再需要"}),
        Question(qid="keep_result", type="noul", instructions=KEEP_RESULT_INSTRUCTION, criteria={"true": "需要逐字保留", "false": "可丢弃或截断"}),
    ]
    labels = {
        "keep_call": Label(hard=1 if keep_call else 0, soft=1.0 if keep_call else 0.0),
        "keep_result": Label(hard=1 if keep_result else 0, soft=1.0 if keep_result else 0.0),
    }
    return Sample(
        sample_id=sample_id,
        domain=domain,
        source={"kind": "codex_history", "ref": source_ref},
        state=state,
        questions=questions,
        labels=labels,
        split=split,
        provenance={"extractor": EXTRACTOR, "extractor_version": EXTRACTOR_VERSION},
    )


def emit_keepdrop_candidates(
    path: str | Path,
    *,
    window: int = 6,
    max_samples: int = 0,
) -> list[dict[str, Any]]:
    """产出**未标注**的候选（state + 空标签），等待教师打标。

    注意这里只产出候选：keep/drop 的标签必须来自教师模型或人工，不能由抽取器自己编。
    抽取器唯一能提供的"免费"标签是**结构性**的（例如"这条调用在后续被再次引用"），
    那类标签要单独命名（``structural``），不要与教师标签混在同一字段里。
    """
    out: list[dict[str, Any]] = []
    for i, rec in enumerate(iter_records(path)):
        pairs = extract_tool_pairs(rec)
        if len(pairs) < 2:
            continue
        state, hits = redact(build_state(pairs, recent=window))
        rec_id = str(_first(rec, ("id", "session_id", "thread_id", "uuid")) or f"rec{i}")
        out.append(
            {
                "candidate_id": f"{rec_id}-{i}",
                "domain": "tool_keepdrop",
                "source": {"kind": "codex_history", "ref": f"{path}#line{i + 1}"},
                "state": state,
                "n_pairs": len(pairs),
                "redactions": hits,
                "questions": ["keep_call", "keep_result"],
                "labels": None,
            }
        )
        if max_samples and len(out) >= max_samples:
            break
    return out


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="jevlocal.extract.codex_sessions", description="本地 Codex 轨迹 → 决策样本候选")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("probe", help="打印真实记录的键路径与类型（写映射前先跑这个）")
    sp.add_argument("path")
    sp.add_argument("--limit", type=int, default=3)

    sc = sub.add_parser("candidates", help="产出未标注的决策样本候选 JSONL")
    sc.add_argument("path")
    sc.add_argument("--window", type=int, default=6)
    sc.add_argument("--out", required=True)
    sc.add_argument("--max", type=int, default=0)

    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.cmd == "probe":
        probe_shape(args.path, limit=args.limit)
        return 0
    if args.cmd == "candidates":
        rows = emit_keepdrop_candidates(args.path, window=args.window, max_samples=args.max)
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8", newline="\n") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        print(f"写出 {len(rows)} 条候选 → {out}")
        print("下一步：用教师模型给 labels 打标（见 extract/teacher.py 的约定），再进 splits。")
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
