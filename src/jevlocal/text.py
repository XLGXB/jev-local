"""两种主干的输入模板（ModernBERT / Qwen-Kev）。

模板不是随手拼接，它决定了决策头**在哪里读数**：

ModernBERT（Laya、openJev-verdict-2.0 范式）
    头部段：``<type> <instructions> [SEP] [MASK]<opt0> [MASK]<opt1> …``
    状态段：`` [SEP] <state> [SEP]``
    候选各自带一个标记前缀；决策头在**标记位**读隐状态（marker-pointer），
    再对 K 个标记位做 softmax。注意 ``[MASK]`` 在这里是**位置锚**，不是要预测的洞。
    预算：``head_max_len`` 管头部段，``max_len - head_max_len`` 管状态段。
    Laya 默认头部 192（英文）/ 256（多语言）；openJev 把总上下文从 1024 砍到 512
    以避免超出训练分布。

Qwen-Kev（kev 范式，训练样本与 API 请求同构）::

    <state> …state…
    <q> instructions <opt> 选项1 </opt> <opt> 选项2 </opt> … <decide>
    <q> … <decide>

    每个问题一块，块尾 ``<decide>`` 是读数位；指针头读该位隐状态算 logits。
    训练预算（kev 原文）：state ≤ 384 token、state + 单问 ≤ 1024 token；
    服务端放宽到 8192，但**训练未覆盖更长上下文**，所以超预算要在抽取期裁。

本模块只用标准库。字符偏移（char offset）是给 tokenizer 的 ``offset_mapping``
做二次映射用的：**不要直接把字符下标当 token 下标**（本仓库有 byte offset ≠ 字符下标
的实测教训）。
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Iterable, Sequence

from .schema import Question

# --------------------------------------------------------------------------- #
# 特殊标记
# --------------------------------------------------------------------------- #
SPECIAL_TOKENS: tuple[str, ...] = (
    "[MASK]",
    "[SEP]",
    "<state>",
    "<q>",
    "<opt>",
    "</opt>",
    "<decide>",
)

STATE_MARKER = "<state>"
QUESTION_MARKER = "<q>"
OPTION_OPEN = "<opt>"
OPTION_CLOSE = "</opt>"
DECIDE_MARKER = "<decide>"

# 预算默认值（来源：Laya / openJev / kev 的公开配方，可按 config 覆盖）
MODERNBERT_HEAD_MAX_LEN_EN = 192
MODERNBERT_HEAD_MAX_LEN_MULTILINGUAL = 256
MODERNBERT_MAX_LEN = 512
QWEN_KEV_STATE_MAX_LEN = 384
QWEN_KEV_TOTAL_MAX_LEN = 1024


# --------------------------------------------------------------------------- #
# ModernBERT 路径
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class ModernBertEncoding:
    """头部段与状态段分开返回：collate 时两段各自按自己的预算切。"""

    head_text: str
    state_text: str
    option_names: list[str]
    marker_spans: list[tuple[int, int]]  # 头部段里每个 [MASK] 的字符区间

    @property
    def n_options(self) -> int:
        return len(self.option_names)

    def full_text(self) -> str:
        return f"{self.head_text}[SEP]{self.state_text}[SEP]"


def build_modernbert_head(
    q: Question,
    *,
    marker: str = "[MASK]",
    include_descriptions: bool = False,
    sep: str = " ",
) -> str:
    """构造头部段：``<type> <instructions> [SEP] [MASK]<opt> …``"""
    names = q.option_names
    descs = q.option_descriptions
    parts: list[str] = [q.type, sep]
    if q.type == "score":
        # 顺序信息必须显式给出：score 是序数问题，层级从低到高
        parts.append("(ordered low to high) ")
    parts.append(q.instructions.strip())
    parts.append(f"{sep}[SEP]{sep}")
    for i, name in enumerate(names):
        parts.append(f"{marker}{name}")
        if include_descriptions:
            d = descs[i] if i < len(descs) else None
            if d:
                parts.append(f"：{d}")
        if i != len(names) - 1:
            parts.append(sep)
    return "".join(parts)


def marker_char_spans(head_text: str, marker: str = "[MASK]") -> list[tuple[int, int]]:
    """头部段里每个标记的字符区间，顺序即候选顺序。

    交给 tokenizer 时用 ``return_offsets_mapping=True``，把落在区间内的 token 下标挑出来
    作为读数位；比按 token 序数硬猜稳。
    """
    spans: list[tuple[int, int]] = []
    start = 0
    while True:
        idx = head_text.find(marker, start)
        if idx < 0:
            break
        spans.append((idx, idx + len(marker)))
        start = idx + len(marker)
    return spans


def encode_modernbert(
    q: Question,
    state: str,
    *,
    head_max_chars: int | None = None,
    include_descriptions: bool = False,
) -> ModernBertEncoding:
    head = build_modernbert_head(q, include_descriptions=include_descriptions)
    if head_max_chars is not None and len(head) > head_max_chars:
        raise ValueError(
            f"{q.qid}: 头部段 {len(head)} 字符超过 head_max_chars={head_max_chars}；"
            "选项过多时应改走 shortlist 或拆问，而不是静默截断候选"
        )
    return ModernBertEncoding(
        head_text=head,
        state_text=state.strip(),
        option_names=q.option_names,
        marker_spans=marker_char_spans(head),
    )


# --------------------------------------------------------------------------- #
# Qwen-Kev 路径
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class QwenKevEncoding:
    text: str
    question_offsets: list[int]  # 每个 <q> 块的起始字符下标
    decide_offsets: list[int]  # 每个 <decide> 标记的起始字符下标（== 读数位）

    def full_text(self) -> str:
        return self.text


def build_qwen_kev_block(q: Question, sep: str = " ") -> tuple[str, int]:
    """返回 (块文本, 块内 <decide> 的相对偏移)。"""
    parts: list[str] = [f"{QUESTION_MARKER}{sep}"]
    if q.type == "score":
        parts.append("(ordered low to high) ")
    parts.append(q.instructions.strip())
    parts.append(sep)
    descs = q.option_descriptions
    for i, name in enumerate(q.option_names):
        parts.append(f"{OPTION_OPEN}{sep}{name}{sep}{OPTION_CLOSE}")
        if descs[i]:
            parts.append(sep)
        if i != len(q.option_names) - 1:
            parts.append(sep)
    block = "".join(parts)
    full = f"{block}{sep}{DECIDE_MARKER}"
    # 偏移必须落在 <decide> 的**首字符**上，不是落在分隔空格上。
    # 少加 len(sep) 会让读数位整体左移一格：形状看着对、训练照跑、指标悄悄变差。
    rel = len(full) - len(DECIDE_MARKER)
    return full, rel


def encode_qwen_kev(state: str, questions: Sequence[Question], *, sep: str = " ") -> QwenKevEncoding:
    """按 kev 的多行格式拼整条训练样本，并定位每个读数位。"""
    parts: list[str] = [f"{STATE_MARKER}{sep}{state.strip()}"]
    cursor = len(parts[0]) + 1  # +1 为换行
    q_offsets: list[int] = []
    d_offsets: list[int] = []
    for q in questions:
        block, rel = build_qwen_kev_block(q, sep=sep)
        q_offsets.append(cursor)
        d_offsets.append(cursor + rel)
        parts.append(block)
        cursor += len(block) + 1
    return QwenKevEncoding(text="\n".join(parts), question_offsets=q_offsets, decide_offsets=d_offsets)


def decide_positions(text: str) -> list[int]:
    """反查文本里所有 ``<decide>`` 的字符下标（用于自检拼装是否与预期一致）。"""
    out: list[int] = []
    start = 0
    while True:
        idx = text.find(DECIDE_MARKER, start)
        if idx < 0:
            break
        out.append(idx)
        start = idx + len(DECIDE_MARKER)
    return out


# --------------------------------------------------------------------------- #
# 状态裁剪与预算
# --------------------------------------------------------------------------- #
def estimate_tokens(text: str) -> int:
    """**粗略估算**，不是 tokenizer。只用于抽取期的预算预筛。

    规则（沿用 fast-jev-compaction / jev-pruner 一族的估法，且刻意估高一点）：
    每 6 个字母 ≈ 1 token，每个数字 ≈ 0.5 token，CJK 每字 ≈ 1 token，其余符号 ≈ 1 token。

    ⚠️ 最终裁剪必须用真实 tokenizer 复核；估算只用来在抽取期把明显超限的样本先挡掉。
    """
    letters = len(re.findall(r"[A-Za-z]", text))
    digits = len(re.findall(r"[0-9]", text))
    cjk = len(re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]", text))
    other = max(0, len(text) - letters - digits - cjk)
    return int(math.ceil(letters / 6.0 + digits * 0.5 + cjk * 1.0 + other * 1.0))


def fit_state(
    state: str,
    *,
    max_chars: int,
    keep_head_ratio: float = 0.5,
    note: str = "[…{n} chars omitted…]",
) -> tuple[str, bool]:
    """把状态裁进字符预算。返回 (裁剪后文本, 是否发生了裁剪)。

    策略：保头保尾。理由（来自 fast-jev-compaction 的分级 fit 与 jev-pruner 的
    实测教训）：长日志/长页面里**错误行常在中段或尾段**，只保头会把关键证据丢掉——
    那次实测默认预览 2,227 字符恰好丢了错误行。
    """
    if max_chars <= 0:
        raise ValueError("max_chars 必须为正")
    if len(state) <= max_chars:
        return state, False

    head = int(max_chars * keep_head_ratio)
    tail = max_chars - head
    omitted = len(state) - head - tail
    if omitted <= 0:
        return state[:max_chars], True
    marker = note.format(n=omitted)
    return f"{state[:head]}{marker}{state[-tail:]}", True
