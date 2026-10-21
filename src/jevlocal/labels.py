"""标签算术、严格 proper scoring rule、教师软标签、温度校准。

RLCD 的全部意义在于：**奖励必须是 proper scoring rule**（严格适当评分规则），
这样"报出真实条件分布"才是最大化奖励的策略。用 argmax 命中率当奖励会奖励自信的
猜测；用 imitation 出来的伪置信度会静默毁掉所有下游阈值（hermes-jev-skills 记过
这条教训：一条每轮被询问、答案却永远无法改变任何行为的轴，是纯成本）。

本模块只用标准库。所有函数都是纯函数，便于单测。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Literal, Mapping, Sequence

from .schema import Question, SchemaError

ScoreKind = Literal["log", "brier"]
EPS = 1e-9


# --------------------------------------------------------------------------- #
# 基础分布操作
# --------------------------------------------------------------------------- #
def uniform(k: int) -> list[float]:
    if k < 2:
        raise ValueError("k 至少为 2")
    return [1.0 / k] * k


def onehot(idx: int, k: int) -> list[float]:
    if not (0 <= idx < k):
        raise ValueError(f"下标 {idx} 超出 [0,{k})")
    return [1.0 if i == idx else 0.0 for i in range(k)]


def normalize(p: Sequence[float]) -> list[float]:
    total = math.fsum(p)
    if total <= 0:
        raise ValueError("概率和为 0，无法归一")
    return [max(0.0, float(x)) / total for x in p]


def entropy(p: Sequence[float]) -> float:
    return -math.fsum(x * math.log(x + EPS) for x in p if x > 0)


def kl(p: Sequence[float], q: Sequence[float]) -> float:
    if len(p) != len(q):
        raise ValueError("KL 两侧长度不一致")
    return math.fsum(pi * math.log((pi + EPS) / (qi + EPS)) for pi, qi in zip(p, q) if pi > 0)


def sym_kl(p: Sequence[float], q: Sequence[float]) -> float:
    """对称 KL —— 用于选项顺序不变性惩罚（openJev 的 permutation-KL）。"""
    return 0.5 * (kl(p, q) + kl(q, p))


def is_uniform(p: Sequence[float], tol: float = 1e-6) -> bool:
    k = len(p)
    return all(abs(x - 1.0 / k) <= tol for x in p)


def softmax(logits: Sequence[float], temperature: float = 1.0) -> list[float]:
    if temperature <= 0:
        raise ValueError("temperature 必须为正")
    z = [float(x) / temperature for x in logits]
    m = max(z)
    exps = [math.exp(x - m) for x in z]
    return normalize(exps)


# --------------------------------------------------------------------------- #
# 严格 proper scoring rule
# --------------------------------------------------------------------------- #
def proper_score(
    kind: ScoreKind,
    probs: Sequence[float],
    hard_idx: int,
) -> float:
    """严格适当评分规则。**奖励报出真实条件分布**，而不是奖励猜对。

    - ``log``   ：对数评分 ``log p_y``，值域 (-inf, 0]。
    - ``brier`` ：Brier 评分 ``1 - Σ_i (p_i - y_i)²``，值域 [0, 1]（K 元时上界 1）。

    两者都是严格 proper。RLCD 的奖励用哪一个都行，但**不要用命中率**。
    """
    k = len(probs)
    if kind == "log":
        p_y = float(probs[hard_idx])
        return math.log(max(p_y, EPS))
    target = onehot(hard_idx, k)
    return 1.0 - math.fsum((float(p) - t) ** 2 for p, t in zip(probs, target))


def soft_target_score(
    kind: ScoreKind,
    probs: Sequence[float],
    target: Sequence[float],
) -> float:
    """对**软目标**的适当评分（log 用交叉熵的负值，brier 用平方距离）。"""
    if len(probs) != len(target):
        raise ValueError("长度不一致")
    if kind == "log":
        return math.fsum(t * math.log(max(float(p), EPS)) for p, t in zip(probs, target))
    return 1.0 - math.fsum((float(p) - float(t)) ** 2 for p, t in zip(probs, target))


def expected_level(probs: Sequence[float]) -> float:
    """序数 score 的期望层级（= 报告里的 ``score``）。"""
    return math.fsum(i * float(p) for i, p in enumerate(probs))


def ordinal_mean_penalty(probs: Sequence[float], hard_idx: int) -> float:
    """序数层级的辅助奖励 ``1 - ((E[level] - y)/(K-1))²``。

    ⚠️ **这不是 proper scoring rule**：它只约束分布的一阶矩，对分布形状不敏感。
    仅用作对齐 ``score MAE`` 这个上报指标的**辅助项**，不能单独作为 RLCD 的奖励。
    """
    k = len(probs)
    if k < 2:
        return 1.0
    return 1.0 - ((expected_level(probs) - hard_idx) / (k - 1)) ** 2


# --------------------------------------------------------------------------- #
# 教师软标签（蒸馏用）
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class TeacherLabel:
    soft: list[float]
    n_samples: int
    agreement: float  # 众数占比，等价于 Laya 所说的"教师自一致天花板"
    model: str

    def ceiling_warning(self) -> str | None:
        """自一致度就是这套标签能教会学生的上限——别把超过它当目标，那是在拟合噪声。"""
        if self.agreement < 0.80:
            return (
                f"教师自一致度 {self.agreement:.3f} 偏低：学生准确率若超过该值，"
                "多出的部分很可能是噪声拟合，不是能力"
            )
        return None


def teacher_soft_from_votes(
    votes: Sequence[int],
    k: int,
    *,
    model: str = "",
    alpha: float = 0.5,
) -> TeacherLabel:
    """把教师的 k 次采样投票变成软分布。

    加 ``alpha`` 的 Laplace 平滑：**不要**让任何候选拿到概率 0。Laya 的实测里
    Jev 在 DAIR Emotion 上对真实标签给出过 0 概率（占 16% 样本），任何按置信度分支的
    下游都会在这种样本上硬失败。
    """
    if not votes:
        raise ValueError("votes 为空")
    counts = [0.0] * k
    for v in votes:
        if not (0 <= int(v) < k):
            raise ValueError(f"投票下标 {v} 超出 [0,{k})")
        counts[int(v)] += 1.0
    denom = len(votes) + alpha * k
    soft = [(c + alpha) / denom for c in counts]
    top = max(counts)
    return TeacherLabel(
        soft=normalize(soft),
        n_samples=len(votes),
        agreement=top / len(votes),
        model=model,
    )


def teacher_to_hard(soft: Sequence[float]) -> int:
    return max(range(len(soft)), key=lambda i: soft[i])


# --------------------------------------------------------------------------- #
# 温度校准
# --------------------------------------------------------------------------- #
def fit_temperature(
    logits: Sequence[Sequence[float]],
    hard_idx: Sequence[int],
    *,
    lo: float = 0.05,
    hi: float = 20.0,
    grid: int = 400,
    refine_rounds: int = 6,
) -> float:
    """一维温度拟合：最小化 NLL ``-Σ log softmax(z/T)[y]``。

    做法是"对数网格粗搜 + 逐轮缩窗精修"，不假设 NLL 对 T 单峰（不依赖三分搜索的
    单峰前提），换来的是几百次纯算术，代价可忽略。

    kev 每个 checkpoint 存了约 2.1–2.4 的温度并由指针头在加载时施加；openJev 做的是
    **按问题类型分别拟合**（type-routed temperature）——见 ``fit_type_routed``。
    """
    if len(logits) != len(hard_idx):
        raise ValueError("logits 与标签数量不一致")
    if not logits:
        raise ValueError("空输入")

    def nll(t: float) -> float:
        total = 0.0
        for z, y in zip(logits, hard_idx):
            p = softmax(z, t)
            total -= math.log(max(p[y], EPS))
        return total / len(logits)

    lo_l, hi_l = math.log(lo), math.log(hi)
    best_t = 1.0
    best_v = math.inf
    for i in range(grid + 1):
        t = math.exp(lo_l + (hi_l - lo_l) * i / grid)
        v = nll(t)
        if v < best_v:
            best_v, best_t = v, t
    span = (hi_l - lo_l) / grid
    for _ in range(refine_rounds):
        span /= 4.0
        for cand in (best_t * math.exp(-span), best_t, best_t * math.exp(span)):
            cand = min(max(cand, lo), hi)
            v = nll(cand)
            if v < best_v:
                best_v, best_t = v, cand
    return best_t


def fit_type_routed(
    logits_by_type: Mapping[str, Sequence[Sequence[float]]],
    labels_by_type: Mapping[str, Sequence[int]],
    *,
    fallback: float = 1.0,
) -> dict[str, float]:
    """按问题类型（或更细的 type×K 桶）分别拟合温度。

    openJev 的双通道校准之一就是"按类型路由的温度缩放"，这也是它把 ECE 压下来的关键。
    类型样本太少时退回 ``fallback``，不要用几十条样本拟合一个温度。
    """
    out: dict[str, float] = {}
    for key, logits in logits_by_type.items():
        labels = labels_by_type.get(key, ())
        if len(logits) != len(labels) or len(logits) < 50:
            out[key] = fallback
            continue
        out[key] = fit_temperature(logits, labels)
    return out


# --------------------------------------------------------------------------- #
# 样本 → 训练张量（无 torch）
# --------------------------------------------------------------------------- #
def sample_to_targets(
    q: Question,
    soft: float | Mapping[str, float] | Sequence[float],
    hard: int | str | None,
    *,
    unanswerable: bool = False,
) -> tuple[list[float], int | None]:
    """把样本标签折成 ``(定长目标分布, hard 下标)``。

    不可回答的样本把目标设为**均匀分布**、hard 设为 None。
    这不是偷懒：当 state 里根本不存在判定证据时，正确的标定行为就是不确定
    （kev 2026-09-21 的那次二次训练做的正是这件事——把"决定证据被移除"的样本训向均匀答案）。
    """
    k = q.n_options
    if unanswerable:
        return uniform(k), None

    if q.type == "noul":
        p = float(soft)  # type: ignore[arg-type]
        target = [p, 1.0 - p]
        return target, (0 if int(hard or 0) == 1 else 1)
    if q.type == "choice":
        if not isinstance(soft, Mapping):
            raise SchemaError(f"{q.qid}: choice 的 soft 必须是映射")
        target = [float(soft[name]) for name in q.option_names]
        return normalize(target), q.option_names.index(str(hard))
    if not isinstance(soft, Sequence) or isinstance(soft, (str, bytes)):
        raise SchemaError(f"{q.qid}: score 的 soft 必须是序列")
    target = [float(x) for x in soft]
    if len(target) != k:
        raise SchemaError(f"{q.qid}: score 的 soft 长度与层级数不符")
    return normalize(target), int(hard)  # type: ignore[arg-type]
