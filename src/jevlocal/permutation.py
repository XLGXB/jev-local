"""选项顺序不变性与扰动统计。

为什么单独成模块：**顺序偏置是决策模型的头号隐性缺陷**。自回归模型把选项当文本读，
把 A/B/C 换成 C/B/A 会改答案——kev 公布过 7.41% 的 argmax 翻转率（N=81），
openJev 用对称 permutation-KL 把它压到 4.76%（N=2,918）。

本模块同时服务两个目的：
1. **正则项**：同一问题在两种选项顺序下应当给出一致的分布（``symmetric_permutation_kl``）。
2. **评测项**：翻转率与概率摆幅必须与准确率**一起**上报（``flip_rate`` / ``top_prob_spread``）。

只用标准库。
"""

from __future__ import annotations

import math
import random
from typing import Sequence

from .labels import sym_kl


def identity(k: int) -> list[int]:
    return list(range(k))


def apply_perm(probs: Sequence[float], perm: Sequence[int]) -> list[float]:
    """按 ``perm`` 重排分布：``out[i] = probs[perm[i]]``。

    语义约定（必须与 ``invert`` 配对使用）：``perm`` 描述的是"新位置 i 上放原位置
    perm[i] 的元素"，即**拉取式**。写成推式（``out[perm[i]] = probs[i]``）会让两次
    重排互相抵消掉一半错误，并且这个错误在形状正确时不会报错。
    """
    if len(probs) != len(perm):
        raise ValueError("probs 与 perm 长度不一致")
    return [float(probs[i]) for i in perm]


def invert(perm: Sequence[int]) -> list[int]:
    inv = [0] * len(perm)
    for new_i, old_i in enumerate(perm):
        inv[old_i] = new_i
    return inv


def is_valid_perm(perm: Sequence[int]) -> bool:
    return sorted(perm) == list(range(len(perm)))


def permutations_for(
    k: int,
    *,
    group_size: int,
    seed: int,
    include_identity: bool = True,
) -> list[list[int]]:
    """给一个 K 元问题生成 ``group_size`` 个选项顺序。

    第一个恒为单位排列（作为参照系）。用固定 seed 的 ``random.Random`` 而不是全局
    ``random``：全局状态会被任何别的调用扰动，导致"同一个 seed 复现不出同一批排列"，
    而这批排列同时是 GRPO 的分组和 KL 的配对，不可复现就等于实验不可复现。
    """
    if k < 2:
        raise ValueError("k 至少为 2")
    if group_size < 1:
        raise ValueError("group_size 至少为 1")
    rng = random.Random(seed)
    out: list[list[int]] = []
    if include_identity:
        out.append(identity(k))
    if k == 2:
        # K=2 只有两种顺序，重复采样会退化成同一个排列
        while len(out) < group_size:
            out.append(list(reversed(out[-1])))
        return out[:group_size]
    while len(out) < group_size:
        cand = identity(k)
        rng.shuffle(cand)
        if cand in out:
            continue
        out.append(cand)
    return out


# --------------------------------------------------------------------------- #
# 正则项
# --------------------------------------------------------------------------- #
def symmetric_permutation_kl(
    dists: Sequence[Sequence[float]],
    perms: Sequence[Sequence[int]],
) -> float:
    """把每个排列下的分布"折回原序"后两两算对称 KL，取均值。

    ``dists[i]`` 是在 ``perms[i]`` 这个顺序下模型给出的分布。要比较分布，必须先把
    它折回**同一个基准顺序**：``folded[i][perms[i][j]] = dists[i][j]``。
    直接用 ``apply_perm`` 会折错方向（拉取 vs 推入），而且不报错——所以这里用
    ``invert`` 显式折回，并在下面做了一次自检断言。
    """
    if len(dists) != len(perms):
        raise ValueError("dists 与 perms 数量不一致")
    if len(dists) < 2:
        return 0.0

    folded: list[list[float]] = []
    for dist, perm in zip(dists, perms):
        if not is_valid_perm(perm):
            raise ValueError(f"非法排列: {perm}")
        back = [0.0] * len(perm)
        for new_i, old_i in enumerate(perm):
            back[old_i] = float(dist[new_i])  # 推入式：折回原序
        folded.append(back)

    k = len(folded[0])
    if len(folded[0]) != k or any(len(f) != k for f in folded):
        raise ValueError("分布长度不一致")

    # 自检：单位排列的折回结果必须等于原分布
    if perms[0] == identity(k):
        assert folded[0] == [float(x) for x in dists[0]], "折回方向错了"

    total = 0.0
    pairs = 0
    for i in range(len(folded)):
        for j in range(i + 1, len(folded)):
            total += sym_kl(folded[i], folded[j])
            pairs += 1
    return total / pairs if pairs else 0.0


# --------------------------------------------------------------------------- #
# 评测项
# --------------------------------------------------------------------------- #
def argmax(dist: Sequence[float]) -> int:
    return max(range(len(dist)), key=lambda i: float(dist[i]))


def flip_rate(dists: Sequence[Sequence[float]]) -> float:
    """翻转率 = 与第一个（参照）分布 argmax 不同的比例。

    分母是 ``len(dists) - 1``：参照自己与自己比恒等，计入分母会人为压低数字。
    """
    if len(dists) < 2:
        return 0.0
    base = argmax(dists[0])
    hits = sum(1 for d in dists[1:] if argmax(d) != base)
    return hits / (len(dists) - 1)


def top_prob_spread(dists: Sequence[Sequence[float]]) -> tuple[float, float]:
    """顶端概率摆幅 ``|p_top(σ) - p_top(identity)|`` 的 (均值, p90)。

    ⚠️ 本仓库对这个量的定义是**我们自己的**：kev 只公布了 0.2486 这个 p90 值，
    没有公开公式，不同实现会得到不同数字。上报时必须同时写清定义，否则数字不可比。
    这里按"参照顺序下的顶端概率相对移动了多少"定义，对顺序扰动最直观。
    """
    if len(dists) < 2:
        return 0.0, 0.0
    base = max(float(x) for x in dists[0])
    deltas = sorted(abs(max(float(x) for x in d) - base) for d in dists[1:])
    mean = math.fsum(deltas) / len(deltas)
    idx = min(len(deltas) - 1, int(math.ceil(0.9 * len(deltas))) - 1)
    return mean, deltas[idx]


def permutation_block(
    dists: Sequence[Sequence[float]],
    perms: Sequence[Sequence[int]],
) -> dict[str, float]:
    """打包成一整块——因为这块必须与准确率一起上报，不能挑着报。"""
    mean, p90 = top_prob_spread(dists)
    return {
        "flip_rate": flip_rate(dists),
        "top_prob_spread_mean": mean,
        "top_prob_spread_p90": p90,
        "n_permutations": float(len(dists)),
        "n_questions": float(len(perms)),
    }


# 公开参照值（第三方公布，用于同表对照；不是本仓库的实测）
KEV_REFERENCE = {"flip_rate": 0.0741, "top_prob_spread_p90": 0.2486, "n": 81}
OPENJEV_REFERENCE = {"flip_rate": 0.0476, "n_perturbations": 2918}
