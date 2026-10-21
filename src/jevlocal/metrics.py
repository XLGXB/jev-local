"""评测指标与**强制性完整上报**。

单独成模块的原因是一条规矩（来源：openJev-verdict-2.0 的 RUNBOOK §7 原话
"Publish these together, never a subset"）：**准确率、软准确率、Brier、序数 MAE、
两个 ECE、顺序稳定性整块、参照下限，必须同时上报。** 只报其中好看的那几个读起来像
在挑指标。所以这里提供 ``assert_complete_report``——报告缺项直接抛错，不给"顺手挑一个"
留后门。

只用标准库。
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

EPS = 1e-9

# 参照下限（来源：openJev RUNBOOK §7 提到的 reports/reference_floors.json）
# TF-IDF + 逻辑回归在这套基准上约 0.65 准确率、ECE 约 0.03。
# 任何头条数字都必须对着这个下限读——0.65 是"不用神经网络的免费分数"。
REFERENCE_FLOORS: dict[str, float] = {
    "tfidf_logreg_accuracy": 0.65,
    "tfidf_logreg_ece": 0.03,
    "majority_class_accuracy": 0.461,
    "random_uniform_accuracy": 0.318,
}

# 同一基准上的第三方参照（来源方公布，非本仓库实测）
VENDOR_REFERENCE: dict[str, float] = {
    "typesafe_jev_accuracy": 0.727,
    "typesafe_jev_brier": 0.1480,
    "typesafe_jev_ece": 0.1440,
    "laya_accuracy": 0.766,
    "laya_brier": 0.0660,
    "laya_ece": 0.2140,
    "openjev_accuracy": 0.7710,
    "openjev_brier": 0.0636,
    "openjev_ece_confidence": 0.0144,
    "teacher_self_agreement_ceiling": 0.735,
}


# --------------------------------------------------------------------------- #
# 基础指标
# --------------------------------------------------------------------------- #
def accuracy(pred_idx: Sequence[int], gold_idx: Sequence[int]) -> float:
    if len(pred_idx) != len(gold_idx):
        raise ValueError("长度不一致")
    if not pred_idx:
        return 0.0
    hit = sum(1 for p, g in zip(pred_idx, gold_idx) if int(p) == int(g))
    return hit / len(pred_idx)


def soft_accuracy(pred_idx: Sequence[int], gold_soft: Sequence[Sequence[float]]) -> float:
    """软准确率 = 学生 argmax 处**教师给出的概率**的均值。

    定义说明：OpenAI 系与 Laya 都报这个数，但都不公开公式。这里取"学生选中的那个候选，
    教师认为它有多对"作为软准确率——它衡量的是"两套分布是否指向同一处"，与硬准确率
    互补：openJev 的 argmax 赢了 Jev（0.766 vs 0.727）而软准确率输了（0.471 vs 0.580），
    单看一个数就会得出相反结论。
    """
    if len(pred_idx) != len(gold_soft):
        raise ValueError("长度不一致")
    if not pred_idx:
        return 0.0
    total = 0.0
    for p, soft in zip(pred_idx, gold_soft):
        if not (0 <= int(p) < len(soft)):
            raise ValueError("预测下标越界")
        total += float(soft[int(p)])
    return total / len(pred_idx)


def brier(probs: Sequence[Sequence[float]], gold_idx: Sequence[int]) -> float:
    """多元 Brier = ``mean Σ_i (p_i - y_i)²``。越小越好。"""
    if len(probs) != len(gold_idx):
        raise ValueError("长度不一致")
    if not probs:
        return 0.0
    total = 0.0
    for p, y in zip(probs, gold_idx):
        total += math.fsum((float(pi) - (1.0 if i == int(y) else 0.0)) ** 2 for i, pi in enumerate(p))
    return total / len(probs)


def expected_level(probs: Sequence[float]) -> float:
    return math.fsum(i * float(p) for i, p in enumerate(probs))


def score_mae(probs: Sequence[Sequence[float]], gold_idx: Sequence[int]) -> float:
    """序数问题（score）的平均绝对误差，用期望层级算。"""
    if len(probs) != len(gold_idx):
        raise ValueError("长度不一致")
    if not probs:
        return 0.0
    return math.fsum(abs(expected_level(p) - int(y)) for p, y in zip(probs, gold_idx)) / len(probs)


def within_one_level(probs: Sequence[Sequence[float]], gold_idx: Sequence[int], tol: float = 1.0) -> float:
    if len(probs) != len(gold_idx):
        raise ValueError("长度不一致")
    if not probs:
        return 0.0
    ok = sum(1 for p, y in zip(probs, gold_idx) if abs(expected_level(p) - int(y)) <= tol)
    return ok / len(probs)


# --------------------------------------------------------------------------- #
# 校准
# --------------------------------------------------------------------------- #
def ece(confidences: Sequence[float], correct: Sequence[bool], *, n_bins: int = 10) -> float:
    """期望校准误差（等宽分箱、按箱加权）。

    ``confidences`` 可以来自两个不同通道，这正是 openJev 报两个 ECE 的原因：

    - **分布通道**：置信度 = ``max(p)``（股票口径的"stock harness metric"）；
    - **正确性通道**：由一个 out-of-fold 训练的 correctness 头给出的"我这次对不对"。

    报一个不报另一个会被读成挑指标。所以 ``assert_complete_report`` 两个都要求。
    """
    if len(confidences) != len(correct):
        raise ValueError("长度不一致")
    if not confidences:
        return 0.0
    bins = [[] for _ in range(n_bins)]
    for c, ok in zip(confidences, correct):
        idx = min(n_bins - 1, max(0, int(float(c) * n_bins)))
        bins[idx].append((float(c), bool(ok)))
    total = len(confidences)
    err = 0.0
    for bucket in bins:
        if not bucket:
            continue
        avg_conf = math.fsum(c for c, _ in bucket) / len(bucket)
        acc = sum(1 for _, ok in bucket if ok) / len(bucket)
        err += (len(bucket) / total) * abs(acc - avg_conf)
    return err


def auroc(scores: Sequence[float], labels: Sequence[int]) -> float:
    """AUROC，用平均秩处理并列（不依赖 sklearn）。

    并列全部给平均秩；不这么做的话，同一批置信度会因为排序稳定性不同而给出不同 AUROC，
    这个差异会被误读成模型差异。
    """
    if len(scores) != len(labels):
        raise ValueError("长度不一致")
    pos = sum(1 for y in labels if int(y) == 1)
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return float("nan")

    order = sorted(range(len(scores)), key=lambda i: float(scores[i]))
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and float(scores[order[j + 1]]) == float(scores[order[i]]):
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0  # 1-based
        for t in range(i, j + 1):
            ranks[order[t]] = avg_rank
        i = j + 1
    rank_sum_pos = math.fsum(ranks[i] for i in range(len(scores)) if int(labels[i]) == 1)
    return (rank_sum_pos - pos * (pos + 1) / 2.0) / (pos * neg)


def selective_coverage_curve(
    confidences: Sequence[float],
    correct: Sequence[bool],
    coverages: Sequence[float] = (0.8, 0.6, 0.5, 0.3),
) -> dict[str, float]:
    """按置信度降序取前 c 比例，报该覆盖率下的准确率。

    这是"把不确定的交给人工"能不能落地的依据。openJev 的实测曲线：
    80% 覆盖 83.44% / 60% 89.00% / 50% 91.40% / 30% 95.17%。
    """
    if len(confidences) != len(correct):
        raise ValueError("长度不一致")
    if not confidences:
        return {}
    order = sorted(range(len(confidences)), key=lambda i: -float(confidences[i]))
    out: dict[str, float] = {}
    for c in coverages:
        k = max(1, int(round(len(order) * float(c))))
        top = order[:k]
        out[f"coverage_{float(c):.2f}"] = sum(1 for i in top if correct[i]) / k
    return out


# --------------------------------------------------------------------------- #
# 强制完整上报
# --------------------------------------------------------------------------- #
MANDATORY_REPORT_KEYS: tuple[str, ...] = (
    "accuracy",
    "soft_accuracy",
    "brier",
    "score_mae",
    "within_one_level",
    "ece_distribution",
    "ece_confidence",
    "permutation",
    "reference_floors",
    "n_samples",
    "n_reads_of_test",
)


def assert_complete_report(report: Mapping[str, Any]) -> None:
    """缺项直接抛错。这是把"不许挑指标"落成代码，而不是写成一句忠告。"""
    missing = [k for k in MANDATORY_REPORT_KEYS if k not in report]
    if missing:
        raise ValueError(
            "报告不完整，缺少: "
            + ", ".join(missing)
            + "。参照值必须同表出现，否则数字无法解读（见 openJev RUNBOOK §7）"
        )
    if not isinstance(report.get("permutation"), Mapping) or "flip_rate" not in report["permutation"]:
        raise ValueError("permutation 块必须包含 flip_rate")


def build_report(
    *,
    probs: Sequence[Sequence[float]],
    gold_idx: Sequence[int],
    gold_soft: Sequence[Sequence[float]],
    permutation: Mapping[str, Any],
    ece_confidence: float,
    confidence_scores: Sequence[float] | None = None,
    correctness: Sequence[bool] | None = None,
    n_reads_of_test: int = 1,
) -> dict[str, Any]:
    """组装一份**完整**报告。任何字段缺失都会在 ``assert_complete_report`` 处失败。"""
    pred_idx = [max(range(len(p)), key=lambda i: float(p[i])) for p in probs]
    confs = confidence_scores if confidence_scores is not None else [max(float(x) for x in p) for p in probs]
    corr = correctness if correctness is not None else [int(p) == int(g) for p, g in zip(pred_idx, gold_idx)]

    report: dict[str, Any] = {
        "n_samples": len(probs),
        "accuracy": accuracy(pred_idx, gold_idx),
        "soft_accuracy": soft_accuracy(pred_idx, gold_soft),
        "brier": brier(probs, gold_idx),
        "score_mae": score_mae(probs, gold_idx),
        "within_one_level": within_one_level(probs, gold_idx),
        "ece_distribution": ece(confs, corr),
        "ece_confidence": float(ece_confidence),
        "auroc_confidence": auroc(confs, [1 if c else 0 for c in corr]),
        "selective_coverage": selective_coverage_curve(confs, corr),
        "permutation": dict(permutation),
        "reference_floors": dict(REFERENCE_FLOORS),
        "vendor_reference": dict(VENDOR_REFERENCE),
        "n_reads_of_test": int(n_reads_of_test),
    }
    assert_complete_report(report)
    return report
