"""双通道校准：类型路由温度 + out-of-fold 正确性头。

为什么必须是**两个**通道（来源：openJev-verdict-2.0 的架构说明）
--------------------------------------------------------------
软标签基准里有一个数学冲突：专家面板本身就带疑虑（平均顶概率 0.659）。
- 若模型贴合软分布，它的顶概率会停在 0.62 附近，而准确率是 0.77 ⇒ 对二元正确性而言
  校准误差升高（Laya：Brier 0.066、ECE 0.214）。
- 若模型把概率磨尖去对齐命中率，就背离面板 ⇒ Brier 变差（Jev：ECE 0.144、Brier 0.148）。

两条通道各自回答一个不同的问题，不能互相替代：
- **分布通道**：``max(p)`` 是否等于真实正确率 —— 用类型路由温度缩放修。
- **正确性通道**：单独一个头回答"我这次对不对" —— 必须 **out-of-fold** 训练，
  否则它会拿同一批数据既训练又校准，给出的 ECE 好看但没意义。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .labels import fit_temperature
from .schema import QUESTION_TYPES


# --------------------------------------------------------------------------- #
# 特征
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class GeometrySpec:
    n_features: int


def geometry_features(
    probs: torch.Tensor,
    qtype_ids: torch.Tensor,
    *,
    n_types: int = len(QUESTION_TYPES),
) -> torch.Tensor:
    """正确性头的输入：**预测几何**，不含任何原始文本。

    这样设计有两条理由：
    1. 头很小（几千参数），在 calib 集上就能训稳；
    2. 它读到的是"模型自己有多犹豫"，而不是"题面长什么样"，因此不会悄悄学到
       与具体数据集耦合的捷径。
    """
    p = probs.float().clamp_min(1e-9)
    logp = p.log()
    top2 = torch.topk(p, k=min(2, p.shape[-1]), dim=-1).values
    top1 = top2[:, 0]
    second = top2[:, 1] if top2.shape[-1] > 1 else torch.zeros_like(top1)
    margin = top1 - second
    ent = -(p * logp).sum(dim=-1)
    k = torch.full_like(top1, float(p.shape[-1]))
    norm_ent = ent / k.clamp_min(2.0).log()
    onehot = F.one_hot(qtype_ids.long(), num_classes=n_types).float()
    feats = torch.stack([top1, second, margin, ent, norm_ent, k.log()], dim=-1)
    return torch.cat([feats, onehot], dim=-1)


def spec_for(n_types: int = len(QUESTION_TYPES)) -> GeometrySpec:
    return GeometrySpec(n_features=6 + n_types)


# --------------------------------------------------------------------------- #
# 正确性头
# --------------------------------------------------------------------------- #
class CorrectnessHead(nn.Module):
    """小 MLP：几何特征 → "这次答对了"的 logit。

    刻意做小（单隐层 32 维）。它要学的只是"模型在什么形态下容易错"，
    容量一大就会去拟合 calib 集的噪声，OOF 里程反而更差。
    """

    def __init__(self, n_features: int, hidden: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return self.net(feats).squeeze(-1)


def oof_fit_correctness_head(
    feats: torch.Tensor,
    correct: torch.Tensor,
    *,
    n_folds: int = 5,
    epochs: int = 200,
    lr: float = 1e-2,
    seed: int = 20260922,
) -> tuple[CorrectnessHead, torch.Tensor]:
    """K 折 out-of-fold 训练，返回 (全量重训的头, OOF 分数)。

    返回的 OOF 分数**才是**用来算 ECE 的东西：每个样本的"正确性置信度"都来自一个
    没见过它的折。用 in-fold 分数算 ECE 会得到一个漂亮但无意义的数字。

    同时把 OOF 分数喂给温度拟合也不行——温度要在独立 calib 上拟合。
    """
    n = feats.shape[0]
    if n < n_folds * 10:
        raise ValueError(f"样本太少（{n}）无法做 {n_folds} 折 OOF，先补 calib 集")

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    folds = torch.chunk(perm, n_folds)
    oof = torch.zeros(n, dtype=torch.float32)

    for i in range(n_folds):
        val_idx = folds[i]
        train_idx = torch.cat([folds[j] for j in range(n_folds) if j != i])
        head = CorrectnessHead(feats.shape[-1])
        opt = torch.optim.AdamW(head.parameters(), lr=lr)
        for _ in range(epochs):
            opt.zero_grad()
            loss = F.binary_cross_entropy_with_logits(
                head(feats[train_idx]), correct[train_idx].float()
            )
            loss.backward()
            opt.step()
        with torch.no_grad():
            oof[val_idx] = torch.sigmoid(head(feats[val_idx]))

    full = CorrectnessHead(feats.shape[-1])
    opt = torch.optim.AdamW(full.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(full(feats), correct.float())
        loss.backward()
        opt.step()
    return full, oof


# --------------------------------------------------------------------------- #
# 类型路由温度
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class TypeRoutedTemperatures:
    """``{类型: 温度}``。kev 每 checkpoint 存一个温度（约 2.1–2.4）；
    openJev 走的是按类型路由，因为 noul / choice / score 三类的过度自信程度并不相同。"""

    by_type: dict[str, float]

    def get(self, qtype: str) -> float:
        return float(self.by_type.get(qtype, 1.0))

    def apply(self, logits: torch.Tensor, qtype: str) -> torch.Tensor:
        return logits.float() / max(self.get(qtype), 1e-6)

    def to_json(self) -> dict[str, float]:
        return dict(self.by_type)


def fit_type_routed_temperatures(
    logits_by_type: dict[str, torch.Tensor],
    hard_by_type: dict[str, torch.Tensor],
    *,
    fallback: float = 1.0,
) -> TypeRoutedTemperatures:
    """逐类型拟合。样本不足的类型退回 1.0 并在日志里点名。

    退回而不是硬拟合：几十条样本拟合出的温度会把 ECE 修得"好看"，但它衡量的是
    那几十条的噪声。``rlcd_readiness_check`` 会把这类问题提前列出来。
    """
    out: dict[str, float] = {}
    for qtype in QUESTION_TYPES:
        logits = logits_by_type.get(qtype)
        hard = hard_by_type.get(qtype)
        if logits is None or hard is None or logits.shape[0] < 50:
            out[qtype] = fallback
            continue
        out[qtype] = fit_temperature(logits.detach().cpu().tolist(), hard.detach().cpu().tolist())
    return TypeRoutedTemperatures(by_type=out)


def temperature_report(temps: TypeRoutedTemperatures) -> dict[str, float]:
    return {"T_" + k: float(v) for k, v in temps.by_type.items()}
