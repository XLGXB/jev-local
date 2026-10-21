"""导出的形状契约与**纯标准库**后处理。

为什么后处理放这里而不是 ``wrapper.py``：服务端拿到的是 ONNX Runtime 的输出
（numpy / list），**运行期根本不需要 torch**。把 mask/归一化/置信度实现成纯 Python，
意味着：

1. 服务端进程不必装 torch（少 2–3 GB 依赖，冷启动从十几秒降到几百毫秒）；
2. 这些逻辑可以在没装 torch 的机器上被单测覆盖——包括 ``confidence`` 这个
   每个下游阈值都依赖的公式。

本模块模块级不导入 torch / onnxruntime / numpy。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

from ..schema import MAX_CHOICE_OPTIONS

#: 固定槽数。256 = 2^8，覆盖 255 个候选后余 1 槽（与 Jev 报告的 256-slot 一致）。
N_SLOTS = 256

HeadKind = Literal["slots", "pointer"]


@dataclass(slots=True)
class ExportSpec:
    """导出契约。引擎构建、服务端后处理、Python 侧必须共用同一份。"""

    head: HeadKind = "slots"
    n_slots: int = N_SLOTS
    max_seq_len: int = 512
    max_batch: int = 8
    embed_dim: int = 768  # ModernBERT-base；large 是 1024
    dtype: str = "fp16"


def assert_export_contract(spec: ExportSpec) -> None:
    """导出前核对契约，避免"引擎建好了但形状不对"这种要跑一遍才知道的错误。"""
    if spec.head != "slots":
        raise ValueError("只有 slots 头有恒定形状契约；pointer 头请另行定义 shape profile")
    if spec.n_slots < MAX_CHOICE_OPTIONS:
        raise ValueError(
            f"槽数 {spec.n_slots} 小于候选上限 {MAX_CHOICE_OPTIONS}，合法问题将无法完整表达"
        )
    if spec.max_batch < 1:
        raise ValueError("max_batch 至少为 1")
    if spec.max_seq_len < 8:
        raise ValueError("max_seq_len 过小")
    if spec.embed_dim < 1:
        raise ValueError("embed_dim 必须为正")


# --------------------------------------------------------------------------- #
# 后处理（纯 Python，服务端热路径）
# --------------------------------------------------------------------------- #
def _softmax(values: Sequence[float]) -> list[float]:
    m = max(values)
    exps = [math.exp(v - m) for v in values]
    total = math.fsum(exps)
    return [e / total for e in exps]


def mask_and_normalize(slot_logits: Sequence[float], k: int, temperature: float = 1.0) -> list[float]:
    """把 256 槽切成有效的前 ``k`` 槽并归一化。

    这是整套部署里**最容易漏、漏了又最难发现**的一步：直接对全 256 槽 softmax
    形状完全正确、不抛任何异常，但概率质量被分给了不存在的候选，
    于是 ``confidence`` 与所有 ``keepThreshold`` 语义全部偏小。
    """
    if not (2 <= k <= MAX_CHOICE_OPTIONS):
        raise ValueError(f"k 必须在 [2,{MAX_CHOICE_OPTIONS}]，得到 {k}")
    if len(slot_logits) < k:
        raise ValueError(f"槽数 {len(slot_logits)} 小于 k={k}")
    if temperature <= 0:
        raise ValueError("temperature 必须为正")
    return _softmax([float(v) / temperature for v in slot_logits[:k]])


def confidence(probs: Sequence[float]) -> float:
    """System One 的 ``confidence``：``c = (p_max - 1/K) / (1 - 1/K)``。

    ⚠️ 这是**适配器算出来的**，不是模型学出来的。``c`` 只衡量"领先答案离均匀分布有多远"，
    **不是**"答案正确的概率"。把它当正确率用，是所有下游阈值出错的总根源；
    要"我这次对不对"，必须用独立训练并 out-of-fold 校准的 correctness 头。
    """
    k = len(probs)
    if k < 2:
        raise ValueError("K 至少为 2")
    return (max(float(p) for p in probs) - 1.0 / k) / (1.0 - 1.0 / k)


def noul_prob(probs: Sequence[float]) -> float:
    """``noul`` 的 P(yes)。约定第 0 槽 = yes。"""
    if len(probs) < 2:
        raise ValueError("noul 需要 2 个候选位")
    return float(probs[0])


def ordinal_expectation(probs: Sequence[float]) -> float:
    """``score`` 的期望层级（= 上游看到的 ``score``），层级从 0 起、低到高。"""
    return math.fsum(i * float(p) for i, p in enumerate(probs))


def entropy(probs: Sequence[float]) -> float:
    return -math.fsum(float(p) * math.log(max(float(p), 1e-12)) for p in probs if p > 0)


def geometry_features(probs: Sequence[float], qtype_id: int, n_types: int = 3) -> list[float]:
    """correctness 头的输入几何（与服务端同构、纯 Python）。

    必须与 ``jevlocal.calibrate.geometry_features``（torch 版）**逐位一致**，
    否则训练时学的头到服务端会拿到分布不同的输入。两边的顺序都固定为
    ``[top1, top2, margin, entropy, norm_entropy, log K, onehot(qtype)]``。
    """
    if len(probs) < 2:
        raise ValueError("至少需要 2 个候选")
    ranked = sorted((float(p) for p in probs), reverse=True)
    top1, top2 = ranked[0], ranked[1]
    k = len(probs)
    ent = entropy(probs)
    norm_ent = ent / math.log(max(k, 2))
    onehot = [1.0 if i == qtype_id else 0.0 for i in range(n_types)]
    return [top1, top2, top1 - top2, ent, norm_ent, math.log(k), *onehot]


CFG_FEATURE_WIDTH = 6  # + n_types


# --------------------------------------------------------------------------- #
# TensorRT shape profile 与构建命令
# --------------------------------------------------------------------------- #
def trt_profile(spec: ExportSpec) -> dict[str, str]:
    """三档 shape profile。**必须覆盖服务期真实会出现的形状**，
    否则 TRT 会拒绝该次推理。"""
    seq_lo, seq_opt, seq_hi = 64, min(256, spec.max_seq_len), spec.max_seq_len
    b_hi = spec.max_batch
    return {
        "minShapes": f"input_ids:1x{seq_lo},attention_mask:1x{seq_lo},read_index:1",
        "optShapes": f"input_ids:2x{seq_opt},attention_mask:2x{seq_opt},read_index:2",
        "maxShapes": f"input_ids:{b_hi}x{seq_hi},attention_mask:{b_hi}x{seq_hi},read_index:{b_hi}",
    }


def trt_command(onnx_path: str | Path, engine_path: str | Path, spec: ExportSpec | None = None) -> list[str]:
    spec = spec or ExportSpec()
    profile = trt_profile(spec)
    timing_cache = str(Path(engine_path).with_suffix(".timing.cache"))
    return [
        "trtexec",
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        "--fp16",
        f"--minShapes={profile['minShapes']}",
        f"--optShapes={profile['optShapes']}",
        f"--maxShapes={profile['maxShapes']}",
        f"--timingCacheFile={timing_cache}",
        "--builderOptimizationLevel=3",
    ]


def engine_cache_key(spec: ExportSpec, *, trt_version: str, cuda_version: str, compute_cap: str) -> str:
    """引擎缓存的指纹。

    TRT 引擎与 (TRT 版本, GPU 算力, CUDA 版本) 三者绑定；换了任一之后加载旧引擎
    要么直接失败，要么**跑出错的结果**（后者更糟）。把三者拼进文件名是最省事的做法。
    """
    return (
        f"{spec.head}-slots{spec.n_slots}-sm{compute_cap.replace('.', '')}"
        f"-trt{trt_version}-cu{cuda_version}-{spec.dtype}"
    )
