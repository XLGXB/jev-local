"""数值一致性检查与**导出后温度重拟合**。

为什么这一步不能省
------------------
训练时拟合的温度（kev 的 checkpoint 里存着约 2.1–2.4；openJev 是按类型路由的）
是在 **PyTorch fp32** 的 logits 上拟合的。导出到 ONNX 再到 TensorRT fp16 之后：

- logits 的**尺度**会变（fp16 舍入、kernel 融合改变了累加顺序）；
- 温度的语义是"把 logits 除以 T"，尺度一变，原来的 T 就不再对应原来的校准。

于是会出现一种很难查的现象：**argmax 准确率几乎没变，但 ECE 明显变差**，
下游所有按 confidence 分支的阈值集体偏了。

结论：导出的 T 必须**在导出的图上重拟合**（用 calib 切分），不能沿用 PyTorch 侧的 T。
本模块的纯函数部分不依赖 torch / onnxruntime，可单独单测。
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from .spec import mask_and_normalize

#: 验收门。fp16 下 logits 尺度差异通常远小于此；超过说明导出出了问题，
#: 常见原因是忘了合并 LoRA、或者读数位偏移算错。
MAX_ABS_LOGIT_DELTA = 0.05
MIN_ARGMAX_AGREEMENT = 0.995
MIN_TOP1_AGREEMENT = 0.99


@dataclass
class ParityReport:
    n: int
    argmax_agreement: float
    top1_agreement: float
    max_abs_delta: float
    mean_abs_delta: float
    temperature_before: float
    temperature_after: float
    ece_before: float
    ece_after: float
    problems: list[str] = field(default_factory=list)

    @property
    def verdict(self) -> str:
        return "FAIL" if self.problems else "PASS"

    def to_json(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "argmax_agreement": self.argmax_agreement,
            "top1_agreement": self.top1_agreement,
            "max_abs_delta": self.max_abs_delta,
            "mean_abs_delta": self.mean_abs_delta,
            "temperature_before": self.temperature_before,
            "temperature_after": self.temperature_after,
            "ece_before": self.ece_before,
            "ece_after": self.ece_after,
            "verdict": self.verdict,
            "problems": self.problems,
        }


# --------------------------------------------------------------------------- #
# 纯函数（可用 list 直接单测）
# --------------------------------------------------------------------------- #
def argmax(row: Sequence[float]) -> int:
    return max(range(len(row)), key=lambda i: float(row[i]))


def compare_predictions(
    reference: Sequence[Sequence[float]],
    candidate: Sequence[Sequence[float]],
) -> tuple[float, float, float, float]:
    """返回 (argmax 一致率, top1 概率一致率, 最大绝对差, 平均绝对差)。

    ``top1`` 一致率用相对容差比较，而不是逐位相等：fp16 下末位必然不同，
    用 ``==`` 会把正常的导出判成失败。
    """
    if len(reference) != len(candidate):
        raise ValueError("两组预测数量不一致")
    if not reference:
        raise ValueError("空输入")

    n = len(reference)
    argmax_hits = 0
    top1_hits = 0
    max_delta = 0.0
    total_delta = 0.0
    for ref, cand in zip(reference, candidate):
        if len(ref) != len(cand):
            raise ValueError("宽度不一致")
        if argmax(ref) == argmax(cand):
            argmax_hits += 1
        r_top = max(float(x) for x in ref)
        c_top = max(float(x) for x in cand)
        if abs(r_top - c_top) <= max(1e-3, 0.02 * abs(r_top)):
            top1_hits += 1
        for a, b in zip(ref, cand):
            d = abs(float(a) - float(b))
            max_delta = max(max_delta, d)
            total_delta += d
    width = len(reference[0])
    return (
        argmax_hits / n,
        top1_hits / n,
        max_delta,
        total_delta / (n * width),
    )


def ece_from_logits(
    logits: Sequence[Sequence[float]],
    labels: Sequence[int],
    ks: Sequence[int],
    temperature: float,
    *,
    n_bins: int = 10,
) -> float:
    """在给定温度下算分布通道的 ECE（置信度 = max p）。"""
    if not (len(logits) == len(labels) == len(ks)):
        raise ValueError("长度不一致")
    bins: list[list[tuple[float, bool]]] = [[] for _ in range(n_bins)]
    for row, y, k in zip(logits, labels, ks):
        probs = mask_and_normalize(row, k, temperature)
        conf = max(probs)
        ok = argmax(probs) == int(y)
        bins[min(n_bins - 1, max(0, int(conf * n_bins)))].append((conf, ok))
    total = len(logits)
    err = 0.0
    for bucket in bins:
        if not bucket:
            continue
        avg_conf = math.fsum(c for c, _ in bucket) / len(bucket)
        acc = sum(1 for _, ok in bucket if ok) / len(bucket)
        err += (len(bucket) / total) * abs(acc - avg_conf)
    return err


def fit_temperature_1d(
    logits: Sequence[Sequence[float]],
    labels: Sequence[int],
    ks: Sequence[int],
    *,
    lo: float = 0.05,
    hi: float = 20.0,
    grid: int = 300,
    refine: int = 6,
) -> float:
    """在**给定量化后的 logits** 上重新拟合温度（对数网格 + 缩窗精修）。

    与 ``jevlocal.labels.fit_temperature`` 同一算法，但这里按 k 截断后归一化，
    与部署路径完全一致——用训练侧的拟合函数会漏掉 mask 这一步。
    """
    if not logits:
        raise ValueError("空输入")

    def nll(t: float) -> float:
        total = 0.0
        for row, y, k in zip(logits, labels, ks):
            p = mask_and_normalize(row, k, t)
            total -= math.log(max(p[int(y)], 1e-12))
        return total / len(logits)

    lo_l, hi_l = math.log(lo), math.log(hi)
    best_t, best_v = 1.0, math.inf
    for i in range(grid + 1):
        t = math.exp(lo_l + (hi_l - lo_l) * i / grid)
        v = nll(t)
        if v < best_v:
            best_t, best_v = t, v
    span = (hi_l - lo_l) / grid
    for _ in range(refine):
        span /= 4.0
        for cand in (best_t * math.exp(-span), best_t, best_t * math.exp(span)):
            cand = min(max(cand, lo), hi)
            v = nll(cand)
            if v < best_v:
                best_t, best_v = cand, v
    return best_t


def build_parity_report(
    reference_logits: Sequence[Sequence[float]],
    exported_logits: Sequence[Sequence[float]],
    labels: Sequence[int],
    ks: Sequence[int],
    *,
    temperature_before: float = 1.0,
) -> ParityReport:
    """把一致性 + 温度重拟合 + 前后 ECE 合成一份报告，并给出 PASS/FAIL 依据。"""
    agreement, top1, max_delta, mean_delta = compare_predictions(reference_logits, exported_logits)

    ece_before = ece_from_logits(exported_logits, labels, ks, temperature_before)
    t_after = fit_temperature_1d(exported_logits, labels, ks)
    ece_after = ece_from_logits(exported_logits, labels, ks, t_after)

    problems: list[str] = []
    if max_delta > MAX_ABS_LOGIT_DELTA:
        problems.append(
            f"最大 logit 绝对差 {max_delta:.4f} 超过 {MAX_ABS_LOGIT_DELTA}——"
            "常见原因：忘了合并 LoRA、读数位偏移算错、或导出用了错误的 opset"
        )
    if agreement < MIN_ARGMAX_AGREEMENT:
        problems.append(f"argmax 一致率 {agreement:.4f} 低于 {MIN_ARGMAX_AGREEMENT}")
    if top1 < MIN_TOP1_AGREEMENT:
        problems.append(f"top1 概率一致率 {top1:.4f} 低于 {MIN_TOP1_AGREEMENT}")
    if ece_after > ece_before + 1e-9:
        problems.append(
            f"重拟合后 ECE 反而变差（{ece_before:.4f} → {ece_after:.4f}），说明拟合过程有 bug，"
            "不要带着这个温度上线"
        )

    return ParityReport(
        n=len(reference_logits),
        argmax_agreement=agreement,
        top1_agreement=top1,
        max_abs_delta=max_delta,
        mean_abs_delta=mean_delta,
        temperature_before=temperature_before,
        temperature_after=t_after,
        ece_before=ece_before,
        ece_after=ece_after,
        problems=problems,
    )


# --------------------------------------------------------------------------- #
# 运行时（需要 onnxruntime / torch，惰性导入）
# --------------------------------------------------------------------------- #
def run_onnx(onnx_path: str | Path, batch: dict[str, Any]) -> Any:
    """跑一次 ONNX 推理。默认优先 TensorRT EP，退回 CUDA，最后 CPU。

    注意记录**实际生效的 provider**：静默退回 CPU 会让延迟差一个数量级，
    而结果仍然"正确"，所以只看结果永远发现不了。
    """
    import numpy as np  # noqa: PLC0415
    import onnxruntime as ort  # noqa: PLC0415

    available = ort.get_available_providers()
    order = [p for p in ("TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider") if p in available]
    if not order:
        raise RuntimeError(f"没有可用的执行提供程序，可用: {available}")

    sess = ort.InferenceSession(str(onnx_path), providers=order)
    feeds = {k: np.asarray(v) for k, v in batch.items()}
    out = sess.run(None, feeds)
    return {"outputs": out, "active_providers": sess.get_providers()}


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="jevlocal.export.parity", description="导出后数值一致性 + 温度重拟合")
    p.add_argument("--reference", required=True, help="torch 侧 logits 的 .json（二维数组）")
    p.add_argument("--exported", required=True, help="导出侧 logits 的 .json（二维数组）")
    p.add_argument("--labels", required=True, help="hard 下标 .json（一维数组）")
    p.add_argument("--ks", required=True, help="每个样本的 K .json（一维数组）")
    p.add_argument("--temperature-before", type=float, default=1.0)
    p.add_argument("--out", default="")
    args = p.parse_args(argv)

    load = lambda path: json.loads(Path(path).read_text(encoding="utf-8"))  # noqa: E731
    report = build_parity_report(
        load(args.reference),
        load(args.exported),
        load(args.labels),
        load(args.ks),
        temperature_before=args.temperature_before,
    )
    text = json.dumps(report.to_json(), ensure_ascii=False, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    return 0 if report.verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
