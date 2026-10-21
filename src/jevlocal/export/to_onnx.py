"""导出：PyTorch → ONNX → TensorRT，外加环境核查。

跑法::

    python -m jevlocal.export.to_onnx check-env
    python -m jevlocal.export.to_onnx modernbert --ckpt artifacts/sft-final --out artifacts/onnx/model.onnx
    python -m jevlocal.export.to_onnx qwen-kev   --ckpt runs/mine --out artifacts/onnx/qwen.onnx
    python -m jevlocal.export.to_onnx trt-cmd    --onnx artifacts/onnx/model.onnx --engine artifacts/trt/model.plan

``check-env`` 是唯一在**没装任何重依赖**的机器上也能跑的入口——它逐项探测并给出结论，
而不是让你跑一半才发现缺 CUDA Toolkit。
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from .spec import N_SLOTS, ExportSpec, assert_export_contract, trt_command, trt_profile

# --------------------------------------------------------------------------- #
# 版本事实（2026-09-22 查证，来源见 docs/DEPLOYMENT.md §1）
# --------------------------------------------------------------------------- #
#: onnxruntime-gpu 有 cp313-win_amd64 轮子，Python 3.11–3.14 覆盖（最新 1.30.0 / 2026-09-10）
ORT_PY_SUPPORT = (3, 11, 3, 14)
#: TensorRT 官方前置：算力 Turing+ / CUDA Toolkit 12.x 或 13.x / Windows 驱动 r537+ / Python 3.10–3.13
TRT_PY_SUPPORT = (3, 10, 3, 13)
TRT_MIN_DRIVER_WINDOWS = 537
TRT_MIN_COMPUTE_CAP = 7.5
#: ONNX-TensorRT 解析器：onnx 1.20.0、opset 支持到 25、官方支持 opset 9+
TRT_ONNX_MODEL_VERSION = "1.20.0"
TRT_MAX_OPSET = 25
EXPORT_OPSET = 17


@dataclass
class EnvReport:
    python: str
    platform: str
    driver_version: str | None
    gpu_name: str | None
    compute_cap: str | None
    cuda_toolkit: str | None
    torch: str | None
    onnx: str | None
    onnxruntime: str | None
    ort_providers: list[str]
    tensorrt: str | None
    problems: list[str]
    warnings: list[str]

    def verdict(self) -> str:
        if self.problems:
            return "BLOCKED"
        if self.warnings:
            return "DEGRADED"
        return "READY"


def _sh(cmd: Sequence[str], timeout: int = 20) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    except Exception:
        return None
    if out.returncode != 0:
        return None
    return (out.stdout or "").strip() or None


def _nvidia_smi() -> tuple[str | None, str | None, str | None]:
    raw = _sh(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,compute_cap",
            "--format=csv,noheader",
        ]
    )
    if not raw:
        return None, None, None
    first = raw.splitlines()[0]
    parts = [p.strip() for p in first.split(",")]
    if len(parts) < 3:
        return None, None, None
    return parts[0], parts[1], parts[2]


def _cuda_toolkit() -> str | None:
    candidates = [
        Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"),
        Path("/usr/local/cuda"),
    ]
    for base in candidates:
        if base.exists():
            versions = sorted(p.name for p in base.iterdir() if p.is_dir())
            if versions:
                return str(base / versions[-1])
            return str(base)
    if os.environ.get("CUDA_HOME"):
        return os.environ["CUDA_HOME"]
    return None


def _module_version(name: str) -> tuple[str | None, Any]:
    try:
        mod = __import__(name)
    except Exception:
        return None, None
    return getattr(mod, "__version__", "unknown"), mod


def check_env() -> EnvReport:
    gpu, driver, cc = _nvidia_smi()
    cuda = _cuda_toolkit()

    torch_v, _ = _module_version("torch")
    onnx_v, _ = _module_version("onnx")
    ort_v, ort = _module_version("onnxruntime")
    trt_v, _ = _module_version("tensorrt")

    providers: list[str] = []
    if ort is not None:
        try:
            providers = list(ort.get_available_providers())
        except Exception:
            providers = []

    problems: list[str] = []
    warnings: list[str] = []

    pv = sys.version_info
    py_tuple = (pv.major, pv.minor)

    if ort_v is None:
        problems.append("未安装 onnxruntime-gpu（GPU 版，不是 CPU 版 onnxruntime）")
    elif not (ORT_PY_SUPPORT[0], ORT_PY_SUPPORT[1]) <= py_tuple <= (ORT_PY_SUPPORT[2], ORT_PY_SUPPORT[3]):
        problems.append(f"onnxruntime 对 Python {pv.major}.{pv.minor} 无轮子（支持 {ORT_PY_SUPPORT}）")

    if trt_v is None:
        problems.append("未安装 tensorrt（pip install tensorrt）")
    elif not (TRT_PY_SUPPORT[0], TRT_PY_SUPPORT[1]) <= py_tuple <= (TRT_PY_SUPPORT[2], TRT_PY_SUPPORT[3]):
        problems.append(
            f"TensorRT 官方 Python 支持区间为 {TRT_PY_SUPPORT}，当前 {pv.major}.{pv.minor}——"
            "需要单独一个 3.11/3.12 的 venv（两个解释器可以共存，互不污染）"
        )

    if torch_v is None:
        problems.append("未安装 torch（导出源就是它）")
    if onnx_v is None:
        warnings.append("未安装 onnx / onnxsim（简化与常量折叠需要）")

    if cuda is None:
        problems.append(
            "未找到 CUDA Toolkit（TensorRT 官方前置要求 CUDA Toolkit 12.x 或 13.x）。"
            "注意：C 盘仅剩 42GB、D 盘 76GB，装之前先算空间，优先装到 E/F/G"
        )

    if driver is not None:
        try:
            if int(float(driver.split(".")[0])) < TRT_MIN_DRIVER_WINDOWS:
                problems.append(f"驱动 {driver} 低于 TensorRT 的 Windows 下限 r{TRT_MIN_DRIVER_WINDOWS}")
        except ValueError:
            warnings.append(f"驱动版本 {driver!r} 无法解析")

    if cc:
        try:
            major, minor = (int(x) for x in cc.split("."))
            if (major, minor) < (7, 5):
                problems.append(f"算力 {cc} 低于 TensorRT 的 Turing 下限（7.5）")
        except ValueError:
            warnings.append(f"算力 {cc!r} 无法解析")

    if ort_v is not None and providers and "CUDAExecutionProvider" not in providers:
        warnings.append(
            "onnxruntime 没给出 CUDAExecutionProvider——通常是 CUDA/cuDNN 版本不匹配，"
            "此时会静默退回 CPU，延迟差一个数量级"
        )
    if ort_v is not None and "TensorrtExecutionProvider" in providers:
        pass
    elif ort_v is not None and trt_v is not None:
        warnings.append("装了 tensorrt 但 ORT 未列出 TensorrtExecutionProvider，检查两者 ABI/版本是否配套")

    return EnvReport(
        python=f"{pv.major}.{pv.minor}.{pv.micro}",
        platform=f"{platform.system()} {platform.release()} ({platform.machine()})",
        driver_version=driver,
        gpu_name=gpu,
        compute_cap=cc,
        cuda_toolkit=cuda,
        torch=torch_v,
        onnx=onnx_v,
        onnxruntime=ort_v,
        ort_providers=providers,
        tensorrt=trt_v,
        problems=problems,
        warnings=warnings,
    )


def print_env(report: EnvReport) -> None:
    print("=" * 76)
    print("导出台架环境核查")
    print("=" * 76)
    for key, value in asdict(report).items():
        if key in ("problems", "warnings", "ort_providers"):
            continue
        print(f"  {key:18} {value}")
    print(f"  {'ort_providers':18} {', '.join(report.ort_providers) or '(none)'}")
    print("-" * 76)
    for w in report.warnings:
        print(f"  [警告] {w}")
    for p in report.problems:
        print(f"  [阻断] {p}")
    print("-" * 76)
    print(f"  结论: {report.verdict()}")
    print("=" * 76)


# --------------------------------------------------------------------------- #
# ONNX 导出
# --------------------------------------------------------------------------- #
def export_modernbert(ckpt: str | Path, out: str | Path, spec: ExportSpec | None = None) -> Path:
    """把 ``DecisionModule`` 导出成 ONNX。

    动态轴只为 **batch** 与 **seq** 开；``read_index`` 只开 batch。
    ``n_slots`` 是常量形状——这正是 slots 头的价值所在：K 变化不改变图。
    """
    import torch  # noqa: PLC0415

    from .wrapper import DecisionModule  # noqa: PLC0415

    spec = spec or ExportSpec()
    assert_export_contract(spec)

    ckpt = Path(ckpt)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    payload = torch.load(ckpt / "decision_module.pt", map_location="cpu", weights_only=False)
    module = DecisionModule(
        payload["backbone"],
        hidden_size=int(payload["hidden_size"]),
        n_slots=spec.n_slots,
        head=spec.head,
    )
    module.load_state_dict(payload["state_dict"], strict=True)
    module.eval()

    b, seq = 1, min(spec.max_seq_len, 128)
    dummy = (
        torch.zeros(b, seq, dtype=torch.int64),
        torch.ones(b, seq, dtype=torch.int64),
        torch.zeros(b, dtype=torch.int64),
    )

    with torch.no_grad():
        torch.onnx.export(
            module,
            dummy,
            str(out),
            input_names=["input_ids", "attention_mask", "read_index"],
            output_names=["slot_logits", "read_hidden"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "seq"},
                "attention_mask": {0: "batch", 1: "seq"},
                "read_index": {0: "batch"},
                "slot_logits": {0: "batch"},
                "read_hidden": {0: "batch"},
            },
            opset_version=EXPORT_OPSET,
            do_constant_folding=True,
            dynamo=False,
        )
    print(f"导出完成 → {out}")
    print("下一步（可选）onnxsim 简化，然后跑 parity 数值一致性检查：")
    print(f"  python -m jevlocal.export.parity --ckpt {ckpt} --onnx {out}")
    return out


def export_qwen_kev(ckpt: str | Path, out: str | Path, spec: ExportSpec | None = None) -> Path:
    """Qwen-Kev 路径导出：**先合 LoRA，再带 KV cache 导出**。

    与 ModernBERT 路径的本质差异（决定了能不能真正"state 只编码一次"）::

        ModernBERT  双向注意力：state 的表示**依赖**问题与候选 token
                    ⇒ 不能缓存 state 再换问题复用，只能整条重算（靠 batch 摊薄）
        Qwen-Kev    因果注意力：state 是**前缀**，问题只读它、它不读问题
                    ⇒ 可以缓存 state 的 KV，逐个问题只跑后缀

    所以：**要"state 编码一次"，就用因果主干（kev 路）；用编码器主干（ModernBERT 路），
    就把多个问题压进同一个 batch。** 这是两条路线在部署侧的真正取舍点。
    """
    import torch  # noqa: PLC0415

    spec = spec or ExportSpec(embed_dim=1024)
    ckpt = Path(ckpt)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    from transformers import AutoModelForCausalLM  # noqa: PLC0415

    model = AutoModelForCausalLM.from_pretrained(str(ckpt), torch_dtype=torch.float32)

    # 1) 合 LoRA。忘了这一步会导出一个"底座 + 未生效的 adapter"，
    #    推理结果与训练时的评测值不符，而且不会报错。
    if hasattr(model, "merge_and_unload"):
        try:
            model = model.merge_and_unload()
            print("已合并 LoRA adapter 到基座权重")
        except Exception as exc:  # 不是 PeftModel 时正常发生
            print(f"未检测到 LoRA（{type(exc).__name__}），按全量权重导出")
    else:
        print("警告：未能合并 LoRA，请确认已在 PEFT 包装器上调用 merge_and_unload()")

    model.eval()
    b, seq = 1, min(spec.max_seq_len, 128)
    dummy = torch.zeros(b, seq, dtype=torch.int64)

    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy,
            str(out),
            input_names=["input_ids"],
            output_names=["logits", "present_key_values"],
            dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "logits": {0: "batch", 1: "seq"}},
            opset_version=EXPORT_OPSET,
            do_constant_folding=True,
            dynamo=False,
        )
    print(f"导出完成 → {out}")
    print("注意：带 KV cache 的图有可变数量的 past/present 输入输出，")
    print("      TensorRT 侧要用动态 shape profile 且逐层命名，容易出错；")
    print("      建议先用 ORT 的 CUDAExecutionProvider 跑通，再决定是否上 TRT。")
    return out


# --------------------------------------------------------------------------- #
# TensorRT 构建（trt_profile / trt_command 在 spec.py，纯标准库）
# --------------------------------------------------------------------------- #
def print_trt_command(onnx_path: str | Path, engine_path: str | Path) -> None:
    cmd = trt_command(onnx_path, engine_path)
    print(" ".join(cmd))
    print()
    print("引擎缓存失效条件（三者任一变化都必须重建，否则会加载失败或结果错误）：")
    print("  • TensorRT 版本    • GPU 架构（算力）    • CUDA 版本")
    print("  建议把这三者拼进引擎文件名，例如 model_sm89_trt10.15_cu12.9.plan")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="jevlocal.export.to_onnx", description="ONNX/TensorRT 导出台架")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("check-env", help="逐项探测环境并给出结论（无重依赖也能跑）")

    mb = sub.add_parser("modernbert", help="导出 ModernBERT 决策模块")
    mb.add_argument("--ckpt", required=True)
    mb.add_argument("--out", required=True)
    mb.add_argument("--n-slots", type=int, default=N_SLOTS)
    mb.add_argument("--max-seq-len", type=int, default=512)
    mb.add_argument("--max-batch", type=int, default=8)
    mb.add_argument("--hidden-size", type=int, default=768)

    qk = sub.add_parser("qwen-kev", help="导出 Qwen-Kev（合 LoRA + KV cache）")
    qk.add_argument("--ckpt", required=True)
    qk.add_argument("--out", required=True)
    qk.add_argument("--max-seq-len", type=int, default=1024)
    qk.add_argument("--max-batch", type=int, default=8)

    tc = sub.add_parser("trt-cmd", help="打印 trtexec 构建命令与失效条件")
    tc.add_argument("--onnx", required=True)
    tc.add_argument("--engine", required=True)
    tc.add_argument("--max-seq-len", type=int, default=512)
    tc.add_argument("--max-batch", type=int, default=8)

    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.cmd == "check-env":
        report = check_env()
        print_env(report)
        if os.environ.get("JEVLOCAL_ENV_JSON"):
            print(json.dumps(asdict(report), ensure_ascii=False, indent=2))
        return 0 if report.verdict() != "BLOCKED" else 1

    if args.cmd == "modernbert":
        spec = ExportSpec(
            n_slots=args.n_slots, max_seq_len=args.max_seq_len, max_batch=args.max_batch, embed_dim=args.hidden_size
        )
        export_modernbert(args.ckpt, args.out, spec)
        return 0

    if args.cmd == "qwen-kev":
        spec = ExportSpec(max_seq_len=args.max_seq_len, max_batch=args.max_batch, embed_dim=1024)
        export_qwen_kev(args.ckpt, args.out, spec)
        return 0

    if args.cmd == "trt-cmd":
        print_trt_command(args.onnx, args.engine)
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
