"""导出层：ONNX / TensorRT。

分层（与 ``jevlocal`` 顶层一致）：

- ``spec.py``   纯标准库 —— 形状契约、mask/归一化/confidence、TRT profile 与构建命令、引擎缓存指纹
- ``wrapper.py`` 需要 torch —— 可导出的 ``DecisionModule`` 与两段式前缀复用解码器
- ``to_onnx.py`` 需要 torch/onnx —— 导出命令与 ``check-env``（后者无重依赖也能跑）
- ``parity.py`` 需要 onnxruntime —— 数值一致性检查与**温度重拟合**
"""

from .spec import (  # noqa: F401
    CFG_FEATURE_WIDTH,
    N_SLOTS,
    ExportSpec,
    assert_export_contract,
    confidence,
    engine_cache_key,
    geometry_features,
    mask_and_normalize,
    noul_prob,
    ordinal_expectation,
    trt_command,
    trt_profile,
)

__all__ = [
    "CFG_FEATURE_WIDTH",
    "N_SLOTS",
    "ExportSpec",
    "assert_export_contract",
    "confidence",
    "engine_cache_key",
    "geometry_features",
    "mask_and_normalize",
    "noul_prob",
    "ordinal_expectation",
    "trt_command",
    "trt_profile",
]
