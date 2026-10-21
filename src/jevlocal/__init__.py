"""jevlocal —— 本地 Jev 类决策栈（数据格式 · 抽取 · SFT/RLCD · 校准 · 评测）。

分层与依赖约定（重要）
----------------------
**纯标准库层**（可在没有 GPU、没装 torch 的机器上跑测试）::

    jevlocal.schema     数据格式唯一真源
    jevlocal.text       两种主干的输入模板
    jevlocal.labels     标签算术 / proper scoring rule / 教师软标签 / 温度拟合
    jevlocal.permutation 顺序不变性
    jevlocal.metrics    指标与强制完整上报
    jevlocal.splits     切分 / 防泄漏 / 目标有效性
    jevlocal.test_vault test 金库一次性读

**torch 层**（只在训练时导入）::

    jevlocal.losses
    jevlocal.rlcd
    jevlocal.calibrate
    jevlocal.train_sft

上层的模块**不在模块级导入 torch**；数据侧应能在不安装 torch 的环境中
独立自测（``python tests/test_core.py``）。
"""

from __future__ import annotations

__version__ = "0.2.1"

from .schema import (  # noqa: F401
    DOMAINS,
    MAX_CHOICE_OPTIONS,
    QUESTION_TYPES,
    SPLITS,
    Label,
    Question,
    Sample,
    SchemaError,
    from_kev_record,
    load_jsonl,
    to_kev_record,
    write_jsonl,
)

__all__ = [
    "__version__",
    "DOMAINS",
    "MAX_CHOICE_OPTIONS",
    "QUESTION_TYPES",
    "SPLITS",
    "Label",
    "Question",
    "Sample",
    "SchemaError",
    "from_kev_record",
    "load_jsonl",
    "to_kev_record",
    "write_jsonl",
]
