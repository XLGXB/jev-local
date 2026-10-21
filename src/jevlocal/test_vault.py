"""test 拆分金库：只读一次，每次读都留回执。

来源：openJev-verdict-2.0 的 RUNBOOK §6 原话 —— "The test split is a vault. Read it once
per candidate, after everything else is frozen"，且"Every read you make is a read you have
to report"。它把这件事做成了一个必须显式带 ``--confirm`` 的命令。

这里把同一纪律落成代码：不带 ``confirm`` 时**打印核对清单并拒绝执行**；
带 ``confirm`` 时先查回执，同一份 test 文件对同一个候选只允许读一次，然后追加回执。
这样做不是为了流程好看——test 读两次之后，第二次的读数就不再是"未见数据上的读数"，
所有后续调参都会被污染。
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from .schema import Sample, load_jsonl

ANTI_LEAK_CHECKLIST: tuple[str, ...] = (
    "1. 所有超参、损失权重、温度拟合都已在 dev 上定完，且**没有**因为 dev 结果反向改 test 相关代码。",
    "2. 温度与正确性头只在 calib 上拟合，calib 与 test 无 state 重叠（assert_no_leak 已过）。",
    "3. 教师软标签只由 teacher 产生，没有任何一条来自本模型自己的输出。",
    "4. 排列种子在训练前已冻结，评测用的排列与训练用的排列来自不同 seed。",
    "5. 我知道这次读完必须连同参照下限（TF-IDF 0.65 / ECE 0.03）一起上报，不能只报赢的指标。",
    "6. 这是我为该候选做的**第 1 次**读取；若有第 2 次，必须在报告里逐次说明。",
)


@dataclass(slots=True)
class TestReadReceipt:
    test_path: str
    test_sha256: str
    candidate_id: str
    n_samples: int
    read_at: str
    reads_so_far: int
    extras: dict[str, Any]


def file_sha256(path: str | Path, *, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def default_receipt_path(root: str | Path) -> Path:
    return Path(root) / "data" / "processed" / "test_reads.jsonl"


def _existing_reads(receipt_path: Path, test_sha: str, candidate_id: str) -> list[dict[str, Any]]:
    if not receipt_path.exists():
        return []
    out: list[dict[str, Any]] = []
    with receipt_path.open("r", encoding="utf-8", newline="") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("test_sha256") == test_sha and rec.get("candidate_id") == candidate_id:
                out.append(rec)
    return out


def print_checklist() -> None:
    print("=" * 72)
    print("test 金库读取前核对清单（逐条确认后再加 --confirm）")
    print("=" * 72)
    for item in ANTI_LEAK_CHECKLIST:
        print("  " + item)
    print("=" * 72)


def read_test_once(
    test_path: str | Path,
    *,
    receipt_path: str | Path,
    candidate_id: str,
    confirm: bool = False,
    allow_reread: bool = False,
    extras: dict[str, Any] | None = None,
) -> tuple[list[Sample] | None, TestReadReceipt | None]:
    """读 test 拆分。返回 ``(samples, receipt)``；未确认时返回 ``(None, None)``。"""
    test_path = Path(test_path)
    receipt_path = Path(receipt_path)
    if not test_path.exists():
        raise FileNotFoundError(f"test 拆分不存在: {test_path}")

    test_sha = file_sha256(test_path)
    prior = _existing_reads(receipt_path, test_sha, candidate_id)

    if not confirm:
        print_checklist()
        print(f"[拒绝读取] test={test_path} sha256={test_sha[:12]}… candidate={candidate_id}")
        print(f"（该候选在此文件上已读过 {len(prior)} 次。确认无污染后加 --confirm 重跑。）")
        return None, None

    if prior and not allow_reread:
        raise RuntimeError(
            f"拒绝重复读取：候选 {candidate_id} 已在 {test_path} 上读过 {len(prior)} 次"
            f"（最近一次 {prior[-1]['read_at']}）。"
            "若确实需要再读，请显式 --allow-reread，并在报告里逐次说明。"
        )

    samples = load_jsonl(test_path)
    receipt = TestReadReceipt(
        test_path=str(test_path),
        test_sha256=test_sha,
        candidate_id=candidate_id,
        n_samples=len(samples),
        read_at=datetime.now(timezone.utc).isoformat(),
        reads_so_far=len(prior) + 1,
        extras=dict(extras or {}),
    )

    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    with receipt_path.open("a", encoding="utf-8", newline="\n") as fh:  # 追加；显式 LF
        fh.write(
            json.dumps(
                {
                    "test_path": receipt.test_path,
                    "test_sha256": receipt.test_sha256,
                    "candidate_id": receipt.candidate_id,
                    "n_samples": receipt.n_samples,
                    "read_at": receipt.read_at,
                    "reads_so_far": receipt.reads_so_far,
                    "pid": os.getpid(),
                    "extras": receipt.extras,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        fh.write("\n")
    return samples, receipt


def n_reads(receipt_path: str | Path, candidate_id: str) -> int:
    receipt_path = Path(receipt_path)
    if not receipt_path.exists():
        return 0
    count = 0
    with receipt_path.open("r", encoding="utf-8", newline="") as fh:
        for line in fh:
            line = line.strip()
            if line and json.loads(line).get("candidate_id") == candidate_id:
                count += 1
    return count
