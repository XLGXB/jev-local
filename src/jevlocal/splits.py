"""切分、防泄漏、目标有效性。

三条不能让步的规矩
------------------
1. **按 state 分组切分**，不是按样本切分。同一个 state 出现在 train 与 test 里是
   最隐蔽的泄漏：模型背下这一段上下文就能"泛化"，报告上的准确率全假。
2. **test 是金库**：只读一次，每次读都要留回执（见 ``test_vault.py``）。
3. **不可回答的样本，正确目标是均匀分布**，不是多数类。kev 2026-09-21 的那次二次训练
   专门把"决定证据被移除"的样本训向均匀答案——把它当普通样本训会教出一个
   在没证据时也很自信的模型，而这类样本恰恰是下游要触发升级/人工的地方。
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from .schema import Sample, SchemaError

DEFAULT_RATIOS: dict[str, float] = {"train": 0.70, "dev": 0.10, "calib": 0.10, "test": 0.10}


# --------------------------------------------------------------------------- #
# 分组切分
# --------------------------------------------------------------------------- #
def assign_splits(
    samples: Sequence[Sample],
    *,
    ratios: Mapping[str, float] = DEFAULT_RATIOS,
    seed: int = 20260922,
    group_by: str = "state_sha256",
) -> list[Sample]:
    """按组切分并**就地返回新 split 的样本副本**（不改入参）。

    组键默认取 ``state_sha256``：同一段上下文的所有问题必须同进同出。
    用固定 seed 的 ``random.Random``，保证复现。
    """
    total_ratio = math.fsum(ratios.values())
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"比例和不为 1: {total_ratio}")

    groups: dict[str, list[Sample]] = defaultdict(list)
    for s in samples:
        key = s.state_sha256 if group_by == "state_sha256" else s.fingerprint()
        groups[key].append(s)

    keys = sorted(groups)  # 先排序再打乱：否则 dict 顺序会让结果依赖输入顺序
    rng = random.Random(seed)
    rng.shuffle(keys)

    # 目标组数（按样本量近似），贪心填入
    per_split_target = {
        name: int(round(len(samples) * r)) for name, r in ratios.items()
    }
    counts = {name: 0 for name in ratios}
    order = sorted(ratios, key=lambda n: -ratios[n])  # 大桶先放，避免碎尾

    assigned: dict[str, str] = {}
    for key in keys:
        size = len(groups[key])
        best = min(order, key=lambda n: (counts[n] / max(per_split_target[n], 1)))
        assigned[key] = best
        counts[best] += size

    out: list[Sample] = []
    for key in keys:
        split = assigned[key]
        for s in groups[key]:
            clone = Sample.from_json(s.to_json())
            clone.split = split
            out.append(clone)
    return out


# --------------------------------------------------------------------------- #
# 防泄漏
# --------------------------------------------------------------------------- #
def assert_no_leak(samples: Sequence[Sample], *, strict_state: bool = True) -> dict[str, int]:
    """跨 split 重复检查。发现重复就抛错——静默去重会掩盖抽取器里的 bug。"""
    by_fp: dict[str, set[str]] = defaultdict(set)
    by_state: dict[str, set[str]] = defaultdict(set)
    for s in samples:
        by_fp[s.fingerprint()].add(s.split)
        by_state[s.state_sha256].add(s.split)

    dup_fp = {k: v for k, v in by_fp.items() if len(v) > 1}
    dup_state = {k: v for k, v in by_state.items() if len(v) > 1}
    if dup_fp:
        first = next(iter(dup_fp))
        raise SchemaError(
            f"样本指纹跨切分重复（{len(dup_fp)} 例），首例 {first} 出现在 {sorted(dup_fp[first])}"
        )
    if strict_state and dup_state:
        first = next(iter(dup_state))
        raise SchemaError(
            f"state 跨切分重复（{len(dup_state)} 例），首例 {first} 出现在 {sorted(dup_state[first])}。"
            "同一段上下文的样本必须同进同出，否则是隐蔽泄漏"
        )
    return {
        "n_samples": len(samples),
        "n_unique_fingerprints": len(by_fp),
        "n_unique_states": len(by_state),
    }


def split_counts(samples: Sequence[Sample]) -> dict[str, int]:
    out: dict[str, int] = defaultdict(int)
    for s in samples:
        out[s.split] += 1
    return dict(sorted(out.items()))


# --------------------------------------------------------------------------- #
# 目标有效性
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class ValidityReport:
    kept: int
    dropped: list[tuple[str, str]]  # (sample_id, 原因)

    @property
    def dropped_count(self) -> int:
        return len(self.dropped)

    def summary(self) -> dict[str, object]:
        reasons: dict[str, int] = defaultdict(int)
        for _, why in self.dropped:
            reasons[why] += 1
        return {"kept": self.kept, "dropped": self.dropped_count, "reasons": dict(reasons)}


def target_validity_filter(
    samples: Iterable[Sample],
    *,
    min_state_chars: int = 16,
    drop_uniform_with_hard: bool = True,
) -> tuple[list[Sample], ValidityReport]:
    """丢掉"目标无法由 state 支撑"的样本。

    NanoJev 报过这个门：10,898 条训练问题里 10,893 条通过（丢 5 条）。
    丢掉的样本必须**留名留因**，不能静默删——静默过滤会把"抽取器坏了"伪装成
    "数据很干净"。
    """
    kept: list[Sample] = []
    dropped: list[tuple[str, str]] = []

    for s in samples:
        if len(s.state.strip()) < min_state_chars:
            dropped.append((s.sample_id, f"state 过短(<{min_state_chars} 字符)"))
            continue
        bad = None
        for q in s.questions:
            lab = s.labels[q.qid]
            if not lab.unanswerable and lab.hard is None:
                bad = f"{q.qid}: 非 unanswerable 却缺 hard 标签"
                break
            if drop_uniform_with_hard and not lab.unanswerable and lab.soft_is_uniform(q):
                bad = f"{q.qid}: soft 均匀却带着 hard 标签（目标自相矛盾）"
                break
        if bad:
            dropped.append((s.sample_id, bad))
            continue
        kept.append(s)

    return kept, ValidityReport(kept=len(kept), dropped=dropped)


def mark_unanswerable(sample: Sample, qids: Sequence[str], *, reason: str) -> Sample:
    """把指定问题标为"state 里没有判定证据"，目标改成均匀分布、hard 置空。

    这是 kev 那次二次训练的做法在数据侧的对应物。
    """
    from .labels import Label, uniform  # 局部导入，避免循环

    clone = Sample.from_json(sample.to_json())
    for qid in qids:
        q = next((x for x in clone.questions if x.qid == qid), None)
        if q is None:
            raise SchemaError(f"{sample.sample_id}: 找不到 qid {qid}")
        k = q.n_options
        if q.type == "noul":
            clone.labels[qid] = Label(hard=None, soft=0.5, unanswerable=True)
        else:
            clone.labels[qid] = Label(hard=None, soft=uniform(k), unanswerable=True)
    clone.provenance.setdefault("notes", "")
    clone.provenance["notes"] = (str(clone.provenance.get("notes", "")) + f" | {reason}").strip(" |")
    return clone
