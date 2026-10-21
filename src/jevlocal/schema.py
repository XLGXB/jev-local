"""决策样本的唯一真源（单一定义处）。

设计约束
--------
- **只用标准库**，不 import torch / transformers。数据格式与校验必须能在
  没有 GPU、也没装 torch 的环境中测试；``schema`` / ``text`` / ``labels`` /
  ``metrics`` 四个模块保持零依赖可跑。
- 新字段先加在这里，再同步 ``data/schema/decision_sample.schema.json``。
- 规范标签形状（三大类型各自固定一种）::

      noul   hard: 0 | 1        soft: float in [0,1]
      choice hard: "选项名"      soft: {选项名: p}        键集必须等于 criteria 键集
      score  hard: 层级下标       soft: [p0 .. p{K-1}]    低到高，长度等于 criteria 长度

  这套形状与 kev 的 JSONL ``label`` 语义一一对应（choice 用选项名、noul 用 bool、
  score 用从 0 起的下标），所以 ``to_kev_record`` 可以无损互转。
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Literal, Mapping, Sequence

QUESTION_TYPES: tuple[str, ...] = ("noul", "choice", "score")
SPLITS: tuple[str, ...] = ("train", "dev", "calib", "test", "ood")
DOMAINS: tuple[str, ...] = (
    "browser_action",
    "stdout_prune",
    "tool_keepdrop",
    "routing",
    "other",
)
SOURCE_KINDS: tuple[str, ...] = (
    "codex_session",
    "codex_history",
    "runtime_json",
    "browser_trace",
    "hand_authored",
    "synthetic",
)

MAX_CHOICE_OPTIONS = 255  # 与 System One 验证层一致（2^8 - 1）
SUM_TOL = 1e-6

QType = Literal["noul", "choice", "score"]


class SchemaError(ValueError):
    """样本违反本文件的形状约定。故意不做静默修正。"""


# --------------------------------------------------------------------------- #
# 工具
# --------------------------------------------------------------------------- #
def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _is_close_sum(values: Sequence[float], target: float = 1.0) -> bool:
    return abs(math.fsum(values) - target) <= SUM_TOL


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise SchemaError(msg)


# --------------------------------------------------------------------------- #
# Question
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class Question:
    qid: str
    type: QType
    instructions: str
    criteria: Any = None

    @property
    def n_options(self) -> int:
        """choice / score 的候选数；noul 恒为 2（是/否）。"""
        if self.type == "noul":
            return 2
        if self.type == "choice":
            return len(self.criteria or {})
        return len(self.criteria or [])

    @property
    def option_names(self) -> list[str]:
        """按固定顺序给出候选名。choice 用 dict 的插入顺序（Python 3.7+ 稳定）。"""
        if self.type == "noul":
            return ["yes", "no"]
        if self.type == "choice":
            return list((self.criteria or {}).keys())
        return [f"level_{i}" for i in range(len(self.criteria or []))]

    @property
    def option_descriptions(self) -> list[str | None]:
        if self.type == "noul":
            crit = self.criteria or {}
            return [crit.get("true"), crit.get("false")]
        if self.type == "choice":
            return [v for v in (self.criteria or {}).values()]
        return [str(x) for x in (self.criteria or [])]

    def validate(self) -> None:
        _require(self.qid != "", "qid 不得为空")
        _require(self.type in QUESTION_TYPES, f"未知问题类型: {self.type!r}")
        _require(bool(self.instructions.strip()), f"{self.qid}: instructions 不得为空")
        if self.type == "noul":
            if self.criteria is not None:
                _require(
                    isinstance(self.criteria, dict),
                    f"{self.qid}: noul 的 criteria 必须是 {{'true':…,'false':…}}",
                )
        elif self.type == "choice":
            _require(
                isinstance(self.criteria, dict) and len(self.criteria) >= 1,
                f"{self.qid}: choice 的 criteria 必须是至少 1 个选项的 dict",
            )
            _require(
                len(self.criteria) <= MAX_CHOICE_OPTIONS,
                f"{self.qid}: 选项数 {len(self.criteria)} 超过 {MAX_CHOICE_OPTIONS}",
            )
        else:  # score
            _require(
                isinstance(self.criteria, list) and len(self.criteria) >= 2,
                f"{self.qid}: score 的 criteria 必须是至少 2 个层级的 list",
            )
            _require(
                len(self.criteria) <= MAX_CHOICE_OPTIONS,
                f"{self.qid}: 层级数 {len(self.criteria)} 超过 {MAX_CHOICE_OPTIONS}",
            )

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "qid": self.qid,
            "type": self.type,
            "instructions": self.instructions,
        }
        if self.criteria is not None:
            out["criteria"] = self.criteria
        return out

    @classmethod
    def from_json(cls, obj: Mapping[str, Any]) -> "Question":
        return cls(
            qid=str(obj["qid"]),
            type=str(obj["type"]),  # type: ignore[arg-type]
            instructions=str(obj["instructions"]),
            criteria=obj.get("criteria"),
        )


# --------------------------------------------------------------------------- #
# Label
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class Label:
    hard: int | str | None
    soft: float | dict[str, float] | list[float]
    unanswerable: bool = False
    teacher: dict[str, Any] | None = None

    def validate(self, q: Question) -> None:
        if self.unanswerable:
            _require(
                self.soft_is_uniform(q),
                f"{q.qid}: unanswerable=True 时 soft 必须是均匀分布",
            )
            _require(
                self.hard is None,
                f"{q.qid}: unanswerable=True 时 hard 必须为 null",
            )
            return

        if q.type == "noul":
            _require(
                isinstance(self.soft, (int, float)) and not isinstance(self.soft, bool),
                f"{q.qid}: noul 的 soft 必须是标量",
            )
            p = float(self.soft)
            _require(0.0 <= p <= 1.0, f"{q.qid}: noul 的 soft 越界: {p}")
            _require(self.hard in (0, 1), f"{q.qid}: noul 的 hard 必须是 0 或 1")
        elif q.type == "choice":
            _require(
                isinstance(self.soft, dict),
                f"{q.qid}: choice 的 soft 必须是 {{选项名: 概率}}",
            )
            keys = set(self.soft)
            want = set(q.option_names)
            _require(keys == want, f"{q.qid}: soft 键集与 criteria 不一致: {keys ^ want}")
            _require(_is_close_sum(list(self.soft.values())), f"{q.qid}: soft 不归一")
            _require(
                isinstance(self.hard, str) and self.hard in want,
                f"{q.qid}: choice 的 hard 必须是 criteria 里的选项名",
            )
        else:  # score
            _require(
                isinstance(self.soft, list),
                f"{q.qid}: score 的 soft 必须是 list",
            )
            _require(
                len(self.soft) == q.n_options,
                f"{q.qid}: score 的 soft 长度 {len(self.soft)} != 层级数 {q.n_options}",
            )
            vals = [float(x) for x in self.soft]
            _require(all(0.0 <= v <= 1.0 for v in vals), f"{q.qid}: score 的 soft 越界")
            _require(_is_close_sum(vals), f"{q.qid}: score 的 soft 不归一")
            _require(
                isinstance(self.hard, int)
                and not isinstance(self.hard, bool)
                and 0 <= self.hard < q.n_options,
                f"{q.qid}: score 的 hard 必须是从 0 起的层级下标",
            )

    def soft_is_uniform(self, q: Question) -> bool:
        if q.type == "noul":
            return abs(float(self.soft) - 0.5) <= SUM_TOL
        if q.type == "choice":
            k = q.n_options
            return all(abs(v - 1.0 / k) <= SUM_TOL for v in self.soft.values())  # type: ignore[union-attr]
        k = q.n_options
        return all(abs(v - 1.0 / k) <= SUM_TOL for v in self.soft)  # type: ignore[union-attr]

    def as_vector(self, q: Question) -> list[float]:
        """统一成定长概率向量，顺序 = q.option_names。"""
        if q.type == "noul":
            p = float(self.soft)
            return [p, 1.0 - p]
        if q.type == "choice":
            return [float(self.soft[name]) for name in q.option_names]  # type: ignore[index]
        return [float(x) for x in self.soft]  # type: ignore[union-attr]

    def hard_index(self, q: Question) -> int | None:
        """把 hard 统一成下标。noul: 0=yes。"""
        if self.hard is None:
            return None
        if q.type == "noul":
            return int(self.hard)
        if q.type == "choice":
            return q.option_names.index(str(self.hard))
        return int(self.hard)

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {"hard": self.hard, "soft": self.soft}
        if self.unanswerable:
            out["unanswerable"] = True
        if self.teacher:
            out["teacher"] = self.teacher
        return out

    @classmethod
    def from_json(cls, obj: Mapping[str, Any]) -> "Label":
        return cls(
            hard=obj.get("hard"),
            soft=obj["soft"],
            unanswerable=bool(obj.get("unanswerable", False)),
            teacher=obj.get("teacher"),
        )


# --------------------------------------------------------------------------- #
# Sample
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class Sample:
    sample_id: str
    domain: str
    source: dict[str, Any]
    state: str
    questions: list[Question]
    labels: dict[str, Label]
    split: str
    provenance: dict[str, Any] = field(default_factory=dict)

    # -- 校验 ---------------------------------------------------------------
    def validate(self) -> None:
        _require(self.sample_id.strip() != "", "sample_id 不得为空")
        _require(self.domain in DOMAINS, f"未知 domain: {self.domain!r}")
        _require(self.split in SPLITS, f"未知 split: {self.split!r}")
        _require(bool(self.state.strip()), f"{self.sample_id}: state 不得为空")
        _require(bool(self.questions), f"{self.sample_id}: 至少要有一个 question")

        kind = self.source.get("kind")
        _require(kind in SOURCE_KINDS, f"{self.sample_id}: 未知 source.kind: {kind!r}")
        _require(bool(self.source.get("ref")), f"{self.sample_id}: source.ref 必填")

        qids = [q.qid for q in self.questions]
        _require(len(set(qids)) == len(qids), f"{self.sample_id}: qid 重复")
        _require(
            set(self.labels) == set(qids),
            f"{self.sample_id}: labels 键与 qid 不一一对应（差 {set(self.labels) ^ set(qids)}）",
        )
        for q in self.questions:
            q.validate()
            self.labels[q.qid].validate(q)

    # -- 派生属性 -----------------------------------------------------------
    @property
    def state_sha256(self) -> str:
        return sha256_text(self.state)

    def fingerprint(self) -> str:
        """规范化指纹。用于去重与泄漏检查。"""
        return sha256_text(json.dumps(self.to_json(), sort_keys=True, ensure_ascii=False))

    @property
    def n_questions(self) -> int:
        return len(self.questions)

    @property
    def max_options(self) -> int:
        return max(q.n_options for q in self.questions)

    def has_answerable_labels(self, q: Question) -> bool:
        return not self.labels[q.qid].unanswerable

    # -- 序列化 -------------------------------------------------------------
    def to_json(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "domain": self.domain,
            "source": self.source,
            "state": self.state,
            "questions": [q.to_json() for q in self.questions],
            "labels": {qid: lab.to_json() for qid, lab in self.labels.items()},
            "split": self.split,
            "provenance": self.provenance,
        }

    @classmethod
    def from_json(cls, obj: Mapping[str, Any]) -> "Sample":
        questions = [Question.from_json(q) for q in obj["questions"]]
        labels = {qid: Label.from_json(lab) for qid, lab in obj["labels"].items()}
        return cls(
            sample_id=str(obj["sample_id"]),
            domain=str(obj["domain"]),
            source=dict(obj["source"]),
            state=str(obj["state"]),
            questions=questions,
            labels=labels,
            split=str(obj["split"]),
            provenance=dict(obj.get("provenance", {})),
        )


# --------------------------------------------------------------------------- #
# 与 kev 的 JSONL 互转（训练/推理同构）
# --------------------------------------------------------------------------- #
def to_kev_record(sample: Sample) -> dict[str, Any]:
    """输出 kev `--data train.jsonl` 能直接吃的记录：同 API 请求形状 + 每题一个 label。

    label 语义（照 kev 原文）：choice 用选项名、noul 用 true/false、score 用从 0 起的下标。
    """
    questions: dict[str, Any] = {}
    for q in sample.questions:
        lab = sample.labels[q.qid]
        if lab.unanswerable:
            # kev 没有"不可回答"这一档；用均匀软标签表达会更拧，故此处显式丢弃该题。
            continue
        entry: dict[str, Any] = {"type": q.type, "instructions": q.instructions}
        if q.criteria is not None:
            entry["criteria"] = q.criteria
        if q.type == "noul":
            entry["label"] = bool(int(lab.hard))  # type: ignore[arg-type]
        elif q.type == "choice":
            entry["label"] = str(lab.hard)
        else:
            entry["label"] = int(lab.hard)  # type: ignore[arg-type]
        questions[q.qid] = entry
    _require(bool(questions), f"{sample.sample_id}: 没有可转 kev 的可回答题目")
    return {"state": sample.state, "questions": questions}


def from_kev_record(record: Mapping[str, Any], *, sample_id: str, domain: str, split: str) -> Sample:
    """``to_kev_record`` 的逆。用于往返自检。"""
    questions: list[Question] = []
    labels: dict[str, Label] = {}
    for qid, entry in record["questions"].items():
        q = Question(
            qid=str(qid),
            type=str(entry["type"]),  # type: ignore[arg-type]
            instructions=str(entry["instructions"]),
            criteria=entry.get("criteria"),
        )
        questions.append(q)
        raw = entry["label"]
        if q.type == "noul":
            y = 1 if bool(raw) else 0
            labels[q.qid] = Label(hard=y, soft=float(y))
        elif q.type == "choice":
            onehot = {name: (1.0 if name == str(raw) else 0.0) for name in q.option_names}
            labels[q.qid] = Label(hard=str(raw), soft=onehot)
        else:
            idx = int(raw)
            soft = [1.0 if i == idx else 0.0 for i in range(q.n_options)]
            labels[q.qid] = Label(hard=idx, soft=soft)
    return Sample(
        sample_id=sample_id,
        domain=domain,
        source={"kind": "hand_authored", "ref": "kev-record"},
        state=str(record["state"]),
        questions=questions,
        labels=labels,
        split=split,
        provenance={"extractor": "from_kev_record", "extractor_version": "1"},
    )


# --------------------------------------------------------------------------- #
# JSONL IO
# --------------------------------------------------------------------------- #
def write_jsonl(path: str | Path, samples: Iterable[Sample]) -> int:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with p.open("w", encoding="utf-8", newline="\n") as fh:  # 显式 LF，见项目行尾约定
        for s in samples:
            fh.write(json.dumps(s.to_json(), ensure_ascii=False, sort_keys=True))
            fh.write("\n")
            n += 1
    return n


def iter_jsonl(path: str | Path) -> Iterator[Sample]:
    with Path(path).open("r", encoding="utf-8", newline="") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield Sample.from_json(json.loads(line))
            except SchemaError:
                raise
            except Exception as exc:  # 定位比裸 json 错误重要
                raise SchemaError(f"{path}:{lineno}: {exc}") from exc


def load_jsonl(path: str | Path, *, validate: bool = True) -> list[Sample]:
    out: list[Sample] = []
    for s in iter_jsonl(path):
        if validate:
            s.validate()
        out.append(s)
    return out
