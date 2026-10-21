"""核心层自测：**不需要 torch，不需要 GPU**。

跑法（已配置的项目 Python）::

    python -B -X utf8 tests/test_core.py

或加 pytest 跑。独立运行可在不安装训练依赖的环境中验证数据格式与指标。
"""

from __future__ import annotations

import json
import math
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jevlocal import schema, text  # noqa: E402
from jevlocal.extract.codex_sessions import redact  # noqa: E402
from jevlocal.labels import (  # noqa: E402
    fit_temperature,
    proper_score,
    teacher_soft_from_votes,
)
from jevlocal.metrics import (  # noqa: E402
    assert_complete_report,
    auroc,
    build_report,
    ece,
)
from jevlocal.permutation import (  # noqa: E402
    apply_perm,
    flip_rate,
    invert,
    permutations_for,
    symmetric_permutation_kl,
)
from jevlocal.schema import Label, Question, Sample, SchemaError  # noqa: E402
from jevlocal.splits import assert_no_leak, assign_splits, target_validity_filter  # noqa: E402


# --------------------------------------------------------------------------- #
# 构造助手
# --------------------------------------------------------------------------- #
def mk_noul_sample(sid: str, state: str, *, split: str = "train", keep: int = 1) -> Sample:
    q = Question(qid="keep_call", type="noul", instructions="还需要这条调用吗？")
    lab = Label(hard=keep, soft=float(keep))
    return Sample(
        sample_id=sid,
        domain="tool_keepdrop",
        source={"kind": "synthetic", "ref": f"test:{sid}"},
        state=state,
        questions=[q],
        labels={q.qid: lab},
        split=split,
        provenance={"extractor": "test", "extractor_version": "1"},
    )


def mk_choice_sample(sid: str, state: str, *, split: str = "train") -> Sample:
    q = Question(
        qid="op",
        type="choice",
        instructions="执行哪个操作？",
        criteria={"CLICK": "点击", "TYPE_TEXT": "写入文字", "DONE": "已完成"},
    )
    lab = Label(hard="CLICK", soft={"CLICK": 0.75, "TYPE_TEXT": 0.1667, "DONE": 0.0833})
    return Sample(
        sample_id=sid,
        domain="browser_action",
        source={"kind": "synthetic", "ref": f"test:{sid}"},
        state=state,
        questions=[q],
        labels={q.qid: lab},
        split=split,
        provenance={"extractor": "test", "extractor_version": "1"},
    )


# --------------------------------------------------------------------------- #
# 1. schema
# --------------------------------------------------------------------------- #
def test_schema_roundtrip() -> None:
    s = mk_noul_sample("t-0001", "t0 Read a.py → ok, 100 chars (omitted)")
    s.validate()
    back = Sample.from_json(json.loads(json.dumps(s.to_json(), ensure_ascii=False)))
    assert back.fingerprint() == s.fingerprint(), "canonical 往返改变了指纹"
    assert back.state_sha256 == s.state_sha256


def test_schema_rejects_bad_distribution() -> None:
    q = Question(qid="op", type="choice", instructions="?", criteria={"a": None, "b": None})
    bad = Label(hard="a", soft={"a": 0.9, "b": 0.5})  # 不归一
    try:
        bad.validate(q)
    except SchemaError:
        return
    raise AssertionError("未归一的分布必须被拒绝")


def test_schema_rejects_unknown_source_kind() -> None:
    s = mk_noul_sample("t-0002", "some state here")
    s.source["kind"] = "not_a_real_kind"
    try:
        s.validate()
    except SchemaError:
        return
    raise AssertionError("未知 source.kind 必须被拒绝")


def test_kev_roundtrip_preserves_labels() -> None:
    s = mk_choice_sample("t-0003", "[1] button Search")
    rec = schema.to_kev_record(s)
    assert rec["questions"]["op"]["label"] == "CLICK", "kev 的 choice label 必须是选项名"
    back = schema.from_kev_record(rec, sample_id="t-0003b", domain="browser_action", split="train")
    back.validate()
    assert back.labels["op"].hard == "CLICK"

    n = mk_noul_sample("t-0004", "state")
    rec_n = schema.to_kev_record(n)
    assert rec_n["questions"]["keep_call"]["label"] is True, "kev 的 noul label 必须是 bool"


def test_kev_score_label_is_index() -> None:
    q = Question(qid="d", type="score", instructions="难度？", criteria=["低", "中", "高"])
    s = Sample(
        sample_id="t-0005",
        domain="routing",
        source={"kind": "synthetic", "ref": "test"},
        state="some state",
        questions=[q],
        labels={"d": Label(hard=2, soft=[0.1, 0.2, 0.7])},
        split="train",
        provenance={"extractor": "test", "extractor_version": "1"},
    )
    assert schema.to_kev_record(s)["questions"]["d"]["label"] == 2


# --------------------------------------------------------------------------- #
# 2. text
# --------------------------------------------------------------------------- #
def test_modernbert_markers_match_options() -> None:
    q = Question(qid="op", type="choice", instructions="?", criteria={"a": None, "b": None, "c": None})
    enc = text.encode_modernbert(q, "state text")
    assert len(enc.marker_spans) == 3, "标记数必须等于候选数"
    assert enc.n_options == 3
    for start, end in enc.marker_spans:
        assert enc.head_text[start:end] == "[MASK]"
    assert "[SEP]" in enc.head_text


def test_modernbert_refuses_silent_truncation() -> None:
    q = Question(qid="op", type="choice", instructions="?", criteria={f"o{i}": None for i in range(30)})
    try:
        text.encode_modernbert(q, "s", head_max_chars=50)
    except ValueError:
        return
    raise AssertionError("头部段超预算时必须报错，而不是静默截掉候选")


def test_qwen_kev_decide_positions() -> None:
    qs = [
        Question(qid="a", type="noul", instructions="是不是？"),
        Question(qid="b", type="choice", instructions="选哪个？", criteria={"x": None, "y": None}),
    ]
    enc = text.encode_qwen_kev("state", qs)
    assert len(enc.decide_offsets) == 2
    assert text.decide_positions(enc.text) == enc.decide_offsets, "决定的读数位与反查不一致"
    for off, q in zip(enc.question_offsets, qs):
        assert enc.text[off:].startswith("<q>"), f"{q.qid} 的 <q> 定位错了"


def test_estimate_tokens_is_positive_and_monotonic() -> None:
    a = text.estimate_tokens("hello world")
    b = text.estimate_tokens("hello world" * 10)
    assert 0 < a < b


def test_fit_state_keeps_head_and_tail() -> None:
    out, changed = text.fit_state("HEAD" + "x" * 500 + "TAIL", max_chars=60)
    assert changed and out.startswith("HEAD") and out.endswith("TAIL"), "保头保尾策略失效"


# --------------------------------------------------------------------------- #
# 3. permutation
# --------------------------------------------------------------------------- #
def test_permutation_roundtrip() -> None:
    perm = [2, 0, 1]
    assert invert(perm) == [1, 2, 0]
    probs = [0.1, 0.2, 0.7]
    assert apply_perm(apply_perm(probs, perm), invert(perm)) == probs


def test_symmetric_perm_kl_identity_is_zero() -> None:
    dist = [0.2, 0.3, 0.5]
    val = symmetric_permutation_kl([dist, dist], [[0, 1, 2], [0, 1, 2]])
    assert val < 1e-12, f"同分布同排列的 KL 必须为 0，得到 {val}"


def test_symmetric_perm_kl_is_permutation_invariant() -> None:
    """同一分布换个顺序描述，KL 仍应为 0——这一条专门抓"折回方向写反"。"""
    dist = [0.2, 0.3, 0.5]
    perm = [1, 2, 0]
    permuted = apply_perm(dist, perm)
    val = symmetric_permutation_kl([dist, permuted], [[0, 1, 2], perm])
    assert val < 1e-12, f"同一分布在换了顺序后应仍为 0，得到 {val}（折回方向可能写反）"


def test_flip_rate_and_group() -> None:
    group = permutations_for(4, group_size=4, seed=7)
    assert group[0] == [0, 1, 2, 3], "第一个必须是单位排列（参照系）"
    assert len({tuple(g) for g in group}) == 4, "组内排列必须互不相同"
    assert flip_rate([[0.9, 0.1], [0.9, 0.1]]) == 0.0
    assert flip_rate([[0.9, 0.1], [0.1, 0.9]]) == 1.0


# --------------------------------------------------------------------------- #
# 4. labels
# --------------------------------------------------------------------------- #
def test_proper_score_prefers_true_distribution() -> None:
    q = [0.6, 0.3, 0.1]
    for kind in ("log", "brier"):
        at_q = proper_score(kind, q, 0)
        at_sharp = proper_score(kind, [0.99, 0.005, 0.005], 1)
        assert at_q > at_sharp, f"{kind}：偏离真实分布的自信预测不该得分更高"


def test_teacher_soft_never_zero() -> None:
    t = teacher_soft_from_votes([0, 0, 0, 0, 0, 1, 1, 1], 3, alpha=0.5)
    assert min(t.soft) > 0, "任何候选都不能拿到 0 概率（Laplace 平滑必须生效）"
    assert abs(math.fsum(t.soft) - 1.0) < 1e-9
    assert abs(t.agreement - 5 / 8) < 1e-9


def test_fit_temperature_recovers_known_scale() -> None:
    p = [0.7, 0.2, 0.1]
    t_true = 2.5
    logits = [[t_true * math.log(x) for x in p]] * 200
    labels = [0] * 140 + [1] * 40 + [2] * 20
    t_hat = fit_temperature(logits, labels)
    assert abs(t_hat - t_true) < 0.2, f"温度应回到约 {t_true}，得到 {t_hat:.3f}"


# --------------------------------------------------------------------------- #
# 5. metrics
# --------------------------------------------------------------------------- #
def test_auroc_known_values() -> None:
    assert abs(auroc([1, 2, 3, 4], [0, 0, 1, 1]) - 1.0) < 1e-9
    assert abs(auroc([1, 2, 3, 4], [1, 1, 0, 0]) - 0.0) < 1e-9


def test_ece_zero_when_perfectly_calibrated() -> None:
    conf = [0.5] * 100
    correct = [True] * 50 + [False] * 50
    assert ece(conf, correct) < 1e-9


def test_report_refuses_partial_submission() -> None:
    probs = [[0.8, 0.2], [0.3, 0.7]]
    gold = [0, 1]
    rep = build_report(
        probs=probs,
        gold_idx=gold,
        gold_soft=[[0.9, 0.1], [0.2, 0.8]],
        permutation={"flip_rate": 0.0, "n_permutations": 1.0},
        ece_confidence=0.01,
    )
    assert_complete_report(rep)
    broken = dict(rep)
    del broken["ece_confidence"]
    try:
        assert_complete_report(broken)
    except ValueError:
        return
    raise AssertionError("缺项报告必须被拒绝（这正是防挑指标的机制）")


# --------------------------------------------------------------------------- #
# 6. splits
# --------------------------------------------------------------------------- #
def test_splits_keep_state_groups_together() -> None:
    shared = "same state for three samples"
    samples = [mk_noul_sample(f"s{i}", shared) for i in range(3)]
    samples += [mk_noul_sample(f"u{i}", f"unique state {i}") for i in range(20)]
    out = assign_splits(samples, seed=1)
    by_state: dict[str, set[str]] = {}
    for s in out:
        by_state.setdefault(s.state_sha256, set()).add(s.split)
    assert all(len(v) == 1 for v in by_state.values()), "同一 state 的样本被切到了不同 split"
    assert_no_leak(out)


def test_assert_no_leak_catches_planted_duplicate() -> None:
    a = mk_noul_sample("x1", "leaked state", split="train")
    b = mk_noul_sample("x2", "leaked state", split="test")
    try:
        assert_no_leak([a, b])
    except SchemaError:
        return
    raise AssertionError("跨切分的重复 state 必须被检出")


def test_target_validity_filter_reports_drops() -> None:
    good = mk_noul_sample("g1", "a sufficiently long state string")
    bad = mk_noul_sample("b1", "short")
    kept, rep = target_validity_filter([good, bad])
    assert len(kept) == 1 and rep.dropped_count == 1
    assert "state 过短" in rep.dropped[0][1], "丢弃原因必须留名"


# --------------------------------------------------------------------------- #
# 7. 脱敏
# --------------------------------------------------------------------------- #
def test_redact_masks_keys() -> None:
    raw = "TYPESAFE_API_KEY=sk-abcdef0123456789abcdef and ghp_" + "a" * 30
    out, hits = redact(raw)
    assert hits >= 2, f"应至少命中 2 处，得到 {hits}"
    assert "sk-abcdef0123456789abcdef" not in out
    assert "[REDACTED]" in out


# --------------------------------------------------------------------------- #
# 8. 示例数据必须真的能过校验
# --------------------------------------------------------------------------- #
def test_example_datasets_validate() -> None:
    canon = ROOT / "data" / "examples" / "modernbert.train.jsonl"
    rows = schema.load_jsonl(canon)
    assert len(rows) == 3, f"示例应为 3 条，得到 {len(rows)}"
    assert_no_leak(rows)

    kev_path = ROOT / "data" / "examples" / "qwen_kev.train.jsonl"
    n = 0
    with kev_path.open("r", encoding="utf-8", newline="") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            for qid, entry in rec["questions"].items():
                assert "label" in entry, f"{qid} 缺 label"
                assert entry["type"] in ("noul", "choice", "score")
            n += 1
    assert n == 3, f"kev 示例应为 3 条，得到 {n}"


def test_configs_are_valid_json() -> None:
    for name in ("modernbert.json", "qwen_kev.json"):
        with (ROOT / "configs" / name).open("r", encoding="utf-8") as fh:
            json.load(fh)


# --------------------------------------------------------------------------- #
# 运行器
# --------------------------------------------------------------------------- #
def _run_all() -> int:
    g = globals()
    names = sorted(n for n in g if n.startswith("test_") and callable(g[n]))
    failed: list[str] = []
    for name in names:
        try:
            g[name]()
            print(f"PASS  {name}")
        except Exception as exc:  # noqa: BLE001
            failed.append(name)
            print(f"FAIL  {name}: {exc}")
            traceback.print_exc()
    print(f"\n{len(names) - len(failed)}/{len(names)} passed")
    if failed:
        print("失败用例: " + ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
