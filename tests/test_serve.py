"""服务层与导出层的自测。**不需要 torch / onnxruntime / GPU。**

跑法::

    python -B -X utf8 tests/test_serve.py

这里唯一需要真起进程的是 HTTP 往返那两条——它们用标准库 ``http.client`` 打真实回环端口，
所以"网关能跑"是被执行过的，不是被声称的。
"""

from __future__ import annotations

import http.client
import json
import sys
import tempfile
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jevlocal.export.parity import (  # noqa: E402
    build_parity_report,
    compare_predictions,
    ece_from_logits,
    fit_temperature_1d,
)
from jevlocal.export.spec import (  # noqa: E402
    N_SLOTS,
    ExportSpec,
    assert_export_contract,
    confidence,
    engine_cache_key,
    mask_and_normalize,
    trt_profile,
)
from jevlocal.serve.system_one import (  # noqa: E402
    EchoDecider,
    build_response,
    parse_request,
    serve,
)
from jevlocal.schema import SchemaError  # noqa: E402


def _payload(state: str = "state text", **questions) -> dict:
    return {"state": state, "questions": questions or {"q": {"type": "noul", "instructions": "?"}}}


# --------------------------------------------------------------------------- #
# 1. mask / 置信度：整套部署里最容易漏的一步
# --------------------------------------------------------------------------- #
def test_mask_excludes_phantom_slots() -> None:
    import math

    logits = [5.0, 1.0] + [4.0] * (N_SLOTS - 2)  # 槽 2..255 全是高分噪声
    probs = mask_and_normalize(logits, 2)
    assert len(probs) == 2, "归一化后长度必须等于 k"
    assert abs(sum(probs) - 1.0) < 1e-9
    # 负控：如果误对全 256 槽 softmax，max 会被 254 个噪声槽稀释到远低于 0.5
    m = max(logits)
    exps = [math.exp(v - m) for v in logits]
    naive = [e / math.fsum(exps) for e in exps]
    assert max(probs) > 0.9 > max(naive), f"mask 失效: {max(probs)} vs {max(naive)}"


def test_confidence_matches_documented_formula() -> None:
    # Jev 文档给的例子：3 选项、最大概率 0.8 → confidence 0.7
    p = [0.8, 0.1, 0.1]
    assert abs(confidence(p) - 0.7) < 1e-9, confidence(p)
    # 均匀分布 → 0；完全确定 → 1
    assert abs(confidence([1 / 3, 1 / 3, 1 / 3])) < 1e-9
    assert abs(confidence([1.0, 0.0]) - 1.0) < 1e-9


def test_mask_rejects_out_of_range_k() -> None:
    for bad in (1, 0, 256, 300):
        try:
            mask_and_normalize([0.0] * N_SLOTS, bad)
        except ValueError:
            continue
        raise AssertionError(f"k={bad} 应被拒绝")


# --------------------------------------------------------------------------- #
# 2. 请求解析
# --------------------------------------------------------------------------- #
def test_parse_request_accepts_three_types() -> None:
    payload = _payload(
        state="x",
        a={"type": "noul", "instructions": "yes?"},
        b={"type": "choice", "instructions": "which?", "criteria": {"p": None, "q": None}},
        c={"type": "score", "instructions": "how?", "criteria": ["lo", "mid", "hi"]},
    )
    state, questions, _ = parse_request(payload)
    assert state == "x" and len(questions) == 3
    assert [q.n_options for q in questions] == [2, 2, 3]


def test_parse_request_accepts_object_state() -> None:
    payload = {"state": {"subject": "s", "body": "b"}, "questions": {"q": {"type": "noul", "instructions": "?"}}}
    state, _, _ = parse_request(payload)
    assert json.loads(state)["subject"] == "s"


def test_parse_request_rejects_over_255_options() -> None:
    too_many = {f"o{i}": None for i in range(256)}
    payload = _payload(b={"type": "choice", "instructions": "?", "criteria": too_many})
    try:
        parse_request(payload)
    except SchemaError:
        return
    raise AssertionError("超过 255 个选项必须被拒绝")


def test_parse_request_rejects_empty_questions() -> None:
    try:
        parse_request({"state": "x", "questions": {}})
    except SchemaError:
        return
    raise AssertionError("空 questions 必须被拒绝")


# --------------------------------------------------------------------------- #
# 3. 响应构造
# --------------------------------------------------------------------------- #
def test_build_response_shape_per_type() -> None:
    payload = _payload(
        state="zzz",
        a={"type": "noul", "instructions": "yes?"},
        b={"type": "choice", "instructions": "which?", "criteria": {"p": None, "q": None}},
        c={"type": "score", "instructions": "how?", "criteria": ["lo", "mid", "hi"]},
    )
    resp = build_response(payload, EchoDecider())
    assert set(resp["results"]) == {"a", "b", "c"}
    assert "noul" in resp["results"]["a"]
    assert resp["results"]["b"]["choice"] in ("p", "q")
    assert 0.0 <= resp["results"]["c"]["score"] <= 2.0
    for qid in ("a", "b", "c"):
        assert abs(sum(resp["results"][qid]["probabilities"]) - 1.0) < 1e-6
        assert 0.0 <= resp["results"][qid]["confidence"] <= 1.0
        assert len(resp["results"][qid]["geometry"]) == 6 + 3, "geometry 宽度必须与训练侧一致"


def test_shadow_flag_propagates() -> None:
    resp = build_response(_payload(), EchoDecider(), shadow=True)
    assert resp["shadow"] is True


def test_decision_log_carries_no_state_text() -> None:
    """隐私断言：日志里不许出现 state 内容，哪怕一个片段。"""
    secret = "SECRET_PAGE_BODY_令牌_ABC123"
    entries = []
    build_response(
        _payload(state=secret + " more text"),
        EchoDecider(),
        shadow=True,
        log_sink=entries.append,
    )
    assert entries, "应当产生日志条目"
    blob = json.dumps([e.to_json() for e in entries], ensure_ascii=False)
    for token in ("SECRET_PAGE_BODY", "令牌", "ABC123", "more text"):
        assert token not in blob, f"日志泄漏了 state 片段: {token}"


def test_long_state_is_truncated_and_marked() -> None:
    """超长 state 走裁剪路径，且**必须显式标记**——静默截断会让下游以为看的是全文。"""
    huge = "字" * 200_000
    resp = build_response({"state": huge, "questions": {"q": {"type": "noul", "instructions": "?"}}}, EchoDecider())
    assert resp.get("degraded") is None
    entry = resp["results"]["q"]
    assert entry.get("degraded") == "state_truncated", entry.get("degraded")


def test_response_degrades_when_request_over_budget() -> None:
    """问题数足够多时，单请求整体超预算 → 顶层 degraded，且不返回任何结果。"""
    questions = {
        f"q{i}": {"type": "noul", "instructions": "?"} for i in range(100)
    }
    resp = build_response(
        {"state": "字" * 20_000, "questions": questions},
        EchoDecider(),
    )
    assert resp.get("degraded") == "request_over_token_budget", resp.get("degraded")
    assert resp["results"] == {}, "超预算时不得返回部分结果，否则调用方会误以为拿到了全集"


# --------------------------------------------------------------------------- #
# 4. HTTP 往返（真起服务、真发请求）
# --------------------------------------------------------------------------- #
def test_http_roundtrip_and_health() -> None:
    server, port = serve(EchoDecider(), port=0, shadow=True, log_path=None)
    try:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)

        def call(method: str, path: str, body: bytes | None = None) -> tuple[int, dict]:
            headers = {"Content-Type": "application/json"} if body else {}
            conn.request(method, path, body=body, headers=headers)
            resp = conn.getresponse()
            # 每次都把 body 读干净：keep-alive 连接上漏读一个响应体，
            # 下一次 getresponse() 会抛 ResponseNotReady，看起来像"服务端坏了"
            raw = resp.read()
            return resp.status, (json.loads(raw.decode("utf-8")) if raw else {})

        status, health = call("GET", "/healthz")
        assert status == 200 and health["ok"] is True
        assert health["n_slots"] == N_SLOTS

        body = json.dumps(_payload(state="hello", a={"type": "noul", "instructions": "?"})).encode("utf-8")
        status, out = call("POST", "/v1/systemone", body)
        assert status == 200, out
        assert "a" in out["results"] and out["shadow"] is True

        assert call("GET", "/nope")[0] == 404
        assert call("POST", "/v1/systemone", b"{not json")[0] == 400

        bad_schema = json.dumps({"state": "x", "questions": {}}).encode("utf-8")
        assert call("POST", "/v1/systemone", bad_schema)[0] == 400
    finally:
        server.shutdown()


def test_server_refuses_placeholder_backend_outside_shadow() -> None:
    try:
        serve(EchoDecider(), port=0, shadow=False, log_path=None)
    except RuntimeError:
        return
    raise AssertionError("非影子模式下必须拒绝占位后端")


def test_binds_loopback_only() -> None:
    server, port = serve(EchoDecider(), port=0, shadow=True, log_path=None)
    try:
        assert server.server_address[0] == "127.0.0.1", server.server_address
        assert port > 0, "port=0 应分配到真实端口"
    finally:
        server.shutdown()


def test_decision_log_written_to_file() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        log = Path(tmp) / "decision_log.jsonl"
        server, port = serve(EchoDecider(), port=0, shadow=True, log_path=log)
        try:
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
            body = json.dumps(_payload(state="hello")).encode("utf-8")
            conn.request("POST", "/v1/systemone", body=body, headers={"Content-Type": "application/json"})
            conn.getresponse().read()
        finally:
            server.shutdown()
        lines = [json.loads(x) for x in log.read_text(encoding="utf-8").splitlines() if x.strip()]
        assert lines and "state" not in lines[0] and "hello" not in json.dumps(lines, ensure_ascii=False)


# --------------------------------------------------------------------------- #
# 5. 导出契约
# --------------------------------------------------------------------------- #
def test_export_contract_rejects_small_slot_count() -> None:
    try:
        assert_export_contract(ExportSpec(n_slots=128))
    except ValueError:
        return
    raise AssertionError("槽数小于 255 必须被拒绝（合法问题会表达不全）")


def test_export_contract_rejects_pointer_head() -> None:
    try:
        assert_export_contract(ExportSpec(head="pointer"))
    except ValueError:
        return
    raise AssertionError("pointer 头没有恒定形状契约，必须被拒绝")


def test_trt_profile_covers_max_shapes() -> None:
    spec = ExportSpec(max_seq_len=512, max_batch=8)
    prof = trt_profile(spec)
    assert "8x512" in prof["maxShapes"], prof
    assert "1x64" in prof["minShapes"], prof
    for key in ("minShapes", "optShapes", "maxShapes"):
        for name in ("input_ids", "attention_mask", "read_index"):
            assert name in prof[key], f"{key} 缺 {name}"


def test_engine_cache_key_changes_with_each_input() -> None:
    spec = ExportSpec()
    base = engine_cache_key(spec, trt_version="10.15", cuda_version="12.9", compute_cap="8.9")
    variants = {
        engine_cache_key(spec, trt_version="10.16", cuda_version="12.9", compute_cap="8.9"),
        engine_cache_key(spec, trt_version="10.15", cuda_version="13.0", compute_cap="8.9"),
        engine_cache_key(spec, trt_version="10.15", cuda_version="12.9", compute_cap="7.5"),
    }
    assert len(variants | {base}) == 4, "三者任一变化都必须产生不同的缓存键"


# --------------------------------------------------------------------------- #
# 6. 一致性检查与温度重拟合
# --------------------------------------------------------------------------- #
def test_compare_predictions_detects_planted_shift() -> None:
    ref = [[3.0, 1.0, 0.5], [0.2, 2.5, 0.1]]
    same = [list(r) for r in ref]
    agr, top1, mx, mn = compare_predictions(ref, same)
    assert agr == 1.0 and mx < 1e-12

    shifted = [[1.0, 3.0, 0.5], [0.2, 2.5, 0.1]]  # 第一行 argmax 被翻掉
    agr2, _, mx2, _ = compare_predictions(ref, shifted)
    assert agr2 < 1.0 and mx2 > 1.0


def test_temperature_refit_improves_ece() -> None:
    # 构造：真实温度 3.0，导出后按 T=1 直接使用 → ECE 偏高；重拟合应明显降下来
    import math

    p_true = [0.7, 0.2, 0.1]
    logits = [[3.0 * math.log(x) for x in p_true]] * 300
    labels = [0] * 210 + [1] * 60 + [2] * 30
    ks = [3] * 300

    t_fit = fit_temperature_1d(logits, labels, ks)
    ece_before = ece_from_logits(logits, labels, ks, 1.0)
    ece_after = ece_from_logits(logits, labels, ks, t_fit)
    assert abs(t_fit - 3.0) < 0.3, f"温度应回到约 3.0，得到 {t_fit:.3f}"
    assert ece_after < ece_before, f"重拟合应降低 ECE: {ece_before:.4f} → {ece_after:.4f}"


def test_parity_report_fails_on_planted_corruption() -> None:
    ref = [[3.0, 1.0, 0.5]] * 60
    good = [list(r) for r in ref]
    labels = [0] * 60
    ks = [3] * 60

    ok = build_parity_report(ref, good, labels, ks)
    assert ok.verdict == "PASS", ok.problems

    bad = [[1.0, 3.0, 0.5]] * 60  # 忘了合 LoRA 那类错误的典型形状
    rep = build_parity_report(ref, bad, labels, ks)
    assert rep.verdict == "FAIL" and rep.problems, "被污染的导出必须判 FAIL"


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
