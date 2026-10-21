"""System One 兼容的本地网关（纯标准库，可无重依赖运行）。

为什么要做成 System One 兼容
----------------------------
12 个生态仓库里有 3 个（``fast-jev-compaction``、``hermes-jev-skills``、``kitze/unclutter``）
是直接打 ``https://api.typesafe.ai/v1/systemone`` 的，另外几个也有 ``baseUrl`` 可改。
网关按同一请求/响应形状实现，这些消费方**改一个 URL 就能指向本地**，不需要改它们的代码。
请求/响应契约见 ``docs/DEPLOYMENT.md`` §4。

三条硬规矩
----------
1. **只绑 127.0.0.1**，拒绝其它 host。本地决策服务没有任何鉴权，绑出去等于把决策面开放。
2. **降级要显式**：模型缺失、超预算、K 超过槽数，都返回可判定的 ``degraded`` 字段，
   不抛异常也不静默截断。消费循环需要一个能分支的信号，而不是一个 500。
3. **决策日志只记元数据**：问题类型、K、所选下标、confidence、延迟。
   **不记 state、不记问题文本**（state 里可能含页面正文与轨迹内容）。

本模块模块级只依赖标准库；``OnnxDecider`` 内部惰性导入 onnxruntime。
"""

from __future__ import annotations

import argparse
import hashlib
import http.server
import json
import math
import socketserver
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

from ..export.spec import N_SLOTS, confidence, geometry_features, mask_and_normalize, noul_prob
from ..schema import MAX_CHOICE_OPTIONS, Question, SchemaError
from ..text import estimate_tokens, fit_state

MODEL_ID = "jevlocal-slots256"

#: 单次请求体的字节上限。网关是无鉴权本地服务，不设上限等于给一个 OOM 开关。
MAX_BODY_BYTES = 4 * 1024 * 1024
#: 单请求 token 预算（estimate_tokens 口径；System One 请求上限是 65536）
MAX_REQUEST_TOKENS = 30_000
#: 单个 state 的 token 预算
MAX_STATE_TOKENS = 4_000


# --------------------------------------------------------------------------- #
# 决策后端
# --------------------------------------------------------------------------- #
class Decider(Protocol):
    """``decide`` 返回 ``{qid: slot_logits(list[float])}``。"""

    model_id: str

    def decide(self, state: str, questions: Sequence[Question]) -> dict[str, list[float]]: ...


@dataclass
class EchoDecider:
    """确定性后端：从 (state, question) 的哈希推导 logits。

    用途有两个，且都不是"假装有模型"：
    - **协议测试**：网关的解析/校验/归一化/置信度/降级路径可以在没装 onnxruntime 的机器上全量测；
    - **影子模式**：先跑通接线与日志，不产生任何真实决策影响。
    它的输出与语义无关，所以**绝不允许**用于真实分支——见 ``assert_not_production``。
    """

    model_id: str = "echo-deterministic"
    production: bool = False

    def decide(self, state: str, questions: Sequence[Question]) -> dict[str, list[float]]:
        out: dict[str, list[float]] = {}
        for q in questions:
            seed = hashlib.sha256(f"{q.qid}|{q.type}|{q.instructions}|{state[:512]}".encode("utf-8")).digest()
            k = q.n_options
            vals = []
            for i in range(N_SLOTS):
                b = seed[(i * 7) % len(seed)] ^ (i * 31 % 251)
                vals.append((b / 255.0) * 8.0 - 4.0)
            out[q.qid] = vals[: max(k, 2)] + vals[max(k, 2) :]
        return out


def assert_not_production(decider: Decider) -> None:
    """把"不许用占位后端上线"变成一条断言，而不是一句注释。"""
    if isinstance(decider, EchoDecider) and not getattr(decider, "production", False):
        raise RuntimeError(
            "EchoDecider 的输出与语义无关，禁止用于真实分支。"
            "仅可用于协议测试或影子模式；真实服务请接 OnnxDecider。"
        )


@dataclass
class OnnxDecider:
    """ONNX Runtime 后端。惰性导入 onnxruntime / transformers。

    ``active_providers`` 会被记进健康检查——静默退回 CPU 时结果仍然正确，
    只有延迟差一个数量级，所以只看输出永远发现不了。
    """

    onnx_path: str
    tokenizer_path: str
    model_id: str = MODEL_ID
    max_seq_len: int = 512
    head_max_len: int = 192
    temperature: float = 1.0
    _sess: Any = None
    _tok: Any = None
    active_providers: list[str] = field(default_factory=list)

    def _load(self) -> None:
        if self._sess is not None:
            return
        import onnxruntime as ort  # noqa: PLC0415
        from transformers import AutoTokenizer  # noqa: PLC0415

        avail = ort.get_available_providers()
        order = [p for p in ("TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider") if p in avail]
        if not order:
            raise RuntimeError(f"没有可用执行提供程序: {avail}")
        self._sess = ort.InferenceSession(self.onnx_path, providers=order)
        self.active_providers = list(self._sess.get_providers())
        self._tok = AutoTokenizer.from_pretrained(self.tokenizer_path)

    def decide(self, state: str, questions: Sequence[Question]) -> dict[str, list[float]]:
        self._load()
        assert self._sess is not None and self._tok is not None
        import numpy as np  # noqa: PLC0415

        out: dict[str, list[float]] = {}
        for q in questions:
            head = f"{q.type} {q.instructions} [SEP] " + " ".join(
                f"[MASK]{name}" for name in q.option_names
            )
            enc = self._tok(
                head,
                truncation=True,
                max_length=self.head_max_len,
                return_offsets_mapping=True,
            )
            # 读数位 = 头部段最后一个 token（slots 头把候选信息放在上下文里，读一位即可）
            read_index = len(enc["input_ids"]) - 1
            state_ids = self._tok(
                state,
                truncation=True,
                max_length=self.max_seq_len - self.head_max_len,
                add_special_tokens=False,
            )["input_ids"]
            ids = enc["input_ids"] + [self._tok.sep_token_id] + state_ids + [self._tok.sep_token_id]
            ids = ids[: self.max_seq_len]
            mask = [1] * len(ids)
            feeds = {
                "input_ids": np.asarray([ids], dtype=np.int64),
                "attention_mask": np.asarray([mask], dtype=np.int64),
                "read_index": np.asarray([read_index], dtype=np.int64),
            }
            slot_logits = self._sess.run(["slot_logits"], feeds)[0][0]
            out[q.qid] = [float(x) for x in slot_logits]
        return out


# --------------------------------------------------------------------------- #
# 请求解析与响应构造（纯函数，可单测）
# --------------------------------------------------------------------------- #
@dataclass
class DecisionLogEntry:
    at: str
    qid: str
    qtype: str
    k: int
    chosen_index: int | None
    confidence: float | None
    latency_ms: float
    degraded: str | None

    def to_json(self) -> dict[str, Any]:
        return {
            "at": self.at,
            "qid": self.qid,
            "qtype": self.qtype,
            "k": self.k,
            "chosen_index": self.chosen_index,
            "confidence": self.confidence,
            "latency_ms": round(self.latency_ms, 3),
            "degraded": self.degraded,
        }


def parse_request(payload: Mapping[str, Any]) -> tuple[str, list[Question], list[str]]:
    """校验并归一化请求。返回 (state, questions, degradations)。"""
    problems: list[str] = []
    state = payload.get("state")
    if isinstance(state, (dict, list)):
        state = json.dumps(state, ensure_ascii=False)
    if not isinstance(state, str):
        raise SchemaError("state 必须是字符串、对象或数组")

    raw_questions = payload.get("questions")
    if not isinstance(raw_questions, Mapping) or not raw_questions:
        raise SchemaError("questions 必须是非空对象")

    questions: list[Question] = []
    for qid, spec in raw_questions.items():
        if not isinstance(spec, Mapping):
            raise SchemaError(f"{qid}: 问题定义必须是对象")
        qtype = spec.get("type")
        if qtype == "score":
            criteria = spec.get("criteria")
            if not isinstance(criteria, list):
                raise SchemaError(f"{qid}: score 的 criteria 必须是数组")
        elif qtype == "choice":
            criteria = spec.get("criteria")
            if not isinstance(criteria, Mapping):
                raise SchemaError(f"{qid}: choice 的 criteria 必须是对象")
            if len(criteria) > MAX_CHOICE_OPTIONS:
                raise SchemaError(f"{qid}: 选项数 {len(criteria)} 超过上限 {MAX_CHOICE_OPTIONS}")
        else:
            criteria = spec.get("criteria")
        q = Question(
            qid=str(qid),
            type=str(qtype),  # type: ignore[arg-type]
            instructions=str(spec.get("instructions", "")),
            criteria=criteria,
        )
        q.validate()
        questions.append(q)
    return state, questions, problems


def _softmax_all(logits: Sequence[float]) -> list[float]:
    m = max(logits)
    exps = [math.exp(float(v) - m) for v in logits]
    total = math.fsum(exps)
    return [e / total for e in exps]


def build_response(
    payload: Mapping[str, Any],
    decider: Decider,
    *,
    temperature: float = 1.0,
    shadow: bool = False,
    log_sink: Callable[[DecisionLogEntry], None] | None = None,
) -> dict[str, Any]:
    """把一次 System One 请求变成响应。纯函数（除 decider 调用与日志回调外无副作用）。"""
    t0 = time.perf_counter()
    state, questions, _ = parse_request(payload)

    truncated = False
    if estimate_tokens(state) > MAX_STATE_TOKENS:
        state, truncated = fit_state(state, max_chars=MAX_STATE_TOKENS * 4)
    if estimate_tokens(state) + 200 * len(questions) > MAX_REQUEST_TOKENS:
        return {
            "model": getattr(decider, "model_id", MODEL_ID),
            "results": {},
            "degraded": "request_over_token_budget",
            "shadow": shadow,
            "latency_ms": (time.perf_counter() - t0) * 1000.0,
        }

    logits_by_qid = decider.decide(state, questions)
    results: dict[str, Any] = {}
    now = datetime.now(timezone.utc).isoformat()

    for q in questions:
        k = q.n_options
        entry: dict[str, Any] = {"type": q.type}
        degraded: str | None = "state_truncated" if truncated else None

        logits = logits_by_qid.get(q.qid)
        if logits is None:
            results[q.qid] = {"type": q.type, "degraded": "model_returned_nothing"}
            continue
        if len(logits) < k:
            if len(logits) < 2:
                results[q.qid] = {"type": q.type, "degraded": "insufficient_slots"}
                continue
            # 槽位不足时只用可用部分，并显式标记——绝不补零（补零会把质量分给不存在的候选）
            k = len(logits)
            degraded = "slots_truncated"

        probs = mask_and_normalize(logits, k, temperature)
        idx = max(range(k), key=lambda i: probs[i])
        conf = confidence(probs)
        entry["probabilities"] = probs
        entry["confidence"] = conf

        if q.type == "choice":
            entry["choice"] = q.option_names[idx]
        elif q.type == "noul":
            entry["noul"] = noul_prob(probs)
        else:
            entry["score"] = math.fsum(i * p for i, p in enumerate(probs))
            entry["legend"] = list(q.criteria or [])
        # 供 correctness 头使用；训练侧与服务侧必须逐位一致（见 spec.geometry_features）
        entry["geometry"] = geometry_features(probs, {"noul": 0, "choice": 1, "score": 2}[q.type])
        if degraded:
            entry["degraded"] = degraded
        results[q.qid] = entry

        if log_sink is not None:
            log_sink(
                DecisionLogEntry(
                    at=now,
                    qid=q.qid,
                    qtype=q.type,
                    k=k,
                    chosen_index=idx,
                    confidence=conf,
                    latency_ms=(time.perf_counter() - t0) * 1000.0,
                    degraded=degraded,
                )
            )

    return {
        "model": getattr(decider, "model_id", MODEL_ID),
        "results": results,
        "shadow": shadow,
        "latency_ms": (time.perf_counter() - t0) * 1000.0,
    }


# --------------------------------------------------------------------------- #
# HTTP 服务
# --------------------------------------------------------------------------- #
def make_handler(
    decider: Decider,
    *,
    temperature: float,
    shadow: bool,
    log_path: Path | None,
) -> type[http.server.BaseHTTPRequestHandler]:
    lock = threading.Lock()
    counters = {"requests": 0, "degraded": 0}

    def sink(entry: DecisionLogEntry) -> None:
        # 只写元数据；不写 state、不写问题文本
        if log_path is None:
            return
        with lock:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8", newline="\n") as fh:
                fh.write(json.dumps(entry.to_json(), ensure_ascii=False, sort_keys=True) + "\n")

    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt: str, *args: Any) -> None:  # 默认会打到 stderr
            return

        def _send(self, code: int, obj: Mapping[str, Any]) -> None:
            body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            try:
                self.send_response(code)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                # 客户端在读到响应前就断了（超时、Ctrl-C、被上层取消）。
                # 这在决策循环里是正常现象，不该污染日志。
                return

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/healthz":
                providers = getattr(decider, "active_providers", [])
                self._send(
                    200,
                    {
                        "ok": True,
                        "model": getattr(decider, "model_id", MODEL_ID),
                        "shadow": shadow,
                        "active_providers": providers,
                        "n_slots": N_SLOTS,
                        "counters": dict(counters),
                    },
                )
                return
            self._send(404, {"error": "not_found"})

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/v1/systemone":
                self._send(404, {"error": "not_found"})
                return
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0 or length > MAX_BODY_BYTES:
                self._send(413, {"error": "body_size", "limit": MAX_BODY_BYTES})
                return
            raw = self.rfile.read(length)
            try:
                payload = json.loads(raw.decode("utf-8"))
            except Exception as exc:
                self._send(400, {"error": "bad_json", "detail": str(exc)})
                return
            try:
                resp = build_response(payload, decider, temperature=temperature, shadow=shadow, log_sink=sink)
            except SchemaError as exc:
                self._send(400, {"error": "schema", "detail": str(exc)})
                return
            counters["requests"] += 1
            if resp.get("degraded") or any("degraded" in r for r in resp.get("results", {}).values()):
                counters["degraded"] += 1
            self._send(200, resp)

    return Handler


class LoopbackServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def serve(
    decider: Decider,
    *,
    port: int = 0,
    temperature: float = 1.0,
    shadow: bool = False,
    log_path: Path | None = None,
    allow_placeholder_backend: bool = False,
) -> tuple[LoopbackServer, int]:
    """起服务。``port=0`` 让系统分配空闲端口——**不要硬编码端口**。

    对占位后端（EchoDecider）的约束是：**只有影子模式才允许**。
    理由很直接——它的输出与语义无关，一旦进入真实分支，下游会拿到貌似合理、
    实际随机的结果，而且不报任何错。把这条写进代码而不是写进文档。
    """
    if isinstance(decider, EchoDecider) and not (shadow or allow_placeholder_backend):
        raise RuntimeError(
            "EchoDecider 的输出与语义无关，只允许在 --shadow 下运行。"
            "真实服务请提供 --onnx + --tokenizer。"
        )
    handler = make_handler(decider, temperature=temperature, shadow=shadow, log_path=log_path)
    server = LoopbackServer(("127.0.0.1", port), handler)  # 只绑回环
    actual = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, actual


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="jevlocal.serve.system_one", description="System One 兼容本地网关")
    p.add_argument("--onnx", default="", help="ONNX/TensorRT 引擎路径；留空则用 echo 后端（仅协议自测）")
    p.add_argument("--tokenizer", default="", help="tokenizer 目录（与 --onnx 配套）")
    p.add_argument("--port", type=int, default=0, help="0 = 自动分配空闲端口")
    p.add_argument("--temperature", type=float, default=1.0, help="导出后重拟合的温度，不是训练侧那个")
    p.add_argument("--shadow", action="store_true", help="只决策并记账，不改变行为")
    p.add_argument("--log", default="data/processed/decision_log.jsonl")
    args = p.parse_args(argv)

    decider: Decider
    if args.onnx:
        if not args.tokenizer:
            print("--onnx 必须配 --tokenizer")
            return 2
        decider = OnnxDecider(onnx_path=args.onnx, tokenizer_path=args.tokenizer, temperature=args.temperature)
    else:
        print("[注意] 未提供 --onnx，使用 echo 后端：输出与语义无关，只用于协议自测/影子模式")
        decider = EchoDecider()

    server, port = serve(
        decider,
        port=args.port,
        temperature=args.temperature,
        shadow=args.shadow,
        log_path=Path(args.log) if args.log else None,
    )
    print(f"已启动: http://127.0.0.1:{port}/v1/systemone")
    print(f"健康检查: http://127.0.0.1:{port}/healthz")
    print("Ctrl-C 停止")
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
