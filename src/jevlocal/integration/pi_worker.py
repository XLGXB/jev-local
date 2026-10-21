"""Bounded, advisory-only Pi stdio consumer for the explicit context-native route.

The executable and config path are supplied by the host. This module neither
selects a checkpoint nor starts a model until a valid request arrives.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import BinaryIO, Callable


# `python -I -B path/to/pi_worker.py` omits the project's src directory.
_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from decision_deploy.protocol import MAX_BYTES as REQUEST_MAX_BYTES
from jevlocal.integration.pi_resources import ResourceFault, ResourceGuard

RESPONSE_MAX_BYTES = 256 * 1024
CONFIG_MAX_BYTES = 1024 * 1024
_OFFLINE_FLAGS = (
    "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE",
    "HF_HUB_DISABLE_TELEMETRY", "TORCH_FORCE_WEIGHTS_ONLY_LOAD",
)


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":")).encode("utf-8")


def _valid_id(value: object) -> str | None:
    from decision_deploy.protocol import _id

    try:
        _id(value)
    except ValueError:
        return None
    return value


def _error(code: str, request_id: str | None = None,
           observation_id: str | None = None) -> dict:
    return {
        "schema_version": 1, "request_id": request_id,
        "observation_id": observation_id, "ok": False,
        "error": {"code": code, "message": {
            "invalid_request": "Invalid request frame or public decision contract.",
            "initialization_failed": "Local predictor initialization failed.",
            "prediction_failed": "Local predictor prediction failed.",
        }[code]},
    }


def _write_packet(output: BinaryIO, packet: dict) -> None:
    raw = _json_bytes(packet) + b"\n"
    if len(raw) > RESPONSE_MAX_BYTES:
        raise ValueError("response byte budget exceeded")
    pending = memoryview(raw)
    while pending:
        written = output.write(pending)
        if written is None or written <= 0:
            raise OSError("protocol stdout write failed")
        pending = pending[written:]
    output.flush()


def _parse_request(frame: bytes) -> tuple[dict, str, str]:
    from decision_deploy.common import unique_pairs
    from decision_deploy.protocol import validate_envelope

    if not frame.endswith(b"\n") or len(frame) - 1 > REQUEST_MAX_BYTES:
        raise ValueError("request frame byte budget or newline invalid")
    text = frame[:-1].decode("utf-8")
    if text.startswith("\ufeff"):
        raise ValueError("UTF-8 BOM forbidden")
    def reject_constant(_value: str) -> None:
        raise ValueError("nonfinite JSON")

    envelope = json.loads(text, object_pairs_hook=unique_pairs,
                          parse_constant=reject_constant)
    validate_envelope(envelope)
    return envelope, envelope["request_id"], envelope["observation_id"]


def _metadata(predictor: object, fingerprint: str) -> dict:
    metadata = predictor.metadata
    if (type(metadata) is not dict or
            metadata.get("runtime_fingerprint") != fingerprint or
            metadata.get("calibrated") is not False or
            metadata.get("calibration") is not None or
            metadata.get("advisory_only") is not True or
            metadata.get("execution_authorized") is not False or
            metadata.get("semantic_quality") != "unqualified"):
        raise ValueError("native predictor metadata contract changed")
    return metadata


class Worker:
    """Serial request handler. A construction/prediction fault poisons this worker."""

    def __init__(self, config: dict, factory: Callable[[dict], object],
                 resource: ResourceGuard | None = None):
        self.config = config
        self.factory = factory
        self.resource = resource
        self.predictor: object | None = None
        self.poisoned = False

    def handle(self, frame: bytes) -> tuple[dict, bool]:
        from decision_deploy.protocol import validate_reply

        if self.poisoned:
            raise RuntimeError("worker is terminal")
        request_id = observation_id = None
        try:
            envelope, request_id, observation_id = _parse_request(frame)
        except (ValueError, UnicodeError, TypeError, OverflowError, RecursionError):
            # Do not echo unvalidated IDs or submitted state in diagnostics.
            try:
                from decision_deploy.common import unique_pairs

                raw = json.loads(frame.decode("utf-8"), object_pairs_hook=unique_pairs)
                if type(raw) is dict:
                    request_id = _valid_id(raw.get("request_id"))
                    observation_id = _valid_id(raw.get("observation_id"))
            except (ValueError, UnicodeError, TypeError, OverflowError, RecursionError):
                pass
            if self.resource is not None:
                try:
                    self.resource.record("request", "invalid_request")
                except Exception:
                    self.poisoned = True
                    return _error("initialization_failed", request_id, observation_id), True
            return _error("invalid_request", request_id, observation_id), False

        if self.predictor is None:
            try:
                with contextlib.redirect_stdout(sys.stderr):
                    if self.resource is not None:
                        self.resource.before_factory()
                    self.predictor = self.factory(self.config)
                    if self.resource is not None:
                        self.resource.after_factory()
                if not callable(getattr(self.predictor, "predict", None)):
                    raise ValueError("predictor interface missing")
            except Exception as exc:
                self.poisoned = True
                if self.resource is not None:
                    try:
                        self.resource.fault("request", "resource_limit" if isinstance(exc, ResourceFault)
                                            else "initialization_failed")
                    except Exception:
                        pass
                return _error("initialization_failed", request_id, observation_id), True

        try:
            request = {"state": envelope["state"], "questions": envelope["questions"]}
            with contextlib.redirect_stdout(sys.stderr):
                if self.resource is not None:
                    self.resource.before_predict()
                result = self.predictor.predict(request)
                if self.resource is not None:
                    self.resource.after_predict()
            if type(result) is not dict:
                raise ValueError("predictor reply must be an object")
            reply = {"request_id": request_id, "observation_id": observation_id, **result}
            # A predictor must not override the binding IDs through its result.
            if reply["request_id"] != request_id or reply["observation_id"] != observation_id:
                raise ValueError("predictor changed binding IDs")
            validate_reply(envelope, reply)
            if any(answer["calibrated"] is not False for answer in reply["answers"].values()):
                raise ValueError("context-native answers cannot claim calibration")
            metadata = _metadata(self.predictor, reply["runtime_fingerprint"])
            packet = {"schema_version": 1, "request_id": request_id,
                      "observation_id": observation_id, "ok": True,
                      "reply": reply, "metadata": metadata}
            if len(_json_bytes(packet)) + 1 > RESPONSE_MAX_BYTES:
                raise ValueError("response byte budget exceeded")
            if self.resource is not None:
                self.resource.record("request")
            return packet, False
        except Exception as exc:
            self.poisoned = True
            if self.resource is not None:
                try:
                    self.resource.fault("request", "resource_limit" if isinstance(exc, ResourceFault)
                                        else "prediction_failed")
                except Exception:
                    pass
            return _error("prediction_failed", request_id, observation_id), True


def run_stream(input: BinaryIO, output: BinaryIO, worker: Worker) -> int:
    """One bounded UTF-8 JSON line in, one bounded line out; EOF is normal."""
    try:
        while not worker.poisoned:
            frame = input.readline(REQUEST_MAX_BYTES + 2)
            if not frame:
                return 0
            packet, fatal = worker.handle(frame)
            _write_packet(output, packet)
            if fatal or not frame.endswith(b"\n"):
                return 1
        return 1
    finally:
        if worker.resource is not None:
            worker.resource.close()


def _load_config(path_text: str) -> tuple[dict, str]:
    from jevlocal.integration import context_native
    from jevlocal.integration.state_json_advisory import _local_path, _pairs, _constant

    path = _local_path(path_text)
    if path.stat().st_size > CONFIG_MAX_BYTES:
        raise ValueError("config byte budget exceeded")
    raw = path.read_bytes()
    if len(raw) > CONFIG_MAX_BYTES:
        raise ValueError("config byte budget exceeded")
    text = raw.decode("utf-8")
    if text.startswith("\ufeff"):
        raise ValueError("UTF-8 BOM forbidden")
    config = json.loads(text, object_pairs_hook=_pairs, parse_constant=_constant)
    return context_native.validate_config(config), hashlib.sha256(raw).hexdigest()


def _set_process_environment(config: dict) -> None:
    for flag in _OFFLINE_FLAGS:
        os.environ[flag] = "1"
    cache = config["expected_base_cache"]
    os.environ["HF_HOME"] = cache["hf_home"]
    os.environ["HF_HUB_CACHE"] = cache["hub_cache"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--resource-policy")
    args = parser.parse_args(argv)
    resource = None
    try:
        # Process-local flags precede even stdlib-only admission imports.
        for flag in _OFFLINE_FLAGS:
            os.environ[flag] = "1"
        with contextlib.redirect_stdout(sys.stderr):
            config, config_sha256 = _load_config(args.config)
        _set_process_environment(config)
        from jevlocal.integration.context_native import build_predictor
        if args.resource_policy is not None:
            resource = ResourceGuard.from_file(args.resource_policy, config_sha256, config)
    except Exception:
        if resource is not None:
            try:
                resource.close()
            except Exception:
                pass
        print("Pi worker explicit local configuration admission failed.", file=sys.stderr)
        return 2
    try:
        # Keep a private protocol handle. Library/C-level writes to fd 1 go to
        # stderr, never into the JSONL channel. This process is single-threaded.
        protocol_fd = os.dup(sys.stdout.fileno())
        os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
        with os.fdopen(protocol_fd, "wb", buffering=0) as protocol_output:
            return run_stream(sys.stdin.buffer, protocol_output,
                              Worker(config, build_predictor, resource))
    except Exception:
        print("Pi worker transport failed.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
