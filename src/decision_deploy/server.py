"""Authenticated loopback-only inference. NO browser tools, shell execution or request-selected paths."""
from __future__ import annotations
import argparse, json, os, secrets, threading
from fastapi import FastAPI, HTTPException, Request
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.concurrency import run_in_threadpool
from .common import read, unique_pairs
from .runtime import Runtime, Predictor
from .protocol import validate_envelope, validate_reply

MAX_BYTES = 128 * 1024

def create_app(predictor, token):
    if len(token) < 32: raise ValueError("LOCAL_DECISION_TOKEN must contain at least 32 characters")
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "[::1]"])
    lock = threading.Lock()
    async def auth(request):
        if request.headers.get("origin") is not None: raise HTTPException(403,"browser-origin requests rejected")
        if not secrets.compare_digest(request.headers.get("authorization", ""), "Bearer " + token): raise HTTPException(401,"unauthorized")
    @app.get("/health")
    async def health(request: Request):
        await auth(request)
        return {"ready":True,"runtime_fingerprint":predictor.runtime.fingerprint}
    @app.post("/v1/decide")
    async def decide(request: Request):
        await auth(request)
        if request.headers.get("content-type", "").split(";")[0] != "application/json": raise HTTPException(415,"JSON required")
        buf = bytearray()
        async for chunk in request.stream():
            buf.extend(chunk)
            if len(buf) > MAX_BYTES: raise HTTPException(413,"request too large")
        try:
            payload = json.loads(buf, object_pairs_hook=unique_pairs)
            validate_envelope(payload)
        except (ValueError,TypeError,RecursionError) as e: raise HTTPException(400,"invalid decision request") from e
        if not lock.acquire(blocking=False): raise HTTPException(503,"model busy; retry only before any action")
        def work():
            try: return predictor.predict({"state":payload["state"],"questions":payload["questions"]})
            finally: lock.release()  # client cancellation must not unlock an in-flight GPU operation
        try: result = await run_in_threadpool(work)
        except ValueError as e: raise HTTPException(422,str(e)) from e
        except Exception as e: raise HTTPException(503,"inference failed; no action authorized") from e
        try:
            reply = {**result, "request_id":payload["request_id"], "observation_id":payload["observation_id"]}
            validate_reply(payload, reply)
        except (ValueError, TypeError) as e: raise HTTPException(503,"invalid inference result; no action authorized") from e
        return reply
    return app

def main():
    p=argparse.ArgumentParser();p.add_argument("--runtime",required=True);p.add_argument("--calibration");p.add_argument("--port",type=int,default=0);p.add_argument("--warm-requests");a=p.parse_args()
    predictor=Predictor(Runtime(read(a.runtime)),read(a.calibration) if a.calibration else None)
    if a.warm_requests:
        from .qualify import requests
        for request in requests(a.warm_requests):
            for _ in range(3): predictor.predict(request)
    import uvicorn
    uvicorn.run(create_app(predictor,os.environ.get("LOCAL_DECISION_TOKEN","")),host="127.0.0.1",port=a.port,workers=1,access_log=False)
if __name__=="__main__":main()
