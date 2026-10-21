import json
import os
import shutil
import socket
import subprocess
import time
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def cdp_endpoint(tmp_path_factory):
    executable = os.environ.get("TEST_CHROMIUM") or shutil.which("chromium")
    if not executable:
        pytest.skip("Chromium unavailable")
    profile = tmp_path_factory.mktemp("browser-profile")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0)); port = sock.getsockname()[1]
    cmd = [executable, "--headless=new", "--disable-dev-shm-usage", "--remote-debugging-address=127.0.0.1",
           f"--remote-debugging-port={port}", f"--user-data-dir={profile}", "--no-first-run", "about:blank"]
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        cmd.append("--no-sandbox")  # Root container fixture only; never a production setting.
    log = open(profile / "chromium.log", "w")
    process = subprocess.Popen(cmd, stdout=log, stderr=log,
                               creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    endpoint = f"http://127.0.0.1:{port}"
    for _ in range(100):
        try:
            with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(endpoint + "/json/version", timeout=.2) as r:
                json.load(r)
            break
        except Exception:
            if process.poll() is not None:
                raise RuntimeError("fixture browser failed to launch; inspect " + str(profile))
            time.sleep(.05)
    else:
        process.terminate(); log.close(); raise RuntimeError("fixture CDP unavailable")
    try:
        yield endpoint
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill(); process.wait()
        log.close()
