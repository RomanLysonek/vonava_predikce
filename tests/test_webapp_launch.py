import json
import os
import re
import socket
import subprocess
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import pytest


ROOT = Path(__file__).resolve().parents[1]
LAUNCH_COMMAND = ["uv", "run", "python", "webapp/server.py"]
PROMO_PATTERN = re.compile(
    rb'<div class="promo-bar">\s*(.*?)\s*</div>',
    flags=re.DOTALL,
)


def _get(base_url: str, path: str) -> tuple[int, bytes]:
    with urlopen(f"{base_url}{path}", timeout=3) as response:
        return response.status, response.read()


@pytest.mark.integration
def test_documented_root_command_serves_complete_submission():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    base_url = f"http://127.0.0.1:{port}"
    process = subprocess.Popen(
        LAUNCH_COMMAND,
        cwd=ROOT,
        env={**os.environ, "NOTINO_DASHBOARD_PORT": str(port)},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if process.poll() is not None:
                output = process.stdout.read() if process.stdout else ""
                pytest.fail(f"server exited before readiness:\n{output}")
            try:
                status, body = _get(base_url, "/")
                if status == 200 and b"Standalone interview submission" in body:
                    break
            except URLError:
                time.sleep(0.2)
        else:
            pytest.fail("server did not become ready within 30 seconds")

        expected = {
            "/": b"Forecast the supplied 30-product panel",
            "/dataset": b"Six facts that changed the modeling design",
            "/evaluation": b"Walk-forward / rolling-origin validation",
        }
        published = json.loads(
            (ROOT / "webapp" / "static" / "results.json").read_text()
        )
        expected.update({
            f"/model/{model['slug']}": b"How this model works in this project"
            for model in published["models"]
        })
        promo_bars = set()
        for route, marker in expected.items():
            status, body = _get(base_url, route)
            assert status == 200
            assert marker in body
            assert b"<title>NOTINO - predikce</title>" in body
            assert body.count(b"description-strip") == 1
            promo = PROMO_PATTERN.findall(body)
            assert len(promo) == 1
            promo_bars.add(promo[0])
        assert len(promo_bars) == 1

        status, body = _get(base_url, "/api/results")
        assert status == 200
        payload = json.loads(body)
        assert payload["selection"]["canonical_model"] == "NeuralNet"
        assert payload["selection"]["canonical_strategy"] == "direct"
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
