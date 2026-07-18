import json
import subprocess
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import pytest


ROOT = Path(__file__).resolve().parents[1]
BASE_URL = "http://127.0.0.1:8999"
LAUNCH_COMMAND = ["uv", "run", "python", "webapp/server.py"]


def _get(path: str) -> tuple[int, bytes]:
    with urlopen(f"{BASE_URL}{path}", timeout=3) as response:
        return response.status, response.read()


@pytest.mark.integration
def test_documented_root_command_serves_complete_submission():
    process = subprocess.Popen(
        LAUNCH_COMMAND,
        cwd=ROOT,
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
                status, body = _get("/")
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
            "/model/neuralnet": b"How this model works in this project",
        }
        for route, marker in expected.items():
            status, body = _get(route)
            assert status == 200
            assert marker in body

        status, body = _get("/api/results")
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
