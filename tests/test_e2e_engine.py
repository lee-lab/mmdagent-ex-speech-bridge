"""End-to-end test against a running MMDAgent-EX (Plugin_API enabled).

Starts the bridge with the offline tone backend, sends SYNTH_START through
the engine's REST messages endpoint and expects SYNTH_EVENT_START followed by
SYNTH_EVENT_STOP on the WebSocket.

    MMDAGENT_HOST=localhost MMDAGENT_REST_PORT=48000 \\
        python -m pytest -m requires_engine tests/test_e2e_engine.py -v

Optional: MMDAGENT_API_KEY, MMDAGENT_E2E_MODEL (model alias, default "gene").
"""

import asyncio
import json
import os
import subprocess
import sys
import time
import urllib.request

import pytest
import websockets

pytestmark = pytest.mark.requires_engine

HOST = os.environ.get("MMDAGENT_HOST", "localhost")
REST_PORT = int(os.environ.get("MMDAGENT_REST_PORT", "48000"))
KEY = os.environ.get("MMDAGENT_API_KEY", "")
MODEL = os.environ.get("MMDAGENT_E2E_MODEL", "gene")
BRIDGE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "speech_bridge.py")


def rest(method, path, body=None):
    req = urllib.request.Request(f"http://{HOST}:{REST_PORT}/api/v1{path}", method=method,
                                 data=json.dumps(body).encode() if body is not None else None)
    req.add_header("Content-Type", "application/json")
    if KEY:
        req.add_header("Authorization", f"Bearer {KEY}")
    with urllib.request.urlopen(req, timeout=5) as r:
        return json.loads(r.read() or b"{}")


def our_stream():
    streams = rest("GET", "/streams").get("streams", [])
    return [s for s in streams if s.get("type") == "audio_playback"]


@pytest.fixture
def bridge():
    try:
        rest("GET", "/status")
    except OSError as e:
        pytest.skip(f"no engine at {HOST}:{REST_PORT}: {e}")
    before = {s["id"] for s in our_stream()}
    proc = subprocess.Popen(
        [sys.executable, BRIDGE, "--host", HOST, "--rest-port", str(REST_PORT),
         "--tts", "tone", "--asr", "none", "-v"]
        + (["--mmdagent-key", KEY] if KEY else []),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8")
    deadline = time.time() + 15
    created = None
    while time.time() < deadline and created is None:
        new = [s for s in our_stream() if s["id"] not in before]
        created = new[0] if new else None
        time.sleep(0.2)
    if created is None:
        proc.kill()
        pytest.fail("bridge did not create its playback stream:\n" + proc.communicate()[0])
    yield proc, created
    proc.terminate()
    try:
        out = proc.communicate(timeout=10)[0]
    except subprocess.TimeoutExpired:
        proc.kill()
        out = proc.communicate()[0]
    print(out)


def test_synth_start_gets_start_and_stop(bridge):
    _, stream = bridge
    assert stream["codec"] == "pcm" and stream["sample_rate"] == 24000

    async def scenario():
        headers = {"Authorization": f"Bearer {KEY}"} if KEY else {}
        async with websockets.connect(f"ws://{HOST}:{REST_PORT + 1}",
                                      additional_headers=headers) as ws:
            await ws.send(json.dumps({"action": "subscribe", "events": ["SYNTH_EVENT_*"]}))
            await asyncio.sleep(0.3)
            t0 = time.monotonic()
            await asyncio.to_thread(rest, "POST", "/messages",
                                    {"type": "SYNTH_START", "args": f"{MODEL}|x|hello"})
            seen = []
            while len(seen) < 2:
                msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
                if msg.get("event", "").startswith("SYNTH_EVENT_"):
                    seen.append((msg["event"], msg.get("args"), time.monotonic() - t0))
            return seen

    seen = asyncio.run(scenario())
    assert [(e, a) for e, a, _ in seen] == [("SYNTH_EVENT_START", MODEL),
                                            ("SYNTH_EVENT_STOP", MODEL)]
    t_start, t_stop = seen[0][2], seen[1][2]
    assert t_start < 2.0                     # playback starts promptly
    assert 0.4 < t_stop - t_start < 3.0      # 0.5 s beep (+1 s timer on older engines)
    print(f"START after {t_start:.2f} s, STOP {t_stop - t_start:.2f} s later")
    # the stream is bound to the model for lip sync
    assert any(s.get("model") == MODEL for s in our_stream())


def test_bridge_deletes_its_stream_on_exit(bridge):
    proc, stream = bridge
    proc.terminate()
    proc.wait(timeout=10)
    time.sleep(0.3)
    assert stream["id"] not in {s["id"] for s in our_stream()}
