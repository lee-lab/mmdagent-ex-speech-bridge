"""Unit tests: no network, no audio hardware, no engine."""

import asyncio
import io
import json
import struct
import types
import wave

import numpy as np
import pytest

import speech_bridge as sb

FRAME_SAMPLES = sb.MIC_RATE * sb.MIC_FRAME_MS // 1000


def tone_frame(dbfs: float) -> bytes:
    """A 20 ms sine frame whose RMS is `dbfs`."""
    amp = 10 ** (dbfs / 20) * 32768 * np.sqrt(2)
    t = np.arange(FRAME_SAMPLES) / sb.MIC_RATE
    return (np.sin(2 * np.pi * 300 * t) * amp).astype("<i2").tobytes()


QUIET = tone_frame(-70)
LOUD = tone_frame(-20)


def feed(vad, frames, gated=False):
    events = []
    for f in frames:
        events += vad.process(f, gated=gated)
    return events


# ----- VAD -----

def calibrated_vad(**kw):
    vad = sb.EnergyVad(**kw)
    assert feed(vad, [QUIET] * vad.calib_frames) == []
    assert vad.state == "idle"
    return vad


def test_frame_dbfs():
    assert sb.frame_dbfs(LOUD) == pytest.approx(-20, abs=0.5)
    assert sb.frame_dbfs(b"\x00\x00" * FRAME_SAMPLES) == -100.0


def test_vad_calibrates_noise_floor():
    vad = calibrated_vad()
    assert vad.floor_db == pytest.approx(-70, abs=0.5)
    assert vad.threshold_dbfs == pytest.approx(-50, abs=0.5)   # max(floor + 10, -50)


def test_vad_onset_needs_three_frames():
    vad = calibrated_vad()
    assert feed(vad, [LOUD, LOUD]) == []
    assert feed(vad, [QUIET]) == []          # run broken
    assert feed(vad, [LOUD, LOUD]) == []
    ev = feed(vad, [LOUD])
    assert [e.kind for e in ev] == ["start"]
    assert vad.in_segment


def test_vad_end_after_silence_includes_preroll():
    vad = calibrated_vad(silence_ms=700, preroll_ms=200)
    feed(vad, [QUIET] * 20)                  # fills the pre-roll
    ev = feed(vad, [LOUD] * 25)              # 500 ms of speech
    assert [e.kind for e in ev] == ["start"]
    ev = feed(vad, [QUIET] * 34)
    assert ev == []                           # 680 ms: not yet
    ev = feed(vad, [QUIET])
    assert [e.kind for e in ev] == ["end"]
    assert ev[0].reason == "silence"
    n_frames = len(ev[0].audio) // (FRAME_SAMPLES * 2)
    # 10 pre-roll + 25 speech + 15 frames (300 ms) of trailing silence
    assert n_frames == 10 + 25 + 15
    assert not vad.in_segment


def test_vad_short_segment_is_cancelled():
    vad = calibrated_vad()
    ev = feed(vad, [LOUD] * 10 + [QUIET] * 35)   # 200 ms < 300 ms
    assert [e.kind for e in ev] == ["start", "cancel"]
    assert ev[1].reason == "short"


def test_vad_cuts_at_max_length():
    vad = calibrated_vad(max_speech_ms=2000)
    ev = feed(vad, [LOUD] * 120)
    kinds = [e.kind for e in ev]
    assert kinds[:2] == ["start", "end"]
    assert ev[1].reason == "max"
    assert len(ev[1].audio) // (FRAME_SAMPLES * 2) == 100


def test_vad_gate_aborts_segment():
    vad = calibrated_vad()
    assert [e.kind for e in feed(vad, [LOUD] * 10)] == ["start"]
    ev = vad.process(LOUD, gated=True)
    assert [(e.kind, e.reason) for e in ev] == [("cancel", "gated")]
    assert feed(vad, [LOUD] * 5, gated=True) == []     # nothing while gated
    assert [e.kind for e in feed(vad, [LOUD] * 3)] == ["start"]   # fresh onset after


def test_vad_fixed_threshold_skips_calibration():
    vad = sb.EnergyVad(threshold_dbfs=-30)
    assert vad.state == "idle"
    assert feed(vad, [tone_frame(-35)] * 10) == []
    assert [e.kind for e in feed(vad, [tone_frame(-25)] * 3)] == ["start"]


def test_vad_absolute_minimum():
    vad = sb.EnergyVad()
    feed(vad, [b"\x00\x00" * FRAME_SAMPLES] * vad.calib_frames)    # digital silence
    assert vad.threshold_dbfs == -50.0
    assert feed(vad, [tone_frame(-60)] * 10) == []


# ----- bus messages -----

@pytest.mark.parametrize("args,expected", [
    ("mei|mei_voice_normal|こんにちは", ("mei", "mei_voice_normal", "こんにちは")),
    ("mei|x|a|b", ("mei", "x", "a|b")),
    ("mei|x", ("mei", "x", "")),
    ("mei", ("mei", "", "")),
    ("", None),
    ("|x|text", None),
])
def test_parse_synth_start(args, expected):
    assert sb.parse_synth_start(args) == expected


@pytest.mark.parametrize("text,expected", [
    ("こんにちは", "こんにちは"),
    ("a|b", "a b"),
    ("line1\nline2\r\n", "line1 line2"),
    ("  many   spaces\t", "many spaces"),
    ("", ""),
])
def test_sanitize_recog_text(text, expected):
    assert sb.sanitize_recog_text(text) == expected


def test_sanitize_recog_text_limit():
    assert len(sb.sanitize_recog_text("あ" * 2000)) == sb.MAX_RECOG_CHARS


# ----- voices -----

def test_resolve_voice():
    voices = sb.OpenAITts.voices
    assert sb.resolve_voice("coral", voices, "marin") == ("coral", False)
    assert sb.resolve_voice("Coral", voices, "marin") == ("coral", False)
    assert sb.resolve_voice("mei_voice_normal", voices, "marin") == ("marin", True)
    assert sb.resolve_voice("", voices, "marin") == ("marin", True)
    assert sb.DEFAULT_VOICE in voices


def test_pick_voice_logs_unknown_once(caplog):
    args = types.SimpleNamespace(host="h", rest_port=1, ws_port=2, mmdagent_key=None,
                                 voice="marin", tts="openai")
    tts = types.SimpleNamespace(voices=sb.OPENAI_VOICES, sample_rate=24000)
    bridge = sb.Bridge(args, tts)
    with caplog.at_level("INFO", logger="speech_bridge"):
        assert bridge._pick_voice("mei_voice_normal") == "marin"
        assert bridge._pick_voice("mei_voice_normal") == "marin"
        assert bridge._pick_voice("nova") == "nova"
    assert sum("mei_voice_normal" in r.message for r in caplog.records) == 1


# ----- binary frames -----

def test_pack_frame_header():
    frame = sb.pack_frame(7, 1234, b"\x01\x02")
    assert len(frame) == 14
    assert struct.unpack("<BBBBII", frame[:12]) == (0x01, 0x05, 0x04, 0, 1234, 7)
    assert frame[12:] == b"\x01\x02"


def test_pack_frame_eos_is_header_only():
    assert len(sb.pack_frame(1, 0, b"")) == 12


def test_stream_number():
    assert sb.stream_number({"id": "aud_03", "stream_id": 3}) == 3
    assert sb.stream_number({"id": "aud_12"}) == 12


def test_rechunk_even_and_bounded():
    pieces = [b"\x01" * 5, b"\x02" * 9000, b"\x03" * 3]
    out = list(sb.rechunk(pieces, size=4096))
    assert all(len(p) % 2 == 0 and 0 < len(p) <= 4096 for p in out)
    assert b"".join(out) == b"".join(pieces)[: (9008 // 2) * 2]


def test_rechunk_passes_data_on_immediately():
    gen = sb.rechunk(iter([b"ab", b"cd"]))
    assert next(gen) == b"ab"


def test_pcm_to_wav_roundtrip():
    pcm = LOUD * 3
    with wave.open(io.BytesIO(sb.pcm_to_wav(pcm))) as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (1, 2, 16000)
        assert w.readframes(w.getnframes()) == pcm


# ----- completion timer -----

def test_fallback_deadline_realtime_synthesis():
    # 2 s of audio pushed within 0.5 s: ends 2 s after the first chunk, + 1 s margin
    assert sb.fallback_deadline(10.0, 10.5, 48000, 24000) == pytest.approx(13.0)


def test_fallback_deadline_slow_synthesis():
    # synthesis slower than real time: cannot end before the last chunk arrived
    assert sb.fallback_deadline(10.0, 15.0, 48000, 24000) == pytest.approx(16.0)


# ----- backends -----

def test_backend_registry():
    assert set(sb.TTS_BACKENDS) == {"openai", "tone"}
    assert set(sb.ASR_BACKENDS) == {"openai", "keyboard"}
    for cls in sb.TTS_BACKENDS.values():
        assert issubclass(cls, sb.TtsBackend)
    for cls in sb.ASR_BACKENDS.values():
        assert issubclass(cls, sb.AsrBackend)
    assert sb.KeyboardAsr.needs_microphone is False
    choices = {a.dest: a.choices for a in sb.build_parser()._actions if a.choices}
    assert choices["tts"] == ["openai", "tone", "none"]
    assert choices["asr"] == ["openai", "keyboard", "none"]


@pytest.mark.parametrize("text,seconds", [("", 0.5), ("あ" * 20, 1.2), ("a" * 500, 6.0)])
def test_tone_length(text, seconds):
    tts = sb.ToneTts(types.SimpleNamespace())
    data = b"".join(tts.synthesize(text, "any"))
    assert len(data) // 2 == int(seconds * 24000)


# ----- CLI -----

def test_cli_defaults(monkeypatch):
    monkeypatch.delenv("MMDAGENT_HOST", raising=False)
    args = sb.build_parser().parse_args([])
    assert (args.host, args.rest_port, args.ws_port) == ("localhost", 48000, None)
    assert args.lang == "ja" and args.voice == sb.DEFAULT_VOICE
    assert args.tts == "openai" and args.asr == "openai"
    assert args.silence_ms == 700


@pytest.fixture(autouse=True)
def keep_signal_handlers(monkeypatch):
    """main() installs a SIGTERM handler; keep it out of the test process."""
    monkeypatch.setattr(sb.signal, "signal", lambda *a: None)


def test_cli_ptt_with_keyboard_is_an_error():
    with pytest.raises(SystemExit) as e:
        sb.main(["--asr", "keyboard", "--ptt", "--tts", "tone"])
    assert e.value.code == 2


def test_cli_missing_openai_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert sb.main(["--openai-key", "", "--asr", "none"]) == 2


# ----- utterance flow against a fake engine -----

class FakeWs:
    def __init__(self):
        self.sent = []

    async def send(self, data):
        self.sent.append(data)

    def commands(self):
        return [(m["type"], m["args"]) for m in
                (json.loads(s) for s in self.sent if isinstance(s, str))]

    def frames(self):
        return [s for s in self.sent if isinstance(s, bytes)]


def make_bridge(rest_responses):
    args = types.SimpleNamespace(host="h", rest_port=1, ws_port=2, mmdagent_key=None,
                                 voice="marin", tts="tone")
    bridge = sb.Bridge(args, sb.ToneTts(args))
    bridge.ws = FakeWs()
    calls = []

    def request(method, path, body=None, timeout=5):
        calls.append((method, path, body))
        return rest_responses(method, path, body)

    bridge.rest.request = request
    return bridge, calls


def engine_without_flush(method, path, body):
    if method == "POST":
        return 202, {"id": "aud_02", "stream_id": 2}
    if method == "PATCH" and body and "flush" in body:
        return 400, {"error": {"code": "MISSING_FIELD"}}
    return 200, {}


async def run_flow(coro_fn):
    bridge, calls = make_bridge(engine_without_flush)
    bridge.loop = asyncio.get_running_loop()
    bridge._tts_lock = asyncio.Lock()
    await bridge._open_stream()
    await coro_fn(bridge)
    return bridge, calls


def test_utterance_completes_by_fallback_timer(monkeypatch):
    monkeypatch.setattr(sb, "FALLBACK_MARGIN_S", 0.0)

    async def flow(bridge):
        monkeypatch.setattr(sb, "fallback_deadline",
                            lambda f, l, s, r, margin=0: l + 0.05)
        await bridge._on_synth_start("gene", "x", "hi")
        await asyncio.wait_for(bridge.current.task, 5)

    bridge, calls = asyncio.run(run_flow(flow))
    assert bridge.ws.commands() == [("SYNTH_EVENT_START", "gene"), ("SYNTH_EVENT_STOP", "gene")]
    frames = bridge.ws.frames()
    assert len(frames[-1]) == 12                         # EOS last
    assert all(len(f) - 12 <= sb.MAX_CHUNK_BYTES for f in frames)
    assert ("PATCH", "/streams/aud_02", {"model": "gene"}) in calls
    assert not bridge.mic_blocked() or bridge._gate_until > 0


def test_utterance_completes_on_drained():
    async def flow(bridge):
        await bridge._on_synth_start("gene", "x", "hi")
        u = bridge.current
        for _ in range(100):
            if u.eos_sent:
                break
            await asyncio.sleep(0.01)
        bridge._on_text(json.dumps({"event": "stream.lifecycle", "stream": "aud_02",
                                    "type": "audio_playback", "status": "drained"}))
        await asyncio.wait_for(u.task, 1.0)          # well before the 1.5 s timer
        assert u.drained.is_set()

    bridge, _ = asyncio.run(run_flow(flow))
    assert [c[0] for c in bridge.ws.commands()] == ["SYNTH_EVENT_START", "SYNTH_EVENT_STOP"]


def test_new_synth_start_replaces_current_and_recreates_stream_without_flush():
    async def flow(bridge):
        await bridge._on_synth_start("gene", "x", "a long first sentence " * 3)
        for _ in range(100):
            if bridge.current.started:
                break
            await asyncio.sleep(0.01)
        await bridge._on_synth_start("mei", "x", "second")
        await asyncio.wait_for(bridge.current.task, 10)

    bridge, calls = asyncio.run(run_flow(flow))
    assert bridge.ws.commands() == [("SYNTH_EVENT_START", "gene"), ("SYNTH_EVENT_STOP", "gene"),
                                    ("SYNTH_EVENT_START", "mei"), ("SYNTH_EVENT_STOP", "mei")]
    methods = [(m, p) for m, p, _ in calls]
    assert ("DELETE", "/streams/aud_02") in methods        # fallback for no flush
    assert bridge.flush_supported is False


def test_synth_stop_only_for_the_speaking_model():
    async def flow(bridge):
        await bridge._on_synth_start("gene", "x", "a long sentence " * 5)
        for _ in range(100):
            if bridge.current.started:
                break
            await asyncio.sleep(0.01)
        await bridge._on_synth_stop("mei")
        assert bridge.current is not None
        await bridge._on_synth_stop("gene")
        assert bridge.current is None

    bridge, _ = asyncio.run(run_flow(flow))
    assert bridge.ws.commands() == [("SYNTH_EVENT_START", "gene"), ("SYNTH_EVENT_STOP", "gene")]


def test_synthesis_failure_still_sends_start_and_stop():
    class Broken(sb.TtsBackend):
        def synthesize(self, text, voice):
            raise ConnectionError("network down")
            yield b""

    async def flow(bridge):
        bridge.tts = Broken(types.SimpleNamespace())
        await bridge._on_synth_start("gene", "x", "hi")
        await asyncio.wait_for(bridge.current.task, 5)

    bridge, _ = asyncio.run(run_flow(flow))
    assert bridge.ws.commands() == [("SYNTH_EVENT_START", "gene"), ("SYNTH_EVENT_STOP", "gene")]
    assert bridge.ws.frames() == []


def test_binding_conflict_unbinds_and_still_speaks(caplog):
    def engine(method, path, body):
        if method == "POST":
            return 202, {"id": "aud_02", "stream_id": 2}
        if method == "PATCH" and body == {"model": "gene"}:
            return 409, {"error": {"code": "CONFLICT"}}
        if method == "GET":
            return 200, {"streams": [{"id": "aud_01", "type": "audio_playback", "model": "gene"}]}
        return 200, {}

    async def flow():
        bridge, calls = make_bridge(engine)
        bridge.loop = asyncio.get_running_loop()
        await bridge._open_stream("mei")
        assert bridge.bound_model == "mei"
        assert await bridge._ensure_stream("gene")
        assert bridge.bound_model == ""
        assert ("PATCH", "/streams/aud_02", {"model": ""}) in calls

    with caplog.at_level("ERROR", logger="speech_bridge"):
        asyncio.run(flow())
    assert any("aud_01" in r.message and "409" in r.message for r in caplog.records)


# ----- microphone fallback to the device's native rate -----

def test_mic_opens_at_native_rate_and_resamples():
    class FakeSd:
        def __init__(self):
            self.calls = []

        def query_devices(self, device=None, kind=None):
            return {"name": "fake", "default_samplerate": 48000, "max_input_channels": 1}

        def RawInputStream(self, samplerate, blocksize, channels, dtype, device, callback):
            self.calls.append((samplerate, blocksize))
            if samplerate == sb.MIC_RATE:
                raise RuntimeError("Invalid sample rate")
            self.callback = callback
            return "stream"

    sd = FakeSd()
    args = types.SimpleNamespace(mic=None, lang="ja", full_duplex=False, ptt=False)
    mic = sb.MicListener(bridge=None, asr=None, args=args)
    stream, native = mic._open(sd)
    assert (stream, native) == ("stream", 48000)
    assert sd.calls == [(16000, 320), (48000, 960)]
    for _ in range(3):
        sd.callback(np.zeros(960, dtype="<i2").tobytes(), 960, None, None)
    frames = [mic.frames.get_nowait() for _ in range(mic.frames.qsize())]
    assert len(frames) == 3 and all(len(f) == 640 for f in frames)
