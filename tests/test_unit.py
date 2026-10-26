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
    assert set(sb.TTS_BACKENDS) == {"openai", "openai-realtime", "tone"}
    assert set(sb.ASR_BACKENDS) == {"openai", "openai-live", "keyboard"}
    for cls in sb.TTS_BACKENDS.values():
        assert issubclass(cls, sb.TtsBackend)
    for cls in sb.ASR_BACKENDS.values():
        assert issubclass(cls, sb.AsrBackend)
    assert sb.KeyboardAsr.needs_microphone is False
    choices = {a.dest: a.choices for a in sb.build_parser()._actions if a.choices}
    assert choices["tts"] == ["openai", "openai-realtime", "tone", "none"]
    assert choices["asr"] == ["openai", "openai-live", "keyboard", "none"]
    assert sb.OpenAILiveAsr.streaming and not sb.OpenAIAsr.streaming


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


@pytest.fixture(autouse=True)
def no_dotenv(monkeypatch):
    """main() reads .env files; a developer's real .env must not leak into the tests."""
    monkeypatch.setattr(sb, "dotenv_paths", list)


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


# ----- replay of SYNTH_START missed while connecting -----

def ev(ts, etype, args):
    return {"timestamp": ts, "type": etype, "args": args}


def test_replay_picks_newest_start_per_alias():
    events = [
        ev(1000, "SYNTH_START", "gene|v|old"),
        ev(1500, "MODEL_EVENT_ADD", "gene"),
        ev(2000, "SYNTH_START", "gene|v|new"),
        ev(2100, "SYNTH_START", "mei|v|hello"),
    ]
    picked = sb.select_replays(events)
    assert [e["args"] for e in picked] == ["gene|v|new", "mei|v|hello"]


def test_replay_skips_stopped_or_answered():
    events = [
        ev(1000, "SYNTH_START", "gene|v|a"),
        ev(1100, "SYNTH_STOP", "gene"),
        ev(1200, "SYNTH_START", "mei|v|b"),
        ev(1300, "SYNTH_EVENT_START", "mei"),          # another provider answered
        ev(1400, "SYNTH_START", "uka|v|c"),
        ev(1500, "SYNTH_EVENT_STOP", "uka"),           # not an answer to this request
        ev(1600, "RECOG_EVENT_STOP", "uka|x"),         # unrelated (bridge's own type)
    ]
    assert [e["args"] for e in sb.select_replays(events)] == ["uka|v|c"]


def test_replay_start_after_stop_is_kept():
    events = [ev(1000, "SYNTH_STOP", "gene"), ev(1100, "SYNTH_START", "gene|v|again")]
    assert [e["args"] for e in sb.select_replays(events)] == ["gene|v|again"]


def test_replay_respects_cutoff():
    events = [ev(4000, "SYNTH_START", "gene|v|seen before"), ev(6500, "SYNTH_START", "mei|v|new")]
    assert [e["args"] for e in sb.select_replays(events, cutoff_ms=5000)] == ["mei|v|new"]


def test_replay_skips_starts_handled_live():
    # WS timestamps are whole seconds; the history keeps milliseconds
    events = [ev(12345, "SYNTH_START", "gene|v|hi"), ev(12400, "SYNTH_START", "mei|v|x")]
    picked = sb.select_replays(events, handled=[("gene|v|hi", 12000.0)])
    assert [e["args"] for e in picked] == ["mei|v|x"]
    # a live one also supersedes an older missed request of the same alias
    events = [ev(11000, "SYNTH_START", "gene|v|missed"), ev(12345, "SYNTH_START", "gene|v|hi")]
    assert sb.select_replays(events, handled=[("gene|v|hi", 12000.0)]) == []


def test_replay_ignores_malformed():
    assert sb.select_replays([ev(1, "SYNTH_START", ""), ev(2, "SYNTH_START", "|v|t")]) == []


def test_same_event_window():
    assert sb.same_event(12000, 11999)       # clock read just before the second turned
    assert sb.same_event(12000, 12999)
    assert sb.same_event(12000, 13500)
    assert not sb.same_event(12000, 10900)
    assert not sb.same_event(12000, 14100)


def test_replayed_start_is_not_handled_again_live():
    bridge, _ = make_bridge(engine_without_flush)
    spawned = []
    bridge._spawn = lambda coro: (coro.close(), spawned.append(1))
    bridge._replayed.append(("gene|v|hi", 12345.0))
    bridge._on_text(json.dumps({"event": "SYNTH_START", "args": "gene|v|hi", "timestamp": 12000}))
    assert spawned == [] and list(bridge._live_starts) == []
    assert bridge._last_ws_ts == 12000
    bridge._on_text(json.dumps({"event": "SYNTH_START", "args": "gene|v|next", "timestamp": 13000}))
    assert spawned == [1] and list(bridge._live_starts) == [("gene|v|next", 13000.0)]


# ----- prebuffer at the start of an utterance -----

def test_prebuffer_holds_until_enough():
    chunks = [b"a" * 100, b"b" * 100, b"c" * 100, b"d" * 50]
    out = list(sb.prebuffer(chunks, 250))
    assert out == [b"a" * 100 + b"b" * 100 + b"c" * 100, b"d" * 50]


def test_prebuffer_short_stream_is_one_block():
    assert list(sb.prebuffer([b"ab", b"cd"], 1000)) == [b"abcd"]


def test_prebuffer_disabled_and_empty():
    assert list(sb.prebuffer([b"ab", b"cd"], 0)) == [b"ab", b"cd"]
    assert list(sb.prebuffer([], 100)) == []


def test_prebuffer_is_lazy_after_the_head():
    pulled = []

    def source():
        for c in (b"x" * 10, b"y" * 10, b"z" * 10):
            pulled.append(c[:1])
            yield c

    gen = sb.prebuffer(source(), 15)
    assert next(gen) == b"x" * 10 + b"y" * 10
    assert pulled == [b"x", b"y"]                 # nothing read beyond the head yet


def test_prebuffered_head_goes_out_first_in_bounded_frames():
    tts = sb.ToneTts(types.SimpleNamespace())
    hold = 24000 * 300 // 1000 * 2
    out = list(sb.rechunk(sb.prebuffer(tts.synthesize("a" * 30, "v"), hold)))
    assert all(0 < len(c) <= sb.MAX_CHUNK_BYTES and len(c) % 2 == 0 for c in out)
    assert b"".join(out) == b"".join(tts.synthesize("a" * 30, "v"))
    assert len(out[0]) == sb.MAX_CHUNK_BYTES                      # head split into full frames


def test_utterance_records_prebuffer(monkeypatch):
    async def flow(bridge):
        bridge.args.prebuffer_ms = 300
        monkeypatch.setattr(sb, "fallback_deadline", lambda f, l, s, r, margin=0: l + 0.05)
        await bridge._on_synth_start("gene", "x", "a" * 30)      # 1.8 s tone
        u = bridge.current
        await asyncio.wait_for(u.task, 5)
        assert u.prebuffered == 24000 * 300 // 1000 * 2           # three 100 ms tone chunks

    bridge, _ = asyncio.run(run_flow(flow))
    assert [c[0] for c in bridge.ws.commands()] == ["SYNTH_EVENT_START", "SYNTH_EVENT_STOP"]


# ----- .env files -----

def test_parse_dotenv():
    text = ("﻿# comment\n"
            "OPENAI_API_KEY=sk-plain\n"
            "export MMDAGENT_HOST = 192.168.0.5   # trailing comment\n"
            "MMDAGENT_API_KEY=\"quoted # not a comment\"  # comment\n"
            "SINGLE='a b'\n"
            "EMPTY=\n"
            "HASH=abc#def\n"
            "not a line\n"
            "1BAD=x\n")
    assert sb.parse_dotenv(text) == {
        "OPENAI_API_KEY": "sk-plain", "MMDAGENT_HOST": "192.168.0.5",
        "MMDAGENT_API_KEY": "quoted # not a comment", "SINGLE": "a b", "EMPTY": "",
        "HASH": "abc#def"}


def test_parse_dotenv_crlf():
    assert sb.parse_dotenv("A=1\r\nB=\"2\"\r\n") == {"A": "1", "B": "2"}


def test_load_dotenv_environment_wins_and_first_file_wins(tmp_path):
    first, second = tmp_path / "a.env", tmp_path / "b.env"
    first.write_text("OPENAI_API_KEY=from-first\nOTHER=x\n", encoding="utf-8")
    second.write_text("OPENAI_API_KEY=from-second\nMMDAGENT_HOST=h2\nMMDAGENT_API_KEY=k2\n",
                      encoding="utf-8")
    env = {"MMDAGENT_API_KEY": "already-set"}
    taken = sb.load_dotenv([str(tmp_path / "missing.env"), str(first), str(second)], env)
    assert env == {"OPENAI_API_KEY": "from-first", "MMDAGENT_HOST": "h2",
                   "MMDAGENT_API_KEY": "already-set"}           # OTHER is not ours
    assert taken == [("OPENAI_API_KEY", str(first)), ("MMDAGENT_HOST", str(second))]


def test_load_dotenv_empty_environment_value_is_missing(tmp_path):
    f = tmp_path / ".env"
    f.write_text("OPENAI_API_KEY=k\n", encoding="utf-8")
    env = {"OPENAI_API_KEY": ""}
    sb.load_dotenv([str(f)], env)
    assert env["OPENAI_API_KEY"] == "k"


def test_main_reads_key_from_dotenv_without_logging_it(tmp_path, monkeypatch, caplog):
    f = tmp_path / ".env"
    f.write_text("OPENAI_API_KEY=sk-secret-value\n", encoding="utf-8")
    monkeypatch.setenv("OPENAI_API_KEY", "")          # restored (or removed) at teardown
    monkeypatch.setattr(sb, "dotenv_paths", lambda: [str(f)])
    seen = {}

    def fake_build_parser():
        seen["key"] = sb.os.environ.get("OPENAI_API_KEY")
        raise SystemExit(0)

    monkeypatch.setattr(sb, "build_parser", fake_build_parser)
    with pytest.raises(SystemExit):
        sb.main([])
    assert seen["key"] == "sk-secret-value"
    assert "sk-secret-value" not in caplog.text


# ----- OpenAI REST backends: language hint, keep-alive, warm-up -----

@pytest.mark.parametrize("model,lang,expected", [
    ("gpt-transcribe", "ja", {"extra_body": {"languages": ["ja"]}}),
    ("gpt-live-transcribe", "en", {"extra_body": {"languages": ["en"]}}),
    ("gpt-4o-mini-transcribe", "ja", {"language": "ja"}),
    ("whisper-1", "en", {"language": "en"}),
    ("gpt-transcribe", None, {}),
])
def test_transcription_language_args(model, lang, expected):
    assert sb.transcription_language_args(model, lang) == expected


def test_cli_model_defaults_are_resolved_by_the_backend():
    args = sb.build_parser().parse_args([])
    assert args.tts_model is None and args.asr_model is None
    args.openai_key = "sk-test"
    assert sb.OpenAIAsr(args).model == sb.DEFAULT_ASR_MODEL == "gpt-transcribe"
    assert sb.OpenAITts(args).model == sb.DEFAULT_TTS_MODEL
    args.asr_model = "whisper-1"
    assert sb.OpenAIAsr(args).model == "whisper-1"


def test_rest_clients_keep_connections_alive_and_are_separate():
    import openai
    hc = sb._keepalive_http_client(openai)
    assert hc is not None
    pool = getattr(getattr(hc, "_transport", None), "_pool", None)
    if pool is not None and hasattr(pool, "_keepalive_expiry"):
        assert pool._keepalive_expiry == sb.HTTP_KEEPALIVE_S
    args = types.SimpleNamespace(openai_key="sk-test", tts_model=None, asr_model=None)
    assert sb.OpenAITts(args).client is not sb.OpenAIAsr(args).client


def test_prepare_warms_only_an_idle_backend(monkeypatch):
    args = types.SimpleNamespace(openai_key="sk-test", asr_model=None)
    asr = sb.OpenAIAsr(args)
    warmed = []
    monkeypatch.setattr(asr, "_warm", lambda startup: warmed.append(startup))

    class SyncThread:
        def __init__(self, target, args=(), **kw):
            self.target, self.args = target, args

        def start(self):
            self.target(*self.args)

    monkeypatch.setattr(sb.threading, "Thread", SyncThread)
    asr._last_use = sb.time.monotonic()
    asr.prepare()
    assert warmed == []
    asr._last_use -= sb.WARM_AFTER_IDLE_S + 1
    asr.prepare()
    assert warmed == [False]


# ----- resampler -----

def test_resampler_rate_and_continuity():
    x = (np.sin(np.arange(16000) / 7.0) * 8000).astype("<i2").tobytes()
    whole = sb.LinearResampler(16000, 24000).process(x)
    assert abs(len(whole) // 2 - 24000) <= 1
    r = sb.LinearResampler(16000, 24000)
    pieces = b"".join(r.process(x[i:i + 640]) for i in range(0, len(x), 640))
    assert pieces == whole                       # chunking does not change the output


def test_resampler_odd_chunks_and_reset():
    r = sb.LinearResampler(16000, 24000)
    assert r.process(b"") == b""
    out = r.process(np.full(320, 1000, "<i2").tobytes())
    assert set(np.frombuffer(out, "<i2")) == {1000}
    r.reset()
    assert len(r.process(np.zeros(2, "<i2").tobytes())) // 2 == 2      # t = 0, 0.667


# ----- VAD-gated streaming (what goes to a streaming recognizer) -----

def run_streamer(frames, gated_at=None, **kw):
    vad = calibrated_vad(**kw)
    st = sb.VadStreamer(vad)
    actions = []
    for i, f in enumerate(frames):
        actions += st.process(f, gated=gated_at is not None and i >= gated_at)
    return st, actions


def kinds(actions):
    return [a.kind for a in actions]


def test_streamer_never_sends_silence():
    st, actions = run_streamer([QUIET] * 500)
    assert actions == [] and st.bytes_appended == 0


def test_streamer_holds_until_min_speech_then_streams_and_commits():
    vad_kw = {"silence_ms": 200}
    st, actions = run_streamer([LOUD] * 50 + [QUIET] * 30, **vad_kw)
    k = kinds(actions)
    assert k[0] == "start" and k[-1] == "commit" and k.count("commit") == 1
    first_append = k.index("append")
    # nothing went out before 300 ms of voiced audio (15 frames after the 3-frame onset)
    assert first_append >= 1 and len(actions[first_append].audio) >= 15 * len(LOUD)
    sent = b"".join(a.audio for a in actions if a.kind == "append")
    # = the segment the REST path would upload: pre-roll + speech + 300 ms tail
    vad = calibrated_vad(**vad_kw)
    rest = [e for f in [LOUD] * 50 + [QUIET] * 30 for e in vad.process(f) if e.kind == "end"]
    assert sent == rest[0].audio and st.bytes_appended == len(sent)
    assert actions[-1].seconds == pytest.approx(len(sent) / 2 / sb.MIC_RATE)
    # streaming: most of the audio went out before the commit's frame
    assert len(sent) - len(actions[-2].audio if actions[-2].kind == "append" else b"") > \
        len(sent) // 2


def test_streamer_short_segment_sends_nothing():
    st, actions = run_streamer([LOUD] * 8 + [QUIET] * 60)
    assert kinds(actions) == ["start", "drop"] and actions[-1].reason == "short"
    assert st.bytes_appended == 0


def test_streamer_pause_is_sent_only_when_speech_resumes():
    frames = [LOUD] * 30 + [QUIET] * 25 + [LOUD] * 20 + [QUIET] * 40      # 500 ms pause
    _st, actions = run_streamer(frames, silence_ms=700)
    assert kinds(actions).count("commit") == 1
    vad = calibrated_vad(silence_ms=700)
    rest = [e for f in frames for e in vad.process(f) if e.kind == "end"]
    assert b"".join(a.audio for a in actions if a.kind == "append") == rest[0].audio


def test_streamer_gate_after_sending_clears():
    _st, actions = run_streamer([LOUD] * 40 + [LOUD] * 5, gated_at=40)
    assert kinds(actions)[-2:] == ["clear", "drop"] and actions[-1].reason == "gated"


def test_streamer_gate_before_sending_needs_no_clear():
    _st, actions = run_streamer([LOUD] * 10 + [LOUD] * 5, gated_at=10)
    assert kinds(actions) == ["start", "drop"]


# ----- live transcription bookkeeping -----

def test_live_turns_out_of_order_completion():
    import concurrent.futures as cf
    lt = sb.LiveTurns()
    f1, f2 = cf.Future(), cf.Future()
    lt.append("a")
    t1 = lt.commit(f1)
    lt.append("b")
    t2 = lt.commit(f2)
    assert (t1.event_id, t2.event_id) == ("commit_1", "commit_2")
    lt.on_committed("item_1", None)
    lt.on_committed("item_2", "item_1")
    lt.on_delta("item_2", "wor")
    assert lt.on_completed("item_2", "world") is t2
    assert f2.result(0) == "world" and not f1.done()
    lt.on_completed("item_1", "hello")
    assert f1.result(0) == "hello" and lt.idle()


def test_live_turns_empty_commit_and_rejected_commit():
    import concurrent.futures as cf
    lt = sb.LiveTurns()
    f0 = cf.Future()
    assert lt.commit(f0) is None and f0.result(0) == ""
    f1, f2 = cf.Future(), cf.Future()
    lt.append("a")
    lt.commit(f1)
    lt.append("b")
    lt.commit(f2)
    assert lt.on_error("commit_1", "buffer too small") is not None
    with pytest.raises(RuntimeError):
        f1.result(0)
    lt.on_committed("item_9")                     # belongs to the second commit
    lt.on_completed("item_9", "second")
    assert f2.result(0) == "second"


def test_live_turns_transcript_before_committed_event():
    import concurrent.futures as cf
    lt = sb.LiveTurns()
    f = cf.Future()
    lt.append("a")
    lt.commit(f)
    assert lt.on_completed("item_1", "early") is None
    lt.on_committed("item_1")
    assert f.result(0) == "early"


def test_live_turns_resend_after_reconnect():
    import concurrent.futures as cf
    lt = sb.LiveTurns()
    f1, f2 = cf.Future(), cf.Future()
    lt.append("a1")
    lt.append("a2")
    lt.commit(f1)
    lt.on_committed("item_1")
    lt.append("b1")
    lt.commit(f2)
    f2.cancel()                                   # the listener gave up on it
    lt.append("c1")                               # the turn being spoken
    out = lt.resend()
    assert [(chunks, t is not None) for chunks, t in out] == [(["a1", "a2"], True), (["c1"], False)]
    lt.on_committed("item_7")
    lt.on_completed("item_7", "again")
    assert f1.result(0) == "again"


def test_resolve_future_ignores_cancelled():
    import concurrent.futures as cf
    f = cf.Future()
    f.cancel()
    sb.resolve_future(f, "late")                  # no InvalidStateError
    assert f.cancelled()


class FakeRealtime:
    """Stands in for RealtimeSession: runs calls inline and records what is sent."""

    def __init__(self, connected=True):
        self.sent = []
        self.connected = connected
        self.fatal = None

    def call(self, fn, *args):
        fn(*args)

    def wait_connected(self, timeout):
        return self.connected

    def send(self, event):
        if not self.connected:
            return False
        self.sent.append(event)
        return True

    def types(self):
        return [e["type"] for e in self.sent]


def make_live_asr(lang="ja"):
    args = types.SimpleNamespace(openai_key="sk-test", asr_model=None, lang=lang)
    asr = sb.OpenAILiveAsr(args)
    asr.rt = FakeRealtime()
    return asr


def test_live_asr_session_config():
    asr = sb.OpenAILiveAsr(types.SimpleNamespace(openai_key="k", asr_model=None, lang="ja"))
    cfg = asr.rt.session
    assert cfg["type"] == "transcription"
    inp = cfg["audio"]["input"]
    assert inp["format"] == {"type": "audio/pcm", "rate": 24000}
    assert inp["transcription"] == {"model": "gpt-live-transcribe", "languages": ["ja"]}
    assert "turn_detection" in inp and inp["turn_detection"] is None
    assert asr.rt.url.endswith("/v1/realtime?intent=transcription")
    auto = sb.OpenAILiveAsr(types.SimpleNamespace(openai_key="k", asr_model=None, lang="auto"))
    assert auto.rt.session["audio"]["input"]["transcription"] == {"model": "gpt-live-transcribe"}


def test_live_asr_append_commit_and_result():
    asr = make_live_asr()
    asr.append(LOUD)
    asr.append(LOUD)
    fut = asr.commit()
    assert asr.rt.types() == ["input_audio_buffer.append"] * 2 + ["input_audio_buffer.commit"]
    audio = b"".join(sb.base64.b64decode(e["audio"]) for e in asr.rt.sent[:2])
    assert abs(len(audio) // 2 - 2 * 480) <= 1                 # 2 x 20 ms at 24 kHz
    assert asr.audio_bytes_sent == len(audio)
    commit = asr.rt.sent[-1]
    asr._on_event({"type": "input_audio_buffer.committed", "item_id": "it1",
                   "previous_item_id": None})
    asr._on_event({"type": "conversation.item.input_audio_transcription.delta",
                   "item_id": "it1", "delta": "こん"})
    asr._on_event({"type": "conversation.item.input_audio_transcription.completed",
                   "item_id": "it1", "transcript": "こんにちは"})
    assert fut.result(0) == "こんにちは" and commit["event_id"] == "commit_1"


def test_live_asr_clear_and_rejected_commit():
    asr = make_live_asr()
    asr.clear()
    assert asr.rt.sent == []                                   # nothing open: nothing to clear
    asr.append(LOUD)
    asr.clear()
    assert asr.rt.types()[-1] == "input_audio_buffer.clear"
    asr.append(LOUD)
    fut = asr.commit()
    asr._on_event({"type": "error", "event_id": "srv_1",
                   "error": {"message": "buffer too small", "event_id": "commit_1"}})
    with pytest.raises(RuntimeError, match="too small"):
        fut.result(0)


def test_live_asr_reconnect_resends_unfinished_turn():
    asr = make_live_asr()
    asr.rt.connected = False                                   # offline while speaking
    asr.append(LOUD)
    fut = asr.commit()
    assert asr.rt.sent == [] and not fut.done()
    asr.rt.connected = True
    asr._on_open()
    assert asr.rt.types() == ["input_audio_buffer.append", "input_audio_buffer.commit"]
    asr._on_event({"type": "input_audio_buffer.committed", "item_id": "x"})
    asr._on_event({"type": "conversation.item.input_audio_transcription.completed",
                   "item_id": "x", "transcript": "ok"})
    assert fut.result(0) == "ok"


class FakeBridge:
    def __init__(self):
        self.emitted = []
        self.tts = None

    def mic_blocked(self):
        return False

    def emit_threadsafe(self, mtype, args):
        self.emitted.append((mtype, args))


def test_mic_listener_streams_through_live_backend():
    """The VAD loop with a streaming backend: start, audio, commit, RECOG_EVENT_STOP."""
    import threading
    asr = make_live_asr()
    bridge = FakeBridge()
    args = types.SimpleNamespace(mic=None, lang="ja", full_duplex=False, ptt=False,
                                 vad_threshold=None, silence_ms=200)
    mic = sb.MicListener(bridge, asr, args)
    frames = [QUIET] * 50 + [LOUD] * 40 + [QUIET] * 30 + [QUIET] * 100
    for f in frames:
        mic.frames.put(f)
    threading.Thread(target=mic._vad_loop, daemon=True).start()
    deadline = sb.time.monotonic() + 5
    while "input_audio_buffer.commit" not in asr.rt.types() and sb.time.monotonic() < deadline:
        sb.time.sleep(0.01)
    assert asr.rt.types().count("input_audio_buffer.commit") == 1
    asr._on_event({"type": "input_audio_buffer.committed", "item_id": "i"})
    asr._on_event({"type": "conversation.item.input_audio_transcription.completed",
                   "item_id": "i", "transcript": "テスト|です"})
    while len(bridge.emitted) < 2 and sb.time.monotonic() < deadline:
        sb.time.sleep(0.01)
    assert bridge.emitted == [("RECOG_EVENT_START", ""), ("RECOG_EVENT_STOP", "テスト です")]
    assert mic.streamer.bytes_appended > 0


# ----- realtime TTS: verbatim check, response routing, cancel / leak guard -----

def test_text_signature():
    assert sb.text_signature("今日の天気は？") == sb.text_signature("今日の天気は?")
    assert sb.text_signature("Hello,  World!") == sb.text_signature("hello world")
    assert sb.text_signature("はい。") != sb.text_signature("いいえ。")


def test_router_cancel_before_created_cancels_the_response_later():
    r = sb.ResponseRouter()
    a = sb.RealtimeRequest("1", "a", "marin")
    r.add(a)
    assert r.cancel("1") == "" and a.cancelled and r.idle()
    req, cancel_now = r.created("1", "resp_a")
    assert req is None and cancel_now
    assert r.audio("resp_a") is None and r.dropped == 1


def test_router_late_delta_of_cancelled_response_is_dropped():
    r = sb.ResponseRouter()
    a, b = sb.RealtimeRequest("1", "a", "v"), sb.RealtimeRequest("2", "b", "v")
    r.add(a)
    assert r.created("1", "resp_a") == (a, False)
    assert r.audio("resp_a") is a
    assert r.cancel("1") == "resp_a"
    r.add(b)
    r.created("2", "resp_b")
    assert r.audio("resp_a") is None                  # never reaches b
    assert r.audio("resp_b") is b
    assert r.done("resp_b") is b and r.idle()


def test_router_without_metadata_assigns_in_order_and_routes_errors():
    r = sb.ResponseRouter()
    a, b = sb.RealtimeRequest("1", "a", "v"), sb.RealtimeRequest("2", "b", "v")
    r.add(a)
    r.add(b)
    assert r.created(None, "x")[0] is a and r.created(None, "y")[0] is b
    assert r.by_event_id("tts_2") is b and r.by_event_id("other") is None
    assert r.drain() == [a, b] and r.idle()


def make_realtime_tts(**kw):
    args = types.SimpleNamespace(openai_key="sk-test", tts_model=None, voice="marin",
                                 instructions=None, **kw)
    tts = sb.OpenAIRealtimeTts(args)
    tts.rt = FakeRealtime()
    return tts


def wait_for(cond, timeout=5.0):
    deadline = sb.time.monotonic() + timeout
    while not cond():
        assert sb.time.monotonic() < deadline, "timed out"
        sb.time.sleep(0.005)


def test_realtime_tts_session_and_request_shape():
    tts = sb.OpenAIRealtimeTts(types.SimpleNamespace(
        openai_key="k", tts_model=None, voice="cedar", instructions="Speak slowly."))
    s = tts.rt.session
    assert tts.rt.url.endswith("/v1/realtime?model=gpt-realtime-2.1-mini")
    assert s["type"] == "realtime" and s["output_modalities"] == ["audio"]
    assert s["audio"]["output"] == {"format": {"type": "audio/pcm", "rate": 24000},
                                    "voice": "cedar"}
    assert s["audio"]["input"]["turn_detection"] is None
    assert "verbatim" in s["instructions"] and s["instructions"].endswith("Speak slowly.")
    assert tts.sample_rate == 24000 and "fable" not in tts.voices


def test_realtime_tts_streams_one_response():
    import threading
    tts = make_realtime_tts()
    out = []
    th = threading.Thread(target=lambda: out.extend(tts.synthesize("こんにちは", "cedar")))
    th.start()
    wait_for(lambda: tts.rt.sent)
    create = tts.rt.sent[0]
    assert create["type"] == "response.create"
    resp = create["response"]
    assert resp["conversation"] == "none" and resp["metadata"] == {"bridge": "1"}
    assert resp["audio"]["output"]["voice"] == "cedar"
    assert resp["input"][0]["content"][0] == {"type": "input_text", "text": "こんにちは"}
    tts._on_event({"type": "response.created",
                   "response": {"id": "r1", "metadata": {"bridge": "1"}}})
    for chunk in (b"\x01\x00" * 10, b"\x02\x00" * 5):
        tts._on_event({"type": "response.output_audio.delta", "response_id": "r1",
                       "delta": sb.base64.b64encode(chunk).decode()})
    tts._on_event({"type": "response.output_audio_transcript.done", "response_id": "r1",
                   "transcript": "こんにちは"})
    tts._on_event({"type": "response.done", "response": {"id": "r1", "status": "completed"}})
    th.join(5)
    assert b"".join(out) == b"\x01\x00" * 10 + b"\x02\x00" * 5
    assert tts.router.idle() and [e["type"] for e in tts.rt.sent] == ["response.create"]


def test_realtime_tts_stop_cancels_and_nothing_leaks_into_the_next_utterance():
    import threading
    tts = make_realtime_tts()
    cancel = threading.Event()
    first = []
    th = threading.Thread(target=lambda: first.extend(
        tts.synthesize_cancellable("one", "marin", cancel)))
    th.start()
    wait_for(lambda: tts.rt.sent)
    tts._on_event({"type": "response.created", "response": {"id": "r1",
                                                           "metadata": {"bridge": "1"}}})
    tts._on_event({"type": "response.output_audio.delta", "response_id": "r1",
                   "delta": sb.base64.b64encode(b"\x01\x00").decode()})
    wait_for(lambda: first)
    cancel.set()                                              # SYNTH_STOP
    th.join(5)
    assert not th.is_alive()
    assert tts.rt.sent[-1] == {"type": "response.cancel", "response_id": "r1"}
    second = []
    th2 = threading.Thread(target=lambda: second.extend(tts.synthesize("two", "marin")))
    th2.start()
    wait_for(lambda: len(tts.rt.sent) == 3)
    # a late delta of the cancelled response arrives after the new request went out
    tts._on_event({"type": "response.output_audio.delta", "response_id": "r1",
                   "delta": sb.base64.b64encode(b"\x09\x00" * 100).decode()})
    tts._on_event({"type": "response.done", "response": {"id": "r1", "status": "cancelled"}})
    tts._on_event({"type": "response.created", "response": {"id": "r2",
                                                           "metadata": {"bridge": "2"}}})
    tts._on_event({"type": "response.output_audio.delta", "response_id": "r2",
                   "delta": sb.base64.b64encode(b"\x02\x00").decode()})
    tts._on_event({"type": "response.done", "response": {"id": "r2", "status": "completed"}})
    th2.join(5)
    assert b"".join(second) == b"\x02\x00" and tts.router.dropped == 1


def test_realtime_tts_errors_reach_the_utterance():
    tts = make_realtime_tts()
    import threading
    errs = []

    def run():
        try:
            list(tts.synthesize("x", "marin"))
        except Exception as e:
            errs.append(e)

    th = threading.Thread(target=run)
    th.start()
    wait_for(lambda: tts.rt.sent)
    tts._on_event({"type": "error", "error": {"message": "bad voice", "event_id": "tts_1"}})
    th.join(5)
    assert errs and "bad voice" in str(errs[0]) and tts.router.idle()

    tts.rt.connected = False
    with pytest.raises(ConnectionError):
        tts.synthesize("x", "marin")


def test_realtime_tts_connection_loss_ends_the_utterance():
    import threading
    tts = make_realtime_tts()
    errs = []

    def run():
        try:
            list(tts.synthesize("x", "marin"))
        except Exception as e:
            errs.append(e)

    th = threading.Thread(target=run)
    th.start()
    wait_for(lambda: tts.rt.sent)
    tts._on_close(ConnectionError("gone"))
    th.join(5)
    assert errs and isinstance(errs[0], ConnectionError)


def test_bridge_uses_the_cancellable_entry_point(monkeypatch):
    seen = []

    async def flow(bridge):
        orig = bridge.tts.synthesize_cancellable

        def spy(text, voice, cancel):
            seen.append(cancel)
            return orig(text, voice, cancel)

        bridge.tts.synthesize_cancellable = spy
        monkeypatch.setattr(sb, "fallback_deadline", lambda f, l, s, r, margin=0: l + 0.05)
        await bridge._on_synth_start("gene", "x", "hi")
        u = bridge.current
        await asyncio.wait_for(u.task, 5)
        assert seen == [u.cancel]

    asyncio.run(run_flow(flow))
