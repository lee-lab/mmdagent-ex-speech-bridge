"""Checks against the real OpenAI API (billed; opt-in).

    OPENAI_API_KEY=... python -m pytest -m requires_openai tests/test_live_openai.py -s

The key may also come from a .env file next to speech_bridge.py. Prints:
  1. whether --tts openai-realtime reads text verbatim (its own transcript and
     a gpt-transcribe round trip of the audio it produced),
  2. a round trip of that audio through --asr openai-live, fed as microphone
     frames in real time through the VAD (and that silence sends nothing),
  3. a latency table: TTS time to first audio, ASR end of speech -> text.
No engine, microphone or audio device is needed.
"""

import difflib
import os
import statistics
import threading
import time
import types

import pytest

import speech_bridge as sb

pytestmark = pytest.mark.requires_openai

_env = dict(os.environ)                    # a copy: collecting this file must not change the
sb.load_dotenv(sb.dotenv_paths(), _env)    # environment the unit tests run in
KEY = _env.get("OPENAI_API_KEY", "")
if not KEY:
    pytest.skip("OPENAI_API_KEY is not set", allow_module_level=True)

SENTENCES = [
    ("ja", "こんにちは、私の名前はメイです。今日は3月15日、気温は23度です。"),
    ("ja", "今日の天気は？"),
    ("ja", "はい。"),
    ("en", "The train leaves at 7:45 and the ticket costs 12 dollars."),
    ("en", "What is the capital of France?"),
    ("en", "Wow!"),
]
LATENCY_TEXT = "今日はいい天気ですね。"
RUNS = 5
FRAME_BYTES = sb.MIC_RATE * sb.MIC_FRAME_MS // 1000 * 2


def make_args(**kw):
    base = dict(openai_key=KEY, tts_model=None, asr_model=None, voice="marin",
                instructions=None, lang="ja")
    base.update(kw)
    return types.SimpleNamespace(**base)


def similarity(a, b):
    return difflib.SequenceMatcher(None, sb.text_signature(a), sb.text_signature(b)).ratio()


def synth_timed(tts, text):
    t0 = time.monotonic()
    first, chunks = None, []
    for c in tts.synthesize(text, "marin"):
        if first is None:
            first = time.monotonic() - t0
        chunks.append(c)
    return b"".join(chunks), first


@pytest.fixture(scope="module")
def realtime_tts():
    tts = sb.OpenAIRealtimeTts(make_args())
    tts.start()
    assert tts.rt.wait_connected(15), tts.rt.fatal
    return tts


@pytest.fixture(scope="module")
def spoken(realtime_tts):
    """(lang, text, pcm 24 kHz, own transcript) for every sentence."""
    out = []
    for lang, text in SENTENCES:
        pcm, _ = synth_timed(realtime_tts, text)
        transcript = next((s for t, s in reversed(realtime_tts.transcripts) if t == text), "")
        out.append((lang, text, pcm, transcript))
    return out


def test_realtime_tts_reads_verbatim(spoken):
    rest = sb.OpenAIAsr(make_args())
    print("\n[1] openai-realtime verbatim check")
    print("    | text | own transcript ok | gpt-transcribe of the audio | sim | audio s |")
    bad = []
    for lang, text, pcm, transcript in spoken:
        heard = rest.transcribe(sb.pcm_to_wav(pcm, sb.REALTIME_RATE), lang)
        own_ok = sb.text_signature(transcript) == sb.text_signature(text)
        sim = similarity(heard, text)
        print(f"    | {text} | {'yes' if own_ok else 'NO: ' + transcript} | {heard} | "
              f"{sim:.2f} | {len(pcm) / 2 / sb.REALTIME_RATE:.2f} |")
        if not own_ok or sim < 0.7:          # the round trip adds ASR formatting ("$12")
            bad.append(text)
    assert not bad, f"not read verbatim: {bad}"


def feed_mic(asr, lang, frames, pace=True):
    """Run the bridge's VAD loop on `frames` as if they came from a microphone."""
    class Bridge:
        tts = None

        def __init__(self):
            self.emitted = []

        def mic_blocked(self):
            return False

        def emit_threadsafe(self, mtype, args):
            self.emitted.append((time.monotonic(), mtype, args))

    bridge = Bridge()
    mic = sb.MicListener(bridge, asr, types.SimpleNamespace(
        mic=None, lang=lang, full_duplex=False, ptt=False, vad_threshold=None, silence_ms=700))
    threading.Thread(target=mic._vad_loop, daemon=True).start()
    t_next = time.monotonic()
    for f in frames:
        mic.frames.put(f)
        if pace:
            t_next += sb.MIC_FRAME_MS / 1000
            time.sleep(max(t_next - time.monotonic(), 0))
    return mic, bridge


def to_frames(pcm24):
    pcm16 = sb.LinearResampler(sb.REALTIME_RATE, sb.MIC_RATE).process(pcm24)
    pcm16 += b"\x00" * (-len(pcm16) % FRAME_BYTES)
    return [pcm16[i:i + FRAME_BYTES] for i in range(0, len(pcm16), FRAME_BYTES)]


SILENCE = b"\x00" * FRAME_BYTES


def vad_keeps(pcm24):
    """Whether the bridge's VAD turns this audio into an utterance (>= 300 ms voiced)."""
    vad = sb.EnergyVad()
    frames = [SILENCE] * 60 + to_frames(pcm24) + [SILENCE] * 60
    return any(e.kind == "end" for f in frames for e in vad.process(f))


def test_live_asr_round_trip_and_silence_sends_nothing(spoken):
    print("\n[2] openai-live round trip (audio of [1] as microphone frames, real time)")
    results = []
    for lang in ("ja", "en"):
        asr = sb.OpenAILiveAsr(make_args(lang=lang))
        asr.start()
        assert asr.rt.wait_connected(15), asr.rt.fatal
        # first silence alone (VAD calibration included): nothing may be sent
        mic, bridge = feed_mic(asr, lang, [SILENCE] * 150, pace=False)
        time.sleep(0.5)
        assert asr.audio_bytes_sent == 0 and mic.streamer.bytes_appended == 0
        print(f"    {lang}: 3.0 s of silence -> {asr.audio_bytes_sent} bytes sent")
        mine = [s for s in spoken if s[0] == lang]
        frames = [SILENCE] * 100                         # 1 s calibration + 1 s silence
        for _, _, pcm, _ in mine:
            frames += to_frames(pcm) + [SILENCE] * 75    # 1.5 s between utterances
        expected = [text for _, text, pcm, _ in mine if vad_keeps(pcm)]
        mic, bridge = feed_mic(asr, lang, frames)
        deadline = time.monotonic() + 20
        while (sum(1 for e in bridge.emitted if e[1] == "RECOG_EVENT_STOP") < len(expected)
               and time.monotonic() < deadline):
            time.sleep(0.05)
        got = [e[2] for e in bridge.emitted if e[1] == "RECOG_EVENT_STOP"]
        sent_s = asr.audio_bytes_sent / 2 / sb.REALTIME_RATE
        fed_s = len(frames) * sb.MIC_FRAME_MS / 1000
        tts_s = sum(len(s[2]) / 2 / sb.REALTIME_RATE for s in mine)
        print(f"    {lang}: fed {fed_s:.1f} s ({tts_s:.2f} s of TTS audio, the rest silence); "
              f"sent {sent_s:.2f} s to OpenAI")
        it = iter(got)
        for _, text, pcm, _ in mine:
            if text not in expected:
                print(f"    | {text} | (dropped by the VAD: under 300 ms voiced, "
                      f"as with --asr openai) | - |")
                continue
            have = next(it, "")
            sim = similarity(have, text)
            print(f"    | {text} | {have} | {sim:.2f} |")
            results.append((text, have, sim))
        assert len(got) == len(expected), got
        assert sent_s < tts_s + 0.6 * len(expected)     # pre-roll 0.26 s + tail 0.3 s each
    # numbers may come back spelled out ("seven forty-five", "三月十五日")
    assert all(sim >= 0.7 for _, _, sim in results), results


def run_live(live, pcm16):
    """Append in real time as if spoken, commit; seconds from commit to the final text."""
    t_next = time.monotonic()
    for i in range(0, len(pcm16), FRAME_BYTES):
        live.append(pcm16[i:i + FRAME_BYTES])
        t_next += sb.MIC_FRAME_MS / 1000
        time.sleep(max(t_next - time.monotonic(), 0))
    t0 = time.monotonic()
    text = live.commit().result(timeout=15)
    return time.monotonic() - t0, text


def test_latency_table(realtime_tts, spoken):
    speech_api = sb.OpenAITts(make_args())
    rest = sb.OpenAIAsr(make_args())
    live = sb.OpenAILiveAsr(make_args())
    for b in (speech_api, rest, live):
        b.start()
    assert live.rt.wait_connected(15)
    time.sleep(1.5)                                    # REST warm-up requests finish
    t_api, t_rt = [], []
    pcm_rt = b""
    for _ in range(RUNS):
        _, first = synth_timed(speech_api, LATENCY_TEXT)
        t_api.append(first)
        pcm_rt, first = synth_timed(realtime_tts, LATENCY_TEXT)
        t_rt.append(first)
    rows = [("TTS openai (Speech API): request -> first PCM", t_api),
            ("TTS openai-realtime: request -> first PCM", t_rt)]
    long_pcm = spoken[0][2]
    for label, pcm24 in (("short", pcm_rt), ("long", long_pcm)):
        pcm16 = sb.LinearResampler(sb.REALTIME_RATE, sb.MIC_RATE).process(pcm24)
        dur = len(pcm16) / 2 / sb.MIC_RATE
        t_rest, t_live = [], []
        for _ in range(RUNS):
            t0 = time.monotonic()
            rest.transcribe(sb.pcm_to_wav(pcm16), "ja")
            t_rest.append(time.monotonic() - t0)
            dt, text = run_live(live, pcm16)
            t_live.append(dt)
            assert text, "empty live transcript"
            time.sleep(1.0)
        rows += [(f"ASR openai (REST gpt-transcribe), {dur:.1f} s utterance: upload + transcribe",
                  t_rest),
                 (f"ASR openai-live, {dur:.1f} s utterance: commit -> final", t_live)]
    med = statistics.median
    print(f"\n[3] latency, median of {RUNS} (TTS text {LATENCY_TEXT!r})")
    print("    | path | median s | runs |")
    for name, xs in rows:
        print(f"    | {name} | {med(xs):.2f} | {', '.join(f'{x:.2f}' for x in xs)} |")
