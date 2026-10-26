#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "openai>=1.68",
#     "websockets>=14",
#     "sounddevice>=0.5.1",
#     "numpy>=1.24",
# ]
# ///
#
# Copyright 2026 Lee Laboratory, Nagoya Institute of Technology
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""mmdagent-ex-speech-bridge: external TTS and ASR for MMDAgent-EX.

Connects to a running MMDAgent-EX through Plugin_API (REST + WebSocket) and
answers the speech messages of its dialogue scripts:

  SYNTH_START|model|voice|text  ->  streamed TTS audio played by the engine,
                                    SYNTH_EVENT_START / SYNTH_EVENT_STOP
  SYNTH_STOP|model              ->  stops the utterance, SYNTH_EVENT_STOP
  microphone speech             ->  RECOG_EVENT_START / RECOG_EVENT_STOP|text

Run:  uv run speech_bridge.py      or      python speech_bridge.py --help
"""

from __future__ import annotations

import abc
import argparse
import asyncio
import collections
import concurrent.futures
import contextlib
import io
import json
import logging
import math
import os
import queue
import re
import signal
import struct
import sys
import threading
import time
import urllib.error
import urllib.request
import wave
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field

import numpy as np
import websockets

__version__ = "0.1.2"

log = logging.getLogger("speech_bridge")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_REST_PORT = 48000
TTS_SAMPLE_RATE = 24000          # OpenAI "pcm" output: 24 kHz mono s16le
MAX_CHUNK_BYTES = 4096           # payload size limit per WebSocket frame
DEFAULT_PREBUFFER_MS = 300       # audio held back before the first push of an utterance
MAX_LEAD_S = 10.0                # never push more audio ahead of playback than this
FALLBACK_MARGIN_S = 1.0          # completion timer slack for engines without "drained"
ECHO_GUARD_S = 0.4               # mic stays muted this long after the bridge stops speaking
MIC_RATE = 16000
MIC_FRAME_MS = 20
MAX_RECOG_CHARS = 500            # keep bus messages well under the engine's buffer size
STATUS_TIMEOUT_S = 5.0
REPLAY_WINDOW_MS = 5000          # = the Lua runtime's wait for SYNTH_EVENT_START
HTTP_KEEPALIVE_S = 300.0         # idle HTTPS connections kept (the SDK default of 5 s closes
                                 # them between dialogue turns: a new TCP+TLS connect each time)
WARM_AFTER_IDLE_S = 240.0        # after this much idleness the REST connection is re-opened when
                                 # speech starts (the server kept idle ones for >= 240 s)

DEFAULT_TTS_MODEL = "gpt-4o-mini-tts"
DEFAULT_ASR_MODEL = "gpt-transcribe"

# Plugin_API binary stream header (dev/plugin-api-audio-playback.md §5.2)
STREAM_HEADER = struct.Struct("<BBBBII")
STREAM_VERSION = 0x01
STREAM_TYPE_AUDIO_PLAYBACK = 0x05
CODEC_PCM16LE = 0x04

OPENAI_VOICES = ("alloy", "ash", "ballad", "cedar", "coral", "echo", "fable",
                 "marin", "nova", "onyx", "sage", "shimmer", "verse")
DEFAULT_VOICE = "marin"

KEY_HINT = ("set OPENAI_API_KEY (in the environment or in a .env file), or pass --openai-key, "
            "or try the offline backends: --tts tone --asr keyboard")

# Settings that may come from a .env file when the environment does not have them
DOTENV_KEYS = ("OPENAI_API_KEY", "MMDAGENT_API_KEY", "MMDAGENT_HOST")


class FatalError(Exception):
    """A configuration problem that retrying cannot fix."""


# ---------------------------------------------------------------------------
# Bus message and audio helpers (pure functions, unit-tested)
# ---------------------------------------------------------------------------

def parse_synth_start(args: str) -> tuple[str, str, str] | None:
    """Split SYNTH_START arguments into (model, voice, text).

    Everything after the second '|' is the text, so a '|' that slipped into
    the text survives. Returns None when the model alias is missing.
    """
    parts = args.split("|", 2)
    parts += [""] * (3 - len(parts))
    model, voice, text = (p.strip() for p in parts)
    return (model, voice, text) if model else None


def sanitize_recog_text(text: str, limit: int = MAX_RECOG_CHARS) -> str:
    """Make recognized text safe as one bus-message field.

    '|' separates fields and a newline ends a message, so both become spaces;
    runs of whitespace collapse to one space.
    """
    text = re.sub(r"[|\r\n\t]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit]


def resolve_voice(requested: str, voices: Sequence[str], default: str) -> tuple[str, bool]:
    """Pick the backend voice for a SYNTH_START voice field.

    Returns (voice, mapped): the requested name when the backend has it
    (case-insensitive), otherwise the default with mapped=True. Old scripts
    say e.g. "mei_voice_normal", which no external backend knows.
    """
    for v in voices:
        if v.lower() == requested.lower():
            return v, False
    return default, True


def pack_frame(stream_id: int, timestamp_ms: int, payload: bytes) -> bytes:
    """Prefix a PCM16LE payload with the 12-byte audio_playback header.

    An empty payload is the end-of-utterance marker (EOS).
    """
    return STREAM_HEADER.pack(STREAM_VERSION, STREAM_TYPE_AUDIO_PLAYBACK, CODEC_PCM16LE, 0,
                              timestamp_ms & 0xFFFFFFFF, stream_id) + payload


def stream_number(stream: dict) -> int:
    """Numeric stream id for the binary header ("stream_id", or NN of "aud_NN")."""
    if "stream_id" in stream:
        return int(stream["stream_id"])
    return int(str(stream["id"]).rsplit("_", 1)[1])


def fallback_deadline(first_send: float, last_send: float, samples: int, rate: int,
                      margin: float = FALLBACK_MARGIN_S) -> float:
    """When to assume playback finished on engines that never send "drained".

    Playback starts with the first chunk and lasts samples/rate; if synthesis
    ran slower than real time the buffer underran, so it cannot end before the
    last chunk arrived either.
    """
    return max(first_send + samples / rate, last_send) + margin


def prebuffer(chunks: Iterable[bytes], nbytes: int) -> Iterator[bytes]:
    """Hold back the start of a stream until `nbytes` are collected (or it ends).

    The first item is that accumulated block; everything after passes through.
    The engine starts playing as soon as audio arrives, so a lone first network
    chunk would underrun (a dropout) before the next one comes. nbytes <= 0
    passes everything through unchanged.
    """
    it = iter(chunks)
    if nbytes > 0:
        head = bytearray()
        for chunk in it:
            head += chunk
            if len(head) >= nbytes:
                break
        if head:
            yield bytes(head)
    yield from it


def rechunk(chunks: Iterable[bytes], size: int = MAX_CHUNK_BYTES) -> Iterator[bytes]:
    """Re-cut a byte stream into even-length pieces of at most `size` bytes.

    Network chunks can split a 16-bit sample; the odd byte is carried over.
    Data is passed on as soon as it arrives (no waiting for a full piece).
    """
    size -= size % 2
    buf = bytearray()
    for chunk in chunks:
        buf += chunk
        while len(buf) >= 2:
            n = min(size, len(buf) - len(buf) % 2)
            yield bytes(buf[:n])
            del buf[:n]


def pcm_to_wav(pcm: bytes, rate: int = MIC_RATE) -> bytes:
    """Wrap mono PCM16LE in an in-memory WAV file."""
    bio = io.BytesIO()
    with wave.open(bio, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)
    return bio.getvalue()


def frame_dbfs(frame: bytes) -> float:
    """RMS level of a PCM16LE frame in dBFS (digital silence -> -100)."""
    x = np.frombuffer(frame, dtype=np.int16).astype(np.float64)
    if x.size == 0:
        return -100.0
    rms = math.sqrt(float(np.mean(x * x))) / 32768.0
    return max(20.0 * math.log10(rms), -100.0) if rms > 0 else -100.0


def same_event(ws_ts: float, buffer_ts: float) -> bool:
    """Could a WS event and an event-history entry be the same message?

    WS events carry whole seconds (time(NULL) * 1000); the history keeps
    milliseconds from a separate clock read, which may fall on either side of
    a second boundary.
    """
    return ws_ts - 1000 <= buffer_ts < ws_ts + 2000


def select_replays(events: Iterable[dict], cutoff_ms: float = 0.0,
                   handled: Iterable[tuple[str, float]] = ()) -> list[dict]:
    """Pick the SYNTH_START requests from the event history that still need an answer.

    `events` are /api/v1/events/recent entries ({timestamp, type, args},
    oldest first). Per model alias the newest SYNTH_START is taken, unless a
    later SYNTH_STOP or SYNTH_EVENT_START for that alias follows it (stopped,
    or already answered by someone). Entries older than `cutoff_ms` (the last
    message the bridge processed over the WebSocket) and SYNTH_STARTs in
    `handled` ((args, ws timestamp) pairs) are ignored. Returns the selected
    entries oldest first.
    """
    handled = list(handled)
    newest: dict[str, dict] = {}
    for e in events:
        etype, args, ts = e.get("type"), str(e.get("args", "")), float(e.get("timestamp", 0))
        alias = args.split("|", 1)[0].strip()
        if etype == "SYNTH_START":
            if ts < cutoff_ms or parse_synth_start(args) is None:
                continue
            if any(a == args and same_event(t, ts) for a, t in handled):
                newest.pop(alias, None)          # superseded by one handled live
                continue
            newest[alias] = e
        elif etype in ("SYNTH_STOP", "SYNTH_EVENT_START"):
            newest.pop(alias, None)
    return sorted(newest.values(), key=lambda e: float(e.get("timestamp", 0)))


def short(text: str, n: int = 40) -> str:
    return text if len(text) <= n else text[: n - 1] + "…"


def transcription_language_args(model: str, lang: str | None) -> dict:
    """Language hint for /v1/audio/transcriptions.

    gpt-transcribe takes a list ('languages'; sending 'language' too is an
    error); the older models take a single 'language'. The list goes through
    extra_body so that SDK versions without the parameter can send it.
    """
    if not lang:
        return {}
    if model.startswith(("gpt-transcribe", "gpt-live-transcribe")):
        return {"extra_body": {"languages": [lang]}}
    return {"language": lang}


# ---------------------------------------------------------------------------
# .env files (pure parser, unit-tested)
# ---------------------------------------------------------------------------

_DOTENV_KEY = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_DOTENV_QUOTED = re.compile(r"""(["'])(.*?)\1\s*(?:#.*)?""")


def parse_dotenv(text: str) -> dict[str, str]:
    """Read KEY=value lines: '#' comments, optional 'export ', optional quotes.

    No variable expansion and no multi-line values. Malformed lines are skipped.
    """
    out: dict[str, str] = {}
    for raw in text.lstrip("﻿").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, sep, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if not sep or not _DOTENV_KEY.fullmatch(key):
            continue
        m = _DOTENV_QUOTED.fullmatch(value)
        if m:
            value = m.group(2)
        else:
            value = re.split(r"\s+#", value, maxsplit=1)[0].rstrip()
        out[key] = value
    return out


def dotenv_paths() -> list[str]:
    """Where .env files are read from: the current directory, then the script's directory."""
    paths: list[str] = []
    for d in (os.getcwd(), os.path.dirname(os.path.abspath(__file__))):
        p = os.path.join(d, ".env")
        if all(os.path.normcase(os.path.realpath(p)) != os.path.normcase(os.path.realpath(q))
               for q in paths):
            paths.append(p)
    return paths


def load_dotenv(paths: Iterable[str], environ, keys: Sequence[str] = DOTENV_KEYS
                ) -> list[tuple[str, str]]:
    """Fill the `keys` that `environ` lacks (or has empty) from .env files.

    The environment always wins; among the files the first one that sets a key
    wins. Returns the (key, path) pairs taken, never the values.
    """
    taken: list[tuple[str, str]] = []
    for path in paths:
        try:
            with open(path, encoding="utf-8-sig") as f:
                values = parse_dotenv(f.read())
        except (OSError, UnicodeDecodeError):
            continue
        for k in keys:
            if values.get(k) and not environ.get(k):
                environ[k] = values[k]
                taken.append((k, path))
    return taken


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------

class TtsBackend(abc.ABC):
    """Text-to-speech backend: streams PCM16LE mono at `sample_rate`."""

    sample_rate: int = TTS_SAMPLE_RATE
    voices: Sequence[str] = ()  # empty = voice is ignored

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args

    @abc.abstractmethod
    def synthesize(self, text: str, voice: str) -> Iterator[bytes]:
        """Yield audio chunks as soon as they are available (any length)."""

    def start(self) -> None:
        """Called once before the first request: connect / warm up (must not block)."""

    def prepare(self) -> None:
        """A request is probably coming soon (the user started speaking); must not block."""

    def describe(self) -> str:
        return type(self).__name__


class AsrBackend(abc.ABC):
    """Speech recognition backend."""

    needs_microphone = True     # False = the backend produces text by itself

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args

    @abc.abstractmethod
    def transcribe(self, wav: bytes, lang: str | None) -> str:
        """Return the text of a 16 kHz mono 16-bit WAV file (lang None = auto)."""

    def start(self) -> None:
        """Called once before the first request: connect / warm up (must not block)."""

    def prepare(self) -> None:
        """A request is probably coming soon (the user started speaking); must not block."""

    def describe(self) -> str:
        return type(self).__name__


def _keepalive_http_client(openai):
    """The SDK's default HTTP client, keeping idle connections for HTTP_KEEPALIVE_S.

    The SDK is built on httpx (openai 1.x / 2.x) or httpx2 (3.x); the Limits
    class is taken from whichever module its client class comes from.
    """
    try:
        cls = openai.DefaultHttpxClient
        mod = next(sys.modules[c.__module__.split(".")[0]] for c in cls.__mro__
                   if c.__name__ == "Client" and c.__module__.split(".")[0] in ("httpx", "httpx2"))
        return cls(limits=mod.Limits(max_connections=20, max_keepalive_connections=5,
                                     keepalive_expiry=HTTP_KEEPALIVE_S))
    except Exception as e:       # an SDK shape we do not know: keep its defaults
        log.debug("keep-alive HTTP client unavailable (%s); using the SDK default", e)
        return None


def make_openai_client(args: argparse.Namespace):
    """A client with its own connection pool (one per backend: TTS and ASR never evict
    each other's keep-alive connection)."""
    try:
        import openai
    except ImportError as e:
        raise FatalError("the 'openai' package is missing: pip install -r requirements.txt") from e
    http_client = _keepalive_http_client(openai)
    kwargs = {"http_client": http_client} if http_client is not None else {}
    return openai.OpenAI(api_key=args.openai_key, timeout=30.0, max_retries=1, **kwargs)


def is_key_error(e: BaseException) -> bool:
    return type(e).__name__ in ("AuthenticationError", "PermissionDeniedError")


class OpenAIRestBackend:
    """Shared by the REST backends: keep the HTTPS connection open and warm."""

    tag = ""

    def _init_rest(self, args: argparse.Namespace, model: str) -> None:
        self.client = make_openai_client(args)
        self.model = model
        self._last_use = 0.0
        self._warming = threading.Lock()

    def _used(self) -> None:
        self._last_use = time.monotonic()

    def _warm(self, startup: bool) -> None:
        """Open the connection with a free request (also checks the key and the model)."""
        if not self._warming.acquire(blocking=False):
            return
        try:
            self._used()
            self.client.models.retrieve(self.model)
        except Exception as e:
            name = type(e).__name__
            if is_key_error(e):
                log.error("[%s] the OpenAI API rejected the key (%s): %s", self.tag, name, KEY_HINT)
            elif name == "NotFoundError" and startup:
                log.warning("[%s] model '%s' is not available to this API key", self.tag,
                            self.model)
            else:
                log.debug("[%s] warm-up failed: %s: %s", self.tag, name, e)
        finally:
            self._warming.release()

    def start(self) -> None:
        threading.Thread(target=self._warm, args=(True,), name=f"warm-{self.tag}",
                         daemon=True).start()

    def prepare(self) -> None:
        if time.monotonic() - self._last_use > WARM_AFTER_IDLE_S:
            threading.Thread(target=self._warm, args=(False,), name=f"warm-{self.tag}",
                             daemon=True).start()


class OpenAITts(OpenAIRestBackend, TtsBackend):
    """OpenAI speech API (/v1/audio/speech), streamed as raw 24 kHz PCM.

    Deprecated by OpenAI: its models shut down on 2027-01-06; the successor is
    the openai-realtime backend.
    """

    voices = OPENAI_VOICES
    tag = "tts"

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)
        self._init_rest(args, getattr(args, "tts_model", None) or DEFAULT_TTS_MODEL)

    def synthesize(self, text: str, voice: str) -> Iterator[bytes]:
        kwargs = {"model": self.model, "voice": voice, "input": text[:4096],
                  "response_format": "pcm"}
        if self.args.instructions:
            kwargs["instructions"] = self.args.instructions
        self._used()
        with self.client.audio.speech.with_streaming_response.create(**kwargs) as resp:
            yield from resp.iter_bytes(MAX_CHUNK_BYTES)
        self._used()

    def describe(self) -> str:
        return f"openai ({self.model}, voice {self.args.voice})"


class ToneTts(TtsBackend):
    """Offline wiring check: a beep about 60 ms per character (0.5-6 s)."""

    FREQ = 440.0
    AMPLITUDE = 0.2
    FADE_S = 0.01

    def synthesize(self, text: str, voice: str) -> Iterator[bytes]:
        seconds = min(max(len(text) * 0.06, 0.5), 6.0)
        n = int(seconds * self.sample_rate)
        t = np.arange(n) / self.sample_rate
        env = np.minimum(1.0, np.minimum(t, seconds - t) / self.FADE_S)
        pcm = (np.sin(2 * np.pi * self.FREQ * t) * env * self.AMPLITUDE * 32767).astype("<i2")
        data = pcm.tobytes()
        step = self.sample_rate // 10 * 2            # 100 ms
        for i in range(0, len(data), step):
            yield data[i:i + step]

    def describe(self) -> str:
        return "tone (offline beep)"


class OpenAIAsr(OpenAIRestBackend, AsrBackend):
    """OpenAI transcription API (/v1/audio/transcriptions), one request per utterance."""

    tag = "asr"

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)
        self._init_rest(args, getattr(args, "asr_model", None) or DEFAULT_ASR_MODEL)

    def transcribe(self, wav: bytes, lang: str | None) -> str:
        self._used()
        try:
            result = self.client.audio.transcriptions.create(
                model=self.model, file=("speech.wav", wav, "audio/wav"),
                **transcription_language_args(self.model, lang))
        finally:
            self._used()
        return getattr(result, "text", "") or ""

    def describe(self) -> str:
        return f"openai ({self.model}, lang {self.args.lang})"


class KeyboardAsr(AsrBackend):
    """Offline wiring check: each line typed in the terminal is a recognition result."""

    needs_microphone = False

    def transcribe(self, wav: bytes, lang: str | None) -> str:
        raise NotImplementedError("the keyboard backend reads the terminal, not audio")

    def describe(self) -> str:
        return "keyboard (type a line + Enter)"


# Registries: add a backend by writing a class and listing it here.
TTS_BACKENDS: dict[str, type[TtsBackend]] = {"openai": OpenAITts, "tone": ToneTts}
ASR_BACKENDS: dict[str, type[AsrBackend]] = {"openai": OpenAIAsr, "keyboard": KeyboardAsr}


# ---------------------------------------------------------------------------
# Voice activity detection (pure state machine)
# ---------------------------------------------------------------------------

@dataclass
class VadEvent:
    kind: str                   # "start" | "end" | "cancel"
    audio: bytes = b""          # "end": the segment (pre-roll included)
    reason: str = ""            # "end": "silence" | "max"; "cancel": "short" | "gated"


class EnergyVad:
    """Energy-based endpointer working on fixed-size PCM16LE frames.

    The noise floor is measured over the first `calibration_ms` (and follows
    slowly while nobody speaks); a frame is speech when its level is
    `factor_db` above the floor and above `min_dbfs`, or above the fixed
    `threshold_dbfs` when one is given.
    """

    def __init__(self, frame_ms: int = MIC_FRAME_MS, threshold_dbfs: float | None = None,
                 factor_db: float = 10.0, min_dbfs: float = -50.0,
                 calibration_ms: int = 1000, onset_frames: int = 3, silence_ms: int = 700,
                 min_speech_ms: int = 300, max_speech_ms: int = 20000, preroll_ms: int = 200,
                 floor_adapt: float = 0.02) -> None:
        self.fixed_threshold = threshold_dbfs
        self.factor_db = factor_db
        self.min_dbfs = min_dbfs
        self.onset_frames = onset_frames
        self.calib_frames = max(1, calibration_ms // frame_ms)
        self.silence_frames = max(1, silence_ms // frame_ms)
        self.min_speech_frames = max(1, min_speech_ms // frame_ms)
        self.max_frames = max(1, max_speech_ms // frame_ms)
        self.tail_frames = self.min_speech_frames      # trailing silence kept in a segment
        self.floor_adapt = floor_adapt
        self.floor_db: float | None = None
        self.state = "idle" if threshold_dbfs is not None else "calibrating"
        self._calib: list[float] = []
        self._pre: collections.deque[bytes] = collections.deque(
            maxlen=max(1, preroll_ms // frame_ms) + onset_frames)
        self._run = 0
        self._seg: list[bytes] = []
        self._onset_index = 0
        self._last_speech = 0
        self._silence = 0

    @property
    def threshold_dbfs(self) -> float:
        if self.fixed_threshold is not None:
            return self.fixed_threshold
        floor = self.floor_db if self.floor_db is not None else -100.0
        return max(floor + self.factor_db, self.min_dbfs)

    @property
    def in_segment(self) -> bool:
        return self.state == "speech"

    def _to_idle(self) -> None:
        self.state = "idle"
        self._pre.clear()
        self._seg = []
        self._run = 0
        self._silence = 0

    def _finish(self, reason: str) -> VadEvent:
        voiced = self._last_speech - self._onset_index
        seg = self._seg
        self._to_idle()
        if voiced < self.min_speech_frames:
            return VadEvent("cancel", reason="short")
        if reason == "silence":
            seg = seg[: self._last_speech + self.tail_frames]
        return VadEvent("end", audio=b"".join(seg), reason=reason)

    def process(self, frame: bytes, gated: bool = False) -> list[VadEvent]:
        """Feed one frame; return the events it caused (usually none)."""
        if gated:
            was = self.in_segment
            if self.state != "calibrating":
                self._to_idle()
            return [VadEvent("cancel", reason="gated")] if was else []
        db = frame_dbfs(frame)
        if self.state == "calibrating":
            self._calib.append(db)
            if len(self._calib) >= self.calib_frames:
                self.floor_db = float(np.median(self._calib))
                self.state = "idle"
            return []
        is_speech = db >= self.threshold_dbfs
        if self.state == "idle":
            self._pre.append(frame)
            if is_speech:
                self._run += 1
            else:
                self._run = 0
                if self.fixed_threshold is None and self.floor_db is not None:
                    self.floor_db += self.floor_adapt * (db - self.floor_db)
            if self._run >= self.onset_frames:
                self.state = "speech"
                self._seg = list(self._pre)
                self._onset_index = len(self._seg) - self._run
                self._last_speech = len(self._seg)
                self._silence = 0
                return [VadEvent("start")]
            return []
        # in a segment
        self._seg.append(frame)
        if is_speech:
            self._silence = 0
            self._last_speech = len(self._seg)
        else:
            self._silence += 1
        if len(self._seg) >= self.max_frames:
            return [self._finish("max")]
        if self._silence >= self.silence_frames:
            return [self._finish("silence")]
        return []


# ---------------------------------------------------------------------------
# Engine REST client
# ---------------------------------------------------------------------------

class EngineRest:
    """Minimal blocking JSON client for the Plugin_API REST endpoints."""

    def __init__(self, base_url: str, key: str | None) -> None:
        self.base = base_url.rstrip("/")
        self.key = key

    def request(self, method: str, path: str, body: dict | None = None,
                timeout: float = STATUS_TIMEOUT_S) -> tuple[int, dict]:
        """Return (HTTP status, JSON body). Raises OSError when unreachable."""
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.base + path, data=data, method=method)
        req.add_header("Content-Type", "application/json")
        if self.key:
            req.add_header("Authorization", f"Bearer {self.key}")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                status, raw = resp.status, resp.read()
        except urllib.error.HTTPError as e:
            status, raw = e.code, e.read()
        try:
            parsed = json.loads(raw) if raw else {}
        except ValueError:
            parsed = {"raw": raw.decode("utf-8", "replace")}
        return status, parsed if isinstance(parsed, dict) else {"data": parsed}


def error_text(body: dict) -> str:
    err = body.get("error")
    if isinstance(err, dict):
        return err.get("message") or err.get("code") or str(err)
    return str(err or body.get("message") or body)


# ---------------------------------------------------------------------------
# The bridge (asyncio side: WebSocket, stream, TTS utterances)
# ---------------------------------------------------------------------------

@dataclass
class Utterance:
    model: str
    voice: str
    text: str
    cancel: threading.Event = field(default_factory=threading.Event)
    drained: asyncio.Event = field(default_factory=asyncio.Event)
    task: asyncio.Task | None = None
    pushed: bool = False        # at least one chunk went to the engine
    started: bool = False       # SYNTH_EVENT_START was sent
    eos_sent: bool = False
    finished: bool = False      # SYNTH_EVENT_STOP handled (by completion or stop)
    prebuffered: int = 0        # bytes held back before the first push


class Bridge:
    def __init__(self, args: argparse.Namespace, tts: TtsBackend | None) -> None:
        self.args = args
        self.tts = tts
        self.rest = EngineRest(f"http://{args.host}:{args.rest_port}/api/v1", args.mmdagent_key)
        self.ws_url = f"ws://{args.host}:{args.ws_port}"
        self.loop: asyncio.AbstractEventLoop | None = None
        self.ws = None
        self.stream: dict | None = None
        self.bound_model = ""
        self.flush_supported: bool | None = None
        self.current: Utterance | None = None
        self._tts_lock: asyncio.Lock | None = None
        self._tasks: set[asyncio.Task] = set()
        self._warned_voices: set[str] = set()
        self._conflicts: set[str] = set()
        self._speaking = False
        self._gate_until = 0.0
        self._was_connected = False
        self._last_ws_ts = 0.0           # timestamp of the last bus event read over the WS
        self._live_starts: collections.deque[tuple[str, float]] = collections.deque(maxlen=64)
        self._replayed: collections.deque[tuple[str, float]] = collections.deque(maxlen=64)
        self.on_connected: Callable[[], None] | None = None

    # ----- thread-safe API used by the microphone / keyboard threads -----

    def mic_blocked(self) -> bool:
        """True while the bridge is speaking (plus the echo guard)."""
        return self._speaking or time.monotonic() < self._gate_until

    def emit_threadsafe(self, mtype: str, args: str) -> None:
        if self.loop is not None and self.loop.is_running():
            asyncio.run_coroutine_threadsafe(self.emit(mtype, args), self.loop)

    # ----- engine I/O -----

    async def _rest(self, method: str, path: str, body: dict | None = None) -> tuple[int, dict]:
        return await asyncio.to_thread(self.rest.request, method, path, body)

    async def emit(self, mtype: str, args: str) -> None:
        """Send a message to the engine's bus (WebSocket, REST as fallback)."""
        log.debug("-> %s|%s", mtype, args)
        if self.ws is not None:
            with contextlib.suppress(websockets.ConnectionClosed):
                await self.ws.send(json.dumps({"action": "command", "type": mtype, "args": args}))
                return
        with contextlib.suppress(OSError):
            await self._rest("POST", "/messages", {"type": mtype, "args": args})

    async def run(self) -> None:
        """Connect, serve, and reconnect forever (until cancelled)."""
        self.loop = asyncio.get_running_loop()
        self._tts_lock = asyncio.Lock()
        delay, waiting = 1.0, False
        while True:
            try:
                await self._session()
                log.info("[engine] connection closed; reconnecting")
                delay, waiting = 1.0, False
            except FatalError:
                raise
            except (OSError, websockets.WebSocketException, asyncio.TimeoutError) as e:
                if self._was_connected:
                    log.info("[engine] connection lost (%s); reconnecting", e or type(e).__name__)
                    delay, waiting = 1.0, False
                elif not waiting:
                    log.info("[engine] waiting for MMDAgent-EX at %s (start it with Plugin_API "
                             "enabled; retrying)", self.rest.base)
                    waiting = True
                log.debug("[engine] connect failed: %s", e)
            await asyncio.sleep(delay)
            delay = min(delay * 2, 5.0)

    async def _session(self) -> None:
        self._was_connected = False
        status, body = await self._rest("GET", "/status")
        if status == 401:
            raise FatalError("the engine rejected the API key (401): pass the engine's "
                             "Plugin_API_Key with --mmdagent-key or MMDAGENT_API_KEY")
        headers = {"Authorization": f"Bearer {self.args.mmdagent_key}"} if self.args.mmdagent_key else {}
        try:
            ws = await websockets.connect(self.ws_url, additional_headers=headers,
                                          open_timeout=STATUS_TIMEOUT_S, max_size=None)
        except websockets.InvalidStatus as e:
            if e.response.status_code == 401:
                raise FatalError("the engine's WebSocket rejected the API key (401): check "
                                 "--mmdagent-key / MMDAGENT_API_KEY") from e
            raise
        self.ws = ws
        cutoff = self._last_ws_ts        # what earlier sessions already saw
        try:
            if self.tts is not None:
                # stream first: a SYNTH_START read while waiting for the ack needs it
                await self._open_stream()
                events = ["SYNTH_START", "SYNTH_STOP", "APIAUDIO_EVENT_DRAINED",
                          "stream.audio_playback.*"]
                await ws.send(json.dumps({"action": "subscribe", "events": events}))
                await self._wait_ack(ws)
            self._was_connected = True
            log.info("[engine] connected to %s (%s %s)%s", self.rest.base,
                     body.get("platform", "?"), body.get("version", ""),
                     ", subscribed to SYNTH_START / SYNTH_STOP" if self.tts else "")
            if self.on_connected:
                self.on_connected()
                self.on_connected = None
            if self.tts is not None:
                await self._replay_missed(cutoff)
            async for msg in ws:
                if isinstance(msg, str):
                    self._on_text(msg)
        finally:
            self.ws = None
            await ws.close()
            self._teardown()

    def _teardown(self) -> None:
        """Forget per-connection state; tell scripts an utterance ended; free the stream."""
        u, self.current = self.current, None
        if u is not None and not u.finished:
            u.finished = True
            u.cancel.set()
            if u.task:
                u.task.cancel()
            if u.started:
                with contextlib.suppress(OSError):
                    self.rest.request("POST", "/messages",
                                      {"type": "SYNTH_EVENT_STOP", "args": u.model}, timeout=2)
        self._end_speaking()
        if self.stream is not None:
            with contextlib.suppress(OSError):
                self.rest.request("DELETE", f"/streams/{self.stream['id']}", timeout=2)
        self.stream, self.bound_model = None, ""

    def _spawn(self, coro) -> None:
        task = asyncio.ensure_future(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _wait_ack(self, ws) -> None:
        """Read until the engine acknowledges the subscription (events seen meanwhile are handled)."""
        deadline = time.monotonic() + STATUS_TIMEOUT_S
        while True:
            msg = await asyncio.wait_for(ws.recv(), timeout=max(deadline - time.monotonic(), 0.1))
            if isinstance(msg, str):
                with contextlib.suppress(ValueError):
                    if json.loads(msg).get("ack") is True:
                        return
                self._on_text(msg)

    async def _replay_missed(self, cutoff: float) -> None:
        """Answer SYNTH_STARTs sent before the subscription was up (engine startup greeting)."""
        try:
            status, data = await self._rest("GET", f"/events/recent?since_ms={REPLAY_WINDOW_MS}")
        except OSError as e:
            log.debug("[tts] event history unavailable: %s", e)
            return
        if status != 200:
            log.debug("[tts] event history unavailable (%d)", status)
            return
        now = float(data.get("now", 0))
        for e in select_replays(data.get("events", []), cutoff, self._live_starts):
            args, ts = str(e["args"]), float(e["timestamp"])
            self._replayed.append((args, ts))
            log.info("[tts] replaying SYNTH_START missed during connect: %s (%.1f s ago)",
                     short(args, 60), (now - ts) / 1000 if now else 0.0)
            await self._on_synth_start(*parse_synth_start(args))

    def _on_text(self, msg: str) -> None:
        try:
            data = json.loads(msg)
        except ValueError:
            return
        event = data.get("event")
        if not event:
            return
        args = str(data.get("args", ""))
        log.debug("<- %s", f"{event}|{args}" if args else msg)
        ws_ts = float(data.get("timestamp", 0) or 0)
        if event != "stream.lifecycle" and ws_ts:
            self._last_ws_ts = max(self._last_ws_ts, ws_ts)
        if event == "SYNTH_START":
            parsed = parse_synth_start(args)
            if parsed is None:
                log.warning("[tts] ignoring SYNTH_START without a model alias: %r", args)
            elif any(a == args and same_event(ws_ts, t) for a, t in self._replayed):
                log.debug("[tts] SYNTH_START already answered from the event history")
            elif self.tts is not None:
                self._live_starts.append((args, ws_ts))
                self._spawn(self._on_synth_start(*parsed))
        elif event == "SYNTH_STOP" and self.tts is not None:
            self._spawn(self._on_synth_stop(args.split("|", 1)[0].strip()))
        elif event == "APIAUDIO_EVENT_DRAINED":
            self._on_drained(args.split("|", 1)[0].strip())
        elif event == "stream.lifecycle":
            ours = self.stream is not None and data.get("stream") == self.stream["id"]
            if data.get("status") == "drained":
                self._on_drained(str(data.get("stream", "")))
            elif data.get("status") == "stopped" and ours:
                log.info("[tts] playback stream %s was stopped by the engine; "
                         "a new one is created for the next utterance", self.stream["id"])
                self.stream, self.bound_model = None, ""

    def _on_drained(self, stream_id: str) -> None:
        u = self.current
        if u and u.eos_sent and self.stream and stream_id == self.stream["id"]:
            u.drained.set()

    # ----- playback stream -----

    async def _open_stream(self, model: str = "") -> bool:
        body = {"type": "audio_playback", "codec": "pcm",
                "sample_rate": self.tts.sample_rate, "channels": 1}
        if model:
            body["model"] = model
        status, data = await self._rest("POST", "/streams", body)
        if status == 409 and model:
            await self._report_conflict(model)
            body.pop("model")
            model = ""
            status, data = await self._rest("POST", "/streams", body)
        if status not in (200, 202):
            log.error("[tts] cannot create the audio_playback stream (%d: %s). Does the engine "
                      "host have an audio output device, and fewer than 4 streams open?",
                      status, error_text(data))
            self.stream = None
            return False
        self.stream, self.bound_model = data, model
        log.debug("[tts] playback stream %s (%d Hz)", data.get("id"), self.tts.sample_rate)
        return True

    async def _delete_stream(self) -> None:
        if self.stream is not None:
            with contextlib.suppress(OSError):
                await self._rest("DELETE", f"/streams/{self.stream['id']}")
        self.stream, self.bound_model = None, ""

    async def _report_conflict(self, model: str) -> None:
        if model in self._conflicts:
            return
        self._conflicts.add(model)
        holder = "another audio_playback stream"
        with contextlib.suppress(OSError):
            _, data = await self._rest("GET", "/streams")
            for s in data.get("streams", []):
                if s.get("type") == "audio_playback" and s.get("model") == model:
                    holder = f"stream {s.get('id')}"
        log.error("[tts] lip sync for '%s' is unavailable: %s already drives its mouth (409). "
                  "Is another bridge running, or did an earlier run crash? Delete that stream "
                  "(DELETE /api/v1/streams/<id>) or restart the engine. Audio still plays.",
                  model, holder)

    async def _ensure_stream(self, model: str) -> bool:
        """Have a playback stream bound to `model` (lip sync follows the binding)."""
        if self.stream is None:
            return await self._open_stream(model)
        if model == self.bound_model:
            return True
        status, data = await self._rest("PATCH", f"/streams/{self.stream['id']}", {"model": model})
        if status == 200:
            self.bound_model = model
            self._conflicts.discard(model)
        elif status == 404:
            self.stream = None
            return await self._open_stream(model)
        elif status == 409:
            await self._report_conflict(model)
            # do not let the previous model's mouth move with this voice
            if self.bound_model:
                await self._rest("PATCH", f"/streams/{self.stream['id']}", {"model": ""})
                self.bound_model = ""
        else:
            log.warning("[tts] binding the stream to '%s' failed (%d: %s)", model, status,
                        error_text(data))
        return True

    async def _flush(self) -> None:
        """Discard audio the engine has buffered but not yet played."""
        if self.stream is None:
            return
        if self.flush_supported is not False:
            status, data = await self._rest("PATCH", f"/streams/{self.stream['id']}",
                                            {"flush": True})
            if status == 200:
                self.flush_supported = True
                return
            if status != 400:
                log.warning("[tts] flush failed (%d: %s)", status, error_text(data))
                return
            self.flush_supported = False
            log.info("[tts] this engine has no stream flush; stopping by re-creating the stream")
        model = self.bound_model
        await self._delete_stream()
        await self._open_stream(model)

    # ----- utterances -----

    def _end_speaking(self) -> None:
        if self._speaking:
            self._gate_until = time.monotonic() + ECHO_GUARD_S
        self._speaking = False

    def _pick_voice(self, requested: str) -> str:
        voice, mapped = resolve_voice(requested, self.tts.voices, self.args.voice)
        if mapped and self.tts.voices and requested and requested not in self._warned_voices:
            self._warned_voices.add(requested)
            log.info("[tts] voice '%s' is unknown to the %s backend; using '%s' (see --voice)",
                     requested, self.args.tts, voice)
        return voice

    async def _on_synth_start(self, model: str, voice: str, text: str) -> None:
        async with self._tts_lock:
            await self._stop_current("replaced")
            u = Utterance(model, self._pick_voice(voice), text)
            self.current = u
            u.task = asyncio.ensure_future(self._run_utterance(u))

    async def _on_synth_stop(self, model: str) -> None:
        async with self._tts_lock:
            if self.current is not None and self.current.model == model:
                await self._stop_current("stopped")

    async def _stop_current(self, reason: str) -> None:
        u = self.current
        if u is None:
            return
        u.cancel.set()
        if u.task is not None and not u.task.done():
            u.task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await u.task
        if self.current is u:
            self.current = None
        if u.finished:
            return
        u.finished = True
        if u.pushed:
            await self._flush()
        if u.started:
            await self.emit("SYNTH_EVENT_STOP", u.model)
        self._end_speaking()
        log.info('[tts] %s: "%s" %s', u.model, short(u.text), reason)

    def _synth_thread(self, u: Utterance, q: asyncio.Queue) -> None:
        """Run the (blocking) backend and hand chunks to the event loop."""
        def put(item) -> None:
            with contextlib.suppress(RuntimeError):      # loop already closed
                self.loop.call_soon_threadsafe(q.put_nowait, item)
        src = gen = None
        try:
            src = self.tts.synthesize(u.text, u.voice)
            hold_ms = getattr(self.args, "prebuffer_ms", DEFAULT_PREBUFFER_MS)
            hold = self.tts.sample_rate * max(hold_ms, 0) // 1000 * 2
            head = prebuffer(src, hold)

            def counted() -> Iterator[bytes]:
                for i, block in enumerate(head):
                    if i == 0 and hold > 0:
                        u.prebuffered = len(block)
                    yield block
            gen = rechunk(counted())
            for chunk in gen:
                if u.cancel.is_set():
                    break
                put(chunk)
            put(None)
        except Exception as e:       # reported on the loop side
            put(e)
        finally:
            for g in (gen, src):     # closing ends the backend's HTTP response early
                with contextlib.suppress(Exception):
                    g.close()

    async def _run_utterance(self, u: Utterance) -> None:
        rate = self.tts.sample_rate
        t_req = time.monotonic()
        samples, t_first, t_last, failed = 0, 0.0, 0.0, False
        try:
            if not u.text:
                raise RuntimeError("empty text")
            if not await self._ensure_stream(u.model):
                raise RuntimeError("no playback stream")
            sid = stream_number(self.stream)
            q: asyncio.Queue = asyncio.Queue()
            threading.Thread(target=self._synth_thread, args=(u, q), daemon=True).start()
            while True:
                item = await q.get()
                if item is None:
                    break
                if isinstance(item, BaseException):
                    raise item
                if u.pushed:
                    lead = samples / rate - (time.monotonic() - t_first)
                    if lead > MAX_LEAD_S:
                        await asyncio.sleep(lead - MAX_LEAD_S)
                await self.ws.send(pack_frame(sid, samples * 1000 // rate, item))
                samples += len(item) // 2
                t_last = time.monotonic()
                if not u.pushed:
                    u.pushed, t_first, self._speaking = True, t_last, True
                    u.started = True
                    await self.emit("SYNTH_EVENT_START", u.model)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            failed = True
            self._log_tts_error(e)
        if u.pushed:
            with contextlib.suppress(websockets.ConnectionClosed, AttributeError):
                await self.ws.send(pack_frame(sid, samples * 1000 // rate, b""))    # EOS
            u.eos_sent = True
            wait = fallback_deadline(t_first, t_last, samples, rate) - time.monotonic()
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(u.drained.wait(), timeout=max(wait, 0.0))
        if u.finished:
            return
        u.finished = True
        if not u.started:
            await self.emit("SYNTH_EVENT_START", u.model)    # so scripts never hang
        await self.emit("SYNTH_EVENT_STOP", u.model)
        if self.current is u:
            self.current = None
        self._end_speaking()
        if u.pushed:
            how = "drained" if u.drained.is_set() else "timer"
            pre = (f", prebuffer {u.prebuffered * 500 // rate} ms"
                   if log.isEnabledFor(logging.DEBUG) else "")
            log.info('[tts] %s: "%s" → %.1f s (first audio after %.2f s, end by %s%s)%s', u.model,
                     short(u.text), samples / rate, t_first - t_req, how, pre,
                     " [synthesis failed midway]" if failed else "")

    def _log_tts_error(self, e: Exception) -> None:
        name = type(e).__name__
        if is_key_error(e):
            log.error("[tts] the OpenAI API rejected the key (%s): %s", name, KEY_HINT)
        else:
            log.error("[tts] synthesis failed: %s: %s", name, e)


# ---------------------------------------------------------------------------
# Microphone, push-to-talk and keyboard input (threads)
# ---------------------------------------------------------------------------

def import_sounddevice():
    try:
        import sounddevice
        return sounddevice
    except (ImportError, OSError) as e:
        raise FatalError(f"cannot load sounddevice / PortAudio ({e}); install it with "
                         "'pip install sounddevice' (Linux also needs libportaudio2)") from e


def list_mics() -> int:
    sd = import_sounddevice()
    devices = sd.query_devices()
    try:
        default_in = sd.default.device[0]
    except Exception:
        default_in = -1
    print("Input devices (use --mic INDEX or --mic NAME-SUBSTRING):")
    found = False
    for i, d in enumerate(devices):
        if d.get("max_input_channels", 0) > 0:
            found = True
            mark = "*" if i == default_in else " "
            api = sd.query_hostapis(d["hostapi"])["name"]
            print(f" {mark} {i:3d}  {d['name']}  ({api}, {int(d['default_samplerate'])} Hz)")
    if not found:
        print("  (none)")
    return 0


def find_mic(sd, spec: str | None) -> int | None:
    """Device index for --mic (index or case-insensitive name substring)."""
    if spec is None or spec == "":
        return None
    if spec.isdigit():
        return int(spec)
    for i, d in enumerate(sd.query_devices()):
        if d.get("max_input_channels", 0) > 0 and spec.lower() in d["name"].lower():
            return i
    raise FatalError(f"no input device matches --mic {spec!r}")


class MicListener:
    """Captures the microphone, segments speech (VAD or push-to-talk), transcribes."""

    def __init__(self, bridge: Bridge, asr: AsrBackend, args: argparse.Namespace) -> None:
        self.bridge = bridge
        self.asr = asr
        self.args = args
        self.lang = None if args.lang == "auto" else args.lang
        self.frames: queue.Queue[bytes] = queue.Queue(maxsize=1000)
        self.worker = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self._ptt_toggle = threading.Event()

    def start(self) -> None:
        threading.Thread(target=self._run, name="mic", daemon=True).start()

    def _gated(self) -> bool:
        return not self.args.full_duplex and self.bridge.mic_blocked()

    def _speech_started(self) -> None:
        self.bridge.emit_threadsafe("RECOG_EVENT_START", "")
        self.asr.prepare()                   # an idle REST connection re-opens meanwhile
        if self.bridge.tts is not None:
            self.bridge.tts.prepare()        # the reply will need it next

    def _open(self, sd):
        device = find_mic(sd, self.args.mic)
        block = MIC_RATE * MIC_FRAME_MS // 1000
        pending = bytearray()

        def callback(indata, frames, time_info, status):     # PortAudio thread
            with contextlib.suppress(queue.Full):
                self.frames.put_nowait(bytes(indata))

        try:
            stream = sd.RawInputStream(samplerate=MIC_RATE, blocksize=block, channels=1,
                                       dtype="int16", device=device, callback=callback)
            return stream, None
        except Exception:
            # some host APIs (e.g. WASAPI shared) only open at the native rate
            info = sd.query_devices(device, "input")
            rate = int(info["default_samplerate"])
            if rate == MIC_RATE:
                raise

            def resampling_callback(indata, frames, time_info, status):
                x = np.frombuffer(bytes(indata), dtype=np.int16).astype(np.float32)
                n = round(len(x) * MIC_RATE / rate)
                y = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x)
                pending.extend(y.astype("<i2").tobytes())
                while len(pending) >= block * 2:
                    with contextlib.suppress(queue.Full):
                        self.frames.put_nowait(bytes(pending[: block * 2]))
                    del pending[: block * 2]

            stream = sd.RawInputStream(samplerate=rate, blocksize=rate * MIC_FRAME_MS // 1000,
                                       channels=1, dtype="int16", device=device,
                                       callback=resampling_callback)
            return stream, rate

    def _run(self) -> None:
        try:
            sd = import_sounddevice()
            stream, native = self._open(sd)
            name = sd.query_devices(stream.device)["name"]
        except Exception as e:
            log.error("[asr] microphone unavailable: %s. Speech recognition is off; "
                      "see --list-mics / --mic, or use --asr keyboard.", e)
            return
        mode = "push-to-talk" if self.args.ptt else "voice activity detection"
        log.info("[asr] listening on '%s'%s (%s)", name,
                 f" at {native} Hz, resampled" if native else "", mode)
        with stream:
            if self.args.ptt:
                self._ptt_loop()
            else:
                self._vad_loop()

    def _vad_loop(self) -> None:
        vad = EnergyVad(threshold_dbfs=self.args.vad_threshold, silence_ms=self.args.silence_ms)
        announced = False
        while True:
            frame = self.frames.get()
            for ev in vad.process(frame, gated=self._gated()):
                if ev.kind == "start":
                    log.debug("[asr] speech started")
                    self._speech_started()
                elif ev.kind == "end":
                    self.worker.submit(self._transcribe, ev.audio)
                else:
                    log.debug("[asr] segment dropped (%s)", ev.reason)
            if not announced and vad.state == "idle":
                announced = True
                log.info("[asr] speech threshold %.0f dBFS (noise floor %s)", vad.threshold_dbfs,
                         "fixed" if vad.floor_db is None else f"{vad.floor_db:.0f} dBFS")

    def _ptt_loop(self) -> None:
        threading.Thread(target=self._ptt_keys, name="ptt", daemon=True).start()
        recording, buf = False, bytearray()
        while True:
            frame = self.frames.get()
            if self._ptt_toggle.is_set():
                self._ptt_toggle.clear()
                recording = not recording
                if recording:
                    buf.clear()
                    self._speech_started()
                    print("[asr] recording... press Enter to stop", flush=True)
                else:
                    if len(buf) >= MIC_RATE * 2 * 3 // 10:                  # >= 300 ms
                        self.worker.submit(self._transcribe, bytes(buf))
                    else:
                        log.info("[asr] too short, ignored")
                    print("[asr] press Enter to talk", flush=True)
            if recording and not self._gated():
                buf.extend(frame)
                if len(buf) >= MIC_RATE * 2 * 20:                           # 20 s cut
                    self.worker.submit(self._transcribe, bytes(buf))
                    buf.clear()

    def _ptt_keys(self) -> None:
        print("[asr] press Enter to talk", flush=True)
        for _ in sys.stdin:
            self._ptt_toggle.set()

    def _transcribe(self, pcm: bytes) -> None:
        t0 = time.monotonic()
        try:
            text = sanitize_recog_text(self.asr.transcribe(pcm_to_wav(pcm), self.lang))
        except Exception as e:
            if is_key_error(e):
                log.error("[asr] the OpenAI API rejected the key: %s", KEY_HINT)
            else:
                log.error("[asr] transcription failed: %s: %s", type(e).__name__, e)
            return
        if not text:
            log.debug("[asr] (nothing recognized)")
            return
        log.info("[asr] %s  (%.1f s audio, %.1f s)", text, len(pcm) / 2 / MIC_RATE,
                 time.monotonic() - t0)
        self.bridge.emit_threadsafe("RECOG_EVENT_STOP", text)


def keyboard_loop(bridge: Bridge) -> None:
    """--asr keyboard: every typed line becomes RECOG_EVENT_STOP|line."""
    print("[asr] type a sentence and press Enter to send it as RECOG_EVENT_STOP", flush=True)
    for line in sys.stdin:
        text = sanitize_recog_text(line)
        if text:
            log.info("[asr] %s", text)
            bridge.emit_threadsafe("RECOG_EVENT_STOP", text)


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="speech_bridge.py",
        description="External text-to-speech and speech recognition for MMDAgent-EX "
                    "(via Plugin_API).")
    g = p.add_argument_group("engine")
    g.add_argument("--host", default=os.environ.get("MMDAGENT_HOST", "localhost"),
                   help="engine host (env MMDAGENT_HOST or .env; default localhost)")
    g.add_argument("--rest-port", type=int, default=DEFAULT_REST_PORT,
                   help="Plugin_API REST port (default 48000)")
    g.add_argument("--ws-port", type=int, default=None,
                   help="Plugin_API WebSocket port (default REST port + 1)")
    g.add_argument("--mmdagent-key", default=os.environ.get("MMDAGENT_API_KEY") or None,
                   help="engine Plugin_API_Key, if set (env MMDAGENT_API_KEY or .env)")
    g = p.add_argument_group("backends")
    g.add_argument("--tts", choices=[*TTS_BACKENDS, "none"], default="openai",
                   help="speech synthesis backend (default openai)")
    g.add_argument("--asr", choices=[*ASR_BACKENDS, "none"], default="openai",
                   help="speech recognition backend (default openai)")
    g.add_argument("--lang", choices=["ja", "en", "auto"], default="ja",
                   help="recognition language hint (default ja; auto = no hint)")
    g.add_argument("--voice", default=DEFAULT_VOICE,
                   help=f"TTS voice when a script's voice is unknown (default {DEFAULT_VOICE})")
    g.add_argument("--instructions", default=None, help="tone-of-voice prompt for the TTS model")
    g.add_argument("--prebuffer-ms", type=int, default=DEFAULT_PREBUFFER_MS, metavar="MS",
                   help="audio collected before an utterance starts playing, against dropouts "
                        f"(default {DEFAULT_PREBUFFER_MS}; 0 = off)")
    g.add_argument("--tts-model", default=None, help=f"default {DEFAULT_TTS_MODEL}")
    g.add_argument("--asr-model", default=None, help=f"default {DEFAULT_ASR_MODEL}")
    g.add_argument("--openai-key", default=os.environ.get("OPENAI_API_KEY") or None,
                   help="OpenAI API key (env OPENAI_API_KEY, or a .env file)")
    g = p.add_argument_group("microphone")
    g.add_argument("--mic", default=None, help="input device index or name substring")
    g.add_argument("--list-mics", action="store_true", help="list input devices and exit")
    g.add_argument("--ptt", action="store_true",
                   help="push-to-talk: Enter starts recording, Enter again stops it")
    g.add_argument("--vad-threshold", type=float, default=None, metavar="DBFS",
                   help="fixed speech threshold in dBFS, e.g. -40 (default: adaptive)")
    g.add_argument("--silence-ms", type=int, default=700,
                   help="silence that ends an utterance (default 700)")
    g.add_argument("--full-duplex", action="store_true",
                   help="keep listening while the character speaks (headphones / barge-in)")
    p.add_argument("-v", "--verbose", action="store_true", help="log all message traffic")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return p


def print_banner(args: argparse.Namespace, tts: TtsBackend | None, asr: AsrBackend | None) -> None:
    key = "set" if args.mmdagent_key else "none"
    asr_desc = asr.describe() if asr else "off"
    if asr and asr.needs_microphone:
        asr_desc += (", push-to-talk" if args.ptt else ", VAD") + \
                    (", full duplex" if args.full_duplex else ", half duplex")
    print(f"mmdagent-ex-speech-bridge {__version__}\n"
          f"  engine : http://{args.host}:{args.rest_port}  ws://{args.host}:{args.ws_port}"
          f"  (API key: {key})\n"
          f"  tts    : {tts.describe() if tts else 'off'}\n"
          f"  asr    : {asr_desc}"
          + (f"\n  mic    : {args.mic or 'default'}" if asr and asr.needs_microphone else ""),
          flush=True)
    if tts:
        print("Try: start MMDAgent-EX with Plugin_API and let a script send "
              "SYNTH_START|<model>|<voice>|<text>" + (" — or just speak." if asr else "."),
              flush=True)
    elif asr:
        print("Try: speak (or type) and watch RECOG_EVENT_STOP arrive in MMDAgent-EX.", flush=True)


def _sigterm(signum, frame) -> None:
    raise KeyboardInterrupt        # same clean shutdown as Ctrl-C (the stream is deleted)


def main(argv: list[str] | None = None) -> int:
    for s in (sys.stdout, sys.stderr):
        with contextlib.suppress(Exception):
            s.reconfigure(errors="replace")
    with contextlib.suppress(ValueError, OSError, AttributeError):   # not in the main thread
        signal.signal(signal.SIGTERM, _sigterm)
    from_dotenv = load_dotenv(dotenv_paths(), os.environ)    # before the defaults are read
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.ws_port is None:
        args.ws_port = args.rest_port + 1
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    for noisy in ("asyncio", "websockets", "httpx", "httpx2", "httpcore", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    for key, path in from_dotenv:
        log.info("[config] %s read from %s", key, path)
    try:
        if args.list_mics:
            return list_mics()
        if args.ptt and args.asr == "keyboard":
            parser.error("--ptt and --asr keyboard both read the terminal; use only one")
        if args.tts == "none" and args.asr == "none":
            parser.error("both --tts and --asr are 'none'; nothing to do")
        if "openai" in (args.tts, args.asr) and not args.openai_key:
            raise FatalError(f"no OpenAI API key: {KEY_HINT}")
        tts = TTS_BACKENDS[args.tts](args) if args.tts != "none" else None
        asr = ASR_BACKENDS[args.asr](args) if args.asr != "none" else None
        if tts and tts.voices and args.voice.lower() not in [v.lower() for v in tts.voices]:
            log.warning("[tts] --voice %s is not a built-in %s voice; passing it through",
                        args.voice, args.tts)
        bridge = Bridge(args, tts)
        print_banner(args, tts, asr)
        for backend in (tts, asr):
            if backend is not None:
                backend.start()
        if asr is not None:
            def start_input() -> None:
                if asr.needs_microphone:
                    MicListener(bridge, asr, args).start()
                else:
                    threading.Thread(target=keyboard_loop, args=(bridge,), daemon=True).start()
            bridge.on_connected = start_input
        asyncio.run(bridge.run())
    except FatalError as e:
        log.error("error: %s", e)
        return 2
    except KeyboardInterrupt:
        print("bye", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
