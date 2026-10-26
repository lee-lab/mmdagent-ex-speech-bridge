# mmdagent-ex-speech-bridge

Single-file Python bridge that gives MMDAgent-EX external TTS and ASR through
Plugin_API (REST + WebSocket). The public MMDAgent-EX tutorial links here, so
the CLI flag names are a public contract: do not rename or remove flags.

## Layout

- `speech_bridge.py` — everything (PEP 723 header; `uv run speech_bridge.py`).
  Sections: helpers (pure, unit-tested) / backends + registries / `EnergyVad` /
  `EngineRest` / `Bridge` (asyncio: WS, playback stream, utterances) /
  microphone + keyboard threads / CLI.
- `tests/test_unit.py` — no network, no hardware, no engine.
- `tests/test_e2e_engine.py` — `requires_engine`, opt-in.
- `tests/test_live_openai.py` — `requires_openai`, opt-in, billed (about $0.05):
  verbatim check of `--tts openai-realtime`, `--asr openai-live` round trip,
  latency table. Run with `-s` to see the tables.

## Tests

```
uv venv && uv pip install -r requirements.txt pytest
.venv/bin/python -m pytest                       # unit tests (engine tests deselected)
MMDAGENT_HOST=127.0.0.1 MMDAGENT_REST_PORT=48000 \
  .venv/bin/python -m pytest -m requires_engine  # needs a running engine
.venv/bin/python -m pytest -m requires_openai -s # OPENAI_API_KEY from env or .env
```

The engine test speaks a beep on that engine and creates / deletes one
audio_playback stream: point it at your own instance (a second instance can
run with `Plugin_API_Port=48100` in its .mdf; WS is REST port + 1).

## Conventions

- Python >= 3.10 at runtime (CI: 3.10 and 3.13 on Linux / Windows / macOS).
- Minimum dependency versions live in two places: the PEP 723 block and
  `requirements.txt`. Keep them identical.
- Logs are English, one line per event at default verbosity; `-v` shows traffic.
- Never log, print or commit an API key (`.env` is gitignored; log only which
  file a key came from).
- Realtime backends (`RealtimeSession`) run their own asyncio loop on a
  thread; their state is touched only on that thread (`call()` from outside).
  OpenAI facts the code relies on (event names, 24 kHz-only PCM, out-of-band
  responses not inheriting the session's `reasoning`, shutdown dates) were
  checked on developers.openai.com on 2026-10-26; recheck before changing them.
- Engine protocol references (in the MMDAgent-EX repository): `dev/plugin-api.md`,
  `dev/plugin-api-audio-playback.md`, `dev/plugin-api-lipsync.md` §8.
- When behaviour or flags change, update both `README.md` and `README.ja.md`.
- Commit messages in English.
