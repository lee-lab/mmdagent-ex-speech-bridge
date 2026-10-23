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

## Tests

```
uv venv && uv pip install -r requirements.txt pytest
.venv/bin/python -m pytest                       # unit tests (engine tests deselected)
MMDAGENT_HOST=127.0.0.1 MMDAGENT_REST_PORT=48000 \
  .venv/bin/python -m pytest -m requires_engine  # needs a running engine
```

The engine test speaks a beep on that engine and creates / deletes one
audio_playback stream: point it at your own instance (a second instance can
run with `Plugin_API_Port=48100` in its .mdf; WS is REST port + 1).

## Conventions

- Python >= 3.10 at runtime (CI: 3.10 and 3.13 on Linux / Windows / macOS).
- Minimum dependency versions live in two places: the PEP 723 block and
  `requirements.txt`. Keep them identical.
- Logs are English, one line per event at default verbosity; `-v` shows traffic.
- Engine protocol references (in the MMDAgent-EX repository): `dev/plugin-api.md`,
  `dev/plugin-api-audio-playback.md`, `dev/plugin-api-lipsync.md` §8.
- When behaviour or flags change, update both `README.md` and `README.ja.md`.
- Commit messages in English.
