# mmdagent-ex-speech-bridge

[日本語](README.ja.md)

Speech for [MMDAgent-EX](https://mmdagent-ex.dev/): a single Python file that
gives the engine **text-to-speech** and **speech recognition** from an external
service (OpenAI by default).

MMDAgent-EX no longer bundles a speech synthesizer or recognizer. Its dialogue
scripts (`.fst` and Lua) still speak with `SYNTH_START` and listen for
`RECOG_EVENT_STOP`; this bridge connects to the engine's Plugin_API and answers
those messages. The character's mouth moves with the voice (lip sync is driven
by the engine from the audio it plays).

- One file, `uv run speech_bridge.py`, Python 3.10 or later
- Windows, macOS, Linux; Japanese and English
- Offline wiring check without any API key (`--tts tone --asr keyboard`)
- Pluggable backends

## Quick start

1. **Get an OpenAI API key** at <https://platform.openai.com/api-keys> and put
   it in the environment:

   ```sh
   export OPENAI_API_KEY=sk-...            # macOS / Linux
   $env:OPENAI_API_KEY = "sk-..."          # Windows PowerShell
   ```

   or in a file named `.env` in the directory you start the bridge from, or
   next to `speech_bridge.py`:

   ```
   OPENAI_API_KEY=sk-...
   ```

   The bridge reads `OPENAI_API_KEY`, `MMDAGENT_API_KEY` and `MMDAGENT_HOST`
   from `.env` when the environment does not set them (simple `KEY=value`
   lines; `#` comments and quotes are allowed). It logs which file a key came
   from, never the key. Keep `.env` out of version control.

2. **Run the bridge.** With [uv](https://docs.astral.sh/uv/) (it installs the
   dependencies by itself):

   ```sh
   uv run speech_bridge.py
   ```

   or with plain pip:

   ```sh
   pip install -r requirements.txt
   python speech_bridge.py
   ```

   You only need `speech_bridge.py` (and `requirements.txt` for pip): clone this
   repository or download the file from
   `https://raw.githubusercontent.com/lee-lab/mmdagent-ex-speech-bridge/main/speech_bridge.py`.

3. **Start MMDAgent-EX** with your content. Plugin_API is part of the standard
   build and listens on ports 48000 (REST) and 48001 (WebSocket); the bridge
   finds it there. The order does not matter: the bridge may be started before
   the engine, and speech requested in the first seconds after the engine
   starts (a startup greeting) is picked up. The bridge reconnects when the
   engine restarts.

When a script sends `SYNTH_START`, the character speaks; when you talk into the
microphone, the script receives `RECOG_EVENT_STOP|<what you said>`. A session
looks like this:

```
14:02:11 [engine] connected to http://localhost:48000/api/v1 (windows 1.0.0)
14:02:11 [asr] listening on 'Microphone (USB Audio)' (voice activity detection)
14:02:15 [asr] こんにちは  (1.3 s audio, 0.6 s)
14:02:17 [tts] gene: "こんにちは！今日はどうしたの？" → 2.4 s (first audio after 0.48 s, end by drained)
```

## Check the wiring without a key

```sh
uv run speech_bridge.py --tts tone --asr keyboard
```

- `--tts tone` answers every `SYNTH_START` with a beep whose length follows the
  text (about 60 ms per character, 0.5 to 6 s), so you can see the
  `SYNTH_EVENT_START` / `SYNTH_EVENT_STOP` timing and the mouth moving.
- `--asr keyboard` sends each line you type in the terminal as
  `RECOG_EVENT_STOP|<line>`.

## Backends and models

| Option | Model | OpenAI API | Price (2026-10) | Status |
|---|---|---|---|---|
| `--tts openai` (default) | `gpt-4o-mini-tts` | Speech (`/v1/audio/speech`, HTTPS) | $0.60 / 1M text tokens in, $12 / 1M audio tokens out | **Deprecated: shuts down 2027-01-06** (also `tts-1`, `tts-1-hd`) |
| `--tts openai-realtime` | `gpt-realtime-2.1-mini` | Realtime (WebSocket) | $0.60 / 1M text tokens in, $20 / 1M audio tokens out | OpenAI's named successor of the Speech API models |
| `--asr openai` (default) | `gpt-transcribe` | Transcription (`/v1/audio/transcriptions`, HTTPS) | $0.0045 / min of audio | Current |
| `--asr openai-live` | `gpt-live-transcribe` | Realtime transcription (WebSocket) | $0.017 / min of audio sent | Current |

- **`--tts openai-realtime`** uses the Realtime voice model as a read-aloud
  engine: each utterance is a separate request whose only input is the text,
  under instructions to read it verbatim, once, in its own language, without
  answering or translating it. In our tests it read Japanese and English
  lines word for word, including numbers and questions such as 「今日の天気は？」;
  `-v` logs the model's own transcript of what it said so you can check.
  `--tts openai` stays the default until you have compared the two voices;
  switch before 2027-01-06.
- **`--asr openai-live`** transcribes while you speak, so the text is ready a
  little sooner after you stop (see below). The bridge's own voice activity detection still
  decides what is sent: nothing is sent until an utterance has 300 ms of
  speech, silence between utterances is never sent, and the pause that ends
  an utterance is not sent beyond a 300 ms tail. It sends the same audio
  that `--asr openai` would upload, but costs about 3.8 times as much per
  minute.
- The transcription models `gpt-4o-mini-transcribe`, `gpt-4o-transcribe` and
  `whisper-1` (selectable with `--asr-model`) shut down on 2027-02-26.
- Both REST backends keep their HTTPS connection open between requests (one
  connection each, opened at startup), and the Realtime backends keep one
  WebSocket open for the whole run, reconnecting by themselves.

### Faster turns

Measured from one machine in Japan on 2026-10-26 (the range of the medians of
three sessions of 3–5 runs each; it varies with the network and OpenAI's load):

| Step | `openai` | `openai-realtime` / `openai-live` |
|---|---|---|
| TTS: request → first audio | 0.6–0.9 s | 0.5–0.85 s |
| ASR: commit → text, 2 s utterance | 0.6–0.65 s | 0.5 s |
| ASR: commit → text, 6 s utterance | 0.65–0.75 s | 0.6–0.65 s |

The Realtime backends gain about 0.1 s per step: most of what remains is
OpenAI's fixed time to the first audio or to the final transcript. The VAD
adds `--silence-ms` before an utterance ends, and the TTS adds
`--prebuffer-ms` before playback starts. On a stable network,
`--prebuffer-ms 150 --silence-ms 500` saves about 0.35 s per turn; raise
them again if speech drops out at its start or if you get cut off in a pause.

## Options

| Option | Default | Meaning |
|---|---|---|
| `--host HOST` | `localhost` (env / `.env` `MMDAGENT_HOST`) | Machine running MMDAgent-EX |
| `--rest-port N` | `48000` | Plugin_API REST port (`Plugin_API_Port`) |
| `--ws-port N` | REST port + 1 | Plugin_API WebSocket port |
| `--mmdagent-key KEY` | env / `.env` `MMDAGENT_API_KEY` | The engine's `Plugin_API_Key`, when it has one |
| `--tts openai\|openai-realtime\|tone\|none` | `openai` | Speech synthesis backend (see [Backends and models](#backends-and-models); `none` = do not answer `SYNTH_START`) |
| `--asr openai\|openai-live\|keyboard\|none` | `openai` | Speech recognition backend (`none` = no microphone) |
| `--lang ja\|en\|auto` | `ja` | Recognition language hint (`auto` = let the model detect it). Synthesis speaks whatever language the text is in. |
| `--voice NAME` | `marin` | TTS voice used when the script's voice name is not one of the backend's voices |
| `--instructions TEXT` | none | Tone-of-voice prompt for the TTS model, e.g. `"Speak cheerfully and a little fast."` (`openai-realtime` adds it to its read-aloud instructions) |
| `--prebuffer-ms MS` | `300` | Audio collected before an utterance starts playing, so playback does not run dry right after it starts (`0` = off; short utterances start when fully synthesized). `150` is enough on a stable network |
| `--tts-model NAME` | `gpt-4o-mini-tts` / `gpt-realtime-2.1-mini` | OpenAI speech model of `--tts openai` / `openai-realtime` |
| `--asr-model NAME` | `gpt-transcribe` / `gpt-live-transcribe` | OpenAI transcription model of `--asr openai` / `openai-live` |
| `--openai-key KEY` | env / `.env` `OPENAI_API_KEY` | OpenAI API key |
| `--mic DEVICE` | system default | Input device, by index or part of its name |
| `--list-mics` | | List input devices and exit |
| `--ptt` | off | Push-to-talk: Enter starts recording, Enter again stops and sends it |
| `--vad-threshold DBFS` | adaptive | Fixed speech level, e.g. `-40`. By default the bridge measures the room noise for 1 s at start and treats sound 10 dB above it (and above -50 dBFS) as speech. |
| `--silence-ms MS` | `700` | Silence that ends an utterance (`500` answers sooner, but may cut you off in a pause) |
| `--full-duplex` | off | Keep listening while the character speaks (use with headphones; allows barge-in) |
| `-v`, `--verbose` | off | Log every message exchanged with the engine |
| `--version` | | Print the version |

### Voices

`voice` in `SYNTH_START|model|voice|text` is used when it names one of the
OpenAI voices (`alloy`, `ash`, `ballad`, `cedar`, `coral`, `echo`, `fable`,
`marin`, `nova`, `onyx`, `sage`, `shimmer`, `verse`). Any other name, such as
`mei_voice_normal` from older content, falls back to `--voice`; the bridge logs
this once per name.

The default is `marin` because OpenAI's text-to-speech guide recommends `marin`
and `cedar` for the best quality, and `marin`'s bright voice suits the female
characters most MMDAgent-EX content uses. OpenAI describes its voices as
optimized for English; they read Japanese text in Japanese, so listen to a few
and pick the one you like. Try others with `--voice` (`cedar` is the male
counterpart). `marin`, `cedar`, `ballad` and `verse` need `gpt-4o-mini-tts`;
with `--tts-model tts-1` pick e.g. `--voice nova`.

`--tts openai-realtime` has `alloy`, `ash`, `ballad`, `coral`, `echo`, `sage`,
`shimmer`, `verse`, `marin` and `cedar` (no `fable`, `nova` or `onyx`).

## How the bridge maps bus messages

| From the engine | What the bridge does | To the engine |
|---|---|---|
| `SYNTH_START\|model\|voice\|text` | Synthesizes `text` and streams it to an `audio_playback` stream bound to `model` | `SYNTH_EVENT_START\|model` when the first audio has been sent, `SYNTH_EVENT_STOP\|model` when playback has ended |
| `SYNTH_START` while that or another model is still speaking | Stops the current utterance first | `SYNTH_EVENT_STOP\|old model`, then as above |
| `SYNTH_STOP\|model` | Discards the rest of `model`'s utterance | `SYNTH_EVENT_STOP\|model` |
| (you start speaking) | Starts collecting audio | `RECOG_EVENT_START` |
| (you stop speaking) | Transcribes the utterance | `RECOG_EVENT_STOP\|text` |

- Everything after the second `|` of `SYNTH_START` is the text.
- In recognized text, `|` and line breaks become spaces; empty results are not sent.
- Requests sent while the bridge was not yet connected (the engine's startup
  greeting, or a bridge restart) are answered after connecting: the bridge reads
  the engine's event history of the last 5 s and replays the newest
  `SYNTH_START` per model that nobody stopped or answered.
- If synthesis fails (network, key), the bridge logs why and still sends
  `SYNTH_EVENT_START` followed by `SYNTH_EVENT_STOP`, so scripts never hang.
- **Half duplex**: from the first audio of an utterance until 0.4 s after it
  ends, microphone input is discarded and a half-captured utterance is dropped,
  so the character does not hear itself. `--full-duplex` turns this off.
- Audio goes to the engine as 24 kHz mono PCM over one playback stream that
  the bridge creates when it connects and deletes when it exits. The stream is
  bound to the speaking model's alias, which is what makes the engine move that
  model's mouth.
- Completion: the bridge marks the end of each utterance, and the engine reports
  when it has played everything (`drained`). Engines from before that feature
  are handled too: the bridge then assumes playback ends 1 s after the audio's
  length, and stops an utterance by re-creating the stream instead of flushing it.

## Adding a backend

A backend is a class plus one line in a registry; the `--tts` / `--asr` choices
follow the registries.

```python
class MyTts(TtsBackend):
    sample_rate = 24000                  # what synthesize() yields
    voices = ["alice", "bob"]            # names a script may pass as `voice`

    def synthesize(self, text: str, voice: str) -> Iterator[bytes]:
        for chunk in my_engine.stream(text, voice):   # mono PCM16LE, any chunk size
            yield chunk

class MyAsr(AsrBackend):
    def transcribe(self, wav: bytes, lang: str | None) -> str:   # 16 kHz mono WAV
        return my_recognizer(wav, language=lang)

TTS_BACKENDS["mytts"] = MyTts
ASR_BACKENDS["myasr"] = MyAsr
```

Yield audio as soon as it is produced: scripts give up waiting for
`SYNTH_EVENT_START` after about 5 s. Backends read their options from
`self.args` (the parsed command line); add flags in `build_parser()` if needed.

Optional hooks: `start()` runs once at startup (open a connection; must not
block), `prepare()` when the user starts speaking (a request will follow),
and a TTS backend that can stop early overrides
`synthesize_cancellable(text, voice, cancel)`, whose `threading.Event` is set
when the utterance is stopped or replaced. A recognizer that takes audio
while the user speaks derives from `StreamingAsrBackend` and implements
`append()`, `commit()` (returns a future with the text) and `clear()`;
`OpenAILiveAsr` is the example.

## Troubleshooting

- **"waiting for MMDAgent-EX"** — the engine is not running or not reachable at
  `--host` / `--rest-port`. Check `http://<host>:48000/api/v1/status` in a browser.
- **401 from the engine** — the engine's `.mdf` sets `Plugin_API_Key`; pass the
  same value with `--mmdagent-key` (or `MMDAGENT_API_KEY`).
- **The engine runs on another PC (Windows)** — allow MMDAgent-EX through
  Windows Defender Firewall for private networks (the prompt appears on its
  first start), so that ports 48000 and 48001 are reachable. Nothing is needed
  when both run on the same PC.
- **A dropout (or a short burst, then a gap) right after speech starts** — the
  engine ran out of audio before the next streamed chunk arrived. Raise
  `--prebuffer-ms` (e.g. `600`); this delays the start of speech by up to that
  amount. WSL/WSLg audio also adds start-up jitter.
- **"cannot create the audio_playback stream"** — the engine's machine has no
  audio output device, or four streams are already open (the engine's limit).
- **"lip sync for 'x' is unavailable ... (409)"** — another playback stream
  already drives that model's mouth: a second bridge, or a stream left behind
  by a bridge that was killed. The voice still plays. Close the other bridge,
  or delete the stream (`DELETE /api/v1/streams/<id>`, the id is in the message)
  or restart the engine.
- **No microphone / "microphone unavailable"** — run `--list-mics` and pick one
  with `--mic`. On macOS, allow microphone access for your terminal app
  (System Settings > Privacy & Security > Microphone); on Windows, enable
  "Let desktop apps access your microphone"; on Linux, install `libportaudio2`.
  Until then you can use `--asr keyboard` or `--ptt`.
- **It reacts to noise / misses quiet speech** — set the level yourself with
  `--vad-threshold` (e.g. `-45` for a quiet voice, `-30` in a noisy room; `-v`
  shows the measured noise floor), or use `--ptt`.
- **The character answers itself (echo)** — keep the default half duplex. With
  `--full-duplex`, use headphones so the microphone does not pick up the voice.
- **It cuts my first word after the character spoke** — the microphone opens
  0.4 s after the character's voice ends; wait a moment, or use headphones and
  `--full-duplex`.
- **"OpenAI realtime connection failed ... retrying"** (`openai-realtime`,
  `openai-live`) — the WebSocket to `wss://api.openai.com` could not be opened;
  check the network or a proxy that blocks WebSockets. The bridge keeps
  retrying, and an utterance requested meanwhile fails after 4 s (the script
  still gets `SYNTH_EVENT_START` / `SYNTH_EVENT_STOP`).

## Cost and privacy

The OpenAI backends are paid services: synthesis is billed by the amount of
text and audio, recognition by the length of audio sent (prices above; see
<https://openai.com/api/pricing/> for the current ones). With voice activity
detection every detected sound longer than 300 ms is sent for transcription,
so a noisy room costs more; `--ptt` sends only what you choose. `openai-live`
streams those same utterances while you speak and nothing in between.
Audio and text are sent to OpenAI and handled under its terms.

## Development

```sh
uv venv && uv pip install -r requirements.txt pytest
.venv/bin/python -m pytest                     # unit tests: no engine, network or audio device
.venv/bin/python -m pytest -m requires_engine  # against a running engine (beeps once)
.venv/bin/python -m pytest -m requires_openai -s   # against OpenAI (billed, about $0.05)
```

`MMDAGENT_HOST` and `MMDAGENT_REST_PORT` select the engine for the second
command. The third needs `OPENAI_API_KEY` (or `.env`) and prints a
verbatim check of `openai-realtime`, a round trip through `openai-live`
and a latency table. The protocol the bridge speaks is described in the
MMDAgent-EX documentation of Plugin_API (`/api/v1/streams`, `audio_playback`).

## License

Apache License 2.0. Copyright 2026 Lee Laboratory, Nagoya Institute of
Technology. See [LICENSE](LICENSE).
