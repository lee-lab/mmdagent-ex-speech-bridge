# mmdagent-ex-speech-bridge

[English](README.md)

[MMDAgent-EX](https://mmdagent-ex.dev/) に外部サービス（既定は OpenAI）の
**音声合成**と**音声認識**をつなぐ、Python ファイル 1 本のブリッジです。

MMDAgent-EX は音声合成器・音声認識器を同梱しなくなりました。対話スクリプト
（`.fst` と Lua）は今までどおり `SYNTH_START` で話し、`RECOG_EVENT_STOP` を待って
聞きます。このブリッジはエンジンの Plugin_API に接続して、これらのメッセージに
応答します。キャラクターの口は声に合わせて動きます（再生する音声からエンジンが
リップシンクを駆動します）。

- ファイル 1 本、`uv run speech_bridge.py`、Python 3.10 以降
- Windows / macOS / Linux、日本語と英語
- API キーなしで配線を確認できるオフライン動作（`--tts tone --asr keyboard`）
- バックエンドは差し替え可能

## クイックスタート

1. **OpenAI の API キーを取得**して（<https://platform.openai.com/api-keys>）、
   環境変数に設定します。

   ```sh
   export OPENAI_API_KEY=sk-...            # macOS / Linux
   $env:OPENAI_API_KEY = "sk-..."          # Windows PowerShell
   ```

2. **ブリッジを起動**します。[uv](https://docs.astral.sh/uv/) を使う場合
   （依存パッケージは uv が自動で入れます）:

   ```sh
   uv run speech_bridge.py
   ```

   pip を使う場合:

   ```sh
   pip install -r requirements.txt
   python speech_bridge.py
   ```

   必要なのは `speech_bridge.py`（pip の場合は `requirements.txt` も）だけです。
   このリポジトリを clone するか、
   `https://raw.githubusercontent.com/lee-lab/mmdagent-ex-speech-bridge/main/speech_bridge.py`
   からファイルをダウンロードしてください。

3. **MMDAgent-EX を起動**してコンテンツを開きます。Plugin_API は標準ビルドに
   含まれ、ポート 48000（REST）と 48001（WebSocket）で待ち受けます。ブリッジは
   そこに接続します。起動の順番は問いません。ブリッジはエンジンの起動を待ち、
   エンジンが再起動すれば再接続します。

スクリプトが `SYNTH_START` を送るとキャラクターが話し、マイクに向かって話すと
スクリプトに `RECOG_EVENT_STOP|<話した内容>` が届きます。動作中のログは次のように
なります。

```
14:02:11 [engine] connected to http://localhost:48000/api/v1 (windows 1.0.0)
14:02:11 [asr] listening on 'Microphone (USB Audio)' (voice activity detection)
14:02:15 [asr] こんにちは  (1.3 s audio, 0.6 s)
14:02:17 [tts] gene: "こんにちは！今日はどうしたの？" → 2.4 s (first audio after 0.48 s, end by drained)
```

## キーなしで配線を確認する

```sh
uv run speech_bridge.py --tts tone --asr keyboard
```

- `--tts tone` は `SYNTH_START` にビープ音で応答します。長さはテキストに比例します
  （1 文字あたり約 60 ms、0.5〜6 秒）。`SYNTH_EVENT_START` / `SYNTH_EVENT_STOP` の
  タイミングと口の動きを確認できます。
- `--asr keyboard` は端末に入力した各行を `RECOG_EVENT_STOP|<行>` として送ります。

## オプション

| オプション | 既定値 | 意味 |
|---|---|---|
| `--host HOST` | `localhost`（環境変数 `MMDAGENT_HOST`） | MMDAgent-EX が動いているマシン |
| `--rest-port N` | `48000` | Plugin_API の REST ポート（`Plugin_API_Port`） |
| `--ws-port N` | REST ポート + 1 | Plugin_API の WebSocket ポート |
| `--mmdagent-key KEY` | 環境変数 `MMDAGENT_API_KEY` | エンジンに `Plugin_API_Key` が設定されている場合のキー |
| `--tts openai\|tone\|none` | `openai` | 音声合成バックエンド（`none` = `SYNTH_START` に応答しない） |
| `--asr openai\|keyboard\|none` | `openai` | 音声認識バックエンド（`none` = マイクを使わない） |
| `--lang ja\|en\|auto` | `ja` | 認識の言語ヒント（`auto` = モデルが判定）。合成はテキストの言語でそのまま話します |
| `--voice NAME` | `marin` | スクリプトの声の名前がバックエンドの声にないときに使う TTS の声 |
| `--instructions TEXT` | なし | TTS モデルへの話し方の指示。例: `"明るく、少し早口で話してください。"` |
| `--tts-model NAME` | `gpt-4o-mini-tts` | OpenAI の音声合成モデル |
| `--asr-model NAME` | `gpt-4o-mini-transcribe` | OpenAI の文字起こしモデル |
| `--openai-key KEY` | 環境変数 `OPENAI_API_KEY` | OpenAI の API キー |
| `--mic DEVICE` | システム既定 | 入力デバイス（番号または名前の一部） |
| `--list-mics` | | 入力デバイスを一覧表示して終了 |
| `--ptt` | オフ | プッシュトゥトーク: Enter で録音開始、もう一度 Enter で停止して送信 |
| `--vad-threshold DBFS` | 自動 | 発話とみなす音量を固定する（例 `-40`）。既定では起動直後の 1 秒で周囲の雑音を測り、それより 10 dB 大きく、かつ -50 dBFS を超える音を発話とみなします |
| `--silence-ms MS` | `700` | 発話の終わりとみなす無音の長さ |
| `--full-duplex` | オフ | キャラクターが話している間も聞き続ける（ヘッドホン使用時向け。割り込み発話ができる） |
| `-v`, `--verbose` | オフ | エンジンとやりとりする全メッセージをログに出す |
| `--version` | | バージョンを表示 |

### 声

`SYNTH_START|model|voice|text` の `voice` が OpenAI の声（`alloy`, `ash`, `ballad`,
`cedar`, `coral`, `echo`, `fable`, `marin`, `nova`, `onyx`, `sage`, `shimmer`,
`verse`）のどれかなら、その声を使います。それ以外の名前（古いコンテンツの
`mei_voice_normal` など）のときは `--voice` の声になり、名前ごとに 1 回ログに
出ます。

既定を `marin` にしたのは、OpenAI の音声合成ガイドが最高品質の声として `marin` と
`cedar` を推奨しており、`marin` の明るい声が MMDAgent-EX のコンテンツに多い女性
キャラクターに合うためです。OpenAI は各声を英語向けに最適化したものとしており、
日本語のテキストも日本語で読みます。いくつか聞き比べて選んでください。ほかの声は
`--voice` で試せます（`cedar` は男性の声）。`marin`・`cedar`・`ballad`・`verse` は
`gpt-4o-mini-tts` 専用です。`--tts-model tts-1` では `--voice nova` などを選んで
ください。

## メッセージの対応

| エンジンから | ブリッジの処理 | エンジンへ |
|---|---|---|
| `SYNTH_START\|model\|voice\|text` | `text` を合成し、`model` に束縛した `audio_playback` ストリームへ流す | 最初の音声を送った時点で `SYNTH_EVENT_START\|model`、再生が終わったら `SYNTH_EVENT_STOP\|model` |
| 発話中（同じモデルでも別のモデルでも）の `SYNTH_START` | 先に今の発話を止める | `SYNTH_EVENT_STOP\|前のモデル`、そのあと上と同じ |
| `SYNTH_STOP\|model` | `model` の発話の残りを破棄する | `SYNTH_EVENT_STOP\|model` |
| （話し始めた） | 音声の収集を始める | `RECOG_EVENT_START` |
| （話し終えた） | 発話を文字起こしする | `RECOG_EVENT_STOP\|text` |

- `SYNTH_START` の 2 つ目の `|` より後ろはすべてテキストとして扱います。
- 認識結果の `|` と改行は空白に置き換えます。結果が空なら送りません。
- 合成に失敗した場合（ネットワーク、キー）は理由をログに出したうえで
  `SYNTH_EVENT_START` と `SYNTH_EVENT_STOP` を続けて送るので、スクリプトが
  止まったままになることはありません。
- **半二重**: 発話の最初の音声から、終わって 0.4 秒後まではマイク入力を捨て、
  途中まで取り込んだ発話も破棄します。キャラクターが自分の声を聞き取らない
  ようにするためです。`--full-duplex` で無効にできます。
- 音声は 24 kHz モノラル PCM で、ブリッジが接続時に作り終了時に削除する再生
  ストリーム 1 本を通ってエンジンへ送られます。このストリームを話している
  モデルのエイリアスに束縛することで、エンジンがそのモデルの口を動かします。
- 再生完了の判定: ブリッジは各発話の終わりを通知し、エンジンはすべて再生し
  終えたことを知らせます（`drained`）。この機能より前のエンジンにも対応して
  おり、その場合は音声の長さ + 1 秒で再生が終わったとみなし、発話の停止は
  ストリームのフラッシュではなく作り直しで行います。

## バックエンドの追加

バックエンドはクラス 1 つと登録 1 行です。`--tts` / `--asr` の選択肢は登録表から
作られます。

```python
class MyTts(TtsBackend):
    sample_rate = 24000                  # synthesize() が返す音声のレート
    voices = ["alice", "bob"]            # スクリプトが voice に書ける名前

    def synthesize(self, text: str, voice: str) -> Iterator[bytes]:
        for chunk in my_engine.stream(text, voice):   # モノラル PCM16LE、チャンク長は任意
            yield chunk

class MyAsr(AsrBackend):
    def transcribe(self, wav: bytes, lang: str | None) -> str:   # 16 kHz モノラル WAV
        return my_recognizer(wav, language=lang)

TTS_BACKENDS["mytts"] = MyTts
ASR_BACKENDS["myasr"] = MyAsr
```

音声はできた端から yield してください。スクリプトは `SYNTH_EVENT_START` を
約 5 秒しか待ちません。バックエンドは設定を `self.args`（解析済みのコマンド
ライン）から読みます。必要なら `build_parser()` にオプションを追加してください。

## トラブルシューティング

- **"waiting for MMDAgent-EX" と出たまま** — エンジンが起動していないか、
  `--host` / `--rest-port` で到達できません。ブラウザで
  `http://<host>:48000/api/v1/status` を開いて確かめてください。
- **エンジンから 401** — エンジンの `.mdf` に `Plugin_API_Key` が設定されています。
  同じ値を `--mmdagent-key`（または `MMDAGENT_API_KEY`）で渡してください。
- **エンジンが別の PC で動いている（Windows）** — Windows Defender ファイアウォールで
  MMDAgent-EX をプライベートネットワークに許可し（初回起動時に確認が出ます）、
  ポート 48000 と 48001 に届くようにしてください。同じ PC で動かすなら設定は
  不要です。
- **"cannot create the audio_playback stream"** — エンジンのマシンに音声出力
  デバイスがないか、ストリームがすでに 4 本（エンジンの上限）開いています。
- **"lip sync for 'x' is unavailable ... (409)"** — そのモデルの口は別の再生
  ストリームが動かしています。2 つ目のブリッジか、強制終了されたブリッジが残した
  ストリームです。声は再生されます。もう一方のブリッジを閉じるか、ストリームを
  削除する（`DELETE /api/v1/streams/<id>`、id はメッセージに出ます）か、エンジンを
  再起動してください。
- **マイクがない / "microphone unavailable"** — `--list-mics` で一覧を見て `--mic` で
  選んでください。macOS ではターミナルアプリにマイクの使用を許可し（システム設定 >
  プライバシーとセキュリティ > マイク）、Windows では「デスクトップ アプリがマイクに
  アクセスできるようにする」を有効にし、Linux では `libportaudio2` を入れてください。
  それまでは `--asr keyboard` や `--ptt` を使えます。
- **雑音に反応する / 小さな声を拾わない** — `--vad-threshold` で音量を指定して
  ください（小さな声なら `-45`、騒がしい部屋なら `-30` など。`-v` で測定した雑音の
  レベルが出ます）。`--ptt` も使えます。
- **キャラクターが自分の声に返事をする（エコー）** — 既定の半二重のままにして
  ください。`--full-duplex` を使うなら、マイクが声を拾わないようにヘッドホンを
  使ってください。
- **キャラクターが話した直後の第一声が切れる** — マイクはキャラクターの声が
  終わって 0.4 秒後に開きます。少し間を置くか、ヘッドホンと `--full-duplex` を
  使ってください。

## 料金とプライバシー

`openai` バックエンドは有料サービスです。合成はテキストの量、認識は音声の長さで
課金されます。現在の料金は <https://openai.com/api/pricing/> を見てください。
発話検出（VAD）では検出した音をすべて文字起こしに送るので、騒がしい部屋ほど
費用がかかります。`--ptt` なら送るのは自分で選んだ区間だけです。音声とテキストは
OpenAI に送られ、同社の規約に従って扱われます。

## 開発

```sh
uv venv && uv pip install -r requirements.txt pytest
.venv/bin/python -m pytest                     # 単体テスト: エンジン・ネットワーク・音声デバイス不要
.venv/bin/python -m pytest -m requires_engine  # 動作中のエンジンに対して実行（ビープが 1 回鳴る）
```

2 つ目のコマンドの接続先は `MMDAGENT_HOST` と `MMDAGENT_REST_PORT` で指定します。
ブリッジが使うプロトコルは MMDAgent-EX の Plugin_API のドキュメント
（`/api/v1/streams`、`audio_playback`）に書かれています。

## ライセンス

Apache License 2.0。Copyright 2026 Lee Laboratory, Nagoya Institute of
Technology。[LICENSE](LICENSE) を参照してください。
