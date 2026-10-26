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

   ブリッジを起動するディレクトリか `speech_bridge.py` と同じディレクトリに置いた
   `.env` ファイルに書いてもかまいません。

   ```
   OPENAI_API_KEY=sk-...
   ```

   ブリッジは環境変数に `OPENAI_API_KEY`・`MMDAGENT_API_KEY`・`MMDAGENT_HOST` が
   ないとき、`.env` から読みます（`KEY=value` の行。`#` のコメントと引用符も
   使えます）。どのファイルから読んだかはログに出しますが、キーの値は出しません。
   `.env` はバージョン管理に入れないでください。

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
   そこに接続します。起動の順番は問いません。ブリッジをエンジンより先に起動
   してもよく、エンジン起動直後の数秒間に要求された発話（起動時のあいさつ）も
   取りこぼしません。エンジンが再起動すれば再接続します。

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

## バックエンドとモデル

| オプション | モデル | OpenAI の API | 料金（2026-10） | 状態 |
|---|---|---|---|---|
| `--tts openai` | `gpt-4o-mini-tts` | Speech（`/v1/audio/speech`、HTTPS） | テキスト入力 100 万トークン $0.60、音声出力 100 万トークン $12 | **非推奨: 2027-01-06 に提供終了**（`tts-1`・`tts-1-hd` も） |
| `--tts openai-realtime`（既定） | `gpt-realtime-2.1-mini` | Realtime（WebSocket） | テキスト入力 100 万トークン $0.60、音声出力 100 万トークン $20 | OpenAI が Speech API のモデルの後継に指定 |
| `--asr openai`（既定） | `gpt-transcribe` | Transcription（`/v1/audio/transcriptions`、HTTPS） | 音声 1 分 $0.0045 | 現行 |
| `--asr openai-live` | `gpt-live-transcribe` | Realtime transcription（WebSocket） | 送った音声 1 分 $0.017 | 現行 |

- **`--tts openai-realtime`**（既定）は Realtime の音声モデルを読み上げエンジンとして使い
  ます。発話ごとにテキストだけを入力とする要求を出し、テキストをその言語のまま
  一字一句 1 回だけ読み、答えたり訳したりしないよう指示しています。私たちの試験
  では、数字や「今日の天気は？」のような問いかけも含め、日本語と英語の文を
  そのとおりに読みました。`-v` を付けるとモデル自身が話した内容の書き起こしが
  ログに出るので確かめられます。`--tts openai`（Speech API）も選べますが、そのモデルは
  2027-01-06 に提供終了します。
- **`--asr openai-live`** は話している間に文字起こしを進めるので、話し終えてから
  テキストが届くまでが少し短くなります（下の表）。何を送るかはブリッジ自身の発話検出（VAD）が
  決めます。発話が 300 ms に達するまでは何も送らず、発話と発話の間の無音は
  送らず、発話を終わらせる無音も 300 ms の余韻を超える分は送りません。送る音声は
  `--asr openai` がアップロードするものと同じですが、1 分あたりの料金は約 3.8 倍
  です。
- 文字起こしモデル `gpt-4o-mini-transcribe`・`gpt-4o-transcribe`・`whisper-1`
  （`--asr-model` で選べます）は 2027-02-26 に提供終了します。
- REST の 2 つのバックエンドは要求の間も HTTPS 接続を保ち（それぞれ 1 本、起動時に
  開く）、Realtime のバックエンドは実行中ずっと WebSocket を 1 本開いておき、
  切れれば自分で再接続します。

### 応答を速くする

2026-10-26 に日本国内の 1 台から測った値です（3〜5 回ずつ 3 回測った中央値の範囲。
ネットワークと OpenAI の混み具合で変わります）。

| 区間 | `openai`（REST） | `openai-realtime` / `openai-live` |
|---|---|---|
| TTS: 要求 → 最初の音声 | 0.6〜0.9 秒 | 0.5〜0.85 秒 |
| ASR: 確定 → テキスト、2 秒の発話 | 0.6〜0.65 秒 | 0.5 秒 |
| ASR: 確定 → テキスト、6 秒の発話 | 0.65〜0.75 秒 | 0.6〜0.65 秒 |

Realtime のバックエンドで縮むのは各区間 0.1 秒ほどです。残りの大半は、最初の音声
や最終的な書き起こしが返るまでに OpenAI 側でかかる一定の時間です。発話の終わりは
VAD が `--silence-ms` の無音を待ってから決まり、TTS は再生開始前に
`--prebuffer-ms` ぶん音声をためます。ネットワークが安定していれば
`--prebuffer-ms 150 --silence-ms 500` で 1 往復あたり約 0.35 秒縮みます。話し始めに
音が途切れたり、言葉の間で切られたりするようなら値を戻してください。

## オプション

| オプション | 既定値 | 意味 |
|---|---|---|
| `--host HOST` | `localhost`（環境変数 / `.env` の `MMDAGENT_HOST`） | MMDAgent-EX が動いているマシン |
| `--rest-port N` | `48000` | Plugin_API の REST ポート（`Plugin_API_Port`） |
| `--ws-port N` | REST ポート + 1 | Plugin_API の WebSocket ポート |
| `--mmdagent-key KEY` | 環境変数 / `.env` の `MMDAGENT_API_KEY` | エンジンに `Plugin_API_Key` が設定されている場合のキー |
| `--tts openai\|openai-realtime\|tone\|none` | `openai-realtime` | 音声合成バックエンド（[バックエンドとモデル](#バックエンドとモデル)を参照。`none` = `SYNTH_START` に応答しない） |
| `--asr openai\|openai-live\|keyboard\|none` | `openai` | 音声認識バックエンド（`none` = マイクを使わない） |
| `--lang ja\|en\|auto` | `ja` | 認識の言語ヒント（`auto` = モデルが判定）。合成はテキストの言語でそのまま話します |
| `--voice NAME` | `marin` | スクリプトの声の名前がバックエンドの声にないときに使う TTS の声 |
| `--instructions TEXT` | なし | TTS モデルへの話し方の指示。例: `"明るく、少し早口で話してください。"`（`openai-realtime` では読み上げの指示に付け足します） |
| `--prebuffer-ms MS` | `300` | 発話の再生を始める前にためておく音声の長さ。再生開始直後に音が途切れるのを防ぎます（`0` = 無効。短い発話は合成が終わった時点で再生を始めます）。ネットワークが安定していれば `150` で足ります |
| `--tts-model NAME` | `gpt-realtime-2.1-mini` / `gpt-4o-mini-tts` | `--tts openai-realtime` / `openai` の OpenAI 音声合成モデル |
| `--asr-model NAME` | `gpt-transcribe` / `gpt-live-transcribe` | `--asr openai` / `openai-live` の OpenAI 文字起こしモデル |
| `--openai-key KEY` | 環境変数 / `.env` の `OPENAI_API_KEY` | OpenAI の API キー |
| `--mic DEVICE` | システム既定 | 入力デバイス（番号または名前の一部） |
| `--list-mics` | | 入力デバイスを一覧表示して終了 |
| `--ptt` | オフ | プッシュトゥトーク: Enter で録音開始、もう一度 Enter で停止して送信 |
| `--vad-threshold DBFS` | 自動 | 発話とみなす音量を固定する（例 `-40`）。既定では起動直後の 1 秒で周囲の雑音を測り、それより 10 dB 大きく、かつ -50 dBFS を超える音を発話とみなします |
| `--silence-ms MS` | `700` | 発話の終わりとみなす無音の長さ（`500` にすると早く応答しますが、言葉の間で切られることがあります） |
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
`--voice` で試せます（`cedar` は男性の声）。`--tts openai` では `marin`・`cedar`・
`ballad`・`verse` は `gpt-4o-mini-tts` 専用です。`--tts-model tts-1` では `--voice nova` などを選んで
ください。

`--tts openai-realtime` の声は `alloy`・`ash`・`ballad`・`coral`・`echo`・`sage`・
`shimmer`・`verse`・`marin`・`cedar` です（`fable`・`nova`・`onyx` はありません）。

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
- ブリッジが接続する前に送られた要求（エンジン起動時のあいさつや、ブリッジの
  再起動中の要求）には接続後に応答します。エンジンのイベント履歴から直近 5 秒を
  読み、止められても応答されてもいない `SYNTH_START` をモデルごとに最新の 1 件
  だけ再生します。
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

任意のフック: `start()` は起動時に 1 回呼ばれ（接続を開く。ブロックしないこと）、
`prepare()` は利用者が話し始めたとき（このあと要求が来る）に呼ばれます。途中で
止められる TTS バックエンドは `synthesize_cancellable(text, voice, cancel)` を
上書きします。発話が止められたり置き換えられたりすると `cancel`
（`threading.Event`）がセットされます。話している間に音声を受け取る認識器は
`StreamingAsrBackend` を継承し、`append()`・`commit()`（テキストを返す future を
返す）・`clear()` を実装します。例は `OpenAILiveAsr` です。

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
- **話し始めた直後に音が途切れる（短く鳴ってから一瞬無音になる）** — 次の音声
  チャンクが届く前にエンジン側の音声が尽きています。`--prebuffer-ms` を上げて
  ください（例 `600`）。最大でそのぶん話し始めが遅れます。WSL/WSLg の音声出力も
  再生開始時の揺らぎを加えます。
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
- **"OpenAI realtime connection failed ... retrying"**（`openai-realtime`、
  `openai-live`）— `wss://api.openai.com` への WebSocket を開けません。ネットワーク
  や、WebSocket を通さないプロキシを確かめてください。ブリッジは再試行を続け、
  その間に要求された発話は 4 秒で失敗します（スクリプトには `SYNTH_EVENT_START` /
  `SYNTH_EVENT_STOP` が届きます）。

## 料金とプライバシー

OpenAI のバックエンドは有料サービスです。合成（既定は Realtime の
`gpt-realtime-2.1-mini`）はテキストと音声の量、認識は送った
音声の長さで課金されます（料金は上の表。最新の料金は
<https://openai.com/api/pricing/> を見てください）。発話検出（VAD）では 300 ms を
超えて検出した音をすべて文字起こしに送るので、騒がしい部屋ほど費用がかかります。
`--ptt` なら送るのは自分で選んだ区間だけです。`openai-live` は同じ発話を話して
いる間に送り、発話の間には何も送りません。音声とテキストは OpenAI に送られ、
同社の規約に従って扱われます。

## 開発

```sh
uv venv && uv pip install -r requirements.txt pytest
.venv/bin/python -m pytest                     # 単体テスト: エンジン・ネットワーク・音声デバイス不要
.venv/bin/python -m pytest -m requires_engine  # 動作中のエンジンに対して実行（ビープが 1 回鳴る）
.venv/bin/python -m pytest -m requires_openai -s   # OpenAI に対して実行（課金。約 $0.05）
```

2 つ目のコマンドの接続先は `MMDAGENT_HOST` と `MMDAGENT_REST_PORT` で指定します。
3 つ目は `OPENAI_API_KEY`（または `.env`）が必要で、`openai-realtime` がそのとおりに
読むかの確認、`openai-live` の往復試験、レイテンシの表を出力します。
ブリッジが使うプロトコルは MMDAgent-EX の Plugin_API のドキュメント
（`/api/v1/streams`、`audio_playback`）に書かれています。

## ライセンス

Apache License 2.0。Copyright 2026 Lee Laboratory, Nagoya Institute of
Technology。[LICENSE](LICENSE) を参照してください。
