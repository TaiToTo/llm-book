# dai-2-kai — 第2回 NLP勉強会デモ

NLP勉強会 第2回の発表用デモスクリプト集です。  
Amazon Reviews 2023 データセットを使い、HuggingFace推論・埋め込み・API呼び出しの効率化テクニックを実演します。

## セットアップ

```bash
cd dai-2-kai
uv sync
```

## ファイル構成

| ファイル | 内容 |
|---|---|
| `dai-2-kai.py` | メインランナー（全セクション実行） |
| `util.py` | 共通ユーティリティ（データ読み込み・タイマー等） |
| `section_intro.py` | §1,3,4 イントロ・NLP振り返り・効率化の概念説明 |
| `section5_hf_inference.py` | §5 HuggingFace推論（逐次 vs バッチ） |
| `section6_embedding.py` | §6 埋め込み（逐次 vs バッチ） |
| `section7_api_inference.py` | §7 API推論（逐次 vs asyncio） |
| `section8_batch_api.py` | §8 Batch API の概念と JSONL 例 |
| `section9_summary.py` | §9 まとめ |
| `chapter-1-introduction.py` | 第1回の復習用スクリプト |
| `download_amazon_review_2023.py` | データセットダウンロード CLI |

## 実行ガイド（何をどの順で確認するか）

### Step 0: 環境準備

```bash
cd dai-2-kai
uv sync
```

依存パッケージが正しくインストールされることを確認。

### Step 1: 第1回の復習（`chapter-1-introduction.py`）

```bash
uv run python chapter-1-introduction.py
```

- **確認ポイント**: 感情分析・NER・要約・埋め込み等 7 つの NLP タスクが順に実行される
- 各タスクで【入力】→【出力】が表示され、パイプラインの基本動作を理解する

### Step 2: データセットの取得確認

```bash
uv run python download_amazon_review_2023.py
```

- **確認ポイント**: Amazon Reviews 2023 (All_Beauty) の JSONL がダウンロードされる
- 初回のみ必要。以降はキャッシュされるので再実行不要

### Step 3: 全セクション通し実行

```bash
uv run python dai-2-kai.py
```

- **確認ポイント**: §1〜§9 が順に実行され、最後に「デモ終了！」と表示される
- 初回はモデルダウンロードに数分かかる

### Step 4: 個別セクションで動作確認（必要に応じて）

気になるセクションだけ単独で実行して出力を確認できる。

```bash
# §5 HuggingFace推論: 逐次 vs バッチの速度比較
uv run python section5_hf_inference.py

# §6 埋め込み: 逐次 vs バッチの速度比較 + コサイン類似度デモ
uv run python section6_embedding.py

# §7 API推論: 逐次 vs asyncio の速度比較（APIキー不要でもシミュレーション動作）
uv run python section7_api_inference.py

# §8 Batch API: 概念説明 + JSONLファイル生成
uv run python section8_batch_api.py
```

| セクション | 確認ポイント |
|---|---|
| §5 | バッチ処理が逐次処理より速いことを数値で確認（スピードアップ倍率が表示される） |
| §6 | 埋め込みのバッチ効果 + 最も類似するレビューペアが出力される |
| §7 | asyncio 並行処理が逐次より速いことを確認（シミュレーションでは約5倍） |
| §8 | `eda_output/batch_request_example.jsonl` が生成される |

### Step 5: OpenAI API を使った実行（オプション）

```bash
export OPENAI_API_KEY="sk-..."
uv run python section7_api_inference.py
```

- **確認ポイント**: 実際の API を使った逐次 vs asyncio 比較。シミュレーションとの違いを体感

## 使用モデル

- **感情分析**: `nlptown/bert-base-multilingual-uncased-sentiment`
- **埋め込み**: `intfloat/multilingual-e5-large`
- **API推論**: `gpt-4o-mini`（OpenAI API キー設定時）
