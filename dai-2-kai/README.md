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
| `download_models.py` | モデル・データ一括ダウンロード |
| `util.py` | 共通ユーティリティ（データ読み込み・タイマー等） |
| `intro.py` | イントロ・NLP振り返り・効率化の概念説明 |
| `hf_inference.py` | HuggingFace推論（逐次 vs バッチ） |
| `embedding.py` | 埋め込み（逐次 vs バッチ） |
| `api_inference.py` | API推論（逐次 vs asyncio） |
| `batch_api.py` | Batch API の概念と JSONL 例 |
| `summary.py` | まとめ |
| `chapter-1-introduction.py` | 教科書 Chapter 1 に対応するデモスクリプト |
| `download_amazon_review_2023.py` | データセットダウンロード CLI |

## 実行ガイド（何をどの順で確認するか）

### Step 0: 環境準備

```bash
cd dai-2-kai
uv sync
```

依存パッケージが正しくインストールされることを確認。

### Step 1: モデル・データの一括ダウンロード

```bash
uv run python download_models.py
```

- **確認ポイント**: 全モデル（9個）と Amazon Reviews データが順にダウンロードされ、最後に「全モデル・データのダウンロード完了！」と表示される
- 初回はダウンロードに数分〜十数分かかる。以降はキャッシュ済み
- ネット環境のあるうちに実行しておけば、デモ時はオフラインでも動作する

### Step 2: 教科書 Chapter 1 のデモ（`chapter-1-introduction.py`）

```bash
uv run python chapter-1-introduction.py
```

- **確認ポイント**: 感情分析・NER・要約・埋め込み等 7 つの NLP タスクが順に実行される
- 各タスクで【入力】→【出力】が表示され、パイプラインの基本動作を理解する

### Step 3: 各デモスクリプトで動作確認

```bash
# HuggingFace推論: 逐次 vs バッチの速度比較
uv run python hf_inference.py

# 埋め込み: 逐次 vs バッチの速度比較 + コサイン類似度デモ
uv run python embedding.py

# API推論: 逐次 vs asyncio の速度比較（APIキー不要でもシミュレーション動作）
uv run python api_inference.py

# Batch API: 概念説明 + JSONLファイル生成
uv run python batch_api.py
```

| スクリプト | 確認ポイント |
|---|---|
| `hf_inference.py` | バッチ処理が逐次処理より速いことを数値で確認（スピードアップ倍率が表示される） |
| `embedding.py` | 埋め込みのバッチ効果 + 最も類似するレビューペアが出力される |
| `api_inference.py` | asyncio 並行処理が逐次より速いことを確認（シミュレーションでは約5倍） |
| `batch_api.py` | `eda_output/batch_request_example.jsonl` が生成される |

### Step 4: OpenAI API を使った実行（オプション）

```bash
export OPENAI_API_KEY="sk-..."
uv run python api_inference.py
```

- **確認ポイント**: 実際の API を使った逐次 vs asyncio 比較。シミュレーションとの違いを体感

## 使用モデル

- **感情分析**: `nlptown/bert-base-multilingual-uncased-sentiment`
- **埋め込み**: `intfloat/multilingual-e5-large`
- **API推論**: `gpt-4o-mini`（OpenAI API キー設定時）
