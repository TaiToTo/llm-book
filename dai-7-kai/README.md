# dai-7-kai — 第7回 NLP勉強会デモ

『大規模言語モデル入門』**第5章（chapter05）** に対応する勉強会用ノートブック集です。
**感情分析（positive / negative の二値分類）** を題材に、

- データの見方（**データ構造・ラベル分布・不均衡**）
- **BERT** のファインチューニング（tokenizer・パディング・訓練・評価・エラー分析）
- **古典手法（TF-IDF + ロジスティック回帰）** での解き方と BERT との比較・**RRF融合**

までを、出力込みで実行済みのノートブックでたどります。

教科書の MARC-ja は元データ（Amazon Review）が非公開化されたため、
本デモは **`McAuley-Lab/Amazon-Reviews-2023`（All_Beauty, 英語）** の星評価を
二値化（★4-5→positive / ★1-2→negative / ★3除外）して使います。
これにより、教科書が語る「**不均衡データ**」を再現します。英語データなので、
CPU でも軽い英語モデル `distilbert-base-uncased` を用います。

## ノートブック

| ノートブック | 内容 |
|---|---|
| `01_amazon_eda.ipynb` | **(1) EDA・ラベル構造**: データ構造／星→二値化／不均衡なラベル分布／テキスト長・語彙 |
| `02_amazon_bert.ipynb` | **(2) BERT**: tokenizer（サブワード・特殊トークン）→ DataCollator による動的パディング → モデル → 訓練（不均衡対応）→ 評価・エラー分析 |
| `03_amazon_logreg.ipynb` | **(補足) TF-IDF + ロジスティック回帰**: BOWの中身を可視化／同じ問題を古典手法で解き(2)のBERTと比較／効いている語(n-gram)／エラー分析比較(相補性)／**RRF**による順位融合 |

`.py` は jupytext のソース（読みやすい・差分が取りやすい）、`.ipynb` が実行済み成果物です。
`util.py` は両ノートが共有するデータ読み込み（`load_amazon_binary`）です。

## セットアップ

```bash
cd dai-7-kai
uv sync
# ノートブック実行用カーネルを（プロジェクトの venv に）登録
uv run python -m ipykernel install --sys-prefix --name python3 \
  --display-name "Python 3 (dai-7-kai)"
```

## 実行

```bash
# 開いて実行（Jupyter / VS Code など）。順番は 01 → 02 → 03
uv run jupyter lab 01_amazon_eda.ipynb
```

`.py` を編集して `.ipynb` を再生成・再実行する場合:

```bash
uv run jupytext --to ipynb 01_amazon_eda.py 02_amazon_bert.py 03_amazon_logreg.py
uv run jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=python3 \
  01_amazon_eda.ipynb 02_amazon_bert.ipynb 03_amazon_logreg.ipynb
```

> **実行順の注意**: `02` は学習結果を `output/nb2_bert_results.json` と
> `output/nb2_bert_preds.npz` に保存し、`03` がそれを読み込んで BERT と比較します。
> **`03` の前に `02` を実行**してください（無い場合 `03` はフォールバック値を使用）。
> `02` は CPU 向けに小さめのサブセット（train 800 / valid 400 × 1エポック）で
> distilbert を2回学習します（数分で完走）。

## 使用データ・モデル

- **データ**: `McAuley-Lab/Amazon-Reviews-2023`（All_Beauty, 英語）
- **モデル**: `distilbert-base-uncased`（英語・軽量, CPU向け）

## メモ

- 図はノートブックにインライン表示（実行済み）。`output/` は中間生成物の置き場
  （`.gitignore` 済み）。
- GPU があれば transformers が自動利用します（サブセットを増やせば本格学習に拡張可）。
