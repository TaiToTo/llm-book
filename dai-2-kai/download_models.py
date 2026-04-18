"""
dai-2-kai: 必要なモデル・データを事前ダウンロード
=================================================
勉強会で使う全モデルとデータセットを一括ダウンロードする。
ネット環境のあるうちに実行しておけば、デモ時はオフラインでも動作する。

  uv run python download_models.py
"""

from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    pipeline,
)

from util import load_amazon_reviews

# chapter-1-introduction.py で使用するモデル
CHAPTER1_MODELS = [
    "llm-book/bert-base-japanese-v3-marc_ja",       # §1 感情分析
    "llm-book/bert-base-japanese-v3-jnli",           # §2 自然言語推論
    "llm-book/bert-base-japanese-v3-jsts",            # §3 意味的類似度
    "llm-book/bert-base-japanese-v3-unsup-simcse-jawiki",  # §4 埋め込み
    "llm-book/bert-base-japanese-v3-ner-wikipedia-dataset", # §5 固有表現認識
]

# chapter-1-introduction.py の要約・生成モデル（AutoModel 系）
CHAPTER1_SEQ2SEQ = "llm-book/t5-base-long-livedoor-news-corpus"  # §6 要約
CHAPTER1_CAUSAL = "rinna/japanese-gpt2-small"                     # §7 テキスト生成

# dai-2-kai デモで使用するモデル
SENTIMENT_MODEL = "nlptown/bert-base-multilingual-uncased-sentiment"  # 感情分析
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"                    # 埋め込み


def main():
    print("=" * 70)
    print("  dai-2-kai: モデル・データ 一括ダウンロード")
    print("=" * 70)

    # --- データセット ---
    print("\n[1/4] Amazon Reviews 2023 (All_Beauty) ...")
    reviews = load_amazon_reviews(n=1)
    print(f"  ✅ データ取得OK ({len(reviews)} 件テスト読み込み)")

    # --- chapter-1 pipeline モデル ---
    print("\n[2/4] chapter-1-introduction.py 用モデル ...")
    for name in CHAPTER1_MODELS:
        print(f"  ⬇  {name}")
        pipeline(model=name)
    print(f"  ⬇  {CHAPTER1_SEQ2SEQ} (Seq2Seq)")
    AutoTokenizer.from_pretrained(CHAPTER1_SEQ2SEQ)
    AutoModelForSeq2SeqLM.from_pretrained(CHAPTER1_SEQ2SEQ)
    print(f"  ⬇  {CHAPTER1_CAUSAL} (CausalLM)")
    AutoTokenizer.from_pretrained(CHAPTER1_CAUSAL)
    AutoModelForCausalLM.from_pretrained(CHAPTER1_CAUSAL)
    print("  ✅ chapter-1 モデル完了")

    # --- 感情分析モデル ---
    print(f"\n[3/4] 感情分析用: {SENTIMENT_MODEL} ...")
    pipeline("sentiment-analysis", model=SENTIMENT_MODEL)
    print("  ✅ 感情分析モデル完了")

    # --- 埋め込みモデル ---
    print(f"\n[4/4] 埋め込み用: {EMBEDDING_MODEL} ...")
    AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
    AutoModel.from_pretrained(EMBEDDING_MODEL)
    print("  ✅ 埋め込みモデル完了")

    print("\n" + "=" * 70)
    print("  全モデル・データのダウンロード完了！")
    print("  各デモスクリプトを個別に実行してください:")
    print("    uv run python chapter-1-introduction.py")
    print("    uv run python hf_inference.py")
    print("    uv run python embedding.py")
    print("    uv run python api_inference.py")
    print("    uv run python batch_api.py")
    print("=" * 70)


if __name__ == "__main__":
    main()

