"""セクション 5: HF推論デモ — 逐次 vs バッチ（感情分析）"""

from transformers import pipeline as hf_pipeline

from util import section, timer, load_amazon_reviews, extract_texts


def section_5_hf_inference_demo(reviews: list[dict]):
    section(5, "HF推論デモ — 逐次 vs バッチ（感情分析）")

    texts = extract_texts(reviews)
    n = len(texts)
    print(f"\n  対象: Amazon Reviews (All_Beauty) {n} 件")

    # --- モデルロード ---
    print("\n  モデル: nlptown/bert-base-multilingual-uncased-sentiment")
    print("  (英語レビューの星1〜5感情分析)")
    sentiment = hf_pipeline(
        "sentiment-analysis",
        model="nlptown/bert-base-multilingual-uncased-sentiment",
        truncation=True,
        max_length=512,
    )

    # --- 逐次推論 ---
    print(f"\n  --- 逐次推論 ({n} 件を1件ずつ) ---")
    with timer(f"逐次推論 ({n} 件)") as t_seq:
        results_seq = []
        for text in texts:
            results_seq.append(sentiment(text)[0])

    print(f"  サンプル結果: {results_seq[0]}")

    # --- バッチ推論 ---
    batch_size = 32
    print(f"\n  --- バッチ推論 ({n} 件, batch_size={batch_size}) ---")
    with timer(f"バッチ推論 ({n} 件)") as t_batch:
        results_batch = sentiment(texts, batch_size=batch_size)

    print(f"  サンプル結果: {results_batch[0]}")

    # --- 比較 ---
    speedup = t_seq.elapsed / t_batch.elapsed if t_batch.elapsed > 0 else float("inf")
    print(f"\n  📊 速度比較:")
    print(f"     逐次:   {t_seq.elapsed:.3f} 秒")
    print(f"     バッチ: {t_batch.elapsed:.3f} 秒")
    print(f"     高速化: {speedup:.1f}x")
    print("""
  💡 ポイント:
     ローカル推論は、まず並列化より「バッチ化」！
     GPU/CPUの並列計算能力を活かすためには、
     データをまとめて渡すのが最も効果的。
""")


if __name__ == "__main__":
    print("📦 Amazon Reviews 2023 (All_Beauty) を読み込み中...")
    reviews = load_amazon_reviews(n=200)
    print(f"   {len(reviews)} 件読み込み完了")
    section_5_hf_inference_demo(reviews)
