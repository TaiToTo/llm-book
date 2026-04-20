"""HF推論デモ — 逐次 vs バッチ vs multiprocessing（感情分析）"""

import multiprocessing as mp
from functools import partial

from transformers import pipeline as hf_pipeline

from util import part, timer, load_amazon_reviews, extract_texts


def _classify_chunk(texts: list[str], model_name: str) -> list[dict]:
    """子プロセスで実行: チャンクごとにパイプラインを作成して推論"""
    pipe = hf_pipeline(
        "sentiment-analysis",
        model=model_name,
        truncation=True,
        max_length=512,
    )
    return pipe(texts, batch_size=32)


def hf_inference_demo(reviews: list[dict]):
    part(5, "HF推論デモ — 逐次 vs バッチ vs multiprocessing（感情分析）")

    texts = extract_texts(reviews)
    n = len(texts)
    model_name = "nlptown/bert-base-multilingual-uncased-sentiment"

    print(f"\n  対象: Amazon Reviews (All_Beauty) {n} 件")
    print(f"  モデル: {model_name}")
    print("  (英語レビューの星1〜5感情分析)")

    # --- モデルロード ---
    sentiment = hf_pipeline(
        "sentiment-analysis",
        model=model_name,
        truncation=True,
        max_length=512,
    )

    # ========== 1. 逐次推論 ==========
    print(f"\n  --- 1. 逐次推論 ({n} 件を1件ずつ) ---")
    with timer(f"逐次推論 ({n} 件)") as t_seq:
        results_seq = []
        for text in texts:
            results_seq.append(sentiment(text)[0])

    print(f"  サンプル結果: {results_seq[0]}")

    # ========== 2. バッチ推論 ==========
    batch_size = 32
    print(f"\n  --- 2. バッチ推論 ({n} 件, batch_size={batch_size}) ---")
    with timer(f"バッチ推論 ({n} 件)") as t_batch:
        results_batch = sentiment(texts, batch_size=batch_size)

    print(f"  サンプル結果: {results_batch[0]}")

    # ========== 3. multiprocessing ==========
    num_workers = min(mp.cpu_count(), 4)
    chunk_size = (n + num_workers - 1) // num_workers
    chunks = [texts[i : i + chunk_size] for i in range(0, n, chunk_size)]

    print(f"\n  --- 3. multiprocessing ({n} 件, {num_workers} プロセス) ---")
    print(f"     チャンク数: {len(chunks)}, 各チャンク ≈ {chunk_size} 件")
    print("     ※ 各プロセスが独立にモデルをロードし、バッチ推論を実行")

    classify = partial(_classify_chunk, model_name=model_name)

    with timer(f"multiprocessing ({n} 件)") as t_mp:
        with mp.Pool(num_workers) as pool:
            chunk_results = pool.map(classify, chunks)
        results_mp = [r for chunk in chunk_results for r in chunk]

    print(f"  サンプル結果: {results_mp[0]}")

    # ========== 比較 ==========
    speedup_batch = t_seq.elapsed / t_batch.elapsed if t_batch.elapsed > 0 else float("inf")
    speedup_mp = t_seq.elapsed / t_mp.elapsed if t_mp.elapsed > 0 else float("inf")
    print(f"\n  📊 速度比較:")
    print(f"     逐次:              {t_seq.elapsed:.3f} 秒")
    print(f"     バッチ:            {t_batch.elapsed:.3f} 秒  ({speedup_batch:.1f}x)")
    print(f"     multiprocessing:   {t_mp.elapsed:.3f} 秒  ({speedup_mp:.1f}x)")
    print("""
  💡 ポイント:
     - バッチ化: GPU/CPU の並列計算能力を活かす最も手軽な方法
     - multiprocessing: 複数プロセスで独立にバッチ推論を回す
       → CPU推論ならコア数に応じて速くなる
       → GPU 1台の場合はメモリ競合でバッチ化のほうが有利なことも
     - まず「バッチ化」、それでも足りなければ multiprocessing を検討
""")


if __name__ == "__main__":
    print("📦 Amazon Reviews 2023 (All_Beauty) を読み込み中...")
    reviews = load_amazon_reviews(n=200)
    print(f"   {len(reviews)} 件読み込み完了")
    hf_inference_demo(reviews)
