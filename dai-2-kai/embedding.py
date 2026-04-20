"""埋め込みデモ — バッチ + multiprocessing で Amazon Reviews を埋め込み・保存"""

import json
import multiprocessing as mp
from functools import partial
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

from util import part, timer, load_amazon_reviews


MODEL_NAME = "intfloat/multilingual-e5-large"
OUTPUT_DIR = Path("eda_output")
EMBEDDINGS_FILE = OUTPUT_DIR / "embeddings.npz"


def _embed_chunk(texts: list[str], model_name: str, batch_size: int = 32) -> np.ndarray:
    """子プロセスで実行: チャンクごとにモデルをロードしバッチ埋め込み"""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inputs = tokenizer(
            batch, return_tensors="pt",
            truncation=True, max_length=512, padding=True,
        )
        with torch.no_grad():
            output = model(**inputs)
        # CLS token embedding
        all_embs.append(output.last_hidden_state[:, 0, :].numpy())
    return np.concatenate(all_embs, axis=0)


def embedding_demo(reviews: list[dict]):
    part(6, "埋め込みデモ — バッチ + multiprocessing → 保存")

    # --- テキスト準備 ---
    texts_raw = [r.get("text", "") or "" for r in reviews]
    valid_indices = [i for i, t in enumerate(texts_raw) if t.strip()]
    valid_reviews = [reviews[i] for i in valid_indices]
    texts = [f"query: {texts_raw[i][:512]}" for i in valid_indices]
    n = len(texts)

    print(f"\n  対象: Amazon Reviews (All_Beauty) {n} 件 (空レビュー除外済み)")
    print(f"  モデル: {MODEL_NAME} (1024次元)")

    # ========== 1. バッチ埋め込み (シングルプロセス) ==========
    print(f"\n  --- 1. バッチ埋め込み ({n} 件, batch_size=32) ---")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()

    with timer(f"バッチ埋め込み ({n} 件)") as t_batch:
        all_embs = []
        for i in range(0, n, 32):
            batch = texts[i : i + 32]
            inputs = tokenizer(
                batch, return_tensors="pt",
                truncation=True, max_length=512, padding=True,
            )
            with torch.no_grad():
                output = model(**inputs)
            all_embs.append(output.last_hidden_state[:, 0, :].numpy())
        embeddings_batch = np.concatenate(all_embs, axis=0)

    print(f"  出力 shape: {embeddings_batch.shape}")

    # モデルを解放してメモリ確保
    del model, tokenizer

    # ========== 2. multiprocessing 埋め込み ==========
    num_workers = min(mp.cpu_count(), 4)
    chunk_size = (n + num_workers - 1) // num_workers
    chunks = [texts[i : i + chunk_size] for i in range(0, n, chunk_size)]

    print(f"\n  --- 2. multiprocessing 埋め込み ({n} 件, {num_workers} プロセス) ---")
    print(f"     チャンク数: {len(chunks)}, 各チャンク ≈ {chunk_size} 件")

    embed = partial(_embed_chunk, model_name=MODEL_NAME, batch_size=32)

    with timer(f"multiprocessing 埋め込み ({n} 件)") as t_mp:
        with mp.Pool(num_workers) as pool:
            chunk_results = pool.map(embed, chunks)
        embeddings_mp = np.concatenate(chunk_results, axis=0)

    print(f"  出力 shape: {embeddings_mp.shape}")

    # ========== 比較 ==========
    speedup = t_batch.elapsed / t_mp.elapsed if t_mp.elapsed > 0 else float("inf")
    print(f"\n  📊 速度比較:")
    print(f"     バッチ:            {t_batch.elapsed:.3f} 秒")
    print(f"     multiprocessing:   {t_mp.elapsed:.3f} 秒  ({speedup:.1f}x)")

    # ========== 保存 ==========
    OUTPUT_DIR.mkdir(exist_ok=True)

    # メタデータを構築: テキスト、rating、タイトルなどを保持
    metadata = []
    for r in valid_reviews:
        metadata.append({
            "text": (r.get("text", "") or "")[:512],
            "rating": r.get("rating", None),
            "title": r.get("title", ""),
            "asin": r.get("asin", ""),
        })

    # npz で埋め込み保存、メタデータは JSON で保存
    np.savez_compressed(
        EMBEDDINGS_FILE,
        embeddings=embeddings_mp,
    )
    metadata_file = OUTPUT_DIR / "metadata.json"
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"\n  💾 保存完了:")
    print(f"     埋め込み: {EMBEDDINGS_FILE}  ({embeddings_mp.shape})")
    print(f"     メタデータ: {metadata_file}  ({len(metadata)} 件)")
    print("""
  💡 ポイント:
     - バッチ化で GPU/CPU の並列計算を活かしつつ、
       multiprocessing でさらに CPU コアを使い切る
     - 埋め込みは .npz、メタデータは .json で保存
       → 可視化・検索など後続タスクで再利用可能
""")


if __name__ == "__main__":
    print("📦 Amazon Reviews 2023 (All_Beauty) を読み込み中...")
    reviews = load_amazon_reviews(n=200)
    print(f"   {len(reviews)} 件読み込み完了")
    embedding_demo(reviews)
