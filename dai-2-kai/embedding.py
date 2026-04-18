"""埋め込みデモ — 逐次 vs バッチ (multilingual-e5-large)"""

import torch
from torch.nn.functional import cosine_similarity
from transformers import AutoTokenizer, AutoModel

from util import part, timer, load_amazon_reviews, extract_texts


def embedding_demo(reviews: list[dict]):
    part(6, "埋め込みデモ — 逐次 vs バッチ (multilingual-e5-large)")

    texts = extract_texts(reviews)
    n = len(texts)
    print(f"\n  対象: Amazon Reviews (All_Beauty) {n} 件")

    model_name = "intfloat/multilingual-e5-large"
    print(f"  モデル: {model_name}")
    print("  (多言語対応の埋め込みモデル, 1024次元)")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"  デバイス: {device}")

    def embed_single(text: str) -> torch.Tensor:
        """1件ずつ埋め込みを生成"""
        inputs = tokenizer(
            f"query: {text}", return_tensors="pt",
            truncation=True, max_length=512, padding=True,
        ).to(device)
        with torch.no_grad():
            output = model(**inputs)
        return output.last_hidden_state[:, 0, :]

    def embed_batch(texts: list[str], batch_size: int = 32) -> torch.Tensor:
        """バッチで埋め込みを生成"""
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = [f"query: {t}" for t in texts[i:i + batch_size]]
            inputs = tokenizer(
                batch, return_tensors="pt",
                truncation=True, max_length=512, padding=True,
            ).to(device)
            with torch.no_grad():
                output = model(**inputs)
            all_embeddings.append(output.last_hidden_state[:, 0, :])
        return torch.cat(all_embeddings, dim=0)

    # --- 逐次埋め込み ---
    print(f"\n  --- 逐次埋め込み ({n} 件を1件ずつ) ---")
    with timer(f"逐次埋め込み ({n} 件)") as t_seq:
        embeddings_seq = []
        for text in texts:
            embeddings_seq.append(embed_single(text))
        embeddings_seq = torch.cat(embeddings_seq, dim=0)

    print(f"  出力 shape: {tuple(embeddings_seq.shape)}")

    # --- バッチ埋め込み ---
    batch_size = 32
    print(f"\n  --- バッチ埋め込み ({n} 件, batch_size={batch_size}) ---")
    with timer(f"バッチ埋め込み ({n} 件)") as t_batch:
        embeddings_batch = embed_batch(texts, batch_size=batch_size)

    print(f"  出力 shape: {tuple(embeddings_batch.shape)}")

    # --- 比較 ---
    speedup = t_seq.elapsed / t_batch.elapsed if t_batch.elapsed > 0 else float("inf")
    print(f"\n  📊 速度比較:")
    print(f"     逐次:   {t_seq.elapsed:.3f} 秒")
    print(f"     バッチ: {t_batch.elapsed:.3f} 秒")
    print(f"     高速化: {speedup:.1f}x")

    # --- 類似度デモ ---
    print("\n  --- 埋め込みの活用例: コサイン類似度 ---")
    sim = cosine_similarity(embeddings_batch[0].unsqueeze(0), embeddings_batch[1:], dim=1)
    top_idx = sim.argmax().item() + 1
    print(f"  レビュー[0] に最も類似: レビュー[{top_idx}] (cos_sim={sim[top_idx-1]:.4f})")
    print(f"  レビュー[0]: {texts[0][:80]}...")
    print(f"  レビュー[{top_idx}]: {texts[top_idx][:80]}...")

    print("""
  💡 ポイント:
     ローカル埋め込みも「バッチ化」が最重要！
     tokenizer の padding で長さを揃え、まとめて GPU に渡す。
     API型埋め込み (OpenAI等) の場合は asyncio が有効。
""")


if __name__ == "__main__":
    print("📦 Amazon Reviews 2023 (All_Beauty) を読み込み中...")
    reviews = load_amazon_reviews(n=200)
    print(f"   {len(reviews)} 件読み込み完了")
    embedding_demo(reviews)
