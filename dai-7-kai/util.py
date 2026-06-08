"""共通ユーティリティ — データ読み込み

ノートブック（02 / 03）が共有する、Amazon-Reviews-2023 の感情分析データ読み込み。
"""

import json
import random

from huggingface_hub import hf_hub_download


def load_amazon_binary(max_lines: int = 120000, seed: int = 42):
    """Amazon-Reviews-2023 (All_Beauty) を星評価から二値ラベル化して読み込む。

    MARC-ja と同じ方針: 星4-5 → positive(0) / 星1-2 → negative(1) / 星3 は除外。
    自然な不均衡（positive 偏重）をそのまま保持する。シャッフル済みの
    (texts, labels, ratings) を返す（seed 固定なので同じ引数なら毎回同じ並び）。
    """
    path = hf_hub_download(
        repo_id="McAuley-Lab/Amazon-Reviews-2023",
        filename="raw/review_categories/All_Beauty.jsonl",
        repo_type="dataset",
    )
    texts, labels, ratings = [], [], []
    with open(path, "r") as f:
        for i, line in enumerate(f):
            if i >= max_lines:
                break
            r = json.loads(line)
            rating = r.get("rating")
            text = (r.get("text") or "").strip()
            if not text or rating is None:
                continue
            if rating >= 4:
                label = 0  # positive
            elif rating <= 2:
                label = 1  # negative
            else:
                continue  # 星3（中立）は除外
            texts.append(text[:512])
            labels.append(label)
            ratings.append(rating)

    # 順序バイアスを避けるためシャッフル
    rng = random.Random(seed)
    idx = list(range(len(texts)))
    rng.shuffle(idx)
    texts = [texts[i] for i in idx]
    labels = [labels[i] for i in idx]
    ratings = [ratings[i] for i in idx]
    return texts, labels, ratings
