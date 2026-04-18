"""共通ユーティリティ — セクション表示・タイマー・データロード"""

import json
import time

from huggingface_hub import hf_hub_download


def section(num: int, title: str):
    print("\n" + "=" * 70)
    print(f"  セクション {num}. {title}")
    print("=" * 70)


def timer(label: str):
    """コンテキストマネージャで経過時間を計測・表示する"""
    class _Timer:
        def __init__(self):
            self.elapsed = 0.0
        def __enter__(self):
            self.start = time.perf_counter()
            return self
        def __exit__(self, *_):
            self.elapsed = time.perf_counter() - self.start
            print(f"  ⏱  {label}: {self.elapsed:.3f} 秒")
    return _Timer()


def load_amazon_reviews(n: int = 200) -> list[dict]:
    """Amazon Reviews 2023 (All_Beauty) から先頭 n 件を読み込む"""
    file_path = hf_hub_download(
        repo_id="McAuley-Lab/Amazon-Reviews-2023",
        filename="raw/review_categories/All_Beauty.jsonl",
        repo_type="dataset",
    )
    records = []
    with open(file_path, "r") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            records.append(json.loads(line))
    return records


def extract_texts(reviews: list[dict], max_len: int = 512) -> list[str]:
    """レビューからテキストを抽出し、空文字列を除去・長さ制限する"""
    texts = [r.get("text", "") or "" for r in reviews]
    return [t[:max_len] for t in texts if t.strip()]
