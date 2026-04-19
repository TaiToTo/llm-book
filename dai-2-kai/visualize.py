"""可視化デモ — 保存済み埋め込みを UMAP で2D可視化"""

import json
from pathlib import Path

import japanize_matplotlib  # noqa: F401  matplotlib 日本語化
import matplotlib.pyplot as plt
import numpy as np
from umap import UMAP

from util import part


INPUT_DIR = Path("eda_output")
EMBEDDINGS_FILE = INPUT_DIR / "embeddings.npz"
METADATA_FILE = INPUT_DIR / "metadata.json"


def visualize_demo():
    part(8, "可視化デモ — UMAP で埋め込みを2D表示")

    # --- データ読み込み ---
    if not EMBEDDINGS_FILE.exists() or not METADATA_FILE.exists():
        print(f"""
  ⚠  埋め込みデータが見つかりません。
     先に embedding.py を実行して埋め込みを生成・保存してください:

       uv run python embedding.py
""")
        return

    data = np.load(EMBEDDINGS_FILE)
    embeddings = data["embeddings"]
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    n = len(embeddings)
    print(f"\n  読み込み: {n} 件, 埋め込み次元: {embeddings.shape[1]}")

    # --- rating を取得 ---
    ratings = np.array([m.get("rating", 0) or 0 for m in metadata], dtype=float)
    has_rating = ratings > 0
    print(f"  rating あり: {has_rating.sum()} 件")

    # --- UMAP で次元削減 ---
    print("\n  UMAP で 2D に次元削減中...")
    reducer = UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    coords = reducer.fit_transform(embeddings)
    print(f"  完了: {coords.shape}")

    # --- プロット ---
    fig, ax = plt.subplots(figsize=(10, 8))

    # rating がある点を色付きプロット
    if has_rating.any():
        sc = ax.scatter(
            coords[has_rating, 0],
            coords[has_rating, 1],
            c=ratings[has_rating],
            cmap="RdYlGn",
            vmin=1,
            vmax=5,
            s=20,
            alpha=0.7,
        )
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label("Rating (星)")

    # rating がない点は灰色
    if (~has_rating).any():
        ax.scatter(
            coords[~has_rating, 0],
            coords[~has_rating, 1],
            c="lightgray",
            s=10,
            alpha=0.4,
            label="rating なし",
        )
        ax.legend()

    ax.set_title(f"Amazon Reviews 埋め込み UMAP 可視化 ({n} 件)")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")

    output_path = INPUT_DIR / "umap_visualization.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\n  💾 画像保存: {output_path}")
    print("""
  💡 ポイント:
     - UMAP は高次元の埋め込みを2Dに写像して構造を視覚化する
     - 似た意味のレビューは近くにクラスタとして現れる
     - rating (星) で色分けすると、感情とクラスタの関係が見える
""")


if __name__ == "__main__":
    visualize_demo()
