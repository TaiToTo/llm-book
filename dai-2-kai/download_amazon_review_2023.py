"""
Amazon Reviews 2023 (McAuley-Lab) のレビューデータをダウンロードするスクリプト。

Usage:
    uv run python download_amazon_review_2023.py
    uv run python download_amazon_review_2023.py --category All_Beauty
    uv run python download_amazon_review_2023.py --category All_Beauty --split meta

Available categories (例):
    All_Beauty, Amazon_Fashion, Appliances, Arts_Crafts_and_Sewing,
    Automotive, Baby_Products, Beauty_and_Personal_Care, Books, ...

ダウンロードされたファイルは Hugging Face のキャッシュ
(~/.cache/huggingface/hub) に保存されます。
返り値としてキャッシュパスを表示します。
"""
import argparse
from huggingface_hub import hf_hub_download

REPO_ID = "McAuley-Lab/Amazon-Reviews-2023"

SPLIT_PATHS = {
    "review": "raw/review_categories/{category}.jsonl",
    "meta":   "raw/meta_categories/meta_{category}.jsonl",
}


def download(category: str, split: str) -> str:
    filename = SPLIT_PATHS[split].format(category=category)
    print(f"Downloading: {filename} ...")
    path = hf_hub_download(
        repo_id=REPO_ID,
        filename=filename,
        repo_type="dataset",
    )
    print(f"Saved to: {path}")
    return path


def main():
    parser = argparse.ArgumentParser(description="Download Amazon Reviews 2023 dataset")
    parser.add_argument(
        "--category", default="All_Beauty",
        help="カテゴリ名 (例: All_Beauty, Books, Electronics)",
    )
    parser.add_argument(
        "--split", default="review", choices=["review", "meta"],
        help="review: ユーザーレビュー / meta: 商品メタデータ (default: review)",
    )
    args = parser.parse_args()
    download(args.category, args.split)


if __name__ == "__main__":
    main()
