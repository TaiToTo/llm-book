"""
第2回 NLP勉強会: 効率化デモスクリプト — メインランナー
=====================================================
全セクションを順番に実行する。個別に実行する場合は各スクリプトを直接実行可能。

  uv run python dai-2-kai.py      # 全セクション
  uv run python section_intro.py  # セクション1,3,4のみ
  uv run python section5_hf_inference.py
  uv run python section6_embedding.py
  uv run python section7_api_inference.py
  uv run python section8_batch_api.py
  uv run python section9_summary.py
"""

from util import load_amazon_reviews
from section_intro import section_1_introduction, section_3_nlp_recap, section_4_concepts
from section5_hf_inference import section_5_hf_inference_demo
from section6_embedding import section_6_embedding_demo
from section7_api_inference import section_7_api_inference_demo
from section8_batch_api import section_8_batch_api_demo
from section9_summary import section_9_summary


def main():
    print("=" * 70)
    print("  第2回 NLP勉強会: 効率よく回す技術")
    print("  — API / ローカル推論 / 埋め込み / 保存の使い分け")
    print("=" * 70)

    # --- データ準備 ---
    print("\n📦 Amazon Reviews 2023 (All_Beauty) を読み込み中...")
    reviews = load_amazon_reviews(n=200)
    print(f"   {len(reviews)} 件読み込み完了\n")

    # --- 各セクション実行 ---
    section_1_introduction()
    section_3_nlp_recap()
    section_4_concepts()
    section_5_hf_inference_demo(reviews)
    section_6_embedding_demo(reviews)
    section_7_api_inference_demo(reviews)
    section_8_batch_api_demo(reviews)
    section_9_summary()

    print("=" * 70)
    print("  デモ終了！")
    print("=" * 70)


if __name__ == "__main__":
    main()

