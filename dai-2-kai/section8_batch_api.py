"""セクション 8: バッチAPIデモ — asyncio vs バッチAPI"""

import json
from pathlib import Path

from util import section, load_amazon_reviews, extract_texts


def section_8_batch_api_demo(reviews: list[dict]):
    section(8, "バッチAPIデモ — asyncio vs バッチAPI")

    print("""
  ■ asyncio (セクション7) と バッチAPI の違い:

    ┌──────────────┬──────────────────────────────────────┐
    │ 方式         │ 仕組み                               │
    ├──────────────┼──────────────────────────────────────┤
    │ asyncio      │ 複数リクエストを「同時に投げる」      │
    │              │ → 待ち時間を重ね合わせる              │
    │              │ → リアルタイムに結果が返る            │
    ├──────────────┼──────────────────────────────────────┤
    │ バッチAPI    │ 全リクエストを1つのファイルにまとめて │
    │              │ 送信し、後でまとめて結果を受け取る    │
    │              │ → 送信単位そのものをまとめる          │
    │              │ → 非同期処理（数時間後に結果取得）    │
    │              │ → 通常 50% 割引（OpenAI の場合）     │
    └──────────────┴──────────────────────────────────────┘

  ■ バッチAPI の使いどころ:
    - 大量データの一括処理 (数百〜数万件)
    - リアルタイム性が不要な場合
    - コストを抑えたい場合

  ■ バッチAPI の流れ (OpenAI の場合):
""")

    # --- バッチリクエストファイルの作成例 ---
    texts = extract_texts(reviews[:5], max_len=200)

    batch_requests = []
    for i, text in enumerate(texts):
        batch_requests.append({
            "custom_id": f"review-{i}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": "Classify the sentiment as positive, negative, or neutral. Reply with one word."},
                    {"role": "user", "content": text},
                ],
                "max_tokens": 5,
            },
        })

    batch_file = Path("eda_output") / "batch_request_example.jsonl"
    batch_file.parent.mkdir(exist_ok=True)
    with open(batch_file, "w") as f:
        for req in batch_requests:
            f.write(json.dumps(req, ensure_ascii=False) + "\n")

    print(f"    1. JSONL形式でリクエストファイルを作成")
    print(f"       → {batch_file} (サンプル {len(batch_requests)} 件)")
    print(f"\n    リクエスト例 (1件目):")
    print(f"    {json.dumps(batch_requests[0], ensure_ascii=False, indent=4)[:400]}...")
    print(f"""
    2. ファイルをアップロード:
       file = client.files.create(file=open("{batch_file}"), purpose="batch")

    3. バッチジョブを作成:
       batch = client.batches.create(
           input_file_id=file.id,
           endpoint="/v1/chat/completions",
           completion_window="24h",
       )

    4. ステータス確認 (完了まで待つ):
       client.batches.retrieve(batch.id)

    5. 結果ダウンロード:
       result = client.files.content(batch.output_file_id)
""")

    print("""
  💡 ポイント:
     - asyncio は「待ち時間を重ねる」工夫 → リアルタイム向き
     - バッチAPI は「送信単位そのものをまとめる」工夫 → 大量処理向き
     - 「並列化」だけでなく「バッチ化」も効率化の手段！
""")


if __name__ == "__main__":
    print("📦 Amazon Reviews 2023 (All_Beauty) を読み込み中...")
    reviews = load_amazon_reviews(n=200)
    print(f"   {len(reviews)} 件読み込み完了")
    section_8_batch_api_demo(reviews)
