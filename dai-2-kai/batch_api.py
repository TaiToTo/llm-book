"""バッチAPIデモ — asyncio vs バッチAPI"""

import argparse
import json
from pathlib import Path

from util import part, load_amazon_reviews, extract_texts

SYSTEM_PROMPT = "Classify the sentiment as positive, negative, or neutral. Reply with one word."


def batch_api_demo(reviews: list[dict]):
    part(8, "バッチAPIデモ — asyncio vs バッチAPI")

    print("""
  ■ asyncio (api_inference.py) と バッチAPI の違い:

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
    batch_requests = _build_requests(texts)

    batch_file = Path("eda_output") / "batch_request_example.jsonl"
    batch_file.parent.mkdir(exist_ok=True)
    _write_jsonl(batch_file, batch_requests)

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


# ─── ヘルパー ─────────────────────────────────────────────────────

def _build_requests(texts: list[str]) -> list[dict]:
    """JSONL 用のバッチリクエストを組み立てる"""
    requests = []
    for i, text in enumerate(texts):
        requests.append({
            "custom_id": f"review-{i}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": text},
                ],
                "max_tokens": 5,
            },
        })
    return requests


def _write_jsonl(path: Path, records: list[dict]):
    path.parent.mkdir(exist_ok=True)
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ─── 実バッチAPI ──────────────────────────────────────────────────

def batch_api_real(reviews: list[dict]):
    """実際に OpenAI Batch API でジョブを投げる"""
    from openai import OpenAI

    client = OpenAI()

    texts = extract_texts(reviews[:5], max_len=200)
    batch_requests = _build_requests(texts)

    # 1. JSONL ファイル作成
    batch_file = Path("eda_output") / "batch_request.jsonl"
    _write_jsonl(batch_file, batch_requests)
    print(f"  1. JSONL ファイル作成: {batch_file} ({len(batch_requests)} 件)")

    # 2. ファイルアップロード
    print("  2. ファイルをアップロード中...")
    with open(batch_file, "rb") as f:
        uploaded = client.files.create(file=f, purpose="batch")
    print(f"     file_id: {uploaded.id}")

    # 3. バッチジョブ作成
    print("  3. バッチジョブを作成中...")
    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    print(f"     batch_id: {batch.id}")
    print(f"     status:   {batch.status}")

    print(f"\n  ✅ ジョブ投入完了！ 結果の取得は:")
    print(f"     uv run python batch_api.py result {batch.id}")
    print(f"     uv run python batch_api.py result {batch.id} --wait  # 完了まで待機")


def batch_api_result(batch_id: str, wait: bool = False):
    """バッチジョブの結果を取得・表示する"""
    import time
    from openai import OpenAI

    client = OpenAI()

    print(f"  バッチジョブ取得中... (batch_id: {batch_id})")
    batch = client.batches.retrieve(batch_id)
    print(f"  ステータス: {batch.status}")

    if wait and batch.status not in ("completed", "failed", "expired", "cancelled"):
        print("  完了を待機中...", end="", flush=True)
        while batch.status not in ("completed", "failed", "expired", "cancelled"):
            time.sleep(5)
            batch = client.batches.retrieve(batch_id)
            print(".", end="", flush=True)
        print(f"\n  最終ステータス: {batch.status}")

    if batch.status != "completed":
        print(f"  ⚠  ステータス: {batch.status}")
        if batch.status in ("failed", "expired", "cancelled") and batch.errors:
            for err in batch.errors.data:
                print(f"     エラー: {err.message}")
        elif batch.status not in ("failed", "expired", "cancelled"):
            print("  まだ処理中です。--wait を付けて再実行するか、しばらく後に再度お試しください。")
        return

    # 結果ダウンロード
    print("  結果をダウンロード中...")
    result_content = client.files.content(batch.output_file_id)
    result_file = Path("eda_output") / "batch_result.jsonl"
    result_file.parent.mkdir(exist_ok=True)
    result_file.write_text(result_content.text)
    print(f"  → {result_file}")

    # 結果表示
    print("\n  📊 結果:")
    for line in result_content.text.strip().split("\n"):
        record = json.loads(line)
        custom_id = record["custom_id"]
        choice = record["response"]["body"]["choices"][0]
        answer = choice["message"]["content"].strip()
        print(f"     {custom_id}: {answer}")

    request_counts = batch.request_counts
    print(f"\n  ✅ 完了 (total: {request_counts.total}, "
          f"completed: {request_counts.completed}, failed: {request_counts.failed})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="バッチAPIデモ")
    subparsers = parser.add_subparsers(dest="command")

    # submit サブコマンド
    sub_submit = subparsers.add_parser("submit", help="バッチジョブを投入する")

    # result サブコマンド
    sub_result = subparsers.add_parser("result", help="バッチジョブの結果を取得する")
    sub_result.add_argument("batch_id", help="OpenAI バッチジョブ ID (batch_...)")
    sub_result.add_argument(
        "--wait", action="store_true",
        help="ジョブが未完了の場合、完了までポーリングして待つ",
    )

    args = parser.parse_args()

    if args.command in ("submit", "result"):
        from dotenv import load_dotenv
        load_dotenv()

    if args.command == "submit":
        print("📦 Amazon Reviews 2023 (All_Beauty) を読み込み中...")
        reviews = load_amazon_reviews(n=200)
        print(f"   {len(reviews)} 件読み込み完了")
        batch_api_real(reviews)
    elif args.command == "result":
        batch_api_result(args.batch_id, wait=args.wait)
    else:
        # サブコマンドなし → シミュレーションモード
        print("📦 Amazon Reviews 2023 (All_Beauty) を読み込み中...")
        reviews = load_amazon_reviews(n=200)
        print(f"   {len(reviews)} 件読み込み完了")
        batch_api_demo(reviews)
