"""LLM API推論デモ — Structured Output + 逐次 vs asyncio"""

import asyncio
import os
import time

from util import part, timer, load_amazon_reviews, extract_texts


SYSTEM_PROMPT = (
    "You are a sentiment analysis assistant. "
    "Classify the sentiment of the given product review."
)


def api_inference_demo(reviews: list[dict]):
    part(7, "LLM API推論デモ — Structured Output + 逐次 vs asyncio")

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("""
  ⚠  OPENAI_API_KEY が設定されていません。
     シミュレーションモードで実行します。

     実際に試すには:
       export OPENAI_API_KEY="sk-..."
       uv run python api_inference.py
""")
        _simulated(reviews)
        return

    _real(reviews, api_key)


# ─── シミュレーション ────────────────────────────────────────────

def _simulated(reviews: list[dict]):
    """API呼び出しをシミュレーション（ネットワーク遅延を time.sleep で再現）"""
    texts = extract_texts(reviews[:20], max_len=200)
    n = len(texts)
    latency = 0.15

    print("  ■ Structured Output とは")
    print("""
    LLM の応答を Pydantic モデルで定義した JSON スキーマに従わせる機能。
    自由テキストではなく、型安全なオブジェクトとして結果を受け取れる。

    from pydantic import BaseModel

    class SentimentResult(BaseModel):
        sentiment: Literal["positive", "negative", "neutral"]
        confidence: float
        summary: str

    → response_format=SentimentResult で呼び出すと
      パース済みの SentimentResult オブジェクトが返る
""")

    print(f"  シミュレーション: {n} 件 / 1回あたり {latency} 秒の遅延を想定\n")

    # --- 逐次 ---
    print(f"  --- 逐次API呼び出し ({n} 件) ---")
    with timer(f"逐次 ({n} 件)") as t_seq:
        for _ in texts:
            time.sleep(latency)

    # --- asyncio ---
    print(f"\n  --- asyncio 並行API呼び出し ({n} 件, 同時実行数=5) ---")

    async def fake_api_call(text: str, sem: asyncio.Semaphore):
        async with sem:
            await asyncio.sleep(latency)
            return {"sentiment": "positive", "confidence": 0.92, "summary": "Good product"}

    async def run_async():
        sem = asyncio.Semaphore(5)
        tasks = [fake_api_call(t, sem) for t in texts]
        return await asyncio.gather(*tasks)

    with timer(f"asyncio ({n} 件)") as t_async:
        asyncio.run(run_async())

    speedup = t_seq.elapsed / t_async.elapsed if t_async.elapsed > 0 else float("inf")
    print(f"\n  📊 速度比較 (シミュレーション):")
    print(f"     逐次:    {t_seq.elapsed:.3f} 秒")
    print(f"     asyncio: {t_async.elapsed:.3f} 秒")
    print(f"     高速化:  {speedup:.1f}x")
    _print_takeaway()


# ─── 実 API ──────────────────────────────────────────────────────

def _real(reviews: list[dict], api_key: str):
    """実際の OpenAI API で Structured Output + 逐次 vs asyncio を比較"""
    try:
        from openai import OpenAI, AsyncOpenAI
        from pydantic import BaseModel
    except ImportError:
        print("  openai / pydantic が未インストールです。 uv add openai pydantic で追加してください。")
        return

    from typing import Literal

    class SentimentResult(BaseModel):
        sentiment: Literal["positive", "negative", "neutral"]
        confidence: float
        summary: str

    texts = extract_texts(reviews[:10], max_len=200)
    n = len(texts)

    client = OpenAI(api_key=api_key)
    aclient = AsyncOpenAI(api_key=api_key)

    print("  ■ Structured Output — Pydantic モデル定義:")
    print("""
    class SentimentResult(BaseModel):
        sentiment: Literal["positive", "negative", "neutral"]
        confidence: float
        summary: str
""")

    def classify_sync(text: str) -> SentimentResult:
        resp = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text},
            ],
            response_format=SentimentResult,
        )
        return resp.choices[0].message.parsed

    async def classify_async(text: str, sem: asyncio.Semaphore) -> SentimentResult:
        async with sem:
            resp = await aclient.beta.chat.completions.parse(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": text},
                ],
                response_format=SentimentResult,
            )
            return resp.choices[0].message.parsed

    # --- 逐次 ---
    print(f"  --- 逐次API呼び出し ({n} 件) ---")
    with timer(f"逐次 ({n} 件)") as t_seq:
        results_seq = [classify_sync(t) for t in texts]

    print(f"  サンプル結果: {results_seq[0]}")

    # --- asyncio ---
    print(f"\n  --- asyncio 並行API呼び出し ({n} 件, 同時実行数=5) ---")

    async def run_async():
        sem = asyncio.Semaphore(5)
        tasks = [classify_async(t, sem) for t in texts]
        return await asyncio.gather(*tasks)

    with timer(f"asyncio ({n} 件)") as t_async:
        results_async = asyncio.run(run_async())

    print(f"  サンプル結果: {results_async[0]}")

    speedup = t_seq.elapsed / t_async.elapsed if t_async.elapsed > 0 else float("inf")
    print(f"\n  📊 速度比較:")
    print(f"     逐次:    {t_seq.elapsed:.3f} 秒")
    print(f"     asyncio: {t_async.elapsed:.3f} 秒")
    print(f"     高速化:  {speedup:.1f}x")
    _print_takeaway()


def _print_takeaway():
    print("""
  💡 ポイント:
     - Structured Output: LLMの応答を Pydantic モデルで型安全に受け取れる
       → 後続処理でパースエラーを気にしなくてよい
     - API呼び出しは通信待ちが大部分 → asyncio で待ち時間を重ね合わせて高速化
     - Semaphore でレート制限を守りつつ並行化する
""")


if __name__ == "__main__":
    print("📦 Amazon Reviews 2023 (All_Beauty) を読み込み中...")
    reviews = load_amazon_reviews(n=200)
    print(f"   {len(reviews)} 件読み込み完了")
    api_inference_demo(reviews)
