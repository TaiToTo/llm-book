"""セクション 7: LLM API推論デモ — 逐次 vs asyncio"""

import asyncio
import os
import time

from util import section, timer, load_amazon_reviews, extract_texts


def section_7_api_inference_demo(reviews: list[dict]):
    section(7, "LLM API推論デモ — 逐次 vs asyncio")

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("""
  ⚠  OPENAI_API_KEY が設定されていません。
     このセクションのデモは実際の API 呼び出しの代わりに
     シミュレーションで実行します。

     実際に試すには:
       export OPENAI_API_KEY="sk-..."
       uv run python section7_api_inference.py
""")
        _simulated(reviews)
        return

    _real(reviews, api_key)


def _simulated(reviews: list[dict]):
    """API呼び出しをシミュレーション（ネットワーク遅延を time.sleep で再現）"""
    texts = extract_texts(reviews[:20], max_len=200)
    n = len(texts)
    latency = 0.15

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
            return "sentiment: positive"

    async def run_async():
        sem = asyncio.Semaphore(5)
        tasks = [fake_api_call(t, sem) for t in texts]
        return await asyncio.gather(*tasks)

    with timer(f"asyncio ({n} 件)") as t_async:
        asyncio.run(run_async())

    speedup = t_seq.elapsed / t_async.elapsed if t_async.elapsed > 0 else float("inf")
    print(f"\n  📊 速度比較 (シミュレーション):")
    print(f"     逐次:   {t_seq.elapsed:.3f} 秒")
    print(f"     asyncio: {t_async.elapsed:.3f} 秒")
    print(f"     高速化: {speedup:.1f}x")
    print("""
  💡 ポイント:
     API呼び出しは通信待ちが大部分。
     asyncio で「待ち時間を重ね合わせる」ことで大幅に高速化。

     ※ 実際のAPIではレート制限があるので、
       Semaphore で同時実行数を制御することが重要。
""")


def _real(reviews: list[dict], api_key: str):
    """実際の OpenAI API で逐次 vs asyncio を比較"""
    try:
        from openai import OpenAI, AsyncOpenAI
    except ImportError:
        print("  openai パッケージが未インストールです。 uv add openai で追加してください。")
        return

    texts = extract_texts(reviews[:10], max_len=200)
    n = len(texts)

    client = OpenAI(api_key=api_key)
    aclient = AsyncOpenAI(api_key=api_key)

    def classify_sync(text: str) -> str:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Classify the sentiment as positive, negative, or neutral. Reply with one word."},
                {"role": "user", "content": text},
            ],
            max_tokens=5,
        )
        return resp.choices[0].message.content.strip()

    async def classify_async(text: str, sem: asyncio.Semaphore) -> str:
        async with sem:
            resp = await aclient.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Classify the sentiment as positive, negative, or neutral. Reply with one word."},
                    {"role": "user", "content": text},
                ],
                max_tokens=5,
            )
            return resp.choices[0].message.content.strip()

    # --- 逐次 ---
    print(f"\n  --- 逐次API呼び出し ({n} 件) ---")
    with timer(f"逐次 ({n} 件)") as t_seq:
        results_seq = [classify_sync(t) for t in texts]
    print(f"  サンプル結果: {results_seq[:3]}")

    # --- asyncio ---
    print(f"\n  --- asyncio 並行API呼び出し ({n} 件, 同時実行数=5) ---")

    async def run_async():
        sem = asyncio.Semaphore(5)
        tasks = [classify_async(t, sem) for t in texts]
        return await asyncio.gather(*tasks)

    with timer(f"asyncio ({n} 件)") as t_async:
        results_async = asyncio.run(run_async())
    print(f"  サンプル結果: {results_async[:3]}")

    speedup = t_seq.elapsed / t_async.elapsed if t_async.elapsed > 0 else float("inf")
    print(f"\n  📊 速度比較:")
    print(f"     逐次:   {t_seq.elapsed:.3f} 秒")
    print(f"     asyncio: {t_async.elapsed:.3f} 秒")
    print(f"     高速化: {speedup:.1f}x")
    print("""
  💡 ポイント:
     API呼び出しは通信待ちなので asyncio が効く！
     Semaphore でレート制限を守りつつ並行化する。
""")


if __name__ == "__main__":
    print("📦 Amazon Reviews 2023 (All_Beauty) を読み込み中...")
    reviews = load_amazon_reviews(n=200)
    print(f"   {len(reviews)} 件読み込み完了")
    section_7_api_inference_demo(reviews)
