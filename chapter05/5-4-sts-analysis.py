# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "datasets<4.0.0",
#   "transformers[ja,torch]<4.41.0",
#   "matplotlib",
#   "japanize-matplotlib",
# ]
# ///

from transformers import pipeline

model_name = "llm-book/bert-base-japanese-v3-jsts"
text_sim_pipeline = pipeline(
    model="llm-book/bert-base-japanese-v3-jsts",
    function_to_apply="none",  # 出力に適用する関数の指定
    device="cuda:0"
)

from datasets import load_dataset

valid_dataset = load_dataset(
    "llm-book/JGLUE", name="JSTS", split="validation"
)

from tqdm import tqdm

# ラベル名の情報を取得するためのClassLabelインスタンス
class_label = valid_dataset.features["label"]

results: list[dict[str, float | str]] = []
for i, example in tqdm(enumerate(valid_dataset)):
    # モデルの予測結果を取得
    model_prediction = text_sim_pipeline({"text": example["sentence1"], "text_pair": example["sentence2"]})

    # resultsに分析に必要な情報を格納
    results.append(
        {
            "example_id": i,
            "pred_score": model_prediction["score"],
            "true_score": example["label"],
        }
    )

import japanize_matplotlib
import matplotlib.pyplot as plt

plt.rcParams["font.size"] = 16  # 文字サイズを設定

plt.scatter(
    [i["true_score"] for i in results],
    [i["pred_score"] for i in results],
    alpha=0.5
)
plt.xlabel("正解スコア")
plt.ylabel("予測スコア")
plt.show()

# モデルの予測誤差が高い順にソート
sorted_results = sorted(
    results, key=lambda x: -abs(x["true_score"] - x["pred_score"])
)
# 高い確率で予測しながら誤った事例の上位を表示
for top_result in sorted_results[:5]:
    sentence1 = valid_dataset[top_result["example_id"]]["sentence1"]
    sentence2 = valid_dataset[top_result["example_id"]]["sentence2"]

    print(f"テキスト１：{sentence1}")
    print(f"テキスト２：{sentence2}")

    print(f"予測スコア：{top_result['pred_score']}")
    print(f"正解スコア：{top_result['true_score']}")
    # print(f"予測確率: {top_result['pred_prob']:.4f}")
    print("----------------")
