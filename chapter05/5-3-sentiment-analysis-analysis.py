# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "datasets<4.0.0",
#   "transformers[ja,torch]<4.41.0",
#   "matplotlib",
#   "scikit-learn",
# ]
# ///

from transformers import pipeline

model_name = "llm-book/bert-base-japanese-v3-marc_ja"
marc_ja_pipeline = pipeline(model=model_name, device="cuda:0")

from datasets import load_dataset

valid_dataset = load_dataset(
    "llm-book/JGLUE", name="MARC-ja", split="validation"
)

from tqdm import tqdm

# ラベル名の情報を取得するためのClassLabelインスタンス
class_label = valid_dataset.features["label"]

results: list[dict[str, float | str]] = []
for i, example in tqdm(enumerate(valid_dataset)):
    # モデルの予測結果を取得
    model_prediction = marc_ja_pipeline(example["sentence"])[0]
    # 正解のラベルIDをラベル名に変換
    true_label = class_label.int2str(example["label"])

    # resultsに分析に必要な情報を格納
    results.append(
        {
            "example_id": i,
            "pred_prob": model_prediction["score"],
            "pred_label": model_prediction["label"],
            "true_label": true_label,
        }
    )

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

plt.rcParams["font.size"] = 18  # 文字サイズを大きくする

# 混同行列の作成
confusion_matrix = confusion_matrix(
    y_true=[result["true_label"] for result in results],
    y_pred=[result["pred_label"] for result in results],
    labels=class_label.names,
)
# 混同行列を画像として表示
ConfusionMatrixDisplay(
    confusion_matrix, display_labels=class_label.names
).plot()

# 予測が誤った事例を収集
failed_results = [
    res for res in results if res["pred_label"] != res["true_label"]
]
# モデルの予測確率が高い順にソート
sorted_failed_results = sorted(
    failed_results, key=lambda x: -x["pred_prob"]
)
# 高い確率で予測しながら誤った事例の上位2件を表示
for top_result in sorted_failed_results[:2]:
    review_text = valid_dataset[top_result["example_id"]]["sentence"]
    print(f"レビュー文：{review_text}")
    print(f"予測：{top_result['pred_label']}")
    print(f"正解：{top_result['true_label']}")
    print(f"予測確率: {top_result['pred_prob']:.4f}")
    print("----------------")

text = "まず，紙ジャケット仕様とありますが，正確には紙ケースです．そのケースはちょうどCD２枚組用のハードケースがぴったり入りそうな紙ケースなのですが，その中にケースより一回り小さな写真集と歌詞カードが入っています．中がすかすかなんですが，私だけでしょうか？これで3780円ですか？？？他のシリーズが出来が良いだけに，なんか残念で☆４つにさせて頂きました"

print(marc_ja_pipeline(text)[0])

print(marc_ja_pipeline(text.replace("☆４つ", ""))[0])

print(marc_ja_pipeline("絶対に買わないで。最悪です。")[0])

print(marc_ja_pipeline("絶対に買わないで。最悪です。星４つ。")[0])
