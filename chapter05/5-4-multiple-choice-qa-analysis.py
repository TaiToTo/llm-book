# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "datasets<4.0.0",
#   "transformers[ja,torch]<4.41.0",
#   "matplotlib",
#   "scikit-learn",
# ]
# ///

from transformers import BatchEncoding

def preprocess_multiple_choice(
    example: dict[str, str]
) -> BatchEncoding:
    """多肢選択式質問応答の事例をIDに変換"""
    # 選択肢の数を"choice"から始まるキーの数として算出
    num_choices = sum(
        key.startswith("choice") for key in example.keys()
    )

    # 質問と選択肢を連結してトークナイザーに渡す
    choice_list = [example[f"choice{i}"] for i in range(num_choices)]
    repeated_question_list = [example["question"]] * num_choices
    encoded_example = tokenizer(
        repeated_question_list, choice_list, max_length=64
    )

    # ラベルが入力に含まれている場合に、出力にも追加
    if "label" in example:
        encoded_example["labels"] = example["label"]
    return encoded_example

import torch
from transformers import BatchEncoding

def collate_fn_multiple_choice(
    features: list[BatchEncoding],
) -> dict[str, torch.Tensor]:
    """選択肢式質問応答の入力からミニバッチを構築"""
    # preprocess_multiple_choice関数に合わせてラベル名を"labels"とする
    label_name = "labels"

    batch_size = len(features)
    num_choices = len(features[0]["input_ids"])

    # 選択肢ごとの入力を一つのlistにまとめる
    flat_features = []
    for feature in features:
        flat_features += [
            {k: v[i] for k, v in feature.items() if k != label_name}
            for i in range(num_choices)
        ]

    # 選択肢ごとの入力についてパディングを行う
    flat_batch = tokenizer.pad(flat_features, return_tensors="pt")

    # 元のバッチごとに選択肢ごとの入力をまとめる
    # （バッチサイズ * 選択肢数, 最大系列長）の形をしたTensorを、
    # （バッチサイズ, 選択肢数, 最大系列長）に変換
    batch = {
        k: v.view(batch_size, num_choices, -1)
        for k, v in flat_batch.items()
    }

    # ラベルが入力に含まれている場合、バッチにまとめてTensorに変換
    if label_name in features[0]:
        labels = [feature[label_name] for feature in features]
        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
    return batch

from transformers import AutoModelForMultipleChoice, AutoTokenizer

model_name = "llm-book/bert-base-japanese-v3-jcommonsenseqa"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMultipleChoice.from_pretrained(model_name)

from transformers import AutoModelForMultipleChoice, AutoTokenizer

def pipeline_multiple_choice(
    examples: dict[str, str] | list[dict[str, str]]
) -> list[dict[str, str]]:
    """多肢選択式質問応答の事例について予測"""
    # 単一のdict入力が与えられたときに、listに格納する
    if isinstance(examples, dict):
        examples = [examples]

    # 事例をモデルの入力形式に変換
    encoded_examples = [
        preprocess_multiple_choice(e) for e in examples
    ]
    batch = collate_fn_multiple_choice(encoded_examples)

    # モデルが使用するデバイス上（CPU/GPU）にデータを移動
    batch = {k: v.to(model.device) for k, v in batch.items()}

    # モデルの前向き計算処理
    model_output = model.forward(**batch)

    # モデルの出力から、選択肢の文字列と予測確率を得る
    predicted_ids = model_output.logits.argmax(dim=-1).tolist()
    probs = torch.softmax(model_output.logits, dim=-1)
    predicted_probs = [ps[i].item() for ps, i in zip(probs, predicted_ids)]
    predictions = [
        {"prediction": e[f"choice{i}"], "pred_prob": p}
        for e, i, p in zip(examples, predicted_ids, predicted_probs)
    ]

    return predictions

from datasets import load_dataset

valid_dataset = load_dataset(
    "llm-book/JGLUE", name="JCommonsenseQA", split="validation"
)

model = model.to("cuda:0")

from tqdm import tqdm

# ラベル名の情報を取得するためのClassLabelインスタンス
class_label = valid_dataset.features["label"]

results: list[dict[str, float | str]] = []
for i, example in tqdm(enumerate(valid_dataset)):
    # モデルの予測結果を取得
    model_prediction = pipeline_multiple_choice(example)[0]
    # 正解の文字列を取得
    true_label = example["label"]
    correct_answer = example[f"choice{true_label}"]
    # resultsに分析に必要な情報を格納
    results.append(
        {
            "example_id": i,
            "pred_prob": model_prediction["pred_prob"],
            "prediction": model_prediction["prediction"],
            "correct_answer": correct_answer,
        }
    )

# 予測が誤った事例を収集
failed_results = [
    res for res in results if res["prediction"] != res["correct_answer"]
]
# モデルの予測確率が高い順にソート
sorted_failed_results = sorted(
    failed_results, key=lambda x: -x["pred_prob"]
)
# 高い確率で予測しながら誤った事例の上位を表示
for top_result in sorted_failed_results[:5]:
    question = valid_dataset[top_result["example_id"]]["question"]

    print(f"問題：{question}")

    print(f"予測：{top_result['prediction']}")
    print(f"正解：{top_result['correct_answer']}")
    print(f"予測確率: {top_result['pred_prob']:.4f}")
    print("----------------")
