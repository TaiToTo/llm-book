# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "transformers[ja,torch]<4.41.0",
#   "datasets<4.0.0",
#   "matplotlib",
#   "japanize-matplotlib",
# ]
# ///

from transformers.trainer_utils import set_seed

# 乱数シードを42に固定
set_seed(42)

from pprint import pprint
from datasets import load_dataset

# Hugging Face Hub上のllm-book/JGLUEのリポジトリから
# JCommonsenseQAのデータを読み込む
train_dataset = load_dataset(
    "llm-book/JGLUE", name="JCommonsenseQA", split="train"
)
valid_dataset = load_dataset(
    "llm-book/JGLUE", name="JCommonsenseQA", split="validation"
)
# pprintで見やすく表示する
pprint(train_dataset[0])

pprint(train_dataset.features)

from transformers import AutoTokenizer

# Hugging Face Hub上のモデル名を指定
model_name = "cl-tohoku/bert-base-japanese-v3"
# モデル名からトークナイザを読み込む
tokenizer = AutoTokenizer.from_pretrained(model_name)

from collections import Counter
import japanize_matplotlib
import matplotlib.pyplot as plt
from datasets import Dataset
from tqdm import tqdm

plt.rcParams["font.size"] = 18  # 文字サイズを大きくする

def visualize_question_length(dataset: Dataset):
    """データセット中の質問のトークン数の分布をグラフとして描画"""
    # データセット中の質問の長さを数える
    length_counter = Counter()
    for data in tqdm(dataset):
        length = len(tokenizer.tokenize(data["question"]))
        length_counter[length] += 1

    # length_counterの値から棒グラフを描画する
    plt.bar(length_counter.keys(), length_counter.values(), width=1.0)
    plt.xlabel("トークン数")
    plt.ylabel("事例数")
    plt.show()

visualize_question_length(train_dataset)
visualize_question_length(valid_dataset)

def visualize_choice_length(dataset: Dataset):
    """データセット中の選択肢のトークン数の分布をグラフとして描画"""
    # データセット中の選択肢の長さを数える
    length_counter = Counter()
    for data in tqdm(dataset):
        for i in range(5):
            length = len(tokenizer.tokenize(data[f"choice{i}"]))
            length_counter[length] += 1

    # length_counterの値から棒グラフを描画する
    plt.bar(length_counter.keys(), length_counter.values(), width=1.0)
    plt.xlabel("トークン数")
    plt.ylabel("事例数")
    plt.show()

visualize_choice_length(train_dataset)
visualize_choice_length(valid_dataset)

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

from transformers import AutoTokenizer

example = train_dataset[0]
encoded_example = preprocess_multiple_choice(example)  # 事例の前処理
for choice in range(5):
    # IDから元の文字列を復元
    print(tokenizer.decode(encoded_example["input_ids"][choice]))

encoded_train_dataset = train_dataset.map(
    preprocess_multiple_choice,
    remove_columns=train_dataset.column_names,
)
encoded_valid_dataset = valid_dataset.map(
    preprocess_multiple_choice,
    remove_columns=valid_dataset.column_names,
)

print(encoded_train_dataset[0])

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

batch_size = 4
encoded_examples = [
    preprocess_multiple_choice(train_dataset[i])
    for i in range(batch_size)
]
batch = collate_fn_multiple_choice(encoded_examples)
pprint({name: tensor.size() for name, tensor in batch.items()})

from transformers import AutoModelForMultipleChoice

model = AutoModelForMultipleChoice.from_pretrained(
    model_name,
    num_labels=train_dataset.features["label"].num_classes,
)

# パラメータをメモリ上に隣接した形で配置
# これを実行しない場合、モデルの保存でエラーになることがある
for param in model.parameters():
    param.data = param.data.contiguous()

model(**batch)

from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="output_jcommonsenseqa",  # 結果の保存フォルダ
    per_device_train_batch_size=32,  # 訓練時のバッチサイズ
    per_device_eval_batch_size=32,  # 評価時のバッチサイズ
    learning_rate=2e-5,  # 学習率
    lr_scheduler_type="linear",  # 学習率スケジューラの種類
    warmup_ratio=0.1,  # 学習率のウォームアップの長さを指定
    num_train_epochs=3,  # エポック数
    save_strategy="epoch",  # チェックポイントの保存タイミング
    logging_strategy="epoch",  # ロギングのタイミング
    evaluation_strategy="epoch",  # 検証セットによる評価のタイミング
    load_best_model_at_end=True,  # 訓練後に開発セットで最良のモデルをロード
    metric_for_best_model="accuracy",  # 最良のモデルを決定する評価指標
    fp16=True,  # 自動混合精度演算の有効化
    report_to="none",  # 外部ツールへのログを無効化
)

import numpy as np

def compute_accuracy(
    eval_pred: tuple[np.ndarray, np.ndarray]
) -> dict[str, float]:
    """予測ラベルと正解ラベルから正解率を計算"""
    predictions, labels = eval_pred
    # predictionsは各ラベルについてのスコア
    # 最もスコアの高いインデックスを予測ラベルとする
    predictions = np.argmax(predictions, axis=1)
    return {"accuracy": (predictions == labels).mean()}

from transformers import Trainer

trainer = Trainer(
    model=model,
    train_dataset=encoded_train_dataset,
    eval_dataset=encoded_valid_dataset,
    data_collator=collate_fn_multiple_choice,
    args=training_args,
    compute_metrics=compute_accuracy,
)
trainer.train()

# 検証セットでモデルを評価
eval_metrics = trainer.evaluate(encoded_valid_dataset)
pprint(eval_metrics)

# Googleドライブをマウントする

# 保存されたモデルをGoogleドライブのフォルダにコピーする

from huggingface_hub import login

login()

# Hugging Face Hubのリポジトリ名
# "YOUR-ACCOUNT"は自らのユーザ名に置き換えてください
repo_name = "YOUR-ACCOUNT/bert-base-japanese-v3-output_jcommonsenseqa"
# トークナイザとモデルをアップロード
tokenizer.push_to_hub(repo_name)
model.push_to_hub(repo_name)
