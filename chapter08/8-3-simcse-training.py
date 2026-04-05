# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "datasets<4.0.0",
#   "scipy",
#   "transformers[ja,torch]<4.41.0",
# ]
# ///

# 必要なパッケージをインストールする

from transformers.trainer_utils import set_seed

# 乱数のシードを設定する
set_seed(42)

from datasets import load_dataset

# Hugging Face Hubのllm-book/jawiki-sentencesのリポジトリから
# Wikipediaの文のデータを読み込み、SimCSEの訓練セットとして使用する
unsup_train_dataset = load_dataset(
    "llm-book/jawiki-sentences", split="train"
)

# 訓練セットの形式と事例数を確認する
print(unsup_train_dataset)

# 訓練セットの内容を確認する
for i, text in enumerate(unsup_train_dataset[:50]["text"]):
    print(i, text)

# 訓練セットから空行の事例を除外する
unsup_train_dataset = unsup_train_dataset.filter(
    lambda example: example["text"].strip() != ""
)

# 訓練セットをシャッフルし、最初の100万事例を取り出す
unsup_train_dataset = unsup_train_dataset.shuffle().select(
    range(1000000)
)
# パフォーマンスの低下を防ぐため、シャッフルされた状態の訓練セットを
# ディスクに書き込む
unsup_train_dataset = unsup_train_dataset.flatten_indices()

# 前処理後の訓練セットの形式と事例数を確認する
print(unsup_train_dataset)

# 前処理後の訓練セットの内容を確認する
for i, text in enumerate(unsup_train_dataset[:10]["text"]):
    print(i, text)

# Hugging Face Hubのllm-book/JGLUEのリポジトリから
# JSTSデータセットの訓練セットと検証セットを読み込み、
# それぞれをSimCSEの検証セットとテストセットとして使用する
valid_dataset = load_dataset(
    "llm-book/JGLUE", name="JSTS", split="train"
)
test_dataset = load_dataset(
    "llm-book/JGLUE", name="JSTS", split="validation"
)

from transformers import AutoTokenizer

# Hugging Face Hubにおけるモデル名を指定する
base_model_name = "cl-tohoku/bert-base-japanese-v3"
# モデル名からトークナイザを初期化する
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

import torch
from torch import Tensor
from transformers import BatchEncoding

def unsup_train_collate_fn(
    examples: list[dict],
) -> dict[str, BatchEncoding | Tensor]:
    """教師なしSimCSEの訓練セットのミニバッチを作成"""
    # ミニバッチに含まれる文にトークナイザを適用する
    tokenized_texts = tokenizer(
        [example["text"] for example in examples],
        padding=True,
        truncation=True,
        max_length=32,
        return_tensors="pt",
    )

    # 文と文の類似度行列における正例ペアの位置を示すTensorを作成する
    # 行列のi行目の事例（文）に対してi列目の事例（文）との組が正例ペアとなる
    labels = torch.arange(len(examples))

    return {
        "tokenized_texts_1": tokenized_texts,
        "tokenized_texts_2": tokenized_texts,
        "labels": labels,
    }

def eval_collate_fn(
    examples: list[dict],
) -> dict[str, BatchEncoding | Tensor]:
    """SimCSEの検証・テストセットのミニバッチを作成"""
    # ミニバッチの文ペアに含まれる文（文1と文2）のそれぞれに
    # トークナイザを適用する
    tokenized_texts_1 = tokenizer(
        [example["sentence1"] for example in examples],
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    tokenized_texts_2 = tokenizer(
        [example["sentence2"] for example in examples],
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )

    # 文1と文2の類似度行列における正例ペアの位置を示すTensorを作成する
    # 行列のi行目の事例（文1）に対して
    # i列目の事例（文2）との組が正例ペアとなる
    labels = torch.arange(len(examples))

    # データセットに付与された類似度スコアのTensorを作成する
    label_scores = torch.tensor(
        [example["label"] for example in examples]
    )

    return {
        "tokenized_texts_1": tokenized_texts_1,
        "tokenized_texts_2": tokenized_texts_2,
        "labels": labels,
        "label_scores": label_scores,
    }

import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from transformers.utils import ModelOutput

class SimCSEModel(nn.Module):
    """SimCSEのモデル"""

    def __init__(
        self,
        base_model_name: str,
        mlp_only_train: bool = False,
        temperature: float = 0.05,
    ):
        """モデルの初期化"""
        super().__init__()

        # モデル名からエンコーダを初期化する
        self.encoder = AutoModel.from_pretrained(base_model_name)
        # パラメータをメモリ上に隣接した形で配置
        # これを実行しない場合、モデルの保存でエラーになることがある
        for param in self.encoder.parameters():
            param.data = param.data.contiguous()
        # MLP層の次元数
        self.hidden_size = self.encoder.config.hidden_size
        # MLP層の線形層
        self.dense = nn.Linear(self.hidden_size, self.hidden_size)
        # MLP層の活性化関数
        self.activation = nn.Tanh()

        # MLP層による変換を訓練時にのみ適用するよう設定するフラグ
        self.mlp_only_train = mlp_only_train
        # 交差エントロピー損失の計算時に使用する温度
        self.temperature = temperature

    def encode_texts(self, tokenized_texts: BatchEncoding) -> Tensor:
        """エンコーダを用いて文をベクトルに変換"""
        # トークナイズされた文をエンコーダに入力する
        encoded_texts = self.encoder(**tokenized_texts)
        # モデルの最終層の出力（last_hidden_state）の
        # [CLS]トークン（0番目の位置のトークン）のベクトルを取り出す
        encoded_texts = encoded_texts.last_hidden_state[:, 0]

        # self.mlp_only_trainのフラグがTrueに設定されていて
        # かつ訓練時でない場合、MLP層の変換を適用せずにベクトルを返す
        if self.mlp_only_train and not self.training:
            return encoded_texts

        # MLP層によるベクトルの変換を行う
        encoded_texts = self.dense(encoded_texts)
        encoded_texts = self.activation(encoded_texts)

        return encoded_texts

    def forward(
        self,
        tokenized_texts_1: BatchEncoding,
        tokenized_texts_2: BatchEncoding,
        labels: Tensor,
        label_scores: Tensor | None = None,
    ) -> ModelOutput:
        """モデルの前向き計算を定義"""
        # 文ペアをベクトルに変換する
        encoded_texts_1 = self.encode_texts(tokenized_texts_1)
        encoded_texts_2 = self.encode_texts(tokenized_texts_2)

        # 文ペアの類似度行列を作成する
        sim_matrix = F.cosine_similarity(
            encoded_texts_1.unsqueeze(1),
            encoded_texts_2.unsqueeze(0),
            dim=2,
        )

        # 交差エントロピー損失を求める
        loss = F.cross_entropy(sim_matrix / self.temperature, labels)

        # 性能評価に使用するため、正例ペアに対するスコアを類似度行列から取り出す
        positive_mask = F.one_hot(labels, sim_matrix.size(1)).bool()
        positive_scores = torch.masked_select(
            sim_matrix, positive_mask
        )

        return ModelOutput(loss=loss, scores=positive_scores)

# 教師なしSimCSEのモデルを初期化する
unsup_model = SimCSEModel(base_model_name, mlp_only_train=True)

from scipy.stats import spearmanr
from transformers import EvalPrediction

def compute_metrics(p: EvalPrediction) -> dict[str, float]:
    """
    モデルが予測したスコアと評価用データのスコアの
    スピアマンの順位相関係数を計算
    """
    scores = p.predictions
    labels, label_scores = p.label_ids

    spearman = spearmanr(scores, label_scores).statistic

    return {"spearman": spearman}

from transformers import TrainingArguments

# 教師なしSimCSEの訓練のハイパーパラメータを設定する
unsup_training_args = TrainingArguments(
    output_dir="outputs_unsup_simcse",  # 結果の保存先フォルダ
    per_device_train_batch_size=64,  # 訓練時のバッチサイズ
    per_device_eval_batch_size=64,  # 評価時のバッチサイズ
    learning_rate=3e-5,  # 学習率
    num_train_epochs=1,  # 訓練エポック数
    evaluation_strategy="steps",  # 検証セットによる評価のタイミング
    eval_steps=250,  # 検証セットによる評価を行う訓練ステップ数の間隔
    logging_steps=250,  # ロギングを行う訓練ステップ数の間隔
    save_steps=250,  # チェックポイントを保存する訓練ステップ数の間隔
    save_total_limit=1,  # 保存するチェックポイントの最大数
    fp16=True,  # 自動混合精度演算の有効化
    load_best_model_at_end=True,  # 最良のモデルを訓練終了後に読み込むか
    metric_for_best_model="spearman",  # 最良のモデルを決定する評価指標
    remove_unused_columns=False,  # データセットの不要フィールドを削除するか
    report_to="none",  # 外部ツールへのログを無効化
)

from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import Trainer

class SimCSETrainer(Trainer):
    """SimCSEの訓練に使用するTrainer"""

    def get_eval_dataloader(
        self, eval_dataset: Dataset | None = None
    ) -> DataLoader:
        """
        検証・テストセットのDataLoaderでeval_collate_fnを使うように
        Trainerのget_eval_dataloaderをオーバーライド
        """
        if eval_dataset is None:
            eval_dataset = self.eval_dataset

        return DataLoader(
            eval_dataset,
            batch_size=64,
            collate_fn=eval_collate_fn,
            pin_memory=True,
        )

# 教師なしSimCSEのTrainerを初期化する
unsup_trainer = SimCSETrainer(
    model=unsup_model,
    args=unsup_training_args,
    data_collator=unsup_train_collate_fn,
    train_dataset=unsup_train_dataset,
    eval_dataset=valid_dataset,
    compute_metrics=compute_metrics,
)

# 教師なしSimCSEの訓練を行う
unsup_trainer.train()

# 検証セットで教師なしSimCSEのモデルの評価を行う
unsup_trainer.evaluate(valid_dataset)

# テストセットで教師なしSimCSEのモデルの評価を行う
unsup_trainer.evaluate(test_dataset)

# 教師なしSimCSEのエンコーダを保存
encoder_path = "outputs_unsup_simcse/encoder"
unsup_model.encoder.save_pretrained(encoder_path)
tokenizer.save_pretrained(encoder_path)

# Googleドライブをマウントする

# 保存されたモデルをGoogleドライブのフォルダにコピーする

# 乱数のシードを設定する
set_seed(42)

# Hugging Face Hubのllm-book/jsnliのリポジトリから
# JSNLIの訓練セットを読み込む
jsnli_dataset = load_dataset("llm-book/jsnli", split="train")

# JSNLIの訓練セットの形式と事例数を確認する
print(jsnli_dataset)

from pprint import pprint

# JSNLIの訓練セットの内容を確認する
pprint(jsnli_dataset[0])
pprint(jsnli_dataset[1])

import csv
import random
from typing import Iterator

# JSNLIの訓練セットから、前提文とラベルごとに仮説文をまとめたdictを作成する
premise2hypotheses = {}

premises = jsnli_dataset["premise"]  # 前提文
hypotheses = jsnli_dataset["hypothesis"]  # 仮説文
labels = jsnli_dataset["label"]  # ラベル

for premise, hypothesis, label in zip(premises, hypotheses, labels):
    if premise not in premise2hypotheses:
        premise2hypotheses[premise] = {
            "entailment": [],
            "neutral": [],
            "contradiction": [],
        }

    premise2hypotheses[premise][label].append(hypothesis)

def generate_sup_train_example() -> Iterator[dict[str, str]]:
    """教師ありSimCSEの訓練セットの事例を生成"""
    # JSNLIのデータから (前提文,「含意」ラベルの仮説文,「矛盾」ラベルの仮説文)
    # の三つ組を生成する
    for premise, hypotheses in premise2hypotheses.items():
        # 「矛盾」ラベルの仮説文が一つもない事例はスキップする
        if len(hypotheses["contradiction"]) == 0:
            continue

        # 「含意」ラベルの仮説文一つにつき、「矛盾」ラベルの仮説文一つを
        # ランダムに関連付ける
        for entailment_hypothesis in hypotheses["entailment"]:
            contradiction_hypothesis = random.choice(
                hypotheses["contradiction"]
            )
            # (前提文,「含意」ラベルの仮説文,「矛盾」ラベルの仮説文) の三つ組を
            # dictとして生成する
            yield {
                "premise": premise,
                "entailment_hypothesis": entailment_hypothesis,
                "contradiction_hypothesis": contradiction_hypothesis,
            }

# 定義したジェネレータ関数を用いて、教師ありSimCSEの訓練セットを構築する
sup_train_dataset = Dataset.from_generator(generate_sup_train_example)

# 訓練セットの形式と事例数を確認する
print(sup_train_dataset)

# 訓練セットの内容を確認する
pprint(sup_train_dataset[0])
pprint(sup_train_dataset[1])

def sup_train_collate_fn(
    examples: list[dict],
) -> dict[str, BatchEncoding | Tensor]:
    """訓練セットのミニバッチを作成"""
    premises = []
    hypotheses = []

    for example in examples:
        premises.append(example["premise"])

        entailment_hypothesis = example["entailment_hypothesis"]
        contradiction_hypothesis = example["contradiction_hypothesis"]

        hypotheses.extend(
            [entailment_hypothesis, contradiction_hypothesis]
        )

    # ミニバッチに含まれる前提文と仮説文にトークナイザを適用する
    tokenized_premises = tokenizer(
        premises,
        padding=True,
        truncation=True,
        max_length=32,
        return_tensors="pt",
    )
    tokenized_hypotheses = tokenizer(
        hypotheses,
        padding=True,
        truncation=True,
        max_length=32,
        return_tensors="pt",
    )

    # 前提文と仮説文の類似度行列における正例ペアの位置を示すTensorを作成する
    # 行列のi行目の事例（前提文）に対して
    # 2*i列目の要素（仮説文）が正例ペアとなる
    labels = torch.arange(0, 2 * len(premises), 2)

    return {
        "tokenized_texts_1": tokenized_premises,
        "tokenized_texts_2": tokenized_hypotheses,
        "labels": labels,
    }

# 教師ありSimCSEのモデルを初期化する
sup_model = SimCSEModel(base_model_name, mlp_only_train=False)

# 教師ありSimCSEの訓練のハイパーパラメータを設定する
sup_training_args = TrainingArguments(
    output_dir="outputs_sup_simcse",  # 結果の保存先フォルダ
    per_device_train_batch_size=128,  # 訓練時のバッチサイズ
    per_device_eval_batch_size=128,  # 評価時のバッチサイズ
    learning_rate=5e-5,  # 学習率
    num_train_epochs=3,  # 訓練エポック数
    evaluation_strategy="steps",  # 検証セットによる評価のタイミング
    eval_steps=250,  # 検証セットによる評価を行う訓練ステップ数の間隔
    logging_steps=250,  # ロギングを行う訓練ステップ数の間隔
    save_steps=250,  # チェックポイントを保存する訓練ステップ数の間隔
    save_total_limit=1,  # 保存するチェックポイントの最大数
    fp16=True,  # 自動混合精度演算の有効化
    load_best_model_at_end=True,  # 最良のモデルを訓練終了後に読み込むか
    metric_for_best_model="spearman",  # 最良のモデルを決定する評価指標
    remove_unused_columns=False,  # データセットの不要フィールドを削除するか
)

# 教師ありSimCSEのTrainerを初期化する
sup_trainer = SimCSETrainer(
    model=sup_model,
    args=sup_training_args,
    data_collator=sup_train_collate_fn,
    train_dataset=sup_train_dataset,
    eval_dataset=valid_dataset,
    compute_metrics=compute_metrics,
)

# 教師ありSimCSEの訓練を行う
sup_trainer.train()

# 検証セットで教師ありSimCSEのモデルの評価を行う
sup_trainer.evaluate(valid_dataset)

# テストセットで教師ありSimCSEのモデルの評価を行う
sup_trainer.evaluate(test_dataset)

# 保存されたモデルをGoogleドライブのフォルダにコピーする
