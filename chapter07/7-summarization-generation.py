# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "datasets<4.0.0",
#   "transformers[ja,torch]<4.41.0",
#   "sentencepiece",
#   "japanize-matplotlib",
#   "mecab-python3",
#   "rouge-score",
#   "sacrebleu",
#   "bert_score",
# ]
# ///

# # 第7章 要約生成

# ## 7.2 データセット

# #### 準備

# #### データセットのダウンロード

from datasets import load_dataset

# データセットを読み込む
dataset = load_dataset("llm-book/livedoor-news-corpus")

# データセットの形式と事例数を確認する
print(dataset)

from pprint import pprint

# 訓練セットの最初の二つの事例を表示する
pprint(list(dataset["train"])[:2])

# #### データセットの分析

from collections import Counter

# 各カテゴリの事例数を確認する
pprint(Counter(dataset["train"]["category"]).most_common())

categories = set() # カテゴリの集合
for data in dataset["train"]: # 訓練セットの各事例を処理する
    category, title = data["category"], data["title"]
    # すでに出現したカテゴリはスキップする
    if category not in categories:
        categories.add(category)
        print(f"{category}: {title}")

from collections import Counter
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizer
import japanize_matplotlib

# フォントサイズを18にする
plt.rcParams["font.size"] = 18

def visualize_num_tokens_distribution(
    dataset: Dataset, tokenizer: PreTrainedTokenizer, column: str
) -> None:
    """トークン数の分布を可視化"""
    # 各事例でトークン数をカウントし、トークン数ごとに結果を集約する
    counter = Counter()
    for data in tqdm(dataset):
        num_tokens = len(tokenizer.tokenize(data[column]))
        counter[num_tokens] += 1

    # トークン数の分布を可視化する
    plt.bar(counter.keys(), counter.values(), width=1.0)
    plt.xlabel("トークン数")
    plt.ylabel("出現頻度")
    plt.gca().yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter("%d"))
    plt.show()

# トークナイザを読み込む
model_name = "retrieva-jp/t5-base-long"
tokenizer = AutoTokenizer.from_pretrained(model_name)
# 記事のトークン数の分布を可視化する
visualize_num_tokens_distribution(
    dataset["train"], tokenizer, "content"
)
# 見出しのトークン数の分布を可視化する
visualize_num_tokens_distribution(
    dataset["train"], tokenizer, "title"
)

# ## 7.3 評価指標

# #### 準備

# ### 7.3.1 ROUGE

# #### ROUGEを算出するための実装

reference = "日本語T5モデルの公開"
prediction1 = "T5モデルの日本語版を公開"
prediction2 = "日本語T5をリリース"
prediction3 = "Japanese T5を発表"

import ipadic
import MeCab

# IPAdicを用いたMeCabを使用して、単語分割を行う
tagger = MeCab.Tagger(f"-O wakati {ipadic.MECAB_ARGS}")
ref_wakati = tagger.parse(reference).strip()
pred_wakati1 = tagger.parse(prediction1).strip()
pred_wakati2 = tagger.parse(prediction2).strip()
pred_wakati3 = tagger.parse(prediction3).strip()
print(f"参照文: {ref_wakati}")
print(f"生成文1: {pred_wakati1}")
print(f"生成文2: {pred_wakati2}")
print(f"生成文3: {pred_wakati3}")

from collections import defaultdict
import pandas as pd
from rouge_score import rouge_scorer, scoring

# pandasの小数点以下の桁数を3に設定する
pd.options.display.precision = 3

def convert_words_to_ids(
    predictions: list[str], references: list[str]
) -> tuple[list[str], list[str]]:
    """単語列をID列に変換"""
    # 単語にユニークなIDを割り当てるためのdefaultdictを作成する
    word2id = defaultdict(lambda: len(word2id))

    # 単語区切りの文字列をID文字列に変換する
    pred_ids = [
        " ".join([str(word2id[w]) for w in p.split()])
        for p in predictions
    ]
    ref_ids = [
        " ".join([str(word2id[w]) for w in r.split()])
        for r in references
    ]
    return pred_ids, ref_ids

def compute_rouge(
    predictions: list[str], references: list[str]
) -> dict[str, dict[str, float]]:
    """ROUGEを算出"""
    # RougeScorerを設定する
    rouge = rouge_scorer.RougeScorer(
        rouge_types=["rouge1", "rouge2", "rougeL"], use_stemmer=False
    )
    aggregator = scoring.BootstrapAggregator()
    # 単語列をID列に変換する
    pred_ids, ref_ids = convert_words_to_ids(predictions, references)
    # ROUGEスコアを計算して結果を集約する
    for pred, ref in zip(pred_ids, ref_ids):
        aggregator.add_scores(rouge.score(ref, pred))
    scores = aggregator.aggregate()
    return {k: v.mid for k, v in scores.items()}

# ROUGEを算出した結果を表示する
rouge_results = {
    "生成文1": compute_rouge([pred_wakati1], [ref_wakati]),
    "生成文2": compute_rouge([pred_wakati2], [ref_wakati]),
    "生成文3": compute_rouge([pred_wakati3], [ref_wakati]),
}
df_list = [
    pd.DataFrame.from_dict(rouge_results[k], orient="index")
    for k in rouge_results.keys()
]
display(pd.concat(df_list, keys=rouge_results.keys(), axis=1).T)

# ### 7.3.2 BLEU

# #### BLEUを算出するための実装

from sacrebleu import corpus_bleu

def compute_bleu(
    predictions: list[str], references: list[list[str]]
) -> dict[str, int | float | list[float]]:
    """BLUEを算出"""
    # BLUEを算出する
    result = corpus_bleu(predictions, references)
    return {
            "score": result.score,
            "counts": result.counts,
            "totals": result.totals,
            "precisions": [round(p, 2) for p in result.precisions],
            "bp": result.bp,
            "sys_len": result.sys_len,
            "ref_len": result.ref_len,
    }

# BLEUを算出した結果を表示する
bleu_results = {
    "生成文1": compute_bleu([pred_wakati1], [[ref_wakati]]),
    "生成文2": compute_bleu([pred_wakati2], [[ref_wakati]]),
    "生成文3": compute_bleu([pred_wakati3], [[ref_wakati]]),
}
df_list = [
    pd.DataFrame.from_dict(bleu_results[k], orient="index")[0]
    for k in bleu_results.keys()
]
display(pd.concat(df_list, keys=bleu_results.keys(), axis=1).T)

# ### 7.3.3 BERTScore

from bert_score import plot_example

# 生成文3と参照文の類似度行列を作成する
plot_example(prediction3, reference, lang="ja")

# #### BERTScoreを算出するための実装

import bert_score

def compute_bertscore(
    predictions: list[str], references: list[str]
) -> dict[str, float]:
    """BERTScoreを算出"""
    # BERTScoreを計算する
    scorer = bert_score.BERTScorer(
        model_type=bert_score.utils.lang2model["ja"],
    )
    p, r, f = scorer.score(
        cands=predictions, refs=references
    )
    return {
        "precision": sum(p.tolist()) / len(p),
        "recall": sum(r.tolist()) / len(r),
        "f1": sum(f.tolist()) / len(f)
    }

# BERTScoreを算出した結果を表示する
bertscore_results = {
    "生成文1": compute_bertscore([prediction1], [reference]),
    "生成文2": compute_bertscore([prediction2], [reference]),
    "生成文3": compute_bertscore([prediction3], [reference]),
}
df_list = [
    pd.DataFrame.from_dict(bertscore_results[k], orient="index")[0]
    for k in bertscore_results.keys()
]
display(pd.concat(df_list, keys=bertscore_results.keys(), axis=1).T)

# ## 7.4 見出し生成モデルの実装

# ### 7.4.1 T5のファインチューニング

# #### データセットの前処理

from typing import Any
from transformers import BatchEncoding, PreTrainedTokenizer

def preprocess_data(
    data: dict[str, Any], tokenizer: PreTrainedTokenizer
) -> BatchEncoding:
    """データの前処理"""
    # 記事のトークナイゼーションを行う
    inputs = tokenizer(
        data["content"], max_length=512, truncation=True
    )
    # 見出しのトークナイゼーションを行う
    # 見出しはトークンIDのみ使用する
    inputs["labels"] = tokenizer(
        data["title"], max_length=128, truncation=True
    )["input_ids"]
    return inputs

# 訓練セットに対して前処理を行う
train_dataset = dataset["train"].map(
    preprocess_data,
    fn_kwargs={"tokenizer": tokenizer},
    remove_columns=dataset["train"].column_names,
)
# 検証セットに対して前処理を行う
validation_dataset = dataset["validation"].map(
    preprocess_data,
    fn_kwargs={"tokenizer": tokenizer},
    remove_columns=dataset["validation"].column_names,
)

# #### モデルのファインチューニング

from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq

# モデルを読み込む
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
# collate関数にDataCollatorForSeq2Seqを用いる
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers.trainer_utils import set_seed

# 乱数シードを42に固定する
set_seed(42)

# Trainerに渡す引数を初期化する
training_args = Seq2SeqTrainingArguments(
    output_dir="output_t5_summarization", # 結果の保存フォルダ
    per_device_train_batch_size=8, # 訓練時のバッチサイズ
    per_device_eval_batch_size=8, # 評価時のバッチサイズ
    learning_rate=1e-4, # 学習率
    lr_scheduler_type="linear", # 学習率スケジューラ
    warmup_ratio=0.1, # 学習率のウォームアップ
    num_train_epochs=5, # 訓練エポック数
    evaluation_strategy="epoch", # 評価タイミング
    save_strategy="epoch", # チェックポイントの保存タイミング
    logging_strategy="epoch", # ロギングのタイミング
    load_best_model_at_end=True, # 訓練後に検証セットで最良のモデルをロード
    report_to="none",  # 外部ツールへのログを無効化
)

# Trainerを初期化する
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    tokenizer=tokenizer,
)

# 訓練する
trainer.train()

# Google ドライブをマウントする

# 保存されたモデルをGoogleドライブのフォルダにコピーする

# ### 7.4.2 見出しの生成とモデルの評価

# モデルを読み込む
model_name = "llm-book/t5-base-long-livedoor-news-corpus"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to("cuda:0")
# パラメータをメモリ上に隣接した形で配置
# これを実行しない場合、モデルの保存でエラーになることがある
for param in model.parameters():
    param.data = param.data.contiguous()

# #### 見出しの生成

from torch.utils.data import DataLoader
from transformers import PreTrainedModel

def convert_list_dict_to_dict_list(
    list_dict: dict[str, list]
) -> list[dict[str, list]]:
    """ミニバッチのデータを事例単位のlistに変換"""
    dict_list = []
    # dictのキーのlistを作成する
    keys = list(list_dict.keys())
    for idx in range(len(list_dict[keys[0]])):  # 各事例で処理する
        # dictの各キーからデータを取り出してlistに追加する
        dict_list.append({key: list_dict[key][idx] for key in keys})
    return dict_list

def run_generation(
    dataloader: DataLoader, model: PreTrainedModel
) -> list[dict[str, Any]]:
    """見出しを生成"""
    generations = []
    for batch in tqdm(dataloader):  # 各ミニバッチを処理する
        batch = {k: v.to(model.device) for k, v in batch.items() if k != "labels"}
        # 見出しのトークンのIDを生成する
        batch["generated_title_ids"] = model.generate(**batch)
        batch = {k: v.cpu().tolist() for k, v in batch.items()}
        # ミニバッチのデータを事例単位のlistに変換する
        generations += convert_list_dict_to_dict_list(batch)
    return generations

# テストセットに対して前処理を行う
test_dataset = dataset["test"].map(
    preprocess_data,
    fn_kwargs={"tokenizer": tokenizer},
    remove_columns=dataset["test"].column_names,
)
test_dataset = test_dataset.remove_columns(["labels"])
# ミニバッチの作成にDataLoaderを用いる
test_dataloader = DataLoader(
    test_dataset,
    batch_size=8,
    shuffle=False,
    collate_fn=data_collator,
)
# 見出しを生成する
generations = run_generation(test_dataloader, model)

# 生成した見出しのトークンのIDのlistをトークンのlistに変換する
tokens = tokenizer.convert_ids_to_tokens(
    generations[0]["generated_title_ids"]
)
print(tokens)

def postprocess_title(
    generations: list[dict[str, Any]],
    dataset: list[dict[str, Any]],
    tokenizer: PreTrainedTokenizer,
):
    """見出しの後処理"""
    results = []
    # 各事例を処理する
    for generation, data in zip(generations, dataset):
        # IDのlistをテキストに変換する
        data["generated_title"] = tokenizer.decode(
            generation["generated_title_ids"],
            skip_special_tokens=True,
        )
        results.append(data)
    return results

# 見出しテキストを生成する
results = postprocess_title(generations, dataset["test"], tokenizer)
print(results[0]["generated_title"])

# #### モデルの評価

# ROUGEを算出して表示する
generated_titles = [
    tagger.parse(r["generated_title"]).strip() for r in results
]
ref_titles = [tagger.parse(r["title"]).strip() for r in results]
rouge_results = compute_rouge(generated_titles, ref_titles)
display(pd.DataFrame.from_dict(rouge_results, orient="index"))

# BLEUを算出して表示する
generated_titles = [
    tagger.parse(r["generated_title"]).strip() for r in results
]
ref_titles = [[tagger.parse(r["title"]).strip() for r in results]]
bleu_results = compute_bleu(generated_titles, ref_titles)
display(pd.DataFrame([bleu_results]).rename(index={0: "BLEU"}).T)

# BERTScoreを算出して表示する
generated_titles = [r["generated_title"].strip() for r in results]
ref_titles = [r["title"].strip() for r in results]
bertscore_results = compute_bertscore(generated_titles, ref_titles)
display(
    pd.DataFrame([bertscore_results]).rename(index={0: "BERTScore"})
)

# DataFrameのセル内の最大出力文字数を指定する
pd.options.display.max_colwidth = 500
# 記事、見出し、生成した見出しを表示する
display(
    pd.DataFrame(results)[:3][["content", "title", "generated_title"]]
)

# ## 7.5 多様な生成方法による見出し生成

content = dataset["test"][434]["content"]
title = dataset["test"][434]["title"]
print(f"記事: {content}")
print(f"見出し: {title}")

from functools import partial
from transformers import pipeline

# 乱数シードを42に再設定する
set_seed(42)

# モデルを固定したpipelineを作成する
fixed_model_pipeline = partial(
    pipeline,
    "summarization",
    model=model,
    tokenizer=tokenizer,
    device="cuda:0",
)

# ### 7.5.1 テキスト生成における探索アルゴリズム

# #### 貪欲法

print(fixed_model_pipeline()(content)[0]["summary_text"])

summarization_pipeline = fixed_model_pipeline(no_repeat_ngram_size=2)
print(summarization_pipeline(content)[0]["summary_text"])

# #### ビームサーチ

summarization_pipeline = fixed_model_pipeline(num_beams=3)
print(summarization_pipeline(content)[0]["summary_text"])

summarization_pipeline = fixed_model_pipeline(
    num_beams=3, num_return_sequences=3
)
for summary in summarization_pipeline(content):
    print(summary["summary_text"])

# ### 7.5.2 サンプリングを用いたテキスト生成

summarization_pipeline = fixed_model_pipeline(do_sample=True, top_k=0)
print(summarization_pipeline(content)[0]["summary_text"])

summarization_pipeline = fixed_model_pipeline(
    do_sample=True, top_k=0, temperature=0.5
)
print(summarization_pipeline(content)[0]["summary_text"])

summarization_pipeline = fixed_model_pipeline(
    do_sample=True, top_k=0, temperature=1.3
)
print(summarization_pipeline(content)[0]["summary_text"])

# #### top-kサンプリング

summarization_pipeline = fixed_model_pipeline(
    do_sample=True, top_k=10, temperature=1.3
)
print(summarization_pipeline(content)[0]["summary_text"])

# #### top-pサンプリング

summarization_pipeline = fixed_model_pipeline(
    do_sample=True, top_k=0, top_p=0.5, temperature=1.3
)
print(summarization_pipeline(content)[0]["summary_text"])

# ### 7.5.3 長さを調整したテキスト生成

summarization_pipeline = fixed_model_pipeline(
    num_beams=3,
    num_return_sequences=3,
    min_new_tokens=5,
    max_new_tokens=5,
)
for summary in summarization_pipeline(content):
    print(summary["summary_text"])

summarization_pipeline = fixed_model_pipeline(
    num_beams=3,
    num_return_sequences=3,
    min_new_tokens=35,
    max_new_tokens=35,
)
for summary in summarization_pipeline(content):
    print(summary["summary_text"])

summarization_pipeline = fixed_model_pipeline(
    num_beams=3,
    num_return_sequences=3,
    min_new_tokens=35,
    max_new_tokens=35,
    do_sample=True,
    temperature=1.3,
    no_repeat_ngram_size=3,
)
for summary in summarization_pipeline(content):
    print(summary["summary_text"])
