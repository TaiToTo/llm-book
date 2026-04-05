# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "datasets<4.0.0",
#   "faiss-cpu",
#   "scipy",
#   "transformers[ja,torch]<4.41.0",
# ]
# ///

from datasets import load_dataset

# Hugging Face Hubのllm-book/jawiki-paragraphsのリポジトリから
# Wikipediaの段落テキストのデータを読み込む
paragraph_dataset = load_dataset(
    "llm-book/jawiki-paragraphs", split="train"
)

# 段落データの形式と事例数を確認する
print(paragraph_dataset)

from pprint import pprint

# 段落データの内容を確認する
pprint(paragraph_dataset[0])
pprint(paragraph_dataset[1])

# 段落データのうち、各記事の最初の段落のみを使うようにする
paragraph_dataset = paragraph_dataset.filter(
    lambda example: example["paragraph_index"] == 0
)

# フィルタリング後の段落データの形式と事例数を確認する
print(paragraph_dataset)

# フィルタリング後の段落データの内容を確認する
pprint(paragraph_dataset[0])
pprint(paragraph_dataset[1])

from transformers import AutoModel, AutoTokenizer

# Hugging Face Hubにアップロードされた
# 教師なしSimCSEのトークナイザとエンコーダを読み込む
model_name = "llm-book/bert-base-japanese-v3-unsup-simcse-jawiki"
tokenizer = AutoTokenizer.from_pretrained(model_name)
encoder = AutoModel.from_pretrained(model_name)

from transformers import AutoModel, AutoTokenizer

# ディスクに保存された教師なしSimCSEのトークナイザとエンコーダを読み込む
model_path = "outputs_unsup_simcse/encoder"
tokenizer = AutoTokenizer.from_pretrained(model_path)
encoder = AutoModel.from_pretrained(model_path)

# 読み込んだモデルをGPUのメモリに移動させる
device = "cuda:0"
encoder = encoder.to(device)

import numpy as np
import torch
import torch.nn.functional as F

def embed_texts(texts: list[str]) -> np.ndarray:
    """SimCSEのモデルを用いてテキストの埋め込みを計算"""
    # テキストにトークナイザを適用
    tokenized_texts = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    ).to(device)

    # トークナイズされたテキストをベクトルに変換
    with torch.inference_mode():
        with torch.cuda.amp.autocast():
            encoded_texts = encoder(
                **tokenized_texts
            ).last_hidden_state[:, 0]

    # ベクトルをNumPyのarrayに変換
    emb = encoded_texts.cpu().numpy().astype(np.float32)
    # ベクトルのノルムが1になるように正規化
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    return emb

# 段落データのすべての事例に埋め込みを付与する
paragraph_dataset = paragraph_dataset.map(
    lambda examples: {
        "embeddings": list(embed_texts(examples["text"]))
    },
    batched=True,
)

# 埋め込みを付与した段落データの形式と事例数を確認する
print(paragraph_dataset)

# 埋め込みを計算した段落データの内容を確認する
pprint(paragraph_dataset[0])

# 埋め込みを付与した段落データをディスクに保存する
paragraph_dataset.save_to_disk(
    "outputs_unsup_simcse/embedded_paragraphs"
)

# 保存された段落データをGoogleドライブのフォルダにコピーする

import faiss

# ベクトルの次元数をエンコーダの設定値から取り出す
emb_dim = encoder.config.hidden_size
# ベクトルの次元数を指定して空のFaissインデックスを作成する
index = faiss.IndexFlatIP(emb_dim)
# 段落データの"embeddings"フィールドのベクトルからFaissインデックスを構築する
paragraph_dataset.add_faiss_index("embeddings", custom_index=index)

query_text = "日本語は、主に日本で話されている言語である。"

# 最近傍探索を実行し、類似度上位10件の事例とスコアを取得する
scores, retrieved_examples = paragraph_dataset.get_nearest_examples(
    "embeddings", embed_texts([query_text])[0], k=10
)
# 取得した事例の内容をスコアとともに表示する
titles = retrieved_examples["title"]
texts = retrieved_examples["text"]
for score, title, text in zip(scores, titles, texts):
    print(score, title, text)
