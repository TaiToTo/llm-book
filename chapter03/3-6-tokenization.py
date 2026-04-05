# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "transformers[ja,sentencepiece,torch]",
# ]
# ///

# 単語とその頻度
word_freqs = {
    "たのしい": 6,
    "たのしさ": 2,
    "うつくしい": 4,
    "うつくしさ": 1,
}
# 語彙を文字で初期化
vocab = sorted(set([char for word in word_freqs for char in word]))
# 単語とその分割状態
splits = {word: [char for char in word] for word in word_freqs}

from collections import Counter

def compute_most_frequent_pair(
    splits: dict[str, list[str]]
) -> tuple[str, str]:
    """
    最も頻度の高い隣接するサブワードの組を計算する
    """
    pair_freqs = Counter()  # サブワードの組のカウンタ
    for word, freq in word_freqs.items():  # すべての単語を処理
        split = splits[word]  # 現在の単語の分割状態を取得
        # すべての隣接したサブワードの組を処理
        for i in range(len(split) - 1):
            pair = (split[i], split[i + 1])
            # サブワードの組の頻度に単語の頻度を加算
            pair_freqs[pair] += freq
    # カウンタから最も頻度の高いサブワードの組を取得
    pair, _ = pair_freqs.most_common(1)[0]
    return pair

def merge_pair(
    target_pair: tuple[str, str], splits: dict[str, list[str]]
) -> dict[str, list[str]]:
    """
    サブワードの組を結合する
    """
    l_str, r_str = target_pair
    for word in word_freqs:  # すべての単語を処理
        split = splits[word]  # 現在の単語の分割状態を取得
        i = 0
        # すべての隣接したサブワードの組を処理
        while i < len(split) - 1:
            # サブワードの組が結合対象と一致したら結合
            if split[i] == l_str and split[i + 1] == r_str:
                split = split[:i] + [l_str + r_str] + split[i + 2 :]
            i += 1
        splits[word] = split  # 現在の結合状態を更新
    return splits

for step in range(9):
    # 最も頻度の高い隣接するサブワードの組を計算
    target_pair = compute_most_frequent_pair(splits)
    # サブワードの組を結合
    splits = merge_pair(target_pair, splits)
    # 語彙にサブワードの組を追加
    vocab.append(target_pair)

print(vocab)

from transformers import AutoTokenizer

mbert_tokenizer = AutoTokenizer.from_pretrained(
    "bert-base-multilingual-cased"
)
print(mbert_tokenizer.tokenize("自然言語処理"))

print(mbert_tokenizer.tokenize("自然言語処理にディープラーニングを使う"))

xlmr_tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
print(xlmr_tokenizer.tokenize("自然言語処理にディープラーニングを使う"))

print(xlmr_tokenizer.tokenize("私は日本で生まれました"))

print(xlmr_tokenizer.tokenize("本日はよろしくお願いいたします"))

bert_ja_tokenizer = AutoTokenizer.from_pretrained(
    "cl-tohoku/bert-base-japanese-v3"
)
print(
    bert_ja_tokenizer.tokenize("自然言語処理にディープラーニングを使う")
)

print(bert_ja_tokenizer.tokenize("私は日本で生まれました"))
