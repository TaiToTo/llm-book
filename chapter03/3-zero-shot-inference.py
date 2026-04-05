# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "transformers[ja,sentencepiece,torch]",
#   "pandas",
# ]
# ///

from transformers import pipeline

# 後続するテキストを予測するpipelineを作成
generator = pipeline(
    "text-generation", model="abeja/gpt2-large-japanese"
)
# "日本で一番高い山は"に続くテキストを生成
outputs = generator("日本で一番高い山は")
print(outputs[0]["generated_text"])

import pandas as pd

# マスクされたトークンを予測するpipelineを作成
fill_mask = pipeline(
    "fill-mask", model="cl-tohoku/bert-base-japanese-v3"
)
masked_text = "日本の首都は[MASK]である"
# [MASK]部分を予測
outputs = fill_mask(masked_text)
# 上位3件をテーブルで表示
display(pd.DataFrame(outputs[:3]))

masked_text = "今日の映画は刺激的で面白かった。この映画は[MASK]。"
outputs = fill_mask(masked_text)
display(pd.DataFrame(outputs[:3]))

# text-to-textで生成するpipelineを作成
t2t_generator = pipeline(
    "text2text-generation", model="retrieva-jp/t5-large-long"
)
# マスクされたスパンを予測
masked_text = "江戸幕府を開いたのは、<extra_id_0>である"
outputs = t2t_generator(masked_text, eos_token_id=32098)
print(outputs[0]["generated_text"])

t2t_generator.tokenizer.convert_tokens_to_ids("<extra_id_1>")

masked_text = "日本で通貨を発行しているのは、<extra_id_0>である"
outputs = t2t_generator(masked_text, eos_token_id=32098)
print(outputs[0]["generated_text"])

"日本銀行" in t2t_generator.tokenizer.vocab
