# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "transformers[ja,sentencepiece,torch]",
# ]
# ///

from transformers import pipeline

text_classification_pipeline = pipeline(
    model="llm-book/bert-base-japanese-v3-marc_ja"
)
positive_text = "世界には言葉がわからなくても感動する音楽がある。"
# positive_textの極性を予測
print(text_classification_pipeline(positive_text)[0])

# negative_textの極性を予測
negative_text = "世界には言葉がでないほどひどい音楽がある。"
print(text_classification_pipeline(negative_text)[0])

nli_pipeline = pipeline(model="llm-book/bert-base-japanese-v3-jnli")
text = "二人の男性がジェット機を見ています"
entailment_text = "ジェット機を見ている人が二人います"

# textとentailment_textの論理関係を予測
print(nli_pipeline({"text": text, "text_pair": entailment_text}))

contradiction_text = "二人の男性が飛んでいます"
# textとcontradiction_textの論理関係を予測
print(nli_pipeline({"text": text, "text_pair": contradiction_text}))

neutral_text = "2人の男性が、白い飛行機を眺めています"
# textとneutral_textの論理関係を予測
print(nli_pipeline({"text": text, "text_pair": neutral_text}))

text_sim_pipeline = pipeline(
    model="llm-book/bert-base-japanese-v3-jsts",
    function_to_apply="none",
)
text = "川べりでサーフボードを持った人たちがいます"
sim_text = "サーファーたちが川べりに立っています"
# textとsim_textの類似度を計算
result = text_sim_pipeline({"text": text, "text_pair": sim_text})
print(result["score"])

dissim_text = "トイレの壁に黒いタオルがかけられています"
# textとdissim_textの類似度を計算
result = text_sim_pipeline({"text": text, "text_pair": dissim_text})
print(result["score"])

from torch.nn.functional import cosine_similarity

sim_enc_pipeline = pipeline(
    model="llm-book/bert-base-japanese-v3-unsup-simcse-jawiki",
    task="feature-extraction",
)

# textとsim_textのベクトルを獲得
text_emb = sim_enc_pipeline(text, return_tensors=True)[0][0]
sim_emb = sim_enc_pipeline(sim_text, return_tensors=True)[0][0]
# textとsim_textの類似度を計算
sim_pair_score = cosine_similarity(text_emb, sim_emb, dim=0)
print(sim_pair_score.item())

# dissim_textのベクトルを獲得
dissim_emb = sim_enc_pipeline(dissim_text, return_tensors=True)[0][0]
# textとdissim_textの類似度を計算
dissim_pair_score = cosine_similarity(text_emb, dissim_emb, dim=0)
print(dissim_pair_score.item())

from pprint import pprint

ner_pipeline = pipeline(
    model="llm-book/bert-base-japanese-v3-ner-wikipedia-dataset",
    aggregation_strategy="simple",
)
text = "大谷翔平は岩手県水沢市出身のプロ野球選手"
# text中の固有表現を抽出
pprint(ner_pipeline(text))

text2text_pipeline = pipeline(
    "text2text-generation",
    model="llm-book/t5-base-long-livedoor-news-corpus"
)
article = "ついに始まった３連休。テレビを見ながら過ごしている人も多いのではないだろうか？　今夜オススメなのは何と言っても、NHKスペシャル「世界を変えた男 スティーブ・ジョブズ」だ。実は知らない人も多いジョブズ氏の養子に出された生い立ちや、アップル社から一時追放されるなどの経験。そして、彼が追い求めた理想の未来とはなんだったのか、ファンならずとも気になる内容になっている。 今年、亡くなったジョブズ氏の伝記は日本でもベストセラーになっている。今後もアップル製品だけでなく、世界でのジョブズ氏の影響は大きいだろうと想像される。ジョブズ氏のことをあまり知らないという人もこの機会にぜひチェックしてみよう。 世界を変えた男　スティーブ・ジョブズ（NHKスペシャル）"
# articleの要約を生成
print(text2text_pipeline(article)[0]["generated_text"])

from transformers import AutoTokenizer

# AutoTokenizerでトークナイザをロードする
tokenizer = AutoTokenizer.from_pretrained("abeja/gpt2-large-japanese")
# 入力文をトークンに分割する
tokenizer.tokenize("今日は天気が良いので")

from transformers import AutoModelForCausalLM

# 生成を行うモデルであるAutoModelForCausalLMを使ってモデルをロードする
model = AutoModelForCausalLM.from_pretrained(
    "abeja/gpt2-large-japanese"
)
# トークナイザを使ってモデルへの入力を作成する
inputs = tokenizer("今日は天気が良いので", return_tensors="pt")
# 後続のテキストを予測
outputs = model.generate(
    **inputs,
    max_length=15,  # 生成する最大トークン数を15に指定
    pad_token_id=tokenizer.pad_token_id  # パディングのトークンIDを指定
)
# generate関数の出力をテキストに変換する
generated_text = tokenizer.decode(
    outputs[0], skip_special_tokens=True
)
print(generated_text)
