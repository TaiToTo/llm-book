# ===== Imports =====
from pprint import pprint
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
)
from torch.nn.functional import cosine_similarity


# ===== 1. テキスト分類（感情分析） =====
print("\n" + "=" * 60)
print("【1. テキスト分類（感情分析）】")
print("=" * 60)

text_classification_pipeline = pipeline(
    model="llm-book/bert-base-japanese-v3-marc_ja"
)

# ポジティブなテキストの分類
positive_text = "世界には言葉がわからなくても感動する音楽がある。"
print(f"\n入力（ポジティブ）: {positive_text}")
positive_result = text_classification_pipeline(positive_text)[0]
print(f"出力: {positive_result}")

# ネガティブなテキストの分類
negative_text = "世界には言葉がでないほどひどい音楽がある。"
print(f"\n入力（ネガティブ）: {negative_text}")
negative_result = text_classification_pipeline(negative_text)[0]
print(f"出力: {negative_result}")


# ===== 2. 自然言語推論（NLI: Natural Language Inference） =====
print("\n" + "=" * 60)
print("【2. 自然言語推論（NLI）】")
print("=" * 60)

nli_pipeline = pipeline(model="llm-book/bert-base-japanese-v3-jnli")
text = "二人の男性がジェット機を見ています"

# 含意関係（entailment）の予測
entailment_text = "ジェット機を見ている人が二人います"
print(f"\n前提文: {text}")
print(f"仮説文（含意）: {entailment_text}")
entailment_result = nli_pipeline({"text": text, "text_pair": entailment_text})
print(f"出力: {entailment_result}")

# 矛盾関係（contradiction）の予測
contradiction_text = "二人の男性が飛んでいます"
print(f"\n前提文: {text}")
print(f"仮説文（矛盾）: {contradiction_text}")
contradiction_result = nli_pipeline({"text": text, "text_pair": contradiction_text})
print(f"出力: {contradiction_result}")

# 中立関係（neutral）の予測
neutral_text = "2人の男性が、白い飛行機を眺めています"
print(f"\n前提文: {text}")
print(f"仮説文（中立）: {neutral_text}")
neutral_result = nli_pipeline({"text": text, "text_pair": neutral_text})
print(f"出力: {neutral_result}")


# ===== 3. テキスト類似度（回帰モデル） =====
print("\n" + "=" * 60)
print("【3. テキスト類似度（回帰モデル）】")
print("=" * 60)

text_sim_pipeline = pipeline(
    model="llm-book/bert-base-japanese-v3-jsts",
    function_to_apply="none",
)

text = "川べりでサーフボードを持った人たちがいます"

# 類似したテキストペアの類似度計算
sim_text = "サーファーたちが川べりに立っています"
print(f"\nテキスト1: {text}")
print(f"テキスト2（類似）: {sim_text}")
sim_result = text_sim_pipeline({"text": text, "text_pair": sim_text})
print(f"出力（類似度スコア）: {sim_result['score']}")

# 非類似なテキストペアの類似度計算
dissim_text = "トイレの壁に黒いタオルがかけられています"
print(f"\nテキスト1: {text}")
print(f"テキスト2（非類似）: {dissim_text}")
dissim_result = text_sim_pipeline({"text": text, "text_pair": dissim_text})
print(f"出力（類似度スコア）: {dissim_result['score']}")


# ===== 4. 文埋め込みベースの類似度（コサイン類似度） =====
print("\n" + "=" * 60)
print("【4. 文埋め込みベースの類似度（コサイン類似度）】")
print("=" * 60)

sim_enc_pipeline = pipeline(
    model="llm-book/bert-base-japanese-v3-unsup-simcse-jawiki",
    task="feature-extraction",
)

# 類似したテキストペアの埋め込みベクトルから類似度を計算
text_emb = sim_enc_pipeline(text, return_tensors=True)[0][0]
sim_emb = sim_enc_pipeline(sim_text, return_tensors=True)[0][0]
print(f"\nテキスト1: {text}")
print(f"テキスト2（類似）: {sim_text}")
sim_pair_score = cosine_similarity(text_emb, sim_emb, dim=0)
print(f"出力（コサイン類似度）: {sim_pair_score.item()}")

# 非類似なテキストペアの埋め込みベクトルから類似度を計算
dissim_emb = sim_enc_pipeline(dissim_text, return_tensors=True)[0][0]
print(f"\nテキスト1: {text}")
print(f"テキスト2（非類似）: {dissim_text}")
dissim_pair_score = cosine_similarity(text_emb, dissim_emb, dim=0)
print(f"出力（コサイン類似度）: {dissim_pair_score.item()}")


# ===== 5. 固有表現抽出（NER: Named Entity Recognition） =====
print("\n" + "=" * 60)
print("【5. 固有表現抽出（NER）】")
print("=" * 60)

ner_pipeline = pipeline(
    model="llm-book/bert-base-japanese-v3-ner-wikipedia-dataset",
    aggregation_strategy="simple",
)

text = "大谷翔平は岩手県水沢市出身のプロ野球選手"
print(f"\n入力: {text}")
ner_result = ner_pipeline(text)
print("出力（固有表現）:")
pprint(ner_result)


# ===== 6. テキスト要約 =====
print("\n" + "=" * 60)
print("【6. テキスト要約】")
print("=" * 60)

summarize_model_name = "llm-book/t5-base-long-livedoor-news-corpus"
summarize_tokenizer = AutoTokenizer.from_pretrained(summarize_model_name)
summarize_model = AutoModelForSeq2SeqLM.from_pretrained(summarize_model_name)

article = "ついに始まった３連休。テレビを見ながら過ごしている人も多いのではないだろうか？　今夜オススメなのは何と言っても、NHKスペシャル「世界を変えた男 スティーブ・ジョブズ」だ。実は知らない人も多いジョブズ氏の養子に出された生い立ちや、アップル社から一時追放されるなどの経験。そして、彼が追い求めた理想の未来とはなんだったのか、ファンならずとも気になる内容になっている。 今年、亡くなったジョブズ氏の伝記は日本でもベストセラーになっている。今後もアップル製品だけでなく、世界でのジョブズ氏の影響は大きいだろうと想像される。ジョブズ氏のことをあまり知らないという人もこの機会にぜひチェックしてみよう。 世界を変えた男　スティーブ・ジョブズ（NHKスペシャル）"
print(f"\n入力（記事）: {article}")

inputs = summarize_tokenizer(article, return_tensors="pt", truncation=True, max_length=512)
outputs = summarize_model.generate(**inputs)
summary = summarize_tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"出力（要約）: {summary}")


# ===== 7. テキスト生成（GPT-2） =====
print("\n" + "=" * 60)
print("【7. テキスト生成（GPT-2）】")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained("rinna/japanese-gpt2-small")
model = AutoModelForCausalLM.from_pretrained("rinna/japanese-gpt2-small")

prompt = "今日は天気が良いので"
print(f"\n入力（プロンプト）: {prompt}")

# トークン分割の確認
tokens = tokenizer.tokenize(prompt)
print(f"トークン分割: {tokens}")

# テキスト生成
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(
    **inputs,
    max_length=15,
    pad_token_id=tokenizer.pad_token_id
)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"出力（生成テキスト）: {generated_text}")

print("\n" + "=" * 60)
