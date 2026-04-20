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
positive_result = text_classification_pipeline(positive_text)[0]
print("\n--- ケース1a: ポジティブなテキスト ---")
print(f"【入力】")
print(f"  データ: {positive_text!r}")
print(f"  型: {type(positive_text).__name__}")
print(f"【出力】")
print(f"  データ構造: {positive_result}")
print(f"  意味: ラベル「{positive_result['label']}」、確信度 {positive_result['score']:.2%}")

# ネガティブなテキストの分類
negative_text = "世界には言葉がでないほどひどい音楽がある。"
negative_result = text_classification_pipeline(negative_text)[0]
print("\n--- ケース1b: ネガティブなテキスト ---")
print(f"【入力】")
print(f"  データ: {negative_text!r}")
print(f"  型: {type(negative_text).__name__}")
print(f"【出力】")
print(f"  データ構造: {negative_result}")
print(f"  意味: ラベル「{negative_result['label']}」、確信度 {negative_result['score']:.2%}")


# ===== 2. 自然言語推論（NLI: Natural Language Inference） =====
print("\n" + "=" * 60)
print("【2. 自然言語推論（NLI）】")
print("=" * 60)

nli_pipeline = pipeline(model="llm-book/bert-base-japanese-v3-jnli")
text = "二人の男性がジェット機を見ています"

# 含意関係（entailment）の予測
entailment_text = "ジェット機を見ている人が二人います"
nli_input_e = {"text": text, "text_pair": entailment_text}
entailment_result = nli_pipeline(nli_input_e)
print("\n--- ケース2a: 含意関係（entailment） ---")
print(f"【入力】")
print(f"  データ: {nli_input_e}")
print(f"  型: {type(nli_input_e).__name__}")
print(f"【出力】")
print(f"  データ構造: {entailment_result}")
print(f"  意味: 前提文から仮説文が「{entailment_result['label']}」と判定（確信度 {entailment_result['score']:.2%}）")

# 矛盾関係（contradiction）の予測
contradiction_text = "二人の男性が飛んでいます"
nli_input_c = {"text": text, "text_pair": contradiction_text}
contradiction_result = nli_pipeline(nli_input_c)
print("\n--- ケース2b: 矛盾関係（contradiction） ---")
print(f"【入力】")
print(f"  データ: {nli_input_c}")
print(f"  型: {type(nli_input_c).__name__}")
print(f"【出力】")
print(f"  データ構造: {contradiction_result}")
print(f"  意味: 前提文から仮説文が「{contradiction_result['label']}」と判定（確信度 {contradiction_result['score']:.2%}）")

# 中立関係（neutral）の予測
neutral_text = "2人の男性が、白い飛行機を眺めています"
nli_input_n = {"text": text, "text_pair": neutral_text}
neutral_result = nli_pipeline(nli_input_n)
print("\n--- ケース2c: 中立関係（neutral） ---")
print(f"【入力】")
print(f"  データ: {nli_input_n}")
print(f"  型: {type(nli_input_n).__name__}")
print(f"【出力】")
print(f"  データ構造: {neutral_result}")
print(f"  意味: 前提文から仮説文が「{neutral_result['label']}」と判定（確信度 {neutral_result['score']:.2%}）")


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
sts_input_s = {"text": text, "text_pair": sim_text}
sim_result = text_sim_pipeline(sts_input_s)
print("\n--- ケース3a: 類似したテキストペア ---")
print(f"【入力】")
print(f"  データ: {sts_input_s}")
print(f"  型: {type(sts_input_s).__name__}")
print(f"【出力】")
print(f"  データ構造: {sim_result}")
print(f"  意味: 類似度スコア {sim_result['score']:.2f}（5段階中）→ 意味が近い")

# 非類似なテキストペアの類似度計算
dissim_text = "トイレの壁に黒いタオルがかけられています"
sts_input_d = {"text": text, "text_pair": dissim_text}
dissim_result = text_sim_pipeline(sts_input_d)
print("\n--- ケース3b: 非類似なテキストペア ---")
print(f"【入力】")
print(f"  データ: {sts_input_d}")
print(f"  型: {type(sts_input_d).__name__}")
print(f"【出力】")
print(f"  データ構造: {dissim_result}")
print(f"  意味: 類似度スコア {dissim_result['score']:.2f}（5段階中）→ 意味が遠い")


# ===== 4. 文埋め込みベースの類似度（コサイン類似度） =====
print("\n" + "=" * 60)
print("【4. 文埋め込みベースの類似度（コサイン類似度）】")
print("=" * 60)

sim_enc_pipeline = pipeline(
    model="llm-book/bert-base-japanese-v3-unsup-simcse-jawiki",
    task="feature-extraction",
)

# ステップ1: 各テキストを埋め込みベクトルに変換
text_emb = sim_enc_pipeline(text, return_tensors=True)[0][0]
sim_emb = sim_enc_pipeline(sim_text, return_tensors=True)[0][0]
dissim_emb = sim_enc_pipeline(dissim_text, return_tensors=True)[0][0]

print("\n--- ステップ1: テキスト → 埋め込みベクトル ---")
print(f"  '{text}'")
print(f"    → shape: {tuple(text_emb.shape)}, 先頭5次元: {text_emb[:5].tolist()}")
print(f"  '{sim_text}'")
print(f"    → shape: {tuple(sim_emb.shape)}, 先頭5次元: {sim_emb[:5].tolist()}")
print(f"  '{dissim_text}'")
print(f"    → shape: {tuple(dissim_emb.shape)}, 先頭5次元: {dissim_emb[:5].tolist()}")
print(f"  ※ 各テキストが{text_emb.shape[0]}次元の実数ベクトルに変換された")

# ステップ2: コサイン類似度で比較 cos(A,B) = A·B / (|A||B|)
sim_pair_score = cosine_similarity(text_emb, sim_emb, dim=0)
dissim_pair_score = cosine_similarity(text_emb, dissim_emb, dim=0)

print("\n--- ステップ2: コサイン類似度 cos(A,B) = A·B / (|A||B|) ---")
print("\n  ケース4a: 類似したテキストペア")
print(f"  【入力】")
print(f"    ベクトルA: text_emb  ('{text}')")
print(f"    ベクトルB: sim_emb   ('{sim_text}')")
print(f"  【出力】")
print(f"    データ構造: tensor({sim_pair_score.item():.6f})")
print(f"    意味: コサイン類似度 {sim_pair_score.item():.4f} → 意味的に近い（1に近い）")

print("\n  ケース4b: 非類似なテキストペア")
print(f"  【入力】")
print(f"    ベクトルA: text_emb    ('{text}')")
print(f"    ベクトルB: dissim_emb  ('{dissim_text}')")
print(f"  【出力】")
print(f"    データ構造: tensor({dissim_pair_score.item():.6f})")
print(f"    意味: コサイン類似度 {dissim_pair_score.item():.4f} → 意味的に遠い（0に近い）")


# ===== 5. 固有表現抽出（NER: Named Entity Recognition） =====
print("\n" + "=" * 60)
print("【5. 固有表現抽出（NER）】")
print("=" * 60)

ner_pipeline = pipeline(
    model="llm-book/bert-base-japanese-v3-ner-wikipedia-dataset",
    aggregation_strategy="simple",
)

text = "大谷翔平は岩手県水沢市出身のプロ野球選手"
ner_result = ner_pipeline(text)
print("\n--- ケース5: 固有表現の抽出 ---")
print(f"【入力】")
print(f"  データ: {text!r}")
print(f"  型: {type(text).__name__}")
print(f"【出力】")
print(f"  データ構造:")
pprint(ner_result, indent=4)
entities = ", ".join(f"「{e['word']}」({e['entity_group']})" for e in ner_result)
print(f"  意味: 抽出された固有表現 → {entities}")


# ===== 6. テキスト要約 =====
print("\n" + "=" * 60)
print("【6. テキスト要約】")
print("=" * 60)

summarize_model_name = "llm-book/t5-base-long-livedoor-news-corpus"
summarize_tokenizer = AutoTokenizer.from_pretrained(summarize_model_name)
summarize_model = AutoModelForSeq2SeqLM.from_pretrained(summarize_model_name)

article = "ついに始まった３連休。テレビを見ながら過ごしている人も多いのではないだろうか？　今夜オススメなのは何と言っても、NHKスペシャル「世界を変えた男 スティーブ・ジョブズ」だ。実は知らない人も多いジョブズ氏の養子に出された生い立ちや、アップル社から一時追放されるなどの経験。そして、彼が追い求めた理想の未来とはなんだったのか、ファンならずとも気になる内容になっている。 今年、亡くなったジョブズ氏の伝記は日本でもベストセラーになっている。今後もアップル製品だけでなく、世界でのジョブズ氏の影響は大きいだろうと想像される。ジョブズ氏のことをあまり知らないという人もこの機会にぜひチェックしてみよう。 世界を変えた男　スティーブ・ジョブズ（NHKスペシャル）"
inputs = summarize_tokenizer(article, return_tensors="pt", truncation=True, max_length=512)
outputs = summarize_model.generate(**inputs)
summary = summarize_tokenizer.decode(outputs[0], skip_special_tokens=True)
print("\n--- ケース6: ニュース記事の要約 ---")
print(f"【入力】")
print(f"  データ: {article!r}")
print(f"  型: {type(article).__name__}（トークン化後のテンソル形状: {tuple(inputs['input_ids'].shape)}）")
print(f"【出力】")
print(f"  データ構造: {summary!r}")
print(f"  意味: 長い記事が「{summary}」に要約された")


# ===== 7. テキスト生成（GPT-2） =====
print("\n" + "=" * 60)
print("【7. テキスト生成（GPT-2）】")
print("=" * 60)

tokenizer = AutoTokenizer.from_pretrained("rinna/japanese-gpt2-small")
model = AutoModelForCausalLM.from_pretrained("rinna/japanese-gpt2-small")

prompt = "今日は天気が良いので"
tokens = tokenizer.tokenize(prompt)
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(
    **inputs,
    max_length=15,
    pad_token_id=tokenizer.pad_token_id
)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("\n--- ケース7: プロンプトからの続き生成 ---")
print(f"【入力】")
print(f"  データ: {prompt!r}")
print(f"  型: {type(prompt).__name__}")
print(f"  トークン分割: {tokens}")
print(f"  トークン化後のテンソル形状: {tuple(inputs['input_ids'].shape)}")
print(f"【出力】")
print(f"  データ構造: {generated_text!r}")
print(f"  意味: プロンプトに続けて「{generated_text}」が生成された")

print("\n" + "=" * 60)
