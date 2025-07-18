# === ✅ 使用者可修改參數區 ===
QUESTION = "你覺得自己是一個怎麼樣的人？"
GROUPED_JSON_PATH = "output/bertopic11.json"
EMBEDDING_MODEL_NAME = "shibing624/text2vec-base-chinese"
OPENAI_MODEL_NAME = "gpt-4-0125-preview"
TOP_N_TOPICS = 5
MAX_REFERENCE_CHAR = 8000       # 超過這個字數才啟用抽樣機制
MAX_SENTENCES = 60              # 抽樣的最大句數（若需要）
MAX_ANSWER_CHAR = 500           # 限制回答在多少個字以內（新增，控制 ChatGPT 回覆長度）

# === ✅ GPT 模型控制參數（已內嵌設定） ===
GPT_TEMPERATURE = 0.8           # 模型創意程度：0.8 創意適中，0.2 趨近穩定事實型
GPT_TOP_P = 1.0                 # 機率截斷，一般固定為 1.0（不修改）
GPT_MAX_TOKENS = 1024           # ChatGPT 回覆的最大 token 長度，約 750～1000 中文字
GPT_PRESENCE_PENALTY = 0.6      # 鼓勵談論新主題（避免回答太雷同）
GPT_FREQUENCY_PENALTY = 0.4     # 降低重複用字的頻率

# === ✅ 套件與初始化 ===
import json
import torch
import numpy as np
from sentence_transformers import SentenceTransformer, util
import os
import random
from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# === ✅ 抽樣控制函式 ===
def limit_reference_texts(reference_texts: list, max_char: int = 8000, max_sentences: int = 60):
    total_char = sum(len(line) for line in reference_texts)
    if total_char <= max_char:
        return reference_texts
    print(f"\n語料總長 {total_char} 字，超過上限 {max_char}，將隨機抽樣 {max_sentences} 句\n")
    return random.sample(reference_texts, min(max_sentences, len(reference_texts)))

# === ✅ 載入資料與模型 ===
with open(GROUPED_JSON_PATH, "r", encoding="utf-8") as f:
    grouped = json.load(f)
print(f"共載入 {len(grouped)} 組主題")

embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

# === ✅ 計算平均向量 ===
topic_mean_vectors = {}
for group_label, sentences in grouped.items():
    if len(sentences) == 0:
        continue
    embeddings = embedding_model.encode(sentences, convert_to_tensor=True)
    mean_vec = torch.mean(embeddings, dim=0)
    topic_mean_vectors[group_label] = mean_vec
print("已完成所有主題的平均向量建立")

# === ✅ 相似主題查找函式 ===
def find_top_n_topics(input_sentence: str, top_n=5):
    input_vec = embedding_model.encode(input_sentence, convert_to_tensor=True)
    similarity_scores = []
    for label, mean_vec in topic_mean_vectors.items():
        sim = util.cos_sim(input_vec, mean_vec).item()
        similarity_scores.append((label, sim))

    similarity_scores.sort(key=lambda x: x[1], reverse=True)

    print(f"\n輸入句子：{input_sentence}")
    print(f"相似度最高的前 {top_n} 組主題：\n")
    for i, (label, score) in enumerate(similarity_scores[:top_n]):
        print(f"{i+1:>2}. {label}（相似度：{score:.4f}）")

    return similarity_scores[:top_n]

# === ✅ GPT 回答函式（新版 openai 語法） ===
def generate_response_with_chatgpt(question: str, reference_texts: list):
    reference_block = '\n'.join(f'"{line}"' for line in reference_texts)
    prompt = f"""請參考reference的內容並模仿他的語氣，回答我問你的question，並將回答控制在 {MAX_ANSWER_CHAR} 字以內。

reference: 
{reference_block}

question: {question}
"""
    response = client.chat.completions.create(
        model=OPENAI_MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=GPT_TEMPERATURE,
        top_p=GPT_TOP_P,
        max_tokens=GPT_MAX_TOKENS,
        presence_penalty=GPT_PRESENCE_PENALTY,
        frequency_penalty=GPT_FREQUENCY_PENALTY
    )
    return response.choices[0].message.content

# === ✅ 主流程 ===
if __name__ == "__main__":
    top_matches = find_top_n_topics(QUESTION, top_n=TOP_N_TOPICS)
    top_label = top_matches[0][0]
    full_texts = grouped[top_label]
    reference_texts = limit_reference_texts(full_texts, max_char=MAX_REFERENCE_CHAR, max_sentences=MAX_SENTENCES)

    reply = generate_response_with_chatgpt(QUESTION, reference_texts)
    print("\nAnswer:", reply)