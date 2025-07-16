import json
import torch
import numpy as np
from sentence_transformers import SentenceTransformer, util

# === 路徑設定 ===
GROUPED_JSON_PATH = "output/bertopic11.json"

# === 載入群組資料（每組是一個主題）===
with open(GROUPED_JSON_PATH, "r", encoding="utf-8") as f:
    grouped = json.load(f)

print(f"共載入 {len(grouped)} 組主題")

# === 初始化 embedding 模型 ===
embedding_model = SentenceTransformer("shibing624/text2vec-base-chinese")
# shibing624/text2vec-base-chinese
# uer/sbert-base-chinese-nli

# === 計算每組主題的平均向量 ===
topic_mean_vectors = {}
for group_label, sentences in grouped.items():
    if len(sentences) == 0:
        continue
    embeddings = embedding_model.encode(sentences, convert_to_tensor=True)
    mean_vec = torch.mean(embeddings, dim=0)
    topic_mean_vectors[group_label] = mean_vec

print("已完成所有主題的平均向量建立")

# === ✅ Top-N 相似主題比對函式 ===
def find_top_n_topics(input_sentence: str, top_n=5):
    input_vec = embedding_model.encode(input_sentence, convert_to_tensor=True)
    similarity_scores = []

    for label, mean_vec in topic_mean_vectors.items():
        sim = util.cos_sim(input_vec, mean_vec).item()
        similarity_scores.append((label, sim))

    similarity_scores.sort(key=lambda x: x[1], reverse=True)

    print(f"\n輸入句子：{input_sentence}")
    print(f"🔍 相似度最高的前 {top_n} 組主題：\n")
    for i, (label, score) in enumerate(similarity_scores[:top_n]):
        print(f"{i+1:>2}. {label}（相似度：{score:.4f}）")

    return similarity_scores[:top_n]

# === 🧪 測試用 ===
if __name__ == "__main__":
    test_sentence = "可以分享你養狗的經驗嗎？"
    top_matches = find_top_n_topics(test_sentence, top_n=5)