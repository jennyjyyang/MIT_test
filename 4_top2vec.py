import json
from top2vec import Top2Vec
from collections import defaultdict, OrderedDict

# === 設定參數 ===
INPUT_PATH = "output/traditional.json"
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
SPEED = "deep-learn"
SAVE_MODEL_PATH = "output/top2vec01_model"
OUTPUT_PATH = "output/top2vec01.json"

# === 載入資料 ===
with open(INPUT_PATH, "r", encoding="utf-8") as f:
    documents = json.load(f)

print(f"載入 {len(documents)} 筆中文回答")

# === 建立模型 ===
print("建立 Top2Vec 模型中...")
model = Top2Vec(
    documents=documents,
    embedding_model=EMBEDDING_MODEL,
    speed=SPEED,
    workers=8,
    min_count=1
)

# === 儲存 .model 檔案（可供日後查詢使用）===
model.save(SAVE_MODEL_PATH + ".model")
print("模型儲存 完成!")

# === 每句話對應的主題 ===
doc_topics_raw = model.get_documents_topics(doc_ids=list(range(len(documents))))
doc_topics = doc_topics_raw[0].tolist()

# === 抓出所有主題關鍵詞（支援舊版 get_topics）===
all_topic_words, _, topic_nums = model.get_topics()
topic_keywords = {
    int(topic_id): list(words[:3])
    for topic_id, words in zip(topic_nums, all_topic_words)
}

# === 組成 {Gx | 關鍵字1, 關鍵字2, ...: [回答1, 回答2]} 結構 ===
grouped = defaultdict(list)
for text, topic in zip(documents, doc_topics):
    grouped[topic].append(text)

output_grouped = OrderedDict()
for topic_id in sorted(grouped.keys()):
    keywords = ", ".join(topic_keywords[topic_id])
    avg_len = round(sum(len(s) for s in grouped[topic_id]) / len(grouped[topic_id]))
    group_label = f"G{topic_id} | {keywords} | {avg_len}字"
    output_grouped[group_label] = grouped[topic_id]

# === 輸出為 JSON 檔（人工瀏覽與審核用）===
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(output_grouped, f, ensure_ascii=False, indent=2)

print("群組分類 完成!")
