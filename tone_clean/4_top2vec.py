import json
from top2vec import Top2Vec
from collections import OrderedDict, defaultdict

# ===== 讀取繁體回答資料 =====
with open("output/traditional.json", "r", encoding="utf-8") as f:
    documents = json.load(f)

# ===== 訓練 Top2Vec 模型 =====
print("建立 Top2Vec 模型中...")
model = Top2Vec(
    documents,
    embedding_model="universal-sentence-encoder-multilingual",
    speed="learn",
    workers=8
)

# ===== 每篇回答對應的主題 =====
doc_topics, _ = model.get_documents_topics(doc_ids=range(len(documents)))

# ===== 建立主題對應內容與關鍵字的字典 =====
topic_keywords = {}
num_topics = model.get_num_topics()
for topic_id in range(num_topics):
    words, _ = model.get_topic_words(topic_id, num_words=3)
    topic_keywords[topic_id] = words

# ===== 將回答歸入群組 =====
grouped = defaultdict(list)
for text, topic in zip(documents, doc_topics):
    grouped[topic].append(text)

# ===== 組合輸出格式：{"TopicX | 關鍵字1, 關鍵字2": [回答]} =====
result = OrderedDict()
for topic_id in sorted(grouped.keys()):
    keywords = ", ".join(topic_keywords[topic_id])
    topic_label = f"Topic{topic_id} | {keywords}"
    result[topic_label] = grouped[topic_id]

# ===== 輸出結果為 JSON 檔 =====
with open("output/top2vec.json", "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print("群組分類 完成!")