import json
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from collections import defaultdict, OrderedDict

# === ✅ 新增：jieba 分詞與中文 CountVectorizer 設定 ===
import jieba
from sklearn.feature_extraction.text import CountVectorizer

stopwords = {"我", "你", "是", "不是", "的", "這樣", "所以", "然後", "就", "還有", "就是"}

def jieba_tokenizer(text):
    return [word for word in jieba.cut(text) if word not in stopwords and len(word) > 1]

vectorizer_model = CountVectorizer(tokenizer=jieba_tokenizer)

# === 基本參數（使用者可自訂） ===
INPUT_PATH = "output/shortened.json"
OUTPUT_PATH = "output/bertopic08.json"
SAVE_MODEL_PATH = "output/bertopic08_model"
EMBEDDING_MODEL = SentenceTransformer("shibing624/text2vec-base-chinese") 
# paraphrase-multilingual-mpnet-base-v2
# shibing624/text2vec-base-chinese
# uer/sbert-base-chinese-nli
MIN_TOPIC_SIZE = 3

# === 載入資料 ===
with open(INPUT_PATH, "r", encoding="utf-8") as f:
    docs = json.load(f)
print(f"載入 {len(docs)} 筆資料")

# === UMAP 設定（符合官方結構）===
umap_model = UMAP(n_neighbors=5,
                  n_components=5,
                  min_dist=0.0,
                  metric="cosine",
                  random_state=42)

# === 建立 BERTopic 模型（✅ 新增 vectorizer_model）===
topic_model = BERTopic(
    embedding_model=EMBEDDING_MODEL,
    umap_model=umap_model,
    vectorizer_model=vectorizer_model,  # ✅ 加入這行！
    min_topic_size=MIN_TOPIC_SIZE,
    language="multilingual",
    calculate_probabilities=False,
    verbose=True
)

# === 執行分群 ===
topics, probs = topic_model.fit_transform(docs)

# === 儲存模型 ===
topic_model.save(SAVE_MODEL_PATH)

# === 轉成分群 JSON 輸出（這部分為自訂，非官網內容）===
grouped = defaultdict(list)
for text, topic in zip(docs, topics):
    if topic != -1:
        grouped[topic].append(text)

output = OrderedDict()
for topic_id in sorted(grouped):
    top_words = topic_model.get_topic(topic_id)
    keywords = ", ".join([word for word, _ in top_words[:3]])
    avg_len = round(sum(len(s) for s in grouped[topic_id]) / len(grouped[topic_id]))
    group_label = f"G{topic_id} | {keywords} | {avg_len}字"
    output[group_label] = grouped[topic_id]

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print("群組分類 完成!")
