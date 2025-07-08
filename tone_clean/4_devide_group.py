import json
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
import hdbscan
from collections import defaultdict, OrderedDict
from sklearn.decomposition import PCA

# ===== 檔案設定 =====
INPUT_FILE = "output/traditional.json"
OUTPUT_FILE = "output/group02.json"

# ===== 載入資料 =====
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    responses = json.load(f)

# ===== 向量化處理 =====
model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
embeddings = model.encode(responses, convert_to_tensor=False, show_progress_bar=True)
embeddings = normalize(embeddings)

# ===== 使用 PCA 將向量降至 50 維 =====
pca = PCA(n_components=50)
embeddings_reduced = pca.fit_transform(embeddings)

# ===== 使用 HDBSCAN 自動分群 =====
clusterer = hdbscan.HDBSCAN(min_cluster_size=2, min_samples=2, metric="euclidean")
labels = clusterer.fit_predict(embeddings_reduced)

# ===== 整理為 {群組名稱: [回答]} 格式 =====
grouped = defaultdict(list)
for text, label in zip(responses, labels):
    group_label = f"G{label}" if label != -1 else "noise"
    grouped[group_label].append(text)

# ===== 群組排序（G0, G1, ..., noise）=====
sorted_keys = sorted(
    [k for k in grouped.keys() if k != "noise"],
    key=lambda x: int(x[1:])  # 按照數字排序
) + (["noise"] if "noise" in grouped else [])

ordered_grouped = OrderedDict((k, grouped[k]) for k in sorted_keys)

# ===== 輸出為 JSON =====
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(ordered_grouped, f, ensure_ascii=False, indent=2)

print("群組分類 完成!")