import json
from tqdm import tqdm
from funasr import AutoModel

# === 載入原始資料 ===
INPUT_PATH = "output/traditional.json"
OUTPUT_PATH = "output/sentence.json"

with open(INPUT_PATH, "r", encoding="utf-8") as f:
    raw_documents = json.load(f)

print(f"載入 {len(raw_documents)} 筆資料")

# === 載入標點模型 ===
model = AutoModel(model="ct-punc", model_revision="v2.0.4", hub="hf")

# === 中文斷字（只針對中文字加空格）===
def char_split(text):
    new_text = []
    for ch in text:
        if '\u4e00' <= ch <= '\u9fff':  # 是中文字
            new_text.append(ch)
            new_text.append(' ')
        else:
            new_text.append(ch)
    return ''.join(new_text).strip()

# === 執行標點符號修正 ===
punctuated_documents = []
for text in tqdm(raw_documents, desc="加標點中"):
    split_text = char_split(text)
    result = model.generate(input=split_text)
    punctuated_text = result[0]["text"]
    punctuated_documents.append(punctuated_text)

# === 儲存結果 ===
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(punctuated_documents, f, ensure_ascii=False, indent=2)

print("加標點符號 完成!")
