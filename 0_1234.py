# 載入模組
import json
from opencc import OpenCC
from tqdm import tqdm
from funasr import AutoModel
import re

# 1. 將簡體中文轉換為台灣繁體中文
# 初始化 OpenCC 轉換器（簡體 → 台灣繁體）
cc = OpenCC('s2t')

# 載入字串 list 格式的 .json 檔
with open("city/output/answer.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# 執行繁體轉換
converted_data = [cc.convert(text) if isinstance(text, str) else text for text in data]

# 儲存為新檔案
with open("city/output/traditional.json", "w", encoding="utf-8") as f:
    json.dump(converted_data, f, ensure_ascii=False, indent=2)

print("簡體轉繁體 完成!")

# 2. 加入標點符號
# === 載入原始資料 ===
INPUT_PATH = "city/output/traditional.json"
OUTPUT_PATH = "city/output/sentence.json"

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

# 3. 去除重複語句
INPUT_PATH = "city/output/sentence.json"
OUTPUT_PATH = "city/output/sentence_clean.json"

def remove_repeats(text):
    # 1. 先簡單清理重複標點
    text = re.sub(r'([。，！？；：、])\1+', r'\1', text)

    # 2. 拆成字元陣列進行清理（處理重複單字）
    cleaned = []
    prev_char = ""
    repeat_count = 0
    for char in text:
        if char == prev_char:
            repeat_count += 1
            if repeat_count < 2:  # 最多保留一次
                cleaned.append(char)
        else:
            cleaned.append(char)
            repeat_count = 0
        prev_char = char
    cleaned_text = ''.join(cleaned)

    # 3. 處理重複詞組（3~6字內連續重複）
    for n in range(6, 1, -1):
        pattern = re.compile(rf'(([\u4e00-\u9fa5a-zA-Z0-9]{{{n}}}))\1+')
        cleaned_text = pattern.sub(r'\1', cleaned_text)

    return cleaned_text

# === 載入資料 ===
with open(INPUT_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"載入 {len(data)} 筆資料，開始清除重複語句...")

# === 執行清理 ===
cleaned_data = [remove_repeats(text) for text in tqdm(data)]

# === 儲存清理後資料 ===
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(cleaned_data, f, ensure_ascii=False, indent=2)

print("去除重複 完成!")

# 4. 切段落
def split_by_punctuation(text, max_len=80):
    """
    將一整段文字用中文標點切開，並根據 max_len 合併成適當長度的段落。
    例如一句很長的話中含有多個「句號」，會先依照句號切開，再依 max_len 合併。
    """
    # 1. 根據中文標點切句（包含句號、問號、驚嘆號）
    sentences = re.split(r'(。|？|！)', text)
    
    # 2. 合併標點與語句：將 ["這是第一句", "。", "這是第二句", "！"] → ["這是第一句。", "這是第二句！"]
    grouped = []
    for i in range(0, len(sentences) - 1, 2):
        grouped.append((sentences[i] + sentences[i+1]).strip())
    if len(sentences) % 2 == 1:
        grouped.append(sentences[-1].strip())  # 若最後一句沒有標點也保留

    # 3. 根據 max_len 合併多個短句成一段（但總長度不超過 max_len）
    result = []
    current = ""
    for seg in grouped:
        # 如果當前段落加上這個句子，還沒超過上限，就接上去
        if len(current) + len(seg) <= max_len:
            current += seg
        else:
            # 如果會超過，就把當前段落加入結果，重新開始新段落
            if current:
                result.append(current.strip())
            current = seg
    if current:
        result.append(current.strip())

    return result

# === 主程式開始 ===
with open("city/output/sentence_clean.json", "r", encoding="utf-8") as f:
    data = json.load(f)  # 資料應該是 list[str]

all_sentences = []
for line in data:
    all_sentences.extend(split_by_punctuation(line, max_len=80))  # 可以依需要調整 max_len

# 輸出新檔案
with open("city/output/shortened.json", "w", encoding="utf-8") as f:
    json.dump(all_sentences, f, ensure_ascii=False, indent=2)

print("切段 完成!")