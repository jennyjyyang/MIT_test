import json
import re
from tqdm import tqdm

INPUT_PATH = "output/sentence.json"
OUTPUT_PATH = "output/sentence_clean.json"

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