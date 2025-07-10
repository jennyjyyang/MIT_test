import json
import re

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
with open("output/sentence_clean.json", "r", encoding="utf-8") as f:
    data = json.load(f)  # 資料應該是 list[str]

all_sentences = []
for line in data:
    all_sentences.extend(split_by_punctuation(line, max_len=80))  # 可以依需要調整 max_len

# 輸出新檔案
with open("output/shortened.json", "w", encoding="utf-8") as f:
    json.dump(all_sentences, f, ensure_ascii=False, indent=2)

print("切段 完成!")