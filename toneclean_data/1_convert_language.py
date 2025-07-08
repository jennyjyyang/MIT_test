import json
from opencc import OpenCC

# 初始化 OpenCC 轉換器（簡體 → 台灣繁體）
cc = OpenCC('s2t')

# 載入字串 list 格式的 .json 檔
with open("output/answer.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# 執行繁體轉換
converted_data = [cc.convert(text) if isinstance(text, str) else text for text in data]

# 儲存為新檔案
with open("output/traditional.json", "w", encoding="utf-8") as f:
    json.dump(converted_data, f, ensure_ascii=False, indent=2)

print("簡體轉繁體 完成!")
