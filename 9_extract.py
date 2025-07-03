import json

# === 自訂輸入與輸出檔案路徑 ===
input_path = "tone_data/output/women/segments_for_dataset.json"              # 輸入的原始 JSON 檔案
output_path = "tone_data/output/answer/women.json"      # 輸出的僅含 answer 欄位的新 JSON 檔案

# === 讀取原始 JSON 檔 ===
with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# === 只保留每筆中的 answer 欄位 ===
answers_only = [item["answer"] for item in data]


# === 儲存成新的 JSON 檔案 ===
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(answers_only, f, ensure_ascii=False, indent=2)

print("單獨分離 完成!")