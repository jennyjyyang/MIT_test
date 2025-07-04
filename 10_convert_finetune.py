import json

# 讀入原始檔案
with open("easydataset.json", "r", encoding="utf-8") as f:
    original_data = json.load(f)

# 格式轉換
converted_data = []
for item in original_data:
    converted_data.append({
        "instruction": item["question"],
        "input": "",
        "output": item["answer"],
        "system": ""
    })

# 寫入新檔案
with open("finetune.json", "w", encoding="utf-8") as f:
    json.dump(converted_data, f, ensure_ascii=False, indent=2)

print("Fine Tune 格式轉換 完成!")