import json
import os

# === 設定 ===
INPUT_PATH = "output/shortened.json"        # 輸入檔案：list[str]
OUTPUT_DIR = "output/answers"           # 輸出資料夾名稱

# === 建立輸出資料夾 ===
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === 讀取回答清單 ===
with open(INPUT_PATH, "r", encoding="utf-8") as f:
    answers = json.load(f)

print(f"載入 {len(answers)} 筆回答，開始轉換...")

# === 一句話一個 JSON 檔（list 格式）===
for i, answer in enumerate(answers, start=1):
    file_name = f"{i:04d}.json"
    output_path = os.path.join(OUTPUT_DIR, file_name)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump([answer], f, ensure_ascii=False, indent=2)

print("一句一檔 完成!")
