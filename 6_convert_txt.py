import os
import json

# === 設定 ===
INPUT_DIR = "output/answers"            # 你剛剛產生的 .json 單句資料夾
OUTPUT_DIR = "output/answers_txt"             # 要輸出的 .txt 資料夾

os.makedirs(OUTPUT_DIR, exist_ok=True)

# === 處理每一個 .json 檔 ===
for filename in os.listdir(INPUT_DIR):
    if filename.endswith(".json"):
        input_path = os.path.join(INPUT_DIR, filename)
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 確認內容是 list 且第一句是你要的
        if isinstance(data, list) and data:
            text = data[0].strip()

            # 儲存成 .txt
            output_filename = filename.replace(".json", ".txt")
            output_path = os.path.join(OUTPUT_DIR, output_filename)
            with open(output_path, "w", encoding="utf-8") as f_out:
                f_out.write(text)

print("json 轉 txt 完成!")
