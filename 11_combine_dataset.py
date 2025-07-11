import os
import json

# === 基本設定 ===
QUESTION_FILE = "output/questions_all.txt"           # 問題來源檔案
ANSWER_FOLDER = "output/answers_txt"                  # answers 資料夾路徑
OUTPUT_FILE = "output/qa_dataset_all.json"           # 輸出 JSON 檔案路徑

# === 讀取 Questions.txt 並解析 ===
with open(QUESTION_FILE, "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f.readlines()]

qa_pairs = []
i = 0
while i < len(lines):
    question = lines[i]

    # 強制全形問號結尾（將句尾標點統一清除後補上？）
    question = question.rstrip("?？.。!！…") + "？"

    # 確認是否跳過空行與取得來源
    if i + 2 < len(lines):
        source_line = lines[i + 2]
        if source_line.startswith("來源:") or source_line.startswith("来源:"):
            source_id = source_line.split(":")[1].strip().split("-")[0] + ".txt"
            answer_path = os.path.join(ANSWER_FOLDER, source_id)

            if os.path.exists(answer_path):
                with open(answer_path, "r", encoding="utf-8") as af:
                    answer = af.read().strip()
            else:
                answer = "[無對應回答]"

            qa_pairs.append({
                "instruction": question,
                "input": "",
                "output": answer,
                "system": ""
            })

            i += 3  # 跳到下一個 QA 區塊
        else:
            i += 1
    else:
        i += 1

# === 輸出 JSON 檔案 ===
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(qa_pairs, f, ensure_ascii=False, indent=2)

print("QA合併 完成!")