# 安裝模組
import json
from datetime import timedelta
import os

# 迴圈批量處理
for i in range(1,10):
    # 修改路徑
    INPUT = f"audio_{i}"

    # 5. 轉換時間格式
    # 輸入和輸出檔案
    input_file = "city/output/" + INPUT + "/merged_output.json"
    output_file = "city/output/" + INPUT + "/segments_for_splitter.json"

    # 檢查輸入檔案是否存在，避免錯誤
    if not os.path.exists(input_file):
        print(f"找不到檔案：{input_file}，跳過...")
        continue

    # 將秒數轉換為 HH:MM:SS,mmm 格式
    def format_timestamp(seconds):
        td = timedelta(seconds=seconds)
        total_seconds = int(td.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        secs = total_seconds % 60
        millis = int((td.total_seconds() - total_seconds) * 1000)
        return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"

    # 載入 JSON 檔案
    with open(input_file, "r", encoding="utf-8") as f:
        segments = json.load(f)

    converted = {"segments": []}

    for seg in segments:
        if "speaker" in seg:
            new_seg = {
                "speaker": seg["speaker"],
                "start": format_timestamp(seg["start"]),
                "end": format_timestamp(seg["end"]),
                "text": seg["text"]
            }
            converted["segments"].append(new_seg)

    # 儲存新的 JSON 檔案
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(converted, f, indent=4, ensure_ascii=False)

    print(f"{INPUT} 可讀格式轉換 完成!")

    # 8.9. 轉換格式
    # 輸入和輸出檔案
    input_file = "city/output/" + INPUT + "/segments_for_splitter.json"
    output_file = "city/output/answer/" + INPUT + ".json"

    # 載入原始 input JSON 檔案（請根據實際檔案路徑修改）
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 提取所有 segments 裡面的 text 欄位，組成 list[str]
    merged_texts = [segment["text"] for segment in data["segments"]]

    # 將結果儲存為新的 JSON 檔案
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(merged_texts, f, ensure_ascii=False, indent=2)

    print(f"{INPUT} QA 格式輸出 完成!")