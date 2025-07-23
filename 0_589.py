# 安裝模組
import json
from datetime import timedelta

# 修改路徑
INPUT="audio_0"
target_speaker = "SPEAKER_06"  # 指定你要作為回答者的 speaker

# 5. 轉換時間格式
# 輸入和輸出檔案
input_file = "city/output/"+INPUT+"/merged_output.json"
output_file = "city/output/"+INPUT+"/segments_for_splitter.json"

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

print("可讀格式轉換 完成!")

# 8. 轉換QA格式
# === 檔案路徑設定 ===
input_path = "city/output/"+INPUT+"/segments_for_splitter.json"
output_path = "city/output/"+INPUT+"/segments_for_dataset.json"

# === 讀取 JSON 檔案 ===
with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

segments = data["segments"]

# === 將連續同一 speaker 的發言合併（避免時間重疊太短造成切段） ===
merged = []
prev_speaker = None
buffer = ""
for seg in segments:
    speaker = seg["speaker"]
    text = seg["text"].strip()

    if speaker == prev_speaker:
        # 若與前一個 speaker 相同，則合併文字
        if text != buffer:  # 避免完全重複
            buffer += text
    else:
        if buffer:
            merged.append({"speaker": prev_speaker, "text": buffer})
        buffer = text
        prev_speaker = speaker

# 加入最後一段
if buffer:
    merged.append({"speaker": prev_speaker, "text": buffer})

# === 組成 Q&A 配對 ===
qa_pairs = []
temp_question = ""

for entry in merged:
    if entry["speaker"] == target_speaker:
        if temp_question.strip():
            qa_pairs.append({
                "question": temp_question.strip(),
                "answer": entry["text"].strip()
            })
            temp_question = ""
    else:
        temp_question += entry["text"].strip()

# === 輸出 JSON ===
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(qa_pairs, f, ensure_ascii=False, indent=2)

print("QA格式轉換 完成！")

# 9. 擷取Answer
# === 自訂輸入與輸出檔案路徑 ===
input_path = "city/output/"+INPUT+"/segments_for_dataset.json"              # 輸入的原始 JSON 檔案
output_path = "city/output/answer/"+INPUT+".json"      # 輸出的僅含 answer 欄位的新 JSON 檔案

# === 讀取原始 JSON 檔 ===
with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# === 只保留每筆中的 answer 欄位 ===
answers_only = [item["answer"] for item in data]


# === 儲存成新的 JSON 檔案 ===
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(answers_only, f, ensure_ascii=False, indent=2)

print("單獨分離 完成!")