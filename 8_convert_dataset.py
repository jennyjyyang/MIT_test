import json

# === 檔案路徑設定 ===
input_path = "city/output/audio_0/segments_for_splitter.json"
output_path = "city/output/audio_0/segments_for_dataset.json"
target_speaker = "SPEAKER_06"  # 指定你要作為回答者的 speaker

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
    # print(merged)
# 加入最後一段
if buffer:
    merged.append({"speaker": prev_speaker, "text": buffer})

# === 組成 Q&A 配對 ===
qa_pairs = []
temp_question = ""

for entry in merged:
    # print(entry["speaker"])
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
