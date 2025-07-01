import json

# 輸入與輸出檔案名稱
input_path = "tone_data/output/autobiography/segments_for_splitter.json"
output_path = "tone_data/output/autobiography/segments_for_dataset.json"

# 讀取原始 JSON
with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

segments = data["segments"]

# 載入原始 JSON
with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

segments = data["segments"]

# 將同 speaker 的對話合併
merged = []
prev_speaker = None
buffer = ""

for seg in segments:
    speaker = seg["speaker"]
    text = seg["text"].strip()

    if speaker == prev_speaker:
        buffer += text
    else:
        if prev_speaker is not None:
            merged.append({"speaker": prev_speaker, "text": buffer})
        buffer = text
        prev_speaker = speaker

# 加入最後一段
if buffer:
    merged.append({"speaker": prev_speaker, "text": buffer})

# 依據 SPEAKER_03 作為回答者，其餘為提問者建立 Q&A
qa_pairs = []
temp_question = ""

for entry in merged:
    if entry["speaker"] == "SPEAKER_05":
        if temp_question:
            qa_pairs.append({
                "question": temp_question,
                "answer": entry["text"]
            })
            temp_question = ""
    else:
        temp_question += entry["text"]

# 輸出 JSON
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(qa_pairs, f, ensure_ascii=False, indent=2)

print("可讀格式轉換 完成!")
