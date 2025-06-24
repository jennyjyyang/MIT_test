import json
from datetime import timedelta

# 輸入和輸出檔案
input_file = "voice_result_speakers.json"
output_file = "segments_for_splitter.json"

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

print("已成功轉換格式為 speaker-splitter 可讀格式: segments_for_splitter.json")
