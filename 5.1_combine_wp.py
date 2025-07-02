import json
from datetime import timedelta

# === 路徑設定 ===
whisper_path = "tone_data/output/autobiography/whisper_output.json"
pyannote_path = "tone_data/output/autobiography/pyannote_output.json"
merged_path = "tone_data/output/autobiography/merged_output.json"
formatted_output_path = "tone_data/output/autobiography/segments_for_splitter.json"

# === 工具：將秒數轉換成 HH:MM:SS,mmm ===
def format_timestamp(seconds):
    td = timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    millis = int((td.total_seconds() - total_seconds) * 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"

# === 載入 whisper 和 pyannote 的 JSON ===
with open(whisper_path, "r", encoding="utf-8") as f:
    whisper_segments = json.load(f)

with open(pyannote_path, "r", encoding="utf-8") as f:
    pyannote_segments = json.load(f)

# === 方法二：根據 pyannote 的區段對 whisper 進行切割 ===
merged = []

for w_seg in whisper_segments:
    w_start = w_seg["start"]
    w_end = w_seg["end"]
    w_text = w_seg["text"]

    for p_seg in pyannote_segments:
        p_start = p_seg["start"]
        p_end = p_seg["end"]
        speaker = p_seg["speaker"]

        # 計算重疊區段
        overlap_start = max(w_start, p_start)
        overlap_end = min(w_end, p_end)

        if overlap_start < overlap_end:
            proportion = (overlap_end - overlap_start) / (w_end - w_start)
            sub_text_len = int(len(w_text) * proportion)
            sub_text = w_text[:sub_text_len].strip()
            w_text = w_text[sub_text_len:].strip()

            merged.append({
                "start": round(overlap_start, 3),
                "end": round(overlap_end, 3),
                "speaker": speaker,
                "text": sub_text
            })

# === 儲存中間合併結果（秒數） ===
with open(merged_path, "w", encoding="utf-8") as f:
    json.dump(merged, f, indent=2, ensure_ascii=False)

# === 轉換格式為 HH:MM:SS,mmm 並輸出 ===
converted = {"segments": []}

for seg in merged:
    new_seg = {
        "speaker": seg["speaker"],
        "start": format_timestamp(seg["start"]),
        "end": format_timestamp(seg["end"]),
        "text": seg["text"]
    }
    converted["segments"].append(new_seg)

with open(formatted_output_path, "w", encoding="utf-8") as f:
    json.dump(converted, f, indent=4, ensure_ascii=False)

print("合併 完成！")
