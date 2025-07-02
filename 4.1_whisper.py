import whisper
import json
import torch
import torchaudio

# === 你要處理的音訊檔案路徑 ===
input_audio_path = "tone_data/vocals/mdx_extra/autobiography/vocals_mono16k.wav"
output_json_path = "tone_data/output/autobiography/whisper_output.json"

# === 1. 載入 Whisper 模型 ===
model = whisper.load_model("large")

# === 2. 讀入單聲道音檔（不裁切）===
audio, sr = torchaudio.load(input_audio_path)
audio = audio[0]  # 確定是單聲道
total_duration = audio.shape[-1] / sr

# === 3. 切成 30 秒片段並解碼 ===
segment_length = 30  # seconds
segments = []

for i in range(0, int(total_duration), segment_length):
    start_sec = i
    end_sec = min(i + segment_length, int(total_duration))
    print(f"處理第 {i // segment_length + 1} 段：{start_sec:.1f} 到 {end_sec:.1f} 秒")
    audio_chunk = audio[int(start_sec * sr):int(end_sec * sr)]
    audio_chunk = whisper.pad_or_trim(audio_chunk)

    # 語音辨識（用 transcribe 才會有 segments）
    result = model.transcribe(audio_chunk, language=None, temperature=0.0, fp16=True, without_timestamps=False, beam_size=5)

    # 語言偵測（僅顯示第一次）
    if i == 0:
        print(f"Detected language: {result['language']}")

    for seg in result["segments"]:
        segments.append({
            "start": float(start_sec + seg["start"]),
            "end": float(start_sec + seg["end"]),
            "text": seg["text"]
        })

# === 4. 輸出為 JSON，含每段資訊 ===
with open(output_json_path, "w", encoding="utf-8") as f:
    json.dump(segments, f, indent=2, ensure_ascii=False)

print("Whisper 完成！")


