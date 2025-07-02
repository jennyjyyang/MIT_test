from pyannote.audio import Pipeline
import torch
import json

# === 初始化 pipeline 與裝置 ===
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token="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")

pipeline.to(torch.device("cuda"))

# === 指定音訊路徑與輸出路徑 ===
input_audio_path = "tone_data/vocals/mdx_extra/autobiography/vocals_mono16k.wav"
output_json_path = "tone_data/output/autobiography/pyannote_output.json"

# === 執行語者分離 ===
diarization = pipeline(input_audio_path)

# === 輸出到螢幕 + 存成 JSON ===
output = []
for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")
    output.append({
        "start": round(turn.start, 3),
        "end": round(turn.end, 3),
        "speaker": speaker
    })

with open(output_json_path, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print("Pyannote 完成!")

