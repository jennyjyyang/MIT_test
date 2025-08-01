import torch
import whisper
from pyannote.audio.pipelines.speaker_diarization import SpeakerDiarization
from pyannote_whisper.utils import diarize_text
import json
import os

# Step 1: 載入模型
# Whisper 模型（建議使用 "large"）
model = whisper.load_model("large")

# Pyannote 模型（替換為你的 Hugging Face token）
pipeline = SpeakerDiarization.from_pretrained("pyannote/speaker-diarization-3.1",
                     use_auth_token=os.getenv("HUGGINGFACE_TOKEN"))
pipeline.to(torch.device("cuda"))

# === ✅ 加上迴圈跑 audio_1 到 audio_9 ===
for i in range(1, 10):
    # 修改路徑
    INPUT = f"audio_{i}"

    # Step 2: 設定音檔路徑
    audio_path = "../city/vocals/mdx_extra/" + INPUT + "/vocals_mono16k.wav"

    # Step 3: 語音辨識（Whisper）
    asr_result = model.transcribe(audio_path)

    # Step 4: 語者分離（Pyannote）
    diarization_result = pipeline(audio_path)

    # Step 5: 合併 Whisper 與 Pyannote 結果
    final_result = diarize_text(asr_result, diarization_result)

    # Step 6: 存成 JSON
    output_data = [
        {
            "start": segment.start,
            "end": segment.end,
            "speaker": speaker,
            "text": sentence
        }
        for segment, speaker, sentence in final_result
    ]

    # 檢查並建立資料夾
    output_folder = "../city/output/" + INPUT
    os.makedirs(output_folder, exist_ok=True)

    with open(output_folder + "/merged_output.json", "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"{INPUT} Pyannote-Whisper 完成!")
