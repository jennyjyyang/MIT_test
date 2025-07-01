import whisperx
import gc
import json
import os
import torch

device = "cuda"
audio_file = "tone_data/vocals/mdx_extra/marriage/vocals_mono16k.wav"
batch_size = 8  # reduce if low on GPU mem
compute_type = "float16"  # change to "int8" if low on GPU mem (may reduce accuracy)

# 建立輸出資料夾
os.makedirs("tone_data/output/marriage", exist_ok=True)

# 1. Transcribe with original whisper (batched)
model = whisperx.load_model("large-v2", device, compute_type=compute_type)
audio = whisperx.load_audio(audio_file)

result = model.transcribe(
    audio,
    batch_size=batch_size,
    language="zh",
)

with open("tone_data/output/marriage/voice_result.json", "w", encoding="utf-8") as f:
    json.dump(result['segments'], f, indent=4, ensure_ascii=False)

# 釋放 GPU 資源
gc.collect()
torch.cuda.empty_cache()
del model

# 2. Align whisper output
model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

with open("tone_data/output/marriage/voice_result_aligned.json", "w", encoding="utf-8") as f:
    json.dump(result["segments"], f, indent=4, ensure_ascii=False)

gc.collect()
torch.cuda.empty_cache()
del model_a

# 3. Assign speaker labels
diarize_model = whisperx.diarize.DiarizationPipeline(
    model_name="pyannote/speaker-diarization-3.1",
    use_auth_token="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",
    device=device
)

diarize_segments = diarize_model(audio)
result = whisperx.assign_word_speakers(diarize_segments, result)

print(diarize_segments)

with open("tone_data/output/marriage/voice_result_speakers.json", "w", encoding="utf-8") as f:
    json.dump(result["segments"], f, indent=4, ensure_ascii=False)

print("WhisperX 完成!")