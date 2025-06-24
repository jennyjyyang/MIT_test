import whisperx
import gc
import json

device = "cuda"
audio_file = "./voice_data/autobiography.mp3"
batch_size = 8 # reduce if low on GPU mem
compute_type = "float16" # change to "int8" if low on GPU mem (may reduce accuracy)

# 1. Transcribe with original whisper (batched)
model = whisperx.load_model("large-v2", device, compute_type=compute_type)

# save model to local path (optional)
# model_dir = "/path/"
# model = whisperx.load_model("large-v2", device, compute_type=compute_type, download_root=model_dir)

audio = whisperx.load_audio(audio_file)
result = model.transcribe(
    audio, 
    batch_size=batch_size,
    language="zh",
    # initial_prompt="臺灣的歷史臺灣的未來這是繁體中文", # optional
)
# print(result["segments"]) # before alignment
with open("voice_result.json", "w") as f:
    json.dump(result['segments'], f, indent=4, ensure_ascii=False)

# delete model if low on GPU resources
import gc; import torch; gc.collect(); torch.cuda.empty_cache(); del model

# 2. Align whisper output
model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

# print(result["segments"]) # after alignment
with open("voice_result_aligned.json", "w") as f:
    json.dump(result["segments"], f, indent=4, ensure_ascii=False)

# delete model if low on GPU resources
import gc; import torch; gc.collect(); torch.cuda.empty_cache(); del model_a

# 3. Assign speaker labels
diarize_model = whisperx.diarize.DiarizationPipeline(
    model_name="pyannote/speaker-diarization-3.1",
    use_auth_token="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", 
    device=device
)

# add min/max number of speakers if known
diarize_segments = diarize_model(audio)
# diarize_model(audio, min_speakers=min_speakers, max_speakers=max_speakers)

result = whisperx.assign_word_speakers(diarize_segments, result)
print(diarize_segments)
# print(result["segments"]) # segments are now assigned speaker IDs
with open("voice_result_speakers.json", "w") as f:
    json.dump(result["segments"], f, indent=4, ensure_ascii=False)

