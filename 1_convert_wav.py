from pydub import AudioSegment

# 原始 mp3 檔案路徑
mp3_path = "./voice_data/autobiography.mp3"

# 轉檔後輸出的 wav 檔案路徑
wav_path = "./voice_data/autobiography.wav"

# 讀取 mp3 並轉成 wav
audio = AudioSegment.from_mp3(mp3_path)
audio.export(wav_path, format="wav")

print("mp3轉wav 完成!")
