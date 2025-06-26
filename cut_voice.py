from pydub import AudioSegment

# 載入 mp3 或 wav 音訊檔案
audio = AudioSegment.from_mp3("x_output-audio-SPEAKER_02.wav")

# 轉換時間字串成毫秒
def timestamp_to_ms(t):
    h, m, s_ms = t.split(":")
    s, ms = s_ms.split(",")
    return (int(h) * 3600 + int(m) * 60 + int(s)) * 1000 + int(ms)

# 設定要切割的時間範圍
start_time = timestamp_to_ms("00:02:55,709")
end_time = timestamp_to_ms("00:03:11,729")

# 切割並輸出
clip = audio[start_time:end_time]
clip.export("speaker02_clip.wav", format="wav")

print("已成功輸出 speaker02_clip！")

