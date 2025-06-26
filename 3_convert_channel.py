import subprocess
import os

# 原始音檔路徑
input_path = "voice_data/vocals/mdx_extra/autobiography/vocals.wav"

# 輸出音檔路徑
output_path = "voice_data/vocals/mdx_extra/autobiography/vocals_mono16k.wav"

# 確保輸出資料夾存在
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# ffmpeg 指令組合
command = [
    "ffmpeg",
    "-y",  # 自動覆蓋
    "-i", input_path,     # 輸入檔案
    "-ac", "1",           # 單聲道
    "-ar", "16000",       # 16kHz
    "-c:a", "pcm_s16le",  # 16-bit PCM
    output_path
]

# 執行指令
try:
    subprocess.run(command, check=True)
    print("參數轉換 完成！")
except subprocess.CalledProcessError as e:
    print("參數轉換 失敗：", e)


