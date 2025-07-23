# 安裝模組
import demucs.separate
import subprocess
import os

# 迴圈批量處理
for i in range(1,10):
    # 修改路徑
    INPUT = f"audio_{i}"

    #2. 分離人聲
    # 使用 demucs 分離人聲
    demucs.separate.main([ 
        "--two-stems", "vocals", 
        "--float32",
        "-n", "mdx_extra", 
        "-d","cuda",
        "city/wav_files/"+INPUT+".wav",
        "-o",
        "city/vocals/",
    ])

    print("人聲分離 完成!")

    # 3. 轉換音檔參數
    # 使用 ffmpeg 將音檔轉換為單聲道 16kHz
    # 原始音檔路徑
    input_path = "city/vocals/mdx_extra/"+INPUT+"/vocals.wav"

    # 輸出音檔路徑
    output_path = "city/vocals/mdx_extra/"+INPUT+"/vocals_mono16k.wav"

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