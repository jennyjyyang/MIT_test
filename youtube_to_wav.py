import yt_dlp
from pydub import AudioSegment
import os

'''
Install required tools:
pip install yt-dlp pydub

You also need FFmpeg (if not already installed):
Ubuntu: sudo apt install ffmpegs
'''

def youtube_to_wav(youtube_url, output_dir="output/wav_files", base_filename="audio", count=0):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    filename = f"{base_filename}_{count}"
    temp_file = os.path.join(output_dir, f"{filename}.webm")
    wav_file = os.path.join(output_dir, f"{filename}.wav")

    ydl_opts = {
        'format': 'bestaudio',
        'outtmpl': temp_file,
        'quiet': True,
    }

    try:
        print(f"\n[{count}] Downloading: {youtube_url}")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])

        print(f"[{count}] Converting to WAV...")
        audio = AudioSegment.from_file(temp_file)
        audio.export(wav_file, format="wav")
        os.remove(temp_file)
        print(f"[{count}] WAV saved to: {wav_file}")
    except Exception as e:
        print(f"[{count}] Error processing {youtube_url}: {e}")

# For testing only, move all links to a text file for easier management in the future
links = [
    "https://www.youtube.com/watch?v=CKfU6Aow8Pc",
    "https://www.youtube.com/watch?v=PBH6T2qdq7s",
    "https://www.youtube.com/watch?v=CCHnS92WcA0",
    "https://www.youtube.com/watch?v=c348T5-v2Cs",
    "https://youtu.be/hMluiDVGxws",
    "https://youtu.be/CKfU6Aow8Pc?si=poLclJN06mQo9tto",
    "https://youtu.be/c348T5-v2Cs?si=T_GygB2GKF1WsDQs",
    "https://youtu.be/CCHnS92WcA0?si=eA-p0l1TWvWdBey1",
    "https://youtu.be/PBH6T2qdq7s?si=K8PRVF984Vz67CMv",
    "https://youtu.be/1uZ03VrrsPM"
]

for i, link in enumerate(links):
    youtube_to_wav(link, count=i)