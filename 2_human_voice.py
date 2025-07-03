import demucs.separate

demucs.separate.main([ 
    "--two-stems", "vocals", 
    "--float32",
    "-n", "mdx_extra", 
    "-d","cuda",
    "tone_data/women.wav",
    "-o",
    "tone_data/vocals/",
])

print("人聲分離 完成!")
# The output will be saved in voice_data/vocals/
