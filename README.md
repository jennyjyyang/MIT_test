# title
## st
### lt

- dot1
- dot2
- [ ] blank  
- [x] check  
`test`

```bash
# before step4
export LD_LIBRARY_PATH=`poetry run python -c 'import os; import nvidia.cublas.lib; import nvidia.cudnn.lib; print(os.path.dirname(nvidia.cublas.lib.__file__) + ":" + os.path.dirname(nvidia.cudnn.lib.__file__))'`
```

```bash
# run step6
poetry run python 6_speaker_splitter.py voice_data/vocals/mdx_extra/autobiography/vocals_mono16k.wav voice_data/output/segments_for_splitter.json 
# poetry run python 5_speaker_splitter.py 路徑/你的音檔.wav 路徑/你的json檔.json
```