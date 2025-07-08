# Fine tuning for Vocal
## TTS Model Pipeline 結論
1. speaker01_clip不能加逐字稿，電腦GPU跑不動。
2. 有加逐字稿比較不會出現error的狀況。(幾乎沒有)
3. 時間切割15s的效果比25s好。(speaker02_clip的效果比speaker01_clip好)
4. 拿掉句號全用逗號可以避免段落合成的電音，但是會比較容易出現結尾迴圈的狀況。

## 操作流程
### .py檔從1跑到7然後接brezzy voice(改檔案路徑)

```bash
#before start
cd ~/Desktop/MIT_test
python3.12 -m venv .venv312
source .venv312/bin/activate
```

```bash
# before step4
export LD_LIBRARY_PATH=`poetry run python -c 'import os; import nvidia.cublas.lib; import nvidia.cudnn.lib; print(os.path.dirname(nvidia.cublas.lib.__file__) + ":" + os.path.dirname(nvidia.cudnn.lib.__file__))'`
```

```bash
# run step6
poetry run python 6_speaker_splitter.py voice_data/vocals/mdx_extra/autobiography/vocals_mono16k.wav voice_data/output/segments_for_splitter.json 
# poetry run python 5_speaker_splitter.py 路徑/你的音檔.wav 路徑/你的json檔.json
```

```bash
# before brezzy voice
deactivate
git clone https://github.com/mtkresearch/BreezyVoice.git
cd ~/Desktop/MIT_test/BreezyVoice
python3.10 -m venv .venv310
source .venv310/bin/activate
# 修改 requirements
pip install -r requirements.txt
```

```bash
# run single inference
poetry run python3 single_inference.py \
  --speaker_prompt_audio_path "../voice_data/output/speaker01_clip.wav" \
  --speaker_prompt_text_transcription "" \
  --content_to_synthesize "你好，我是陳文茜，很開心認識你。" \
  --output_path "../voice_data/output/result.wav"
```

```bash
# run batch inference
poetry run python3 batch_inference.py \
  --csv_file ./batch_files.csv \
  --speaker_prompt_audio_folder ../voice_data/output \
  --output_audio_folder ../voice_data/output/batch
```

## Test Sample

```bash
#test context
poetry run python3 single_inference.py \
  --speaker_prompt_audio_path "../voice_data/output/speaker01_clip.wav" \
  --speaker_prompt_text_transcription "" \
  --content_to_synthesize "也沒有什麼挑戰，我的外婆是對我過度自信、過度袒護。我外婆有幾個心理，雖然我平常可能不太教功課、可能有我自己的、就會活在我自己的幻想世界。" \
  --output_path "../voice_data/output/result00.wav"

poetry run python3 single_inference.py \
  --speaker_prompt_audio_path "../voice_data/output/speaker01_clip.wav" \
  --speaker_prompt_text_transcription "" \
  --content_to_synthesize "媽媽說她很容易緊張，因為我是她第一個小孩。那時我發燒，她不知道該怎麼照顧，就讓外婆接手。外婆說我水土不服，從花蓮帶我回台中。其實她一直很疼我，怕我被媽媽帶走。有時候還會說：「我帶妳去買禮物，但妳不能跟妳媽媽講話喔。」" \
  --output_path "../voice_data/output/result01.wav"

poetry run python3 single_inference.py \
  --speaker_prompt_audio_path "../voice_data/output/speaker01_clip.wav" \
  --speaker_prompt_text_transcription "" \
  --content_to_synthesize "媽媽說她是個很容易 nervous 的人啦，因為我是她的 first kid。她那時候不知道怎麼照顧我，就讓外婆幫忙。外婆說我水土不服，就從花蓮帶回台中，還說：「我買禮物給妳，但不能跟妳媽講話喔。」" \
  --output_path "../voice_data/output/result02.wav"

poetry run python3 single_inference.py \
  --speaker_prompt_audio_path "../voice_data/output/speaker01_clip.wav" \
  --speaker_prompt_text_transcription "" \
  --content_to_synthesize "My mom said she was really nervous because I was her first child. When I got sick, she didn’t know how to take care of me, so my grandma stepped in. Grandma thought I couldn’t adjust to the environment, so she brought me back from Hualien to Taichung and took care of me ever since." \
  --output_path "../voice_data/output/result03.wav"

poetry run python3 single_inference.py \
  --speaker_prompt_audio_path "../voice_data/output/speaker01_clip.wav" \
  --speaker_prompt_text_transcription "" \
  --content_to_synthesize "嗯…媽媽她說她很緊張啦，因為我是她第一個小孩嘛。然後…我那時候發燒，她就…嗯，不知道怎麼辦，就讓外婆來照顧我。外婆就說可能是水土不服吧，就從花蓮帶我回台中這樣。後來她還…會說「妳乖乖我就帶妳去買襪子喔，可是妳不能跟妳媽媽講話喔～」" \
  --output_path "../voice_data/output/result04.wav"

poetry run python3 single_inference.py \
  --speaker_prompt_audio_path "../voice_data/output/speaker01_clip.wav" \
  --speaker_prompt_text_transcription "" \
  --content_to_synthesize "媽媽說她很容易緊張因為我是她第一個小孩那時我發燒她不知道該怎麼照顧就讓外婆接手外婆說我水土不服從花蓮帶我回台中其實她一直很疼我怕我被媽媽帶走有時候還會說我帶妳去買禮物但妳不能跟妳媽媽講話喔" \
  --output_path "../voice_data/output/result05.wav"

poetry run python3 single_inference.py \
  --speaker_prompt_audio_path "../voice_data/output/speaker01_clip.wav" \
  --speaker_prompt_text_transcription "我會去買Live雜誌，然後每天活在我的外太空，想像有一天我要變成太空人，然後我要，所以我常常在想說，如果我生在美國、或是俄國我可能會去NASA，你知道嗎太空總署上班，但是我沒有太忠於自己OK，那我就覺得說可能，這個在我們台灣這個社會不是很practical這樣，那我外婆只是說她在這個世界上她沒有給你任何的壓力，它差別在什麼地方呢，第一個。" \
  --content_to_synthesize "嗯媽媽她說她很緊張啦，因為我是她第一個小孩嘛，然後我那時候發燒，她就嗯，不知道怎麼辦，就讓外婆來照顧我，外婆就說可能是水土不服吧，就從花蓮帶我回台中這樣，後來她還會說，妳乖乖我就帶妳去買襪子喔，可是妳不能跟妳媽媽講話喔。" \
  --output_path "../voice_data/output/result06.wav"

poetry run python3 single_inference.py \
  --speaker_prompt_audio_path "../voice_data/output/speaker01_clip.wav" \
  --speaker_prompt_text_transcription "" \
  --content_to_synthesize "嗯媽媽她說她很緊張啦，因為我是她第一個小孩嘛，然後我那時候發燒，她就嗯，不知道怎麼辦，就讓外婆來照顧我，外婆就說可能是水土不服吧，就從花蓮帶我回台中這樣，後來她還會說，妳乖乖我就帶妳去買襪子喔，可是妳不能跟妳媽媽講話喔。" \
  --output_path "../voice_data/output/result07.wav"

poetry run python3 single_inference.py \
  --speaker_prompt_audio_path "../voice_data/output/speaker02_clip.wav" \
  --speaker_prompt_text_transcription "我會去買Live雜誌，然後每天活在我的外太空，想像有一天我要變成太空人，然後我要，所以我常常在想說，如果我生在美國、或是俄國我可能會去NASA，你知道嗎太空總署上班，但是我沒有太忠於自己OK，那我就覺得說。" \
  --content_to_synthesize "嗯媽媽她說她很緊張啦，因為我是她第一個小孩嘛，然後我那時候發燒，她就嗯，不知道怎麼辦，就讓外婆來照顧我，外婆就說可能是水土不服吧，就從花蓮帶我回台中這樣，後來她還會說，妳乖乖我就帶妳去買襪子喔，可是妳不能跟妳媽媽講話喔。" \
  --output_path "../voice_data/output/result08.wav"

poetry run python3 single_inference.py \
  --speaker_prompt_audio_path "../voice_data/output/speaker02_clip.wav" \
  --speaker_prompt_text_transcription "" \
  --content_to_synthesize "嗯媽媽她說她很緊張啦，因為我是她第一個小孩嘛，然後我那時候發燒，她就嗯，不知道怎麼辦，就讓外婆來照顧我，外婆就說可能是水土不服吧，就從花蓮帶我回台中這樣，後來她還會說，妳乖乖我就帶妳去買襪子喔，可是妳不能跟妳媽媽講話喔。" \
  --output_path "../voice_data/output/result09.wav"
```

# Fine tuning for tone
## Easy DataSet 操作流程
### .py檔跑1-2-3-4.1-4.2-5.1(改檔案路徑)(效果不佳)

```bash
#before start
cd ~/Desktop/MIT_test
source .venv312/bin/activate
```

```bash
# before step4.1
deactivate
python3.9 -m venv .venv309w
source .venv309w/bin/activate
pip install -U openai-whisper
pip install git+https://github.com/openai/whisper.git 
pip install --upgrade --no-deps --force-reinstall git+https://github.com/openai/whisper.git
sudo apt update && sudo apt install ffmpeg
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
pip install setuptools-rust
```

```bash
# before step4.2
deactivate
python3.10 -m venv .venv310p
source .venv310p/bin/activate
pip install pyannote.audio
# 修改speaker number
```

```bash
# before step5.1
deactivate
source .venv312/bin/activate
```

### .py檔跑1-2-3-4pw-5-8-9(改檔案路徑)(效果最好)

```bash
#before start
cd ~/Desktop/MIT_test
source .venv312/bin/activate
```

```bash
# before step4pw
deactivate
git clone https://github.com/yinruiqing/pyannote-whisper.git
cd ~/Desktop/MIT_test/pyannote-whisper
pyenv local 3.9.13
python -m venv .venv309
source .venv309/bin/activate
pip install -r ../requirements_pw.txt
pip install -e .

deactivate
cd ~/Desktop/MIT_test/pyannote-whisper
source .venv309/bin/activate
```

```bash
# before step5
deactivate
cd ~/Desktop/MIT_test
source .venv312/bin/activate
```

## Dataset 清理目標

```bash
cd ~/Desktop/MIT_test/tone_clean
```

1. 簡體轉繁體 convert_language.py
2. 加標點與斷句
3. 刪除語者分離錯誤重複語句
4. 對每筆回答進行分類 4_devide_group.py 4_top2vec.py
5. 清除低品質或錯誤的語料
6. 微調資料不要過長
7. 對同一個回答產生多個不同問題（反覆問答訓練）
8. 添加原始模型訓練句

### 1️⃣ 簡體轉繁體
🎯 目的：統一語言格式為台灣用字，避免訓練語氣混亂
📌 子任務：
將所有簡體字轉為繁體（使用台灣常見用語）

🛠️ 工具：
opencc-python-reimplemented，轉換模式：s2twp.json

✅ 自動化：✅ 批次處理 answer 欄位即可
📝 備註：
可加入簡體詞偵測機制，跳過已是繁體的資料

### 2️⃣ 加標點與斷句
🎯 目的：使回答更接近真實口語語調，提升語氣學習效果
📌 子任務：
自動為回答句子補標點與斷句（如逗號、句號、問號）

🛠️ 工具：
deep-punctuation-zh

可搭配 jieba 做詞切割

✅ 自動化：✅ 支援批次處理
📝 備註：
模型預訓練語料大多含標點 → 若無標點將影響 fine-tune 效果

### 3️⃣ 刪除語者分離錯誤重複語句
🎯 目的：去除語料雜訊，避免模型學到錯誤語調或重複話術
📌 子任務：
比對每筆回答與上下文是否高度相似，去除時間接近且重複的句子

🛠️ 工具：
Python difflib.SequenceMatcher

fuzzywuzzy（模糊字串比對）

✅ 自動化：✅ 可設定相似度閾值自動過濾
📝 備註：
建議保留較長的版本，刪除重複且短的內容

### 4️⃣ 對每筆回答進行分類
🎯 目的：標記語氣風格類型，方便後續分組訓練或生成問題
📌 子任務：
為回答打上語氣類別，例如：「感性」、「評論」、「反問」、「冷靜敘述」、「主持語」

🛠️ 工具：
使用 GPT-4 / ChatGPT API，分類 prompt 可定義語氣分類邏輯

✅ 自動化：✅ 可批次分類，儲存為 type 欄位
📝 備註：
可配合選擇性訓練策略，例如僅訓練評論類回答

### 5️⃣ 清除低品質或錯誤的語料
🎯 目的：避免訓練資料出現語病、非語氣句或空白內容
📌 子任務：
移除過短（< 20 字）、非語句（「嗯」、「呵呵」）、拼音或亂碼

檢查格式是否合法（JSON 結構完整）

🛠️ 工具：
Python 長度判斷 + 關鍵詞排除 + json 檢查

✅ 自動化：✅ 可批次刪除或標記
📝 備註：
可額外輸出「刪除資料表」供人工複查

### 6️⃣ 微調資料不要過長
🎯 目的：避免訓練時輸入長度超過 token 限制、造成語氣崩壞
📌 子任務：
對 instruction + output 長度加總後，限制於 1024~2048 tokens 內

🛠️ 工具：
Python + tokenizer（例如 transformers.AutoTokenizer.from_pretrained("Qwen")）

✅ 自動化：✅ 可自動裁切、移除、警告
📝 備註：
太長的回答會導致 attention 不集，風格學習不集中

### 7️⃣ 對同一個回答產生多個不同問題（反覆問答訓練）
🎯 目的：強化語氣穩定性，不管問法怎麼變，回答語氣一致
📌 子任務：
對同一筆回答產出 2~3 種不同但語意相關的提問（paraphrase）

🛠️ 工具：
ChatGPT API、paraphrasing model（如 Pegasus、BART 中文化）

✅ 自動化：✅ 可批次處理（多對一）
📝 備註：
請保留回答不變，instruction 變化，才能學出語氣穩定

### 8️⃣ 添加原始模型訓練句（Qwen instruction anchor）
🎯 目的：避免模型失去基本對話能力、保持語言結構正常
📌 子任務：
從 Qwen2.5 原始 instruction-tuning 資料集中擷取中性範例（如：社交、問答、小知識）

🛠️ 工具：
Hugging Face Qwen2.5 SFT 資料（或我可幫你提取）

✅ 自動化：✅ 可抽樣並加入到原始 dataset 中（混合比例建議 5–10%）
📝 備註：
可作為語氣 anchor，也能預防過擬合

## Llama-Factory 操作流程
### 用chatgpt生成QA然後.py檔跑10

http://10.100.1.124:7860
