# Fine tune for tone
## Dataset 清理目標

```bash
cd ~/Desktop/MIT_test
poetry env activate
source .venv/bin/activate
```

1. 簡體轉繁體 1_convert_language.py
2. 加標點與斷句 2_seperate_sentence.py
3. 刪除語者分離錯誤重複語句 3_remove_repeat.py
4. 對每筆回答進行分類 4_devide_group.py 4_top2vec.py 4_bertopic.py
5. 清除低品質或錯誤的語料
6. 微調資料不要過長 6_seperate_part.py
7. 對同一個回答產生多個不同問題（反覆問答訓練）7_folder.py 7_convert_txt.py
8. 添加原始模型訓練句

### Sample

| group | input | mini size | label |
|:---:|:---:|:---:|:---:|
| 00 | answer | 5 | x |
| 01 | answer | 2 | x |
| 02 | answer | 2 | v |
| 03 | sentence | 2 | v |
| 04 | sentence_clean | 2 | v |
| 05 | shortened | 2 | v |

| bertopic | MODEL | 關鍵字 | MIN_TOPIC_SIZE |
|:---:|:---:|:---:|:---:|
| 01 | shibing624/text2vec-base-chinese | x | 2 |
| 02 | shibing624/text2vec-base-chinese | c-TF-ITF | 2 |
| 03 | shibing624/text2vec-base-chinese | n-gram | 2 |
| 04 | paraphrase-multilingual-mpnet-base-v2 | c-TF-ITF | 2 |
| 05 | shibing624/text2vec-base-chinese | c-TF-ITF | 5 |
| 06 | uer/sbert-base-chinese-nli | c-TF-ITF | 2 |
| 07 | uer/sbert-base-chinese-nli | c-TF-ITF | 3 |
| 08 | shibing624/text2vec-base-chinese | c-TF-ITF | 3 |

個人感覺06效果比較好

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
