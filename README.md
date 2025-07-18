# Fine tune for style
## Before QA Dataset Generation
### 操作流程

```bash
cd ~/Desktop/MIT_test
poetry env activate
source .venv/bin/activate
```

1. 簡體轉繁體 1_convert_language.py
2. 加標點與斷句 2_seperate_sentence.py
3. 刪除語者分離錯誤重複語句 3_remove_repeat.py
4. 微調資料不要過長 4_seperate_part.py
5. 對每筆回答進行分類 5_devide_group.py 5_top2vec.py 5_bertopic.py
6. 輸出回答 6_folder.py 6_convert_txt.py

### Sample

| group | input | mini size | label |
|:---:|:---:|:---:|:---:|
| 00 | answer | 5 | x |
| 01 | answer | 2 | x |
| 02 | answer | 2 | v |
| 03 | sentence | 2 | v |
| 04 | sentence_clean | 2 | v |
| 05 | shortened | 2 | v |

n_neighbors=5  
n_components=5  
min_dist=0.0  

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

## QA Dataset Generation
### 領域樹主題

1. 社會與政治
    1. 社會議題
    2. 國際局勢
    3. 經濟產業
    4. 政治思想
2. 自我與情感
    1. 原生家庭
    2. 愛情觀點
    3. 身份認同
    4. 心理情緒
3. 藝術與哲學
    1. 藝術觀點
    2. 音樂喜好
    3. 哲學反思
    4. 歷史反思
4. 教育與生活
    1. 語言學習
    2. 教育經驗
    3. 日常生活
    4. 旅行經驗

### Instruction

你是問題制造機。
先從給予的資料中辨識該內容的主旨，再將主旨轉為問題的形式。要求以第二人稱的視角，一次只產生一個問句，且產生的問題敘述要合理、可涵蓋的答案範圍要大，不要太過具體也不要使用是非題。以繁體中文回復。
question: 當然，of course。我就告訴你說，我其實沒有女性主義者那種什麼悲憤啊，怎麼樣，怎麼樣，因為我覺得這種事情就是依照每一個人的狀況，有的時候你會碰到對你很適合的，有的時候你碰到沒有那麼適合的，那也不是百分之百適合或是百分之百不適合嘛，那這是一個老故事，就是說我們可能從小都被創造了對愛情的過度期待對不對，然後呢，有些人就覺得叫壞面，那我覺得其實是清醒到一個年齡。那我到五十歲還不清醒我不是完了嗎?
answer: 你覺得面對愛情與人生，我們什麼時候該學會清醒？

## After QA Dataset Generation
### 用Easy DataSet生成QA

0. 清除低品質或錯誤的語料
0. 對每筆回答進行分類
1. 對同一個回答產生多個不同問題（反覆問答訓練）11_combine_dataset.py
2. 添加原始模型訓練句

## Llama-Factory
### 操作流程

http://10.100.1.124:7860

```bash
# 複製檔案到資料夾
cp output/qa_dataset_all.json ~/Documents/LLaMA-Factory/data/
# 開啟檔案修改
nano ~/Documents/LLaMA-Factory/data/dataset_info.json
# Ctrl + O 儲存
# Ctrl + X 離開
```

舊版:用chatgpt生成QA然後.py檔跑10 (main)

### Sample

參考WeClone https://github.com/xming521/WeClone/blob/master/settings.template.jsonc

---

train_01  

input: qa_dataset_all  
model: Qwen2.5-7B-Instruct  

Learning rate: 1e-4  
Epochs: 2  
Max samples: 10000  
Compute type: fp16  

Cutoff length: 2048  
Batch size: 8  
Gradient accumulation: 4  
LR scheduler: cosine  

Logging steps: 10   
Save steps: 100  
Warmup steps: 1  

LoRA rank: 4  
LoRA dropout: 0.3  
LoRA+ LR ratio: 16  

---

train_02  

input: qa_dataset_part  
model: Qwen2.5-7B-Instruct  

Learning rate: 1e-4  
Epochs: 2  
Max samples: 10000  
Compute type: fp16  

Cutoff length: 2048  
Batch size: 8  
Gradient accumulation: 4  
LR scheduler: cosine  

Logging steps: 10   
Save steps: 100  
Warmup steps: 1  

LoRA rank: 4  
LoRA dropout: 0.3  
LoRA+ LR ratio: 16  

---

螢幕截圖順序

| Model | Top-p | Temperature |
|:---:|:---:|:---:|
| train_01 | 0.65 | 0.5 |
| train_01 | 0.65 | 0.95 |
| train_01 | 0.7 | 0.95 |
| train_02 | 0.65 | 0.5 |
| train_02 | 0.65 | 0.95 |
| train_02 | 0.7 | 0.95 |

# Fine tune for Context
## Answers 分組
### 操作流程

5_bertopic.py
5_similarity.py (已合併到0)
0_context.py

```bash
cd ~/Desktop/MIT_test
poetry env activate
source .venv/bin/activate
export OPENAI_API_KEY="你的金鑰"
```

### Sample

關鍵字: c-TF-ITF  

| bertopic | MODEL | MIN_TOPIC_SIZE | n_neighbors=5 | n_components | min_dist |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 09 | uer/sbert-base-chinese-nli | 5 | 10 | 3 | 0.1 |
| 10 | uer/sbert-base-chinese-nli | 5 | 5 | 3 | 0.1 |
| 11 | shibing624/text2vec-base-chinese | 5 | 5 | 5 | 0.0 |