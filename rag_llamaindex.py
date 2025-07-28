import os
from dotenv import load_dotenv

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.node_parser import SentenceSplitter

from llama_index.llms.openai import OpenAI
from llama_index.embeddings.langchain import LangchainEmbedding
from langchain.embeddings import HuggingFaceEmbeddings

# Step 1: Load env and set OpenAI key
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Step 2: Set embedding model (Traditional Chinese supported)
embedding_model = LangchainEmbedding(
    HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
)

# Step 3: Set LLM (OpenAI GPT-4)
llm = OpenAI(
    model="gpt-4",
    temperature=1.0,
    max_tokens=500, # hard cutoff for response length
    system_prompt = (
    "你是陳文茜，一位67歲的臺灣媒體人、政治評論者與作家，曾任立法委員，熟悉國際政治與社會觀察。"
    "你的語氣帶有歷練與思辨，說話直白、帶情緒、有時具詩意，常以個人經驗、時事觀察或具體例子開展觀點。"
    "你使用繁體中文回答所有問題，語氣自然、口語化，適度加入台灣常見語助詞如「嗯」、「其實」、「你知道嗎」、「然後」來展現思考過程。"
    "回答中避免空泛陳述，盡量引用你過去曾發表過的觀點、經驗或案例，若無法提供明確回答，也請誠實指出，並補充你會如何思考此問題，甚至提出反問，引導對話深入。"
    "**請控制回答在200字以內。**"
    )
)

# Step 4: Configure global Settings
Settings.llm = llm
Settings.embed_model = embedding_model
Settings.node_parser = SentenceSplitter(chunk_size=200, chunk_overlap=20)

# Step 5: Load documents (assumes preprocessed into a .txt or list of paragraphs)
documents = SimpleDirectoryReader(input_files=["output/shortened.json"]).load_data()

# Step 6: Create index
index = VectorStoreIndex.from_documents(documents)

# Step 7: Create query engine
retriever = index.as_retriever(similarity_top_k=5)
query_engine = RetrieverQueryEngine(retriever=retriever)

# Step 8: Run chat loop
while True:
    query = input("\n🔍 請輸入你對城市科學或永續發展的問題：\n> ")
    if query.lower() in ["exit", "quit", "q"]:
        break
    response = query_engine.query(query)
    print("\n🔎 使用的參考資料：")
    for source in response.source_nodes:
        print(source.get_content())
    print("\n🧠 回答：")
    print(response.response)
