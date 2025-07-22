# 載入必要套件
from typing import Literal, Annotated
import typing
typing.Annotated = Annotated
globals()["Annotated"] = Annotated

from pydantic import BaseModel, Field, ConfigDict  # ✅ 使用 pydantic v2 的方式
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain.tools.retriever import create_retriever_tool
from langgraph.graph import MessagesState, StateGraph, START, END
from langchain_community.chat_models import ChatOpenAI
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool, Tool
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import AIMessage, HumanMessage
import json

# 讀取的 JSON 檔
with open("output/shortened.json", "r", encoding="utf-8") as f:
    lines = json.load(f)  # lines 會是 list[str]

# 將每一行轉為一筆 Document
docs = [Document(page_content=line) for line in lines if line.strip()]
print(f"共載入 {len(docs)} 筆句子")

# 2. Create a retriever tool
# 用 OpenAI 模型進行向量化，建立 in-memory 向量資料庫
vectorstore = InMemoryVectorStore.from_documents(
    documents=docs,
    embedding=OpenAIEmbeddings(model="text-embedding-3-large")
)

# 轉換成語意搜尋器 retriever
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 10, "score_threshold": 0.5},  # 修改回傳筆數與相似度閾值
    search_type="similarity"                          # 使用語意相似度搜尋
)

# 封裝 retriever 為 tool 函式
class RetrieverInput(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    query: str = Field(description="使用者的提問內容")

@tool(args_schema=RetrieverInput)
def retrieve_qa_snippets(input: RetrieverInput) -> str:
    """根據語者 QA 資料回傳相關的知識語句。"""
    return f"這裡是模擬回傳與「{input.query}」有關的 QA 資料"

# 包裝成 LangChain tool，可以被 Agent 使用
# retriever_tool = Tool(
    # name="retrieve_qa_snippets",
    # func=retrieve_qa_snippets,
    # description="根據語者 QA 資料回傳相關的知識語句。")

# 3. Generate query
# 用 GPT 模型生成回應或觸發檢索工具，設定只學習內容
response_model = ChatOpenAI(
    model="gpt-4-0125-preview",   # 較新版本的 GPT-4.1 模型名稱
    temperature=0.0,             # 不加創意，只給準確內容
    top_p=1.0,
    max_tokens=1024,
    presence_penalty=0.0,
    frequency_penalty=0.0
)

# 根據目前的 messages 狀態，用 GPT 模型生成回答
def generate_query_or_respond(state: MessagesState):
    messages = state["messages"]  # ✅ 這裡應該是 list[BaseMessage]
    response = response_model.invoke(
        messages,
        tools=[retrieve_qa_snippets],
    )
    return {"messages": messages + [response]}

# 4. Grade documents
# 使用 GPT 模型判斷「剛剛檢索到的語句」是否真的有回答問題
# 評分提示：告訴模型只需回傳 是/否，判斷有無相關
GRADE_PROMPT = (
    "你是一個評分者，負責判斷某段檢索資料與使用者問題是否相關。\n\n"
    "以下是檢索到的資料內容：\n{context}\n\n"
    "以下是使用者的提問：\n{question}\n\n"
    "請依據資料中是否包含與提問相關的關鍵詞或語意，評估這段資料是否有幫助。\n"
    "如果有幫助，請回覆 '是'；如果無關或幫助不大，請回覆 '否'。"
)

# 定義結構化輸出格式：只允許 是 或 否
class GradeDocuments(BaseModel):
    binary_score: str = Field(
        description="回覆 '是' 表示相關，回覆 '否' 表示不相關。"
    )

# 建立評分用的 GPT 模型，同樣只學習內容
grader_model = ChatOpenAI(
    model="gpt-4-0125-preview",
    temperature=0.0,
    top_p=1.0,
    max_tokens=1024,
    presence_penalty=0.0,
    frequency_penalty=0.0
)

# 判斷 retrieved context 是否能回答問題
def grade_documents(state: MessagesState) -> Literal["generate_answer", "rewrite_question"]:
    question = state["messages"][0].content           # 取得使用者原始問題
    context = state["messages"][-1].content           # 最後一段是 retriever 回傳的內容

    prompt = GRADE_PROMPT.format(question=question, context=context)
    response = (
        grader_model
        .with_structured_output(GradeDocuments)
        .invoke([{"role": "user", "content": prompt}])
    )

    # 根據模型回傳結果決定接下來流程方向
    if response.binary_score == "是":
        return "generate_answer"
    else:
        return "rewrite_question"

# 5. Rewrite question
# 若檢索無法回答問題，請 GPT 協助改寫使用者原始問題
REWRITE_PROMPT = (
    "請思考以下提問背後的語意與真正想了解的內容，並試著幫使用者重新表達問題。\n"
    "以下是原始問題：\n-------\n{question}\n-------\n"
    "請將它改寫為更清楚、更有助於搜尋的提問："
)

# 改寫使用者原始問題，讓 retriever 下次查得更準
def rewrite_question(state: MessagesState):
    messages = state["messages"]
    question = messages[0].content                              # 取得原始使用者問題
    prompt = REWRITE_PROMPT.format(question=question)           # 套用提示模版
    response = response_model.invoke([{"role": "user", "content": prompt}])  # 交給 GPT 改寫問題
    return {"messages": [{"role": "user", "content": response.content}]}     # 包成新的 user 訊息

# 6. Generate answer
# 根據使用者原始問題與 context，給出最終答案
GENERATE_PROMPT = (
    "你是一個知識助理，請參考下方提供的語句來回答使用者的問題。\n"
    "如果不知道答案，就回覆不知道，不要編造。\n"
    "請用口語化的方式回答，最多不要超過300字。\n"
    "請用繁體中文回答問題。\n"
    "問題：{question} \n"
    "參考內容：{context}"
)

def generate_answer(state: MessagesState):
    question = state["messages"][0].content
    context = state["messages"][-1].content
    prompt = GENERATE_PROMPT.format(question=question, context=context)
    response = response_model.invoke([{"role": "user", "content": prompt}])
    return {"messages": [response]}

# 7. Assemble graph
# 使用 LangGraph 組裝整體對話流程圖
workflow = StateGraph(MessagesState)

# 加入各個節點
workflow.add_node(generate_query_or_respond)        # Step 1: 問題初步處理
workflow.add_node("retrieve", ToolNode([retrieve_qa_snippets]))  # Step 2: 工具查資料
workflow.add_node(rewrite_question)                 # Step 3: 改寫問題
workflow.add_node(generate_answer)                  # Step 4: 產生最終答案

# 設定流程起點
workflow.add_edge(START, "generate_query_or_respond")

# 根據 tool_calls 判斷是否需要查資料
workflow.add_conditional_edges(
    "generate_query_or_respond",
    tools_condition,
    {
        "tools": "retrieve",  # 有 tool_call 就去查資料
        END: END              # 否則直接結束
    },
)

# 查完資料後請評分是否有幫助（是：回答；否：改寫）
workflow.add_conditional_edges("retrieve", grade_documents)

# 接續流程路徑
workflow.add_edge("generate_answer", END)                      # 有幫助 → 回答 → 結束
workflow.add_edge("rewrite_question", "generate_query_or_respond")  # 沒幫助 → 改寫再問一次

# 編譯整個流程圖
graph = workflow.compile()

# 8. Run the agentic RAG
# 實際執行整個對話流程，從使用者輸入開始
for chunk in graph.stream(
    {
        "messages": [   # 初始對話訊息
            {
                "role": "user",
                "content": "你認為甚麼是全球化?"  # 使用者問題
            }
        ]
    }
):
    for node, update in chunk.items():               # 每個流程節點的回傳結果
        print("Update from node", node)              # 顯示目前是哪個節點在執行
        update["messages"][-1].pretty_print()        # 美化顯示目前回覆的內容
        print("\n\n")                                # 換行方便閱讀