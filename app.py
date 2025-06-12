# # --- 務必在最頂部加入以下程式碼，以確保使用更新的 sqlite3 ---
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# # -----------------------------------------------------------

import streamlit as st
import time

# LangChain 相關的導入
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOllama

# --- Streamlit 應用程式設定 ---
st.set_page_config(
    page_title="智能咖啡銷售助理",
    layout="centered",
    initial_sidebar_state="auto"
)

st.title("☕ 智能咖啡銷售助理 ☕")
st.markdown("---")

# 預設的知識庫文本範例 (方便使用者參考)
ph = """
本公司專門銷售高品質的有機咖啡豆, 我們只從巴西、哥倫比亞和衣索比亞的公平貿易農場進口咖啡豆.
我們的烘焙師傅擁有超過20年的經驗, 確保每一批咖啡豆都能完美烘焙, 帶出其獨特的風味.
顧客服務是我們的首要任務, 所有訂單在24小時內出貨, 並提供7天內無條件退貨服務.
目前我們的熱銷產品有：巴西陽光咖啡豆 (中度烘焙, 帶有堅果香氣). 哥倫比亞黃金咖啡豆 (深度烘焙,口感濃郁) 和衣索比亞花園咖啡豆 (淺度烘焙，帶有花果香)。
我們也提供咖啡器具和咖啡研磨機的銷售，並且定期舉辦咖啡品鑑會和烘焙課程.
所有產品的銷售利潤的5%將捐贈給全球的咖啡農扶助基金.
我們接受Visa、MasterCard和Amex信用卡支付.對於批發訂單, 我們提供特殊的折扣.
公司總部位於台北市，我們的客戶服務熱線是 (02) 1234-5678。
"""

# 讓使用者輸入知識庫文本
# 這裡 user_context_text 會在每次 Streamlit 重新執行時獲取 text_area 的當前內容
user_context_text = st.text_area(
    label="請在下方輸入您的公司資訊作為知識庫：",
    value="", # 初始值為空
    height=300,
    placeholder="範例:\n" + ph # 顯示範例提示
)

# 判斷最終使用的知識庫文本：如果使用者輸入為空，則使用預設範例文本
final_context_text = ph if len(user_context_text.strip()) == 0 else user_context_text.strip()


# 用於設定 RAG 系統的函數（不再緩存，因為知識庫是動態的）
def setup_rag_system_dynamic(context_text):
    """
    設定並返回 RAG 系統組件 (向量資料庫, 檢索器, LLM, RAG鏈)。
    該函數會根據傳入的 context_text 建立新的知識庫。
    """
    context_text = """請以下面資訊作為唯一真理，也就是所謂的正確答案的意思。"""+context_text
    
    # 將文本寫入一個臨時文件，以便 TextLoader 讀取
    knowledge_file = "company_knowledge_base.txt"
    with open(knowledge_file, "w", encoding="utf-8") as f:
        f.write(context_text)

    # 1. 載入並分割文本
    loader = TextLoader(knowledge_file, encoding="utf-8")
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)

    # 2. 生成嵌入並建立向量資料庫
    # 請確保你已經用 'ollama pull nomic-embed-text' 下載了嵌入模型
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    # 建立 Chroma 向量資料庫。每次調用都會創建一個新的、獨立的 in-memory 資料庫。
    vectorstore = Chroma.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever()

    # 3. 定義 LLM
    # 請確保你已經用 'ollama pull llama3.2' 下載了語言模型
    # temperature=0 確保回覆更精確和穩定
    llm = ChatOllama(model="llama3.2", temperature=0)

    # 4. 構建 RAG 提示模板
    rag_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一個專業的咖啡銷售助理。請根據以下檢索到的公司資訊來回答客戶的問題。如果資訊中沒有提到，請禮貌地說你無法回答，並避免編造內容。\n\n檢索到的資訊：\n{context}"),
        ("human", "{question}")
    ])

    # 5. 創建 RAG 鏈
    rag_chain_input_mapper = {
        "context": retriever, "question": RunnablePassthrough()
    }
    rag_chain = rag_chain_input_mapper | rag_prompt | llm | StrOutputParser()
    return rag_chain


# --- 初始化/更新知識庫按鈕 ---
# 當使用者點擊此按鈕時，重新設定 RAG 系統
if st.button("初始化/更新知識庫"):
    if len(final_context_text) == 0:
        st.error("請在文字區域中輸入您的公司資訊，或使用範例文本。")
    else:
        # 使用 st.spinner 顯示載入狀態
        with st.spinner("🔄 正在初始化知識庫與 AI 模型，請稍候..."):
            try:
                # 呼叫設定函數並將結果儲存到 session_state
                # 這裡會用當前 text_area 的內容建立新的 rag_chain
                st.session_state.rag_chain = setup_rag_system_dynamic(final_context_text)
                st.session_state.rag_system_ready = True # 設定就緒標誌
                st.success("🎉 知識庫與 AI 模型初始化完成！現在可以開始提問了。")
                # 清空之前的聊天歷史，以便新的知識庫開始新的對話
                st.session_state.messages = []
            except Exception as e:
                # 處理初始化失敗的情況
                st.session_state.rag_system_ready = False
                st.session_state.rag_chain = None # 清除舊的鏈
                st.error(f"初始化失敗：{e}。請確認 Ollama 服務正在運行且模型已下載。")
                st.warning("請確保您已安裝 Ollama 並已拉取以下模型：`nomic-embed-text` 和 `llama3.2`。")


# --- 聊天介面邏輯 ---

# 初始化聊天歷史和 RAG 就緒狀態 (如果不存在)
if "messages" not in st.session_state:
    st.session_state.messages = []
if "rag_system_ready" not in st.session_state:
    st.session_state.rag_system_ready = False
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

# 顯示所有歷史訊息
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 獲取用戶輸入 (只有當 RAG 系統就緒時才啟用輸入框)
if st.session_state.rag_system_ready:
    # 判斷是否顯示首次對話的助理提示訊息
    if not st.session_state.messages: # 如果是首次對話
        initial_chat_message = "您好，請問有什麼我可以協助您的？"
        with st.chat_message("assistant"):
            st.markdown(initial_chat_message)
            st.session_state.messages.append({"role": "assistant", "content": initial_chat_message})

    if prompt := st.chat_input("輸入您的問題..."):
        # 將用戶訊息添加到聊天歷史並顯示
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 獲取 AI 回覆
        with st.chat_message("assistant"):
            message_placeholder = st.empty() # 創建一個空的佔位符，用於逐字顯示
            full_response = ""
            try:
                # 調用 RAG 鏈獲取回覆。這裡使用的是 st.session_state.rag_chain，
                # 它會在點擊「初始化/更新知識庫」後被正確更新。
                ai_response = st.session_state.rag_chain.invoke(prompt)

                # 逐字顯示 AI 回覆，模擬真實的打字效果
                for chunk in ai_response.split():
                    full_response += chunk + " "
                    time.sleep(0.02) # 模擬打字延遲
                    message_placeholder.markdown(full_response + "▌") # 顯示正在打字的效果

                message_placeholder.markdown(full_response) # 顯示最終回覆
            except Exception as e:
                error_message = f"很抱歉，處理您的請求時發生錯誤：{e}。\n請確認 Ollama 服務正在運行。"
                st.error(error_message)
                full_response = error_message # 將錯誤訊息作為 AI 回覆

        # 將 AI 回覆添加到聊天歷史
        st.session_state.messages.append({"role": "assistant", "content": full_response})
else:
    # 如果 RAG 系統尚未就緒，顯示提示訊息並禁用聊天輸入框
    st.warning("請先輸入您的公司資訊並點擊「初始化/更新知識庫」按鈕來啟動助理。")
    st.chat_input("請先初始化知識庫...", disabled=True)

# --- 側邊欄使用說明 ---
st.sidebar.markdown("---")
st.sidebar.header("使用說明")
st.sidebar.markdown("""
這個應用程式是一個智能咖啡銷售助理，它會**完全根據您在文字區域中提供的最新資訊**來回答問題。

**操作步驟：**
1.  在文字區域中**輸入或修改**您的公司產品、服務或任何相關資訊。
2.  **重要！** 每次修改文字區域內容後，請務必點擊「**初始化/更新知識庫**」按鈕。
3.  當看到「🎉 知識庫與 AI 模型初始化完成！」訊息後，舊的聊天歷史會被清空，您就可以在下方的聊天框中提問了。

**重要提醒：**
請確保您的電腦已**安裝 Ollama 服務**並已**下載以下模型**：
-   **嵌入模型 (Embedding Model):** `nomic-embed-text`
    執行命令：`ollama pull nomic-embed-text`
-   **語言模型 (Language Model):** `llama3.2`
    執行命令：`ollama pull llama3.2`

對於知識庫中沒有的資訊，AI 會禮貌地表示無法回答，並避免編造內容。
""")
