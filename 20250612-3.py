import streamlit as st
import random # 導入 random 模組
# LangChain 相關的導入
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOllama

# region --- Streamlit 設定 ---
st.set_page_config(page_title="智能助理", layout="centered", initial_sidebar_state="auto")
st.title("智能助理")
st.markdown("---")

# 生成 0.0 到 1.0 之間的隨機溫度值
random_initial_temperature = round(random.uniform(0.0, 1.0), 2) # 四捨五入到小數點後兩位，以便滑桿顯示

temperature = st.slider(
    "AI temperature (數值越大，AI提供越熱情的服務態度)",
    max_value=1.0,
    min_value=0.0,
    step=0.01,
    value=random_initial_temperature # 使用隨機生成的初始值
)
# 預設的知識庫文本範例 (方便使用者參考)
ph = """本公司專門銷售高品質的有機咖啡豆, 我們只從巴西、哥倫比亞和衣索比亞的公平貿易農場進口咖啡豆.我們的烘焙師傅擁有超過20年的經驗, 確保每一批咖啡豆都能完美烘焙, 帶出其獨特的風味.顧客服務是我們的首要任務, 所有訂單在24小時內出貨, 並提供7天內無條件退貨服務.目前我們的熱銷產品有：巴西陽光咖啡豆 (中度烘焙, 帶有堅果香氣). 哥倫比亞黃金咖啡豆 (深度烘焙,口感濃郁) 和衣索比亞花園咖啡豆 (淺度烘焙，帶有花果香)。
        我們也提供咖啡器具和咖啡研磨機的銷售，並且定期舉辦咖啡品鑑會和烘焙課程.所有產品的銷售利潤的5%將捐贈給全球的咖啡農扶助基金.我們接受Visa、MasterCard和Amex信用卡支付.對於批發訂單, 我們提供特殊的折扣.公司總部位於台北市，我們的客戶服務熱線是 (02) 1234-5678。"""

# 讓使用者輸入知識庫文本
user_context_text = st.text_area(
    label="請在下方輸入您的資訊作為知識庫：",
    value="",  # 初始值為空
    height=300,
    placeholder="以一間咖啡銷售公司為範例:\n" + ph  # 顯示範例提示
)

# 判斷最終使用的知識庫文本：如果使用者輸入為空，則使用預設範例文本
final_context_text = ph if len(user_context_text.strip()) == 0 else user_context_text.strip()


# 用於設定 RAG 系統的函數（不再緩存，因為知識庫是動態的）
def setup_rag_system_dynamic(context_text, temp):
    """ 設定並返回 RAG 系統組件 (向量資料庫, 檢索器, LLM, RAG鏈)。該函數會根據傳入的 context_text 建立新的知識庫。"""
    knowledge_file = "company_knowledge_base.txt"  # 0. 將文本寫入一個臨時文件，以便 TextLoader 讀取
    with open(knowledge_file, "w", encoding="utf-8") as f:
        f.write(context_text)
    loader = TextLoader(knowledge_file, encoding="utf-8")  # 1. 載入並分割文本
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)

    # 2. 生成嵌入並建立向量資料庫
    embeddings = OllamaEmbeddings(model="nomic-embed-text")  # 請確保你已經用 'ollama pull nomic-embed-text' 下載了嵌入模型
    vectorstore = Chroma.from_documents(chunks, embeddings)  # 建立 Chroma 向量資料庫。每次調用都會創建一個新的、獨立的 in-memory 資料庫。
    retriever = vectorstore.as_retriever()

    # 3. 定義多個 LLM 和其對應的 RAG 鏈
    # 注意：這裡定義模型處理的順序，將會在後面隨機打亂
    models_config = [
        ("llama3.2", ChatOllama(model="llama3.2", temperature=temp)),
        ("gemma3:4b", ChatOllama(model="gemma3:4b", temperature=temp)),
        ("phi4", ChatOllama(model="phi4", temperature=temp))
    ]

    # 隨機打亂模型順序
    random.shuffle(models_config)
    model_order = [name for name, _ in models_config]

    rag_chains = {}
    
    for model_name, llm_instance in models_config: # 使用隨機打亂後的順序
        # 提示模板現在需要能夠接收「前一個模型的回答」
        # 我們將使用 "previous_answer" 作為額外的上下文變數
        rag_prompt = ChatPromptTemplate.from_messages([
            ("system", f"""你是一個專業的助理，現在你使用的模型是 {model_name}。
              請根據以下檢索到的資訊來回答問題。
              
              如果你是第二個或第三個回答的模型，你可以參考「前一個模型回答的內容」來補充、修正或提供不同視角，讓回答更完整豐富。
              
              如果資訊中沒有提到，請禮貌地說你無法回答，並避免編造內容。
              
              \n\n檢索到的資訊：\n{{context}}
              \n\n前一個模型回答的內容：\n{{previous_answer}}
              """),
            ("human", "{question}")
        ])
        
        # input_mapper 現在需要包含 previous_answer，即使它是空的
        rag_chain_input_mapper = {
            "context": retriever,
            "question": RunnablePassthrough(),
            "previous_answer": RunnablePassthrough() # 這裡需要一個佔位符，實際值在 invoke 時傳入
        }
        rag_chains[model_name] = rag_chain_input_mapper | rag_prompt | llm_instance | StrOutputParser()

    return rag_chains, model_order # 返回 RAG 鏈字典和模型的隨機順序

# 由於不再生成關鍵字，此函數將不再被使用，但為了程式碼完整性暫時保留
def highlight_important_tokens(text, keywords):
    """
    用紅色粗體標記文本中的關鍵字。
    keywords 應該是一個列表或集合，包含要標記的關鍵字。
    """
    highlighted_text = text
    # Split keywords string into a list
    keyword_list = [k.strip() for k in keywords.split(',') if k.strip()]

    for keyword in keyword_list:
        # Using a simple replace. For more robust highlighting (e.g., avoiding partial word matches),
        # you might need regex with word boundaries.
        highlighted_text = highlighted_text.replace(
            keyword, f'<span style="color:red;font-weight:bold;">{keyword}</span>'
        )
    return highlighted_text


# endregion

# region --- 初始變數 ---
if "messages" not in st.session_state:  # 歷史訊息
    st.session_state.messages = []
if "rag_system_ready" not in st.session_state:  # 當系統準備好的時候，也就是代表使用者已經確認將資料做初始化，接著方能開啟問答
    st.session_state.rag_system_ready = False
if "rag_chains" not in st.session_state:  # rag鍊核心，我們是利用 rag_chains[model_name].invoke() 來取得 ai 資訊
    st.session_state.rag_chains = {}
if "model_order" not in st.session_state: # 儲存模型的回答順序
    st.session_state.model_order = []
if "initial_kb_loaded" not in st.session_state:  # 追蹤是否已首次載入知識庫
    st.session_state.initial_kb_loaded = False
if "show_confirm_modal" not in st.session_state:  # 是否顯示再確認方框
    st.session_state.show_confirm_modal = False
if "meta_data_by_model" not in st.session_state: # 儲存所有模型的元數據 (現在將不再使用)
    st.session_state.meta_data_by_model = {}
# endregion

# region --- 初始按鈕 ---
if st.button("初始化/更新知識庫"):  # 當使用者點擊此按鈕時，根據狀態決定是直接初始化還是顯示確認方框
    if not st.session_state.initial_kb_loaded:  # 第一次點擊：直接初始化知識庫
        if len(final_context_text) == 0:
            st.error("請在文字區域中輸入您的資訊，或使用範例文本。")
        else:
            with st.spinner("🔄 正在初始化知識庫與 AI 模型，請稍候..."):
                try:
                    rag_chains, model_order = setup_rag_system_dynamic(final_context_text, temperature)
                    st.session_state.rag_chains = rag_chains
                    st.session_state.model_order = model_order # 儲存模型順序
                    st.session_state.meta_data_by_model = {} # 確保清空或保持為空
                    st.session_state.rag_system_ready = True
                    st.session_state.initial_kb_loaded = True  # 標記為已載入
                    st.success("🎉 知識庫與 AI 模型初始化完成！現在可以開始提問了。")
                    st.session_state.messages = []  # 清空聊天歷史
                except Exception as e:
                    st.session_state.rag_system_ready = False
                    st.session_state.rag_chains = {}
                    st.session_state.model_order = []
                    st.session_state.meta_data_by_model = {} # 清空元數據
                    st.error(f"初始化失敗：{e}。請確認 Ollama 服務正在運行且所需模型已下載。")
                    st.warning("請確保您已安裝 Ollama 並已拉取以下所有模型：`nomic-embed-text`、`llama3.2`、`gemma3:4b` 和 `phi4`。")
                    st.warning("❗ **重要提示：** 儘管 `gemma3:4b` 支援多模態，但您的應用目前僅支援文本RAG。")
    else:
        st.session_state.show_confirm_modal = True  # 如果已經初始化過，則跳到 show_confirm_modal 確認方框邏輯(下方)
        st.rerun()  # 強制 Streamlit 重新執行，以立即顯示確認方框
# endregion

# region --- 方框邏輯 ---
if st.session_state.show_confirm_modal:  # 只有當 st.session_state.show_confirm_modal 為 True 時才顯示
    st.markdown("---")  # 分隔線
    st.info("ℹ️ 您已經初始化過知識庫。再次點擊表示您要**更新**知識庫。")
    st.write("是否確定要用當前文字區域的資訊來更新知識庫？這將清除當前對話歷史，並且確認您提供的資訊**是否前後有牴觸***。")
    col_confirm, col_cancel = st.columns(2)  # 設置兩個按鈕，用於確認或取消操作
    with col_confirm:
        if st.button("確定更新", key="confirm_update_kb_button"):  # 點擊「確定更新」按鈕
            if len(final_context_text) == 0:  # 執行知識庫更新的邏輯
                st.error("請在文字區域中輸入您的資訊，或使用範例文本。")
                st.session_state.show_confirm_modal = False  # 出錯時隱藏方框
                st.rerun()  # 強制重新執行以更新UI
            else:
                with st.spinner("🔄 正在更新知識庫與 AI 模型，請稍候..."):
                    try:
                        rag_chains, model_order = setup_rag_system_dynamic(final_context_text, temperature)
                        st.session_state.rag_chains = rag_chains
                        st.session_state.model_order = model_order # 儲存模型順序
                        st.session_state.meta_data_by_model = {} # 確保清空或保持為空
                        st.session_state.rag_system_ready = True
                        st.success("✅ 知識庫已成功更新！")
                        st.session_state.messages = []  # 清空聊天歷史
                        st.session_state.show_confirm_modal = False  # 隱藏確認方框
                        st.rerun()  # 強制重新執行以更新UI
                    except Exception as e:
                        st.session_state.rag_system_ready = False
                        st.session_state.rag_chains = {}
                        st.session_state.model_order = []
                        st.session_state.meta_data_by_model = {} # 清空元數據
                        st.error(f"更新失敗：{e}。請確認 Ollama 服務正在運行且所需模型已下載。")
                        st.session_state.show_confirm_modal = False  # 出錯時隱藏方框
                        st.rerun()  # 強制重新執行以更新UI
    with col_cancel:
        if st.button("取消", key="cancel_update_kb_button"):  # 點擊「取消」按鈕
            st.session_state.show_confirm_modal = False  # 隱藏確認方框
            st.rerun()  # 強制重新執行以更新UI
# endregion

# region --- 顯示摘要和關鍵字 (此區塊已移除) ---
# 由於元數據的生成和顯示功能已移除，此區塊的內容不再需要
# endregion

# region --- 聊天介面邏輯 ---
# 顯示所有歷史訊息，區分用戶和不同模型的助理回覆
for message in st.session_state.messages:
    if message["role"] == "user":
        with st.chat_message("user"):
            st.markdown(message["content"])
    elif message["role"] == "assistant":
        with st.chat_message("assistant"):
            # 如果是多模型回覆，則顯示模型名稱
            if "model" in message:
                st.markdown(f"**🤖 {message['model']} 的回覆：**")
            st.markdown(message["content"])

if st.session_state.rag_system_ready:  # 獲取用戶輸入 (只有當 RAG 系統就緒時才啟用輸入框)
    if not st.session_state.messages:  # 判斷是否顯示首次對話的助理提示訊息 (僅在聊天歷史為空且助理已準備好時顯示)
        initial_chat_message = "您好，請問有什麼我可以協助您的？"
        with st.chat_message("assistant"):
            st.markdown(initial_chat_message)
            st.session_state.messages.append({"role": "assistant", "content": initial_chat_message})
    if prompt := st.chat_input("輸入您的問題..."):  # 將用戶訊息添加到聊天歷史並顯示
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        previous_model_answer = "" # 初始化前一個模型的回答
        
        # 迭代所有 RAG 鏈並獲取回覆，依序傳遞前一個模型的回答
        # 使用隨機打亂後的順序 st.session_state.model_order
        for model_name in st.session_state.model_order:
            rag_chain = st.session_state.rag_chains[model_name]
            with st.chat_message("assistant"):
                try:
                    st.markdown(f"**🤖 {model_name} 正在思考中...**") # 顯示模型正在處理的提示
                    
                    # 調用 RAG 鏈時，傳遞 prompt 和 previous_model_answer
                    ai_response = rag_chain.invoke({
                        "question": prompt,
                        "previous_answer": previous_model_answer
                    })
                    
                    st.markdown(f"**🤖 {model_name} 的回覆：**") # 最終顯示模型名稱
                    st.markdown(ai_response)  # 顯示最終回覆
                    
                    # 將當前模型的回答儲存為下一個模型的「前一個回答」
                    previous_model_answer = ai_response
                    
                    # 將回覆添加到聊天歷史，包含模型名稱
                    st.session_state.messages.append({"role": "assistant", "model": model_name, "content": ai_response})
                except Exception as e:
                    error_message = f"很抱歉，**{model_name}** 處理您的請求時發生錯誤：{e}。\n請確認 Ollama 服務正在運行且模型已下載。"
                    st.error(error_message)  # 將錯誤訊息作為 AI 回覆
                    # 將錯誤訊息添加到聊天歷史
                    st.session_state.messages.append({"role": "assistant", "model": model_name, "content": error_message})
else:  # 如果 RAG 系統尚未就緒，顯示提示訊息並禁用聊天輸入框
    st.warning("請先輸入您的資訊並點擊「初始化/更新知識庫」按鈕來啟動助理。")
    st.chat_input("請先初始化知識庫...", disabled=True)


# endregion

# region --- 側欄說明 ---
st.sidebar.markdown("---")
st.sidebar.header("使用說明")
st.sidebar.markdown("""
這個應用程式是一個智能助理，它會**完全根據您在文字區域中提供的最新資訊**來回答問題。

**操作步驟：**
1.  在文字區域中**輸入或修改**您的資訊。
2.  **重要！**
    * **第一次**點擊「**初始化/更新知識庫**」按鈕，將會載入資訊並啟動助理。
    * **後續**點擊此按鈕，會彈出一個確認方框，詢問您是否要**更新**知識庫。點擊「確定更新」將會清除舊對話並使用新資訊。
3.  當看到「🎉 知識庫與 AI 模型初始化完成！」或「✅ 知識庫已成功更新！」訊息後，您就可以在下方的聊天框中提問了。

**模型行為：**
* 系統會使用三個語言模型（`llama3.2`、`gemma3:4b`、`phi4`）依序回答您的問題。
* **每次初始化或重新執行應用程式時，這三個模型的回答順序都會是隨機的。**
* 每個模型在回答時，都可以**參考前一個模型給出的答案**，這有助於答案的補充、修正或提供不同視角。

**重要提醒：**
請確保您的電腦已**安裝 Ollama 服務**並已**下載以下所有模型**：
-   **嵌入模型 (Embedding Model):** `nomic-embed-text`
    執行命令：`ollama pull nomic-embed-text`
-   **語言模型 (Language Model) 組合：**
    * `llama3.2`
        執行命令：`ollama pull llama3.2`
    * `gemma3:4b`
        執行命令：`ollama pull gemma3:4b`
    * `phi4`
        執行命令：`ollama pull phi4`

**關於模型能力的注意事項：**
* `gemma3:4b` 是一個支援**多模態 (圖像與文本)** 的優秀模型！然而，在目前這個應用程式中，**您的知識庫是純文本，且程式碼邏輯也只處理文字輸入**。若要實現圖片輸入和基於圖片的 RAG，需要對程式碼進行進一步的開發和修改。
* 對於知識庫中沒有的資訊，AI 會禮貌地表示無法回答，並避免編造內容。
""")
# endregion