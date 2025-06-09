import streamlit as st
import random
import time

st.set_page_config(page_title="簡易聊天機器人範例", layout="centered")

st.title("Streamlit 聊天機器人範例")

# 檢查 session_state 中是否有 'messages'，如果沒有則初始化為空列表
# 這是用來儲存聊天歷史的關鍵
if "messages" not in st.session_state:
    st.session_state.messages = []

# 顯示所有歷史訊息
# 迭代 st.session_state.messages 列表中的每個訊息
for message in st.session_state.messages:
    # 使用 st.chat_message 顯示訊息，並指定發送者 (role)
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 獲取用戶輸入
# st.chat_input 是一個專為聊天應用設計的輸入框
if prompt := st.chat_input("您好，請問有什麼我可以協助您的？"):
    # 將用戶輸入的訊息添加到聊天歷史中
    st.session_state.messages.append({"role": "user", "content": prompt})
    # 在介面上顯示用戶的訊息
    with st.chat_message("user"):
        st.markdown(prompt)

    # 模擬 AI 回覆
    with st.chat_message("assistant"):
        message_placeholder = st.empty() # 創建一個空的佔位符，用於逐字顯示
        full_response = ""
        # 模擬 AI 的回覆邏輯 (這裡只是一個簡單的隨機回覆)
        if "你好" in prompt or "您好" in prompt:
            assistant_response = "您好！很高興為您服務。"
        elif "謝謝" in prompt:
            assistant_response = "不客氣！"
        elif "時間" in prompt:
            import datetime
            assistant_response = f"現在是台灣時間 {datetime.datetime.now().strftime('%Y年%m月%d日 %H時%M分%S秒')}。"
        elif "天氣" in prompt:
            assistant_response = "很抱歉，我目前無法提供即時天氣資訊。"
        else:
            assistant_response = random.choice(
                [
                    "抱歉，我還在學習中，可能無法完全理解您的意思。",
                    "您可以問我其他問題。",
                    "請提供更多資訊，我會盡力幫助您。",
                    "這個問題很有趣，但我需要更多上下文。",
                ]
            )

        # 逐字顯示 AI 回覆，模擬真實的打字效果
        for chunk in assistant_response.split():
            full_response += chunk + " "
            time.sleep(0.05) # 模擬延遲
            message_placeholder.markdown(full_response + "▌") # 顯示正在打字的效果

        # 最終顯示完整的 AI 回覆
        message_placeholder.markdown(full_response)

    # 將 AI 回覆添加到聊天歷史中
    st.session_state.messages.append({"role": "assistant", "content": full_response})

st.sidebar.markdown("---")
st.sidebar.markdown("**關於這個範例：**")
st.sidebar.markdown("這是一個使用 `st.chat_input` 和 `st.chat_message` 構建的基礎 Streamlit 聊天應用程式。它會將對話內容儲存在 `st.session_state` 中，以便在重新整理頁面後也能保留歷史訊息。")
st.sidebar.markdown("您可以試著輸入一些問題，例如：'你好'、'謝謝'、'現在時間'。")