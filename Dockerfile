# ----------------------------------------------------------------------------
# Dockerfile: 用於建置 Streamlit 應用程式的 Docker 映像檔
# ----------------------------------------------------------------------------

# STEP 1: 設定基礎映像檔
# 使用官方的 Python 3.9 輕量級映像檔，它基於 Debian Buster
FROM python:3.9-slim-buster

# STEP 2: 設定容器內的工作目錄
# 所有後續的命令將在此目錄下執行
WORKDIR /app

# STEP 3: 複製 requirements.txt 並安裝所有 Python 依賴
# 先複製 requirements.txt 並安裝，以利用 Docker 的層緩存機制，提高建置效率
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# STEP 4: 複製所有應用程式程式碼
# 將本地專案的所有檔案（包括 app.py）複製到容器的 /app 目錄中
COPY . .

# STEP 5: 暴露 Streamlit 應用程式的端口
# Streamlit 預設運行在 8501 端口，這裡聲明容器將監聽這個端口
EXPOSE 8501

# STEP 6: 定義容器啟動時執行的命令
# 這行指定了當基於此映像檔創建並啟動一個容器時，應該執行的預設命令。
# 它會啟動您的 Streamlit 應用程式。
# --server.port=8501 確保應用程式在預期的端口上運行。
# --server.enableCORS=false 和 --server.enableXsrfProtection=false
# 是在某些部署環境中可能需要設置的選項，以避免跨域或 CSRF 保護導致的問題。
# 在生產環境中，請根據實際需求和安全性考量來配置這些選項。
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.enableCORS=false", "--server.enableXsrfProtection=false"]

# ----------------------------------------------------------------------------
# 關於 Ollama 的重要說明：
# 這個 Docker 映像檔只包含您的 Streamlit 應用程式。
# 它會嘗試連接到您啟動 Streamlit 應用程式的機器上正在運行的 Ollama 服務。
# 部署此容器的機器上需要單獨安裝並運行 Ollama 服務，
# 並且您的 Streamlit 應用程式才能成功調用 nomic-embed-text 和 llama3.2 模型。
# 確保 Ollama 服務可在 Streamlit 應用程式所在的網路環境中被訪問（預設是 localhost:11434）。
# ----------------------------------------------------------------------------
