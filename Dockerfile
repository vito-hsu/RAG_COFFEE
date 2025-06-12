# ----------------------------------------------------------------------------
# Dockerfile: 用於建置 Streamlit 應用程式的 Docker 映像檔
# ----------------------------------------------------------------------------

# STEP 1: 設定基礎映像檔
FROM python:3.9-slim-buster

# STEP 2: 設定容器內的工作目錄
WORKDIR /app

# STEP 3: 安裝系統級別的編譯工具和開發庫
# 這些工具對於某些 Python 套件（例如需要編譯 C/C++ 擴展的套件）的安裝是必需的。
# build-essential 包含了常見的編譯工具（如 gcc, g++）。
# libsqlite3-dev 提供了 SQLite 的開發庫，儘管使用了 pysqlite3-binary，
# 但有些情況下仍然可能需要它來確保兼容性。
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsqlite3-dev \
    && rm -rf /var/lib/apt/lists/* # 清理 apt 快取以減小映像檔大小

# STEP 4: 複製 requirements.txt 並安裝所有 Python 依賴
# 先複製 requirements.txt 並安裝，以利用 Docker 的層緩存機制，提高建置效率
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# STEP 5: 複製所有應用程式程式碼
# 將本地專案的所有檔案（包括 app.py）複製到容器的 /app 目錄中
COPY . .

# STEP 6: 暴露 Streamlit 應用程式的端口
# Streamlit 預設運行在 8501 端口，這裡聲明容器將監聽這個端口
EXPOSE 8501

# STEP 7: 定義容器啟動時執行的命令
# 這行指定了當基於此映像檔創建並啟動一個容器時，應該執行的預設命令。
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.enableCORS=false", "--server.enableXsrfProtection=false"]

# ----------------------------------------------------------------------------
# 關於 Ollama 的重要說明：
# 這個 Docker 映像檔只包含您的 Streamlit 應用程式。
# 它會嘗試連接到您啟動 Streamlit 應用程式的機器上正在運行的 Ollama 服務。
# 部署此容器的機器上需要單獨安裝並運行 Ollama 服務，
# 並且您的 Streamlit 應用程式才能成功調用 nomic-embed-text 和 llama3.2 模型。
# 確保 Ollama 服務可在 Streamlit 應用程式所在的網路環境中被訪問（預設是 localhost:11434）。
# ----------------------------------------------------------------------------
