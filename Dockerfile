# ----------------------------------------------------------------------------
# Dockerfile: Streamlit + LangChain + SQLite (for Ollama client) 容器建置檔案
# ----------------------------------------------------------------------------

FROM python:3.9-slim-buster

WORKDIR /app

# 安裝必要系統相依套件（含 SQLite 開發庫）
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsqlite3-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# 安裝 Python wheel 工具
RUN pip install --no-cache-dir wheel

# 安裝 Python 相依
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 複製應用程式原始碼
COPY . .

# 開放 Streamlit 預設端口
EXPOSE 8501

# 啟動應用程式（停用 CORS 和 XSRF 保護，適用於本機開發）
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.enableCORS=false", "--server.enableXsrfProtection=false"]
