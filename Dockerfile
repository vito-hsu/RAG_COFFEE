FROM python:3.10-slim

# 安裝系統工具與必要的編譯環境
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libffi-dev \
    libssl-dev \
    libsqlite3-dev \
    && rm -rf /var/lib/apt/lists/*

# 設定工作目錄
WORKDIR /app

# 複製依賴檔案
COPY requirements.txt .

# 安裝 Python 套件
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# 複製專案檔案
COPY . .

# 預設執行指令 (根據你使用的是 Streamlit)
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.enableCORS=false"]
