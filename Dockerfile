# 使用Python 3.10作为基础镜像
FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 复制项目文件
COPY backend/ ./backend/
COPY core/ ./core/
COPY frontend/ ./frontend/

# 创建必要的目录
RUN mkdir -p uploads chroma_db

# 暴露端口
EXPOSE 8000

# 设置环境变量
ENV APP_PORT=8000
ENV PYTHONUNBUFFERED=1

# 启动命令
CMD ["python", "backend/main.py"]
