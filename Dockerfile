FROM python:3.12-slim

# 安装 ffmpeg、libmagic 和构建依赖
RUN apt-get update && apt-get install -y ffmpeg libmagic1 gcc && apt-get clean

# 设置工作目录
WORKDIR /app

# 拷贝依赖文件
COPY requirements.txt .

# 安装 Python 依赖
RUN pip install --no-cache-dir -r requirements.txt

# 拷贝代码
COPY . .

# 启动 FastAPI 服务
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
