FROM python:3.12-slim

# 安装依赖（含 libmagic）
RUN apt-get update && apt-get install -y libmagic1 gcc

# 设置工作目录
WORKDIR /app

# 拷贝依赖文件
COPY requirements.txt .

# 安装 Python 依赖
RUN pip install --no-cache-dir -r requirements.txt

# 拷贝项目代码
COPY . .

# 启动 FastAPI 应用
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
