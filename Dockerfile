# ── PaddleOCR-VL Layout Parsing Server ──────────────────────────────
# 本地 PaddleOCRVL + vLLM 模式 / AI Studio 托管模式均支持。
# 构建体积较大（paddlepaddle ~500MB+），首次构建请耐心等待。
FROM python:3.12-slim

# OpenCV / paddle 运行时系统库
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 先复制安装清单以利用 Docker 层缓存（paddle 依赖安装很慢）
COPY pyproject.toml ./
COPY server.py gen_token.py ./
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -e .

# 复制其余源码（.dockerignore 已排除 .venv / uploads 等）
COPY . .

# 非 root 运行
RUN useradd -m -u 1001 app
USER app

ENV PYTHONUNBUFFERED=1 \
    PORT=8399 \
    VLLM_SERVER_URL=http://localhost:8000/v1 \
    UPLOAD_DIR=/app/data/uploads \
    JOBS_DIR=/app/data/uploads/jobs \
    USER_DB_PATH=/app/data/user.json

EXPOSE 8399

CMD ["python", "server.py"]
