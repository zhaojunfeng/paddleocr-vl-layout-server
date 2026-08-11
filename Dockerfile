# ── PaddleOCR-VL Layout Parsing Server ──────────────────────────────
# 本地 PaddleOCRVL + vLLM 模式 / AI Studio 托管模式均支持。
# 构建体积较大（paddlepaddle ~500MB+），首次构建请耐心等待。
FROM python:3.12-slim
RUN sed -i s@deb.debian.org@mirrors.cloud.tencent.com@g /etc/apt/sources.list.d/debian.sources
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
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --no-cache-dir --upgrade pip setuptools wheel \
    && pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --no-cache-dir -e .

# 复制其余源码（.dockerignore 已排除 .venv / uploads 等）
COPY . .

# 非 root 运行
# 预创建数据目录并归属 app，否则命名卷首次挂载到不存在的目录时由 root 创建，
# 容器内 app 用户将无写入权限（PermissionError: /app/data/uploads）
RUN useradd -m -u 1001 app \
    && mkdir -p /app/data/uploads/jobs \
    && chown -R app:app /app/data \
    && mkdir -p /home/app/.paddlex \
    && chown -R app:app /home/app
USER app

ENV PYTHONUNBUFFERED=1 \
    PORT=8399 \
    VLLM_SERVER_URL=http://localhost:8000/v1 \
    UPLOAD_DIR=/app/data/uploads \
    JOBS_DIR=/app/data/uploads/jobs \
    USER_DB_PATH=/app/data/user.json

EXPOSE 8399

CMD ["python", "server.py"]
