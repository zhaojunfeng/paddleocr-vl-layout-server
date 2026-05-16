#!/bin/bash
cd "$(dirname "$0")"

export VLLM_SERVER_URL="${VLLM_SERVER_URL:-http://localhost:8080/v1}"
export JWT_SECRET="${JWT_SECRET:-442afaf92fb27ccba6a4daa9f7972ebe979a5c4bd695ccd7c7ab8c2e5f34705e}"
export USER_DB_PATH="${USER_DB_PATH:-user.json}"
export PORT=6006
echo "VLLM_SERVER_URL=$VLLM_SERVER_URL"
echo "JWT_SECRET=${JWT_SECRET:0:8}..."
echo "USER_DB_PATH=$USER_DB_PATH"

if [ -f server.pid ] && kill -0 "$(cat server.pid)" 2>/dev/null; then
    echo "Server already running (PID $(cat server.pid))"
    exit 1
fi

echo "Starting PaddleOCR-VL Layout Server..."
nohup python server.py > server.log 2>&1 &
echo $! > server.pid
echo "Started (PID $(cat server.pid)), logs: server.log"
