#!/bin/bash
cd "$(dirname "$0")"

if [ ! -f server.pid ]; then
    echo "No server.pid found, server may not be running"
    exit 1
fi

PID=$(cat server.pid)
if kill -0 "$PID" 2>/dev/null; then
    echo "Stopping server (PID $PID)..."
    kill "$PID"
    rm -f server.pid
    echo "Stopped"
else
    echo "Process $PID not found, cleaning up"
    rm -f server.pid
fi
