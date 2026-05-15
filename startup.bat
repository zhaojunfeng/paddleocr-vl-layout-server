@echo off
cd /d "%~dp0"

if "%VLLM_SERVER_URL%"=="" set VLLM_SERVER_URL=http://134.199.132.159/v1
if "%JWT_SECRET%"=="" (
    for /f %%i in ('python -c "import secrets;print(secrets.token_hex(32))"') do set JWT_SECRET=%%i
)
if "%USER_DB_PATH%"=="" set USER_DB_PATH=user.json

echo VLLM_SERVER_URL=%VLLM_SERVER_URL%
echo JWT_SECRET=%JWT_SECRET:~0,8%...
echo USER_DB_PATH=%USER_DB_PATH%

echo Starting PaddleOCR-VL Layout Server...
start /b python server.py > server.log 2>&1
echo Server started, logs: server.log
