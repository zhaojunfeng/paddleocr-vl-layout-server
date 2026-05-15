@echo off
echo Stopping server...
taskkill /f /im python.exe /fi "WINDOWTITLE eq server.py*" >nul 2>&1
for /f "tokens=2" %%a in ('tasklist /fi "imagename eq python.exe" ^| findstr /i "python"') do (
    wmic process where "ProcessId=%%a" get CommandLine 2>nul | findstr /i "server.py" >nul && taskkill /f /pid %%a >nul 2>&1
)
echo Done
