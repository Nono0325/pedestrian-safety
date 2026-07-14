@echo off
chcp 65001 > nul
title 先行一步 (One Step Ahead) - 一鍵啟動

echo ==========================================
echo    先行一步 (One Step Ahead) 啟動中...
echo ==========================================
echo.

cd /d %~dp0

REM 檢查虛擬環境
if not exist "venv" (
    echo [ERROR] 找不到 venv 虛擬環境。
    echo 請先執行環境設定:
    echo   python -m venv venv
    echo   .\venv\Scripts\activate
    echo   pip install -r pi/requirements.txt
    pause
    exit /b 1
)

echo [1/1] 正在啟動 AI 辨識 + Web 儀表板 (Port 8000)...
echo       (AI 辨識已整合於 dashboard，無需分開啟動)
echo ------------------------------------------
echo   提示: 瀏覽器訪問 http://localhost:8000
echo   按 Ctrl+C 可停止系統。
echo ------------------------------------------
echo.

venv\Scripts\python.exe pi\dashboard.py

echo.
echo 系統已結束。
pause
