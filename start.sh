#!/bin/bash
# ==========================================
# 先行一步 (One Step Ahead) - 一鍵啟動腳本
# ==========================================

# 取得腳本所在目錄
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

echo "=========================================="
echo "   先行一步 (One Step Ahead) 啟動中..."
echo "=========================================="
echo ""

# 檢查虛擬環境
if [ ! -d "venv" ]; then
    echo "[ERROR] 找不到 venv 虛擬環境，請先執行 install.sh"
    exit 1
fi

# 啟動 AI 辨識 + Web 儀表板（已整合為單一程式）
echo "[1/1] 正在啟動 AI 辨識 + Web 儀表板 (Port 8000)..."
echo "      (AI 辨識已整合於 dashboard，無需分開啟動)"
echo "------------------------------------------"
echo "  瀏覽器訪問: http://$(hostname -I | awk '{print $1}'):8000"
echo "  按 Ctrl+C 停止系統。"
echo "------------------------------------------"
echo ""

venv/bin/python3 pi/dashboard.py

echo ""
echo "系統已結束。"
