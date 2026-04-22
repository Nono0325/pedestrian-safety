#!/bin/bash

# ==========================================================
# 先行一步 (One Step Ahead) - Raspberry Pi 5 自動化部署腳本
# ==========================================================

set -e # Exit on error

echo "-------------------------------------------------------"
echo "開始部署: 單鏡頭 AI 行人意圖辨識與斑馬線 LED 預警系統"
echo "-------------------------------------------------------"

# 1. 系統檢查與更新
echo "[1/4] 正在檢查環境與安裝必要的硬體套件..."

# 檢查是否為 Raspberry Pi
if [ ! -f /etc/rpi-issue ]; then
    echo "警告: 本系統偵測到不屬於 Raspberry Pi 環境，部分硬體依賴(如 lgpio) 可能會安裝失敗。"
fi

sudo apt-get update -y
sudo apt-get install -y python3-venv python3-pip liblgpio-dev python3-opencv git libgl1

# 建立必要目錄
mkdir -p pi/templates
mkdir -p pi/static

# 2. 建立專案目錄與下載碼 (如果不在目錄內)
PROJECT_DIR="pedestrian-safety"
if [ ! -d ".git" ]; then
    echo "[2/4] 正在從 GitHub 下載程式碼..."
    if [ ! -d "$PROJECT_DIR" ]; then
        git clone https://github.com/Nono0325/pedestrian-safety.git
    fi
    cd "$PROJECT_DIR"
fi

# 3. 建立虛擬環境與安裝 Python 依賴
echo "[3/4] 正在建立虛擬環境並安裝 AI 模型依賴 (強制 CPU 輕量版)..."

# 清理舊的殘留與快取以確保空間充足
sudo apt-get clean
rm -rf ~/.cache/pip

python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip

# 關鍵優化：強制安裝 CPU 版本 torch，避免包含 NVIDIA CUDA 庫 (省下 1.5GB 空間)
echo "正在安裝核心 AI 套件 (CPU Only)..."
pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 安裝剩餘套件
pip install --no-cache-dir -r pi/requirements.txt
pip install --no-cache-dir ultralytics

# 4. 預載模型與環境檢查
echo "[4/4] 正在初始化環境..."
# 下載預設模型以避免第一次執行時等待過久
python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

echo "-------------------------------------------------------"
echo "部署完成！"
echo "-------------------------------------------------------"
echo "使用方式:"
echo "1. 進入目錄: cd pedestrian-safety"
echo "2. 啟動後台辨識: venv/bin/python3 pi/main.py"
echo "3. 啟動 Web 儀表板: venv/bin/python3 pi/dashboard.py"
echo ""
echo "注意: 執行前請確保 pi/config.json 中的 IP 位址已設為正確的 ESP32 IP。"
echo "-------------------------------------------------------"
