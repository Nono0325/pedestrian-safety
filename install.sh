#!/bin/bash

# ==========================================================
# 先行一步 (One Step Ahead) - Raspberry Pi 5 自動化部署腳本
# ==========================================================

set -e # Exit on error

# 強制設定語系為 UTF-8，避免系統預設編碼問題
export LANG=C.UTF-8
export LC_ALL=C.UTF-8

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
sudo apt-get install -y python3-venv python3-pip python3-dev swig liblgpio-dev python3-opencv git libgl1

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

# 定義自動修復 METADATA 編碼問題的函數 (避免 Python 3.13+ 的 UnicodeDecodeError)
fix_metadata_encoding() {
    echo "正在檢查與修復 Python 套件 METADATA 編碼..."
    python3 -c "
import glob
for path in glob.glob('venv/lib/python*/site-packages/**/*.dist-info/METADATA', recursive=True):
    try:
        with open(path, 'rb') as f:
            content = f.read()
        content.decode('utf-8')
    except UnicodeDecodeError:
        print(f'修復檔案編碼: {path}')
        with open(path, 'w', encoding='utf-8', errors='replace') as f:
            f.write(content.decode('utf-8', errors='replace'))
"
}

# 清理舊的殘留與快取以確保空間充足
sudo apt-get clean
rm -rf ~/.cache/pip

python3 -m venv venv
source venv/bin/activate

# 立即修復先前遺留的任何編碼問題
fix_metadata_encoding

pip install --upgrade pip
fix_metadata_encoding

# 預先安裝並修復 scipy，避免其 METADATA 編碼問題干擾後續 pip 解析
echo "正在預先安裝並修復 scipy..."
pip install --no-cache-dir scipy
fix_metadata_encoding

# 關鍵優化：強制安裝 CPU 版本 torch，避免包含 NVIDIA CUDA 庫 (省下 1.5GB 空間)
echo "正在安裝核心 AI 套件 (CPU Only)..."
pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fix_metadata_encoding

# 安裝剩餘套件
pip install --no-cache-dir -r pi/requirements.txt
fix_metadata_encoding

pip install --no-cache-dir ultralytics
fix_metadata_encoding

# 4. 預載模型與環境檢查
echo "[4/4] 正在初始化環境..."
# 下載預設模型以避免第一次執行時等待過久
python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

echo "-------------------------------------------------------"
echo "部署完成！"
echo "-------------------------------------------------------"
echo "使用方式:"
echo "  1. 進入目錄:  cd pedestrian-safety"
echo "  2. 啟動系統:  chmod +x start.sh && ./start.sh"
echo ""
echo "  或手動啟動（已整合 AI + Web 儀表板）："
echo "    venv/bin/python3 pi/dashboard.py"
echo ""
echo "  瀏覽器開啟:  http://<Pi-IP>:8000"
echo ""
echo "注意事項:"
echo "  - 執行前請確保 pi/config.json 中的 ESP32 IP 已填正確"
echo "  - AI 辨識與 Web 儀表板已整合為單一程式，無需分開啟動"
echo "  - 霍爾感測器狀態可在儀表板即時監看"
echo "-------------------------------------------------------"
