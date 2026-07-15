#!/bin/bash
# ==========================================================
# 先行一步 (One Step Ahead) — 一鍵更新到最新版
# ==========================================================
# 使用方式（在樹莓派上執行）：
#   curl -sSL https://raw.githubusercontent.com/Nono0325/pedestrian-safety/main/update.sh | bash
# 或已有專案目錄：
#   bash update.sh
# ==========================================================

set -e
export LANG=C.UTF-8
export LC_ALL=C.UTF-8

REPO_URL="https://github.com/Nono0325/pedestrian-safety.git"
PROJECT_DIR="pedestrian-safety"

echo "========================================================"
echo "  先行一步 (One Step Ahead) — 更新腳本"
echo "========================================================"
echo ""

# ── 1. 找到或建立專案目錄 ──────────────────────────────────
find_project_dir() {
    # 已在專案目錄內
    if [ -f "pi/dashboard.py" ]; then
        echo "$(pwd)"
        return
    fi
    # 常見安裝路徑搜尋
    for path in \
        "$HOME/$PROJECT_DIR" \
        "/home/pi/$PROJECT_DIR" \
        "/opt/$PROJECT_DIR" \
        "$HOME/pedestrian-safety"; do
        if [ -f "$path/pi/dashboard.py" ]; then
            echo "$path"
            return
        fi
    done
    echo ""
}

PROJ=$(find_project_dir)

if [ -z "$PROJ" ]; then
    echo "[INFO] 找不到現有安裝，將重新 clone 專案..."
    cd "$HOME"
    if [ -d "$PROJECT_DIR" ]; then
        rm -rf "$PROJECT_DIR"
    fi
    git clone "$REPO_URL"
    PROJ="$HOME/$PROJECT_DIR"
fi

echo "[INFO] 專案目錄: $PROJ"
cd "$PROJ"

# ── 2. 停止目前運行中的舊版程式 ───────────────────────────
echo ""
echo "[1/5] 正在停止舊版程式..."
pkill -f "python3 pi/dashboard.py"  2>/dev/null && echo "      已停止 dashboard.py" || true
pkill -f "python3 pi/main.py"       2>/dev/null && echo "      已停止 main.py（舊版）" || true
pkill -f "uvicorn"                  2>/dev/null && echo "      已停止 uvicorn" || true
sleep 1
echo "      完成"

# ── 3. 更新程式碼 ──────────────────────────────────────────
echo ""
echo "[2/5] 正在從 GitHub 拉取最新版本..."

if [ -d ".git" ]; then
    # 已有 git 倉庫 → 強制更新
    git fetch origin
    git reset --hard origin/main
    git pull origin main
    echo "      git pull 完成，已更新至最新版"
else
    # 非 git 安裝 → 備份舊 config 後重新 clone
    echo "      [警告] 目前目錄非 git 倉庫，將備份 config 後重新安裝..."
    if [ -f "pi/config.json" ]; then
        cp pi/config.json /tmp/config_backup.json
        echo "      已備份 pi/config.json 到 /tmp/config_backup.json"
    fi
    cd ..
    rm -rf "$PROJ"
    git clone "$REPO_URL"
    cd "$PROJECT_DIR"
    PROJ="$(pwd)"
    if [ -f "/tmp/config_backup.json" ]; then
        cp /tmp/config_backup.json pi/config.json
        echo "      已還原 pi/config.json"
    fi
fi

# ── 4. 確保 venv 與套件是最新 ─────────────────────────────
echo ""
echo "[3/5] 正在更新 Python 環境..."

fix_metadata_encoding() {
    python3 -c "
import glob
for path in glob.glob('venv/lib/python*/site-packages/**/*.dist-info/METADATA', recursive=True):
    try:
        with open(path, 'rb') as f: content = f.read()
        content.decode('utf-8')
    except UnicodeDecodeError:
        with open(path, 'w', encoding='utf-8', errors='replace') as f:
            f.write(content.decode('utf-8', errors='replace'))
" 2>/dev/null || true
}

if [ ! -d "venv" ]; then
    echo "      建立虛擬環境..."
    python3 -m venv venv
fi

source venv/bin/activate
fix_metadata_encoding

echo "      安裝/更新 Python 套件..."
pip install --upgrade pip --quiet
pip install --no-cache-dir -r pi/requirements.txt --quiet
fix_metadata_encoding

if ! python3 -c "import ultralytics" 2>/dev/null; then
    echo "      安裝 ultralytics (YOLOv8)..."
    pip install --no-cache-dir ultralytics --quiet
    fix_metadata_encoding
else
    echo "      ultralytics 已安裝，跳過"
fi

# ── 5. 確保 YOLOv8 模型已下載 ─────────────────────────────
echo ""
echo "[4/5] 正在確認 YOLOv8 模型..."
if [ ! -f "yolov8n.pt" ]; then
    echo "      下載模型中（約 6MB）..."
    python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt')" 2>/dev/null || true
else
    echo "      模型已存在，跳過"
fi

# ── 6. 啟動最新版系統 ──────────────────────────────────────
echo ""
echo "[5/5] 更新完成，正在啟動系統..."
echo ""
echo "========================================================"
echo "  最新版本資訊："
git log --oneline -3 2>/dev/null || true
echo "========================================================"
echo ""
echo "  瀏覽器訪問: http://$(hostname -I | awk '{print $1}' 2>/dev/null || echo 'localhost'):8000"
echo "  按 Ctrl+C 可停止"
echo "========================================================"
echo ""

chmod +x start.sh
./start.sh
