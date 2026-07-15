# 先行一步 (One Step Ahead)：邊緣運算 AI 行人安全系統

[![Competition](https://img.shields.io/badge/競賽-115年人本環境全國大專院校競賽-FFB300.svg)]()
[![Status](https://img.shields.io/badge/狀態-原型完成-green.svg)]()
[![Platform](https://img.shields.io/badge/平台-Raspberry_Pi_5-C51A4A.svg)]()
[![License](https://img.shields.io/badge/授權-MIT-blue.svg)]()

> 利用邊緣運算 AI 即時偵測行人過馬路意圖，於危險發生前自動觸發 LED 警示燈，降低人車衝突風險。

---

## 目錄

- [核心功能](#-核心功能)
- [系統架構](#-系統架構)
- [技術棧](#-技術棧)
- [硬體清單](#-硬體清單)
- [效能基準](#-效能基準)
- [快速開始](#-快速開始)
- [模擬測試（無硬體）](#-模擬測試無硬體)
- [專案結構](#-專案結構)
- [設定說明](#-設定說明)
- [安全性說明](#-安全性說明)
- [網路排錯](#-網路排錯)
- [競賽資訊](#-競賽資訊)

---

## 🌟 核心功能

| 功能 | 說明 |
| :--- | :--- |
| **AI 即時偵測** | YOLOv8n + ByteTrack，透過 Homography 投影精準計算行人與路口的實際距離 |
| **危險意圖判斷** | 綜合速度向量、距離門檻與持續幀數，過濾誤觸發 |
| **自動化警示** | 偵測到危險行為時，透過 HTTP 遠端觸發 ESP32 LED 警示燈（含冷卻機制）|
| **整合式儀表板** | FastAPI Web UI，提供即時 AI 標記影像串流、人數統計、警報歷史與系統診斷 |
| **多攝影機管理** | 同時管理多台 ESP32-CAM，支援透過設定頁新增 / 刪除攝影機 |
| **行動 AP 模式** | 無路由器環境下，Pi 可自行建立 Wi-Fi 熱點供展示使用 |
| **低延遲架構** | 非同步協程 + 多執行緒，影像端到端延遲低於 300ms |

---

## 🏗️ 系統架構

```
  ┌─────────────────────┐
  │  ESP32-CAM  (x1~4)  │  ← MJPEG 影像串流 (HTTP)
  │  + LED 警示燈        │  ← 接收 GPIO 12 控制訊號
  └────────┬────────────┘
           │ Wi-Fi (HTTP)
           ▼
  ┌──────────────────────────────────────────┐
  │         Raspberry Pi 5  (核心)            │
  │                                          │
  │  ┌─────────────────┐  ┌───────────────┐  │
  │  │   AI Worker      │  │  FastAPI       │  │
  │  │  YOLOv8n        │  │  Web Server    │  │
  │  │  ByteTrack      │◄─►  Port 8000    │  │
  │  │  Homography     │  │  Jinja2 UI    │  │
  │  └─────────────────┘  └───────────────┘  │
  │         dashboard.py（單一啟動點）         │
  └──────────────────────┬───────────────────┘
                         │ HTTP
                         ▼
            瀏覽器 http://<Pi-IP>:8000
           （手機 / 電腦 / 平板皆可）
```

---

## 🛠️ 技術棧

| 領域 | 使用技術 |
| :--- | :--- |
| **AI 推論** | YOLOv8n (Ultralytics)、ByteTrack、OpenCV |
| **空間分析** | Homography 透視變換（影像座標 → 真實世界座標）|
| **後端** | Python 3.11+、FastAPI、Uvicorn、Zeroconf (mDNS)、psutil |
| **嵌入式** | C++、Arduino ESP32 Core、esp_http_server、mDNS Responder |
| **前端** | HTML5、CSS3 (Glassmorphism)、JavaScript、Jinja2 |
| **部署** | Linux Bash、Python venv、systemd（可選）|

---

## 📦 硬體清單

| 零件 | 角色 | 規格說明 |
| :--- | :--- | :--- |
| **Raspberry Pi 5** | 核心運算與管理站 | 建議 8GB RAM，配主動散熱風扇 |
| **Hailo-8L AI Kit** | NPU 推論加速（選配）| 官方 M.2 HAT+，需啟用 PCIe Gen 3 |
| **ESP32-CAM** | 影像採集 + LED 控制 | AI-Thinker 模組，支援 OV2640 |
| **LED 警示燈條** | 行人警示顯示 | 5V/12V LED Strip，接 **GPIO 12** |
| **防水外殼 / 支架** | 戶外防護固定 | 建議架設高度 3.5m ~ 5m |

---

## 📊 效能基準

> 數據由 `pi_benchmark.py` 與 `stream4_benchmark.py` 模擬，
> 基準為 x86 實測推論時間（~25ms/幀），套用各硬體縮放因子與熱節流模型。

### 單路 ESP32-CAM

| 硬體平台 | 平均 FPS | 平均延遲 | 達標 (≥10 fps) |
| :--- | :---: | :---: | :---: |
| Pi 5（CPU only） | 6.3 fps | 158 ms | ❌ |
| Pi 5 + ONNX INT8 | 10.0 fps | 100 ms | ⚠️ 邊界 |
| **Pi 5 + Hailo-8L** ⭐ | ~110 fps | ~9 ms | ✅ |
| Pi 5 + Hailo-8 (26T) | ~220 fps | ~4.5 ms | ✅ |
| Jetson Orin Nano 4GB | ~50 fps | ~20 ms | ✅ |
| Jetson Orin NX 8GB | ~110 fps | ~9 ms | ✅ |

### 4 路 ESP32-CAM 並行

| 硬體平台 | 每路 FPS | 建議台數 | 預估成本 (USD) |
| :--- | :---: | :---: | :---: |
| Pi 5 CPU only | 1.8 fps | ❌ 4 台仍不足 | $320 |
| Pi 5 + ONNX INT8 | 2.8 fps | ⚠️ 4 台各跑 1 路 | $320 |
| **Pi 5 + Hailo-8L** ⭐ | **44.5 fps** | **1 台** | **$150** |
| Pi 5 + Hailo-8 (26T) | 89 fps | 1 台 | $210 |
| Jetson Orin Nano | 21.8 fps | 1 台 | $199 |
| Jetson Orin NX 8GB | 49 fps | 1 台 | $399 |

> ⭐ **競賽推薦方案**：Pi 5 (8GB) + Hailo-8L
> - 總成本約 $150 USD（≈ NT$4,700），官方套件開箱即用
> - 4 路串流每路達 44.5 fps，餘裕高達 4.45 倍
> - 功耗僅 8W，適合長時間戶外部署

### 執行效能模擬腳本

```bash
# 安裝依賴
pip install ultralytics psutil opencv-python numpy

# 單路 Pi 5 效能模擬（產生 HTML 報表）
python pi_benchmark.py

# 多硬體效能對比
python hardware_benchmark.py

# 4 路 ESP32-CAM 並行需求計算
python stream4_benchmark.py
```

---

## 🚀 快速開始

### A. Raspberry Pi 端（處理中心）

**方法一：自動化安裝腳本（推薦）**
```bash
curl -sSL https://raw.githubusercontent.com/Nono0325/pedestrian-safety/main/install.sh | bash
```

**方法二：手動安裝**
```bash
git clone https://github.com/Nono0325/pedestrian-safety.git
cd pedestrian-safety
python3 -m venv venv
source venv/bin/activate
pip install -r pi/requirements.txt
pip install ultralytics
```

**啟動系統**
```bash
chmod +x start.sh
./start.sh
```

瀏覽器訪問 `http://<Pi-IP>:8000` 即可看到整合 AI 的管理介面。

> **注意**：AI 辨識與 Web 儀表板已整合為單一程式 `pi/dashboard.py`，無需分開啟動。

### 🔄 更新到最新版（一鍵更新）

若已安裝舊版，執行以下指令即可強制更新並重啟系統：

```bash
# 方式一：若已在專案目錄內
bash update.sh

# 方式二：從網路直接更新（不需要進入目錄）
curl -sSL https://raw.githubusercontent.com/Nono0325/pedestrian-safety/main/update.sh | bash
```

`update.sh` 會自動：
1. 停止舊版程式
2. 從 GitHub 拉取最新程式碼
3. 更新 Python 套件
4. 啟動新版系統

---

### B. ESP32-CAM 端（採集端）

1. 使用 **Arduino IDE** 開啟 `esp32/camera_stream.ino`
2. 複製並填寫設定檔：
   ```bash
   cp esp32/secrets.h.example esp32/secrets.h
   # 編輯 secrets.h，填入 Wi-Fi SSID / 密碼 / API Key
   ```
3. Arduino IDE 選擇開發板：**AI Thinker ESP32-CAM**，燒錄韌體
4. **硬體接線**：LED 警示燈接 **GPIO 12**（已迴避 SD 卡腳位衝突）

---

### C. Windows 一鍵啟動

直接雙擊 **`start.bat`** 即可。

---

## 🧪 模擬測試（無硬體）

不需要實體硬體，在 Windows / macOS / Linux 均可預覽完整功能。

**第一步：建立環境並安裝依賴**
```powershell
python -m venv venv
.\venv\Scripts\activate        # Windows
# source venv/bin/activate     # Linux / macOS
pip install -r pi/requirements.txt
pip install ultralytics
```

**第二步：啟動 ESP32 模擬器**（另開一個終端機）
```powershell
python pi/mock_esp32.py
# 模擬器將在 http://127.0.0.1:8080 提供：
#   /stream   → 模擬 MJPEG 串流
#   /alarm    → 模擬警示燈控制
#   /status   → 模擬感測器狀態
```

**第三步：啟動管理儀表板**
```powershell
python pi/dashboard.py
```

**第四步**：瀏覽器訪問 `http://localhost:8000`，
在「**系統設定**」頁面新增攝影機 IP：`127.0.0.1:8080`

---

## 🏕️ 行動 AP 模式

在無路由器的展示環境，可將 Pi 設為 Wi-Fi 基地台：

```bash
sudo chmod +x pi/setup_ap.sh
sudo ./pi/setup_ap.sh
```

| 項目 | 預設值 |
| :--- | :--- |
| SSID | `OneStepAhead_AP` |
| 密碼 | `NonoSafety@2026` |
| Pi IP | `192.168.4.1` |
| 儀表板 | `http://192.168.4.1:8000` |

啟動後，首頁將出現「Pi 基地台狀態」卡片，顯示當前熱點資訊。

---

## 📂 專案結構

```text
pedestrian-safety/
├── docs/                        # 專案文件
│   ├── proposal.md              # 計畫書
│   └── reports/                 # 效能模擬報表（auto-generated）
├── esp32/                       # ESP32-CAM 韌體（C++）
│   ├── camera_stream.ino        # 主韌體：串流 + 警示燈控制 + API 驗證
│   ├── camera_pins.h            # GPIO 腳位定義
│   └── secrets.h                # Wi-Fi 與 API Key（已加入 .gitignore）
├── pi/                          # Raspberry Pi 端程式
│   ├── templates/               # Web UI 模板
│   │   ├── index.html           # 主儀表板
│   │   └── settings.html        # 系統設定頁
│   ├── dashboard.py             # 核心程式：AI 推論 + FastAPI Web 服務
│   ├── main.py                  # 獨立 AI 辨識模組（除錯用）
│   ├── mock_esp32.py            # ESP32-CAM 軟體模擬器（開發測試用）
│   ├── utils.py                 # 工具函式（Homography 計算等）
│   ├── config.json              # 系統設定（攝影機 IP、API Key 等）
│   ├── requirements.txt         # Python 依賴清單
│   └── setup_ap.sh              # Wi-Fi AP 模式設定腳本
├── pi_benchmark.py              # Pi 5 單路效能模擬
├── hardware_benchmark.py        # 多硬體效能對比
├── stream4_benchmark.py         # 4 路並行需求計算
├── install.sh                   # 自動化環境安裝（Linux / Pi）
├── start.sh                     # 一鍵啟動（Linux / Pi）
├── start.bat                    # 一鍵啟動（Windows）
├── yolov8n.pt                   # YOLOv8n 模型權重（已加入 .gitignore）
└── README.md                    # 專案說明文件
```

---

## ⚙️ 設定說明

所有系統設定集中於 `pi/config.json`，修改此檔案後重啟服務即可生效：

```json
{
  "cameras": [
    { "id": 1, "name": "西側校門",       "ip": "192.168.1.101" },
    { "id": 2, "name": "宿舍區行穿線",   "ip": "192.168.1.102" },
    { "id": 99,"name": "模擬測試機",     "ip": "127.0.0.1:8080" }
  ],
  "server": {
    "port": 8000,
    "debug": false
  },
  "security": {
    "api_key": "your_api_key_here"
  }
}
```

> **重要**：`api_key` 是唯一的金鑰來源，所有模組（`dashboard.py`、`main.py`、`mock_esp32.py`）皆從此處讀取，**只需修改這一個地方**即可同步全部設定。

---

## 🔒 安全性說明

1. **API 金鑰驗證**：所有對 ESP32 的 HTTP 請求必須攜帶 `auth` 參數
   ```
   http://{ESP32-IP}/stream?auth={API_KEY}
   http://{ESP32-IP}/alarm?state=on&auth={API_KEY}
   ```
2. **金鑰集中管理**：修改金鑰只需更新 `pi/config.json` 的 `api_key` 欄位，以及 `esp32/secrets.h` 中的 `#define API_KEY`
3. **敏感檔案保護**：`esp32/secrets.h` 與 `yolov8n.pt` 已加入 `.gitignore`，不會上傳至版本控制

---

## 🌐 網路排錯

| 症狀 | 解決方式 |
| :--- | :--- |
| 影像無法顯示 | 確認 Pi 與 ESP32 在同一 Wi-Fi，IP 網段相同 |
| TCP 拒絕連線 | 在「系統設定」頁使用「連線測試」功能診斷 |
| mDNS 找不到設備 | 確認路由器未封鎖 mDNS，設備名稱為 `esp32-safety` |
| 無線隔離問題 | 關閉路由器的 AP Isolation（硬體隔離）功能 |
| 無實體硬體測試 | 執行 `python pi/mock_esp32.py`，新增 `127.0.0.1:8080` 作為攝影機 |

---

## 🎓 競賽資訊

| 項目 | 內容 |
| :--- | :--- |
| 競賽名稱 | 115 年人本環境全國大專院校學生競賽 |
| 隊伍名稱 | nono-pi-4g |
| 開發期間 | 2026 年 4 月 ~ 2026 年 7 月 |

---

*本專案以 MIT 授權釋出，歡迎引用與改作，請保留原作者資訊。*
