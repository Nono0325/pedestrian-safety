# 先行一步 (One Step Ahead)：邊緣運算 AI 交通監控系統

[![Competition](https://img.shields.io/badge/Competition-115%E5%B9%B4%E4%BA%BA%E6%9C%AC%E7%92%B0%E5%A2%83%E5%85%A8%E5%9C%8B%E5%A4%A7%E5%B0%88%E9%99%A2%E6%A0%A1%E5%AD%B8%E7%94%9F%E7%AB%B6%E8%B3%BD-FFB300.svg)](https://example.com)
[![Status](https://img.shields.io/badge/Status-Prototype_Ready-green.svg)]()
[![Platform](https://img.shields.io/badge/Platform-Raspberry_Pi_5-C51A4A.svg)]()

本專案致力於透過邊緣運算 (Edge AI) 技術提升路口行人安全。透過即時影像辨識行人動態，並在必要時觸發邊緣端的 LED 警示看板，降低人車衝突風險。

---

## 🌟 核心功能
- **AI 即時偵測**：使用 YOLOv8n + ByteTrack，結合 Homography 投影技術精準計算行人與路口的距離。
- **邊緣端串流技術**：Raspberry Pi 5 作為核心處理站，同時管理多台 ESP32-CAM 影像採集節點。
- **自動化警示**：辨識到行人進入危險區域時，自動遠端觸發 ESP32 上的 LED 警示燈。
- **現代化管理介面**：基於 FastAPI 的 Web Dashboard，支援即時影像看版、參數設定與網路連線診斷。
- **低延遲優化**：採用非同步協程與多執行緒處理，確保影像延遲保持在 300ms 以下。

---

## 🛠️ 技術棧 (Technological Stack)
| 領域 | 使用技術 |
| :--- | :--- |
| **AI & 影像辨識** | YOLOv8 (Ultralytics), ByteTrack (追蹤), OpenCV, Homography (透視變換) |
| **後端架構** | Python 3.11+, FastAPI, Uvicorn, Zeroconf (mDNS 自動發現) |
| **嵌入式開發** | C++, Arduino ESP32 Core, esp_http_server, mDNS Responder |
| **前端介面** | HTML5, CSS3 (Glassmorphism), JavaScript (Async Fetch), Jinja2 |
| **部署環境** | Linux Bash Shell, Python Virtual Environment (venv) |

---

## 📋 零件清單 (Bill of Materials)
| 硬體名稱 | 角色 | 建議規格 |
| :--- | :--- | :--- |
| **Raspberry Pi 5** | 核心運算與管理站 | 8GB RAM, 建議搭配散熱風扇 |
| **ESP32-CAM** | 邊緣影像與 LED 控制站 | AI-Thinker 模組 |
| **LED 警示模組** | 行人警示顯示 | 5V/12V LED Strip (接 **GPIO 12**) |
| **外殼/支架** | 硬體防護與固定 | 建議高度 3.5m - 5m |

---

## 🚀 快速開始 (Quick Start)

### A. Raspberry Pi 端 (處理中心)
在您的 Raspberry Pi 終端機執行以下指令進行自動化安裝：
```bash
curl -sSL https://raw.githubusercontent.com/Nono0325/pedestrian-safety/main/install.sh | bash
```
安裝完成後執行：
1. **啟動辨識主程式**: `venv/bin/python3 pi/main.py`
2. **啟動管理介面**: `venv/bin/python3 pi/dashboard.py` (瀏覽器訪問：http://localhost:8000)

### B. ESP32-CAM 端 (採集端)
1. 使用 Arduino IDE 開啟 `esp32/camera_stream.ino`。
2. 修改程式碼中的 Wi-Fi SSID 與 Password。
3. 開發板選擇 AI Thinker ESP32-CAM 並燒錄。
4. **硬體接線**：將 LED 警示燈接在 **GPIO 12** (避開 SD 卡衝突)。

---

## ⚡ 一鍵啟動 (One-Click Start)

為了方便現場演示，您可以直接執行以下腳本同時啟動 AI 辨識與管理介面：

- **Linux / Raspberry Pi**:
  ```bash
  chmod +x start.sh
  ./start.sh
  ```
- **Windows**: 直接雙擊執行 `start.bat` 即可。

---

## 🏕️ 行動部署模式 (Portable AP Mode)

為了在沒有路由器的環境下進行展示，本系統支援將 Raspberry Pi 設為 Wi-Fi 基地台：

1. **執行設定腳本**:
   ```bash
   sudo chmod +x pi/setup_ap.sh
   sudo ./pi/setup_ap.sh
   ```
2. **預設熱點資訊**:
   - **SSID**: OneStepAhead_AP
   - **Password**: NonoSafety@2026
   - **Pi IP**: 192.168.4.1
3. **儀表板監控**: 啟動後，首頁將出現「Pi 基地台狀態」卡片，顯示當前熱點名稱與連線狀態。

---

## 🧪 模擬與測試 (Windows/Judge Testing)
如果您在沒有實體硬體的情況下想要預覽功能：
1. **環境配置**:
   ```powershell
   python -m venv venv
   .\venv\Scripts\activate
   pip install -r pi/requirements.txt
   ```
2. **啟動模擬器**: `python pi/mock_esp32.py`
3. **啟動管理介面**: `python pi/dashboard.py`
4. 訪問 http://localhost:8000 即可看到虛擬測試畫面與網路診斷數據。

---

## 📂 專案結構
```text
├── docs/             # 專案文件與設計說明
├── esp32/            # ESP32-CAM 韌體源碼 (C++)
├── pi/
│   ├── templates/    # Web 管理介面 UI (HTML/CSS)
│   ├── main.py       # AI 辨識主核心
│   ├── dashboard.py  # FastAPI 後端管理系統
│   ├── mock_esp32.py # 軟體模擬器
│   └── utils.py      # 通用工具函式
├── install.sh        # 環境安裝腳本 (Linux)
├── start.sh          # 一鍵啟動腳本 (Linux/Pi)
├── start.bat         # 一鍵啟動腳本 (Windows)
└── README.md         # 專案說明文件
```

---

## 🎓 專案背景：本作品參加 **115年人本環境全國大專院校學生競賽**
- **隊伍名稱**: nono-pi-4g
- **參賽 ID**: 50915133
- **開發時間**: 2026年4月

---

## 🔒 安全性說明 (Security)

本專案已實作基本的資安防護機制：

1. **隱私保護**：Wi-Fi 與 API 金鑰存放在 `esp32/secrets.h`，已加入 `.gitignore`，請勿上傳至公開倉庫。
2. **API 認證**：所有對 ESP32-CAM 的 HTTP 請求必須攜帶 `auth` 參數，範例：`http://{IP}/stream?auth={KEY}`
3. **金鑰同步**：若要更改金鑰，請同時修改 `esp32/secrets.h` 與 `pi/config.json` 中的 `api_key` 欄位。
4. **CORS 安全**：後端已限制僅允許特定來源訪問。若需遠端管理，建議搭配 VPN 或 SSH Tunnel。

---

## 🌐 網路排錯指南 (Network Troubleshooting)

若影像無法顯示，請檢查以下幾點：

1. **檢查網段是否一致**：確保 Pi 與 ESP32-CAM 在同一個 Wi-Fi 下，IP 網段相同。
2. **使用連線測試工具**：在「系統設定」頁中使用「連線測試 (TCP)」功能測試攝影機 IP。
3. **mDNS 自動發現**：確保路由器沒有阻擋 mDNS 封包（設備名稱通常為 esp32-safety）。
4. **硬體隔離 (AP Isolation)**：部分路由器的無線隔離功能會導致 Pi 無法連線 ESP32，請關閉此功能。
5. **模擬器測試**：可先執行 `python pi/mock_esp32.py` 並新增 `127.0.0.1:8080` 作為攝影機進行測試。
