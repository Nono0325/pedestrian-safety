# 先行一步 (One Step Ahead)：單鏡頭 AI 行人意圖辨識與斑馬線預警系統

[![Competition](https://img.shields.io/badge/Competition-115%E5%B9%B4%E4%BA%BA%E6%9C%AC%E7%92%B0%E5%A2%83%E5%85%A8%E5%9C%8B%E5%A4%A7%E5%B0%88%E9%99%A2%E6%A0%A1%E5%AD%B8%E7%94%9F%E7%AB%B6%E8%B3%BD-FFB300.svg)](https://example.com)
[![Status](https://img.shields.io/badge/Status-Prototype_Ready-green.svg)]()
[![Platform](https://img.shields.io/badge/Platform-Raspberry_Pi_5-C51A4A.svg)]()

本專案致力於提升校園與社區路口的「人本交通」品質，透過邊緣運算 (Edge AI) 技術即時辨識行人穿越意圖，並自動驅動斑馬線警示 LED，減少人車衝突風險。

---

## 🌟 核心功能
- **AI 意圖辨識**：採用 YOLOv8n + ByteTrack，結合單鏡頭 Homography 投影技術，精準計算行人速度與趨近距離。
- **邊緣中繼架構**：Raspberry Pi 5 作為核心運算端，支援同時連線 5 台 ESP32-CAM 進行廣域監控。
- **智慧預警系統**：偵測到行人嘗試穿越時，自動觸發遠端斑馬線 LED 警示。
- **現代化儀表板**：提供玻璃擬態設計的 Web Dashboard，即時監控訊號強度、設備狀態與感測數據。
- **動態管理**：不需要重啟系統，即可透過 Web 介面動態新增或修改攝影機配置。

---

## 🛠️ 技術棧 (Technological Stack)
| 類別 | 使用技術 |
| :--- | :--- |
| **AI & 視覺辨識** | YOLOv8 (物件偵測), ByteTrack (追蹤), OpenCV, Homography (單鏡頭測距) |
| **後端開發** | Python 3.11+, FastAPI, Uvicorn, Zeroconf (mDNS 自動發現) |
| **嵌入式韌體** | C++, Arduino ESP32 Core, esp_http_server, mDNS Responder |
| **前端介面** | HTML5, CSS3 (Glassmorphism), JavaScript (Async Fetch), Jinja2 |
| **環境部署** | Linux Bash Shell, Python Virtual Environment (venv) |

---

## 🛠️ 硬體清單 (Bill of Materials)
| 元件名稱 | 用途 | 規格建議 |
| :--- | :--- | :--- |
| **Raspberry Pi 5** | 核心運算與後台伺服器 | 8GB RAM, 建議搭配主動式散熱 |
| **ESP32-CAM** | 邊緣攝影與 LED 驅動端 | AI-Thinker 模組 |
| **LED 警示燈條** | 路面警示提示 | 5V/12V LED Strip (接 GPIO 13) |
| **外殼/支架** | 安裝於電線桿或牆面 | 離地高度建議 3.5m - 5m |

---

## 🚀 快速開始 (Quick Start)

### A. Raspberry Pi 端 (部署與啟動)
在您的 Raspberry Pi 終端機執行一行指令即可完成環境安裝：
```bash
curl -sSL https://raw.githubusercontent.com/Nono0325/pedestrian-safety/main/install.sh | bash
```
安裝完成後，啟動系統：
1. **後台辨識引擎**: `venv/bin/python3 pi/main.py`
2. **監控中心儀表板**: `venv/bin/python3 pi/dashboard.py` (網址: `http://localhost:8000`)

### B. ESP32-CAM 端 (韌體燒錄)
1. 使用 Arduino IDE 開啟 `esp32/camera_stream.ino`。
2. 填入您的 `Wi-Fi SSID` 與 `Password`。
3. 板子選擇 `AI Thinker ESP32-CAM` 並燒錄。
4. **硬體接線**：將警示 LED 接在 **GPIO 13**。

---

## 💻 虛擬模擬測試 (Windows/Judge Testing)
如果您手邊沒有硬體，仍可透過模擬器驗證系統功能：
1. **環境設定**:
   ```powershell
   python -m venv venv
   .\venv\Scripts\activate
   pip install -r pi/requirements.txt
   ```
2. **啟動模擬器**: `python pi/mock_esp32.py`
3. **啟動儀表板**: `python pi/dashboard.py`
4. 訪問 `http://localhost:8000` 即可看到「模擬測試機」的即時畫面與警示燈動作。

---

## 📂 專案結構
```text
├── docs/             # 競賽計畫書與設計說明
├── esp32/            # ESP32-CAM 韌體源碼 (C++)
├── pi/
│   ├── templates/    # Web 儀表板 UI (HTML/CSS)
│   ├── main.py       # AI 辨識主程序
│   ├── dashboard.py  # FastAPI 後端伺服器
│   ├── mock_esp32.py # 系统模擬器
│   └── utils.py      # 幾何變換與數學工具函式
├── install.sh        # 一鍵部署腳本 (Linux)
└── README.md         # 本文件
```

---

## 📜 聯絡與作者
本專案為 **115年人本環境全國大專院校學生競賽** B組創意構想組作品。

- **團隊名稱**: nono-pi-4g
- **開發者 ID**: 50915133
- **競賽時間**: 2026年4月
