# 先行一步 (One Step Ahead)
### 單鏡頭 AI 行人意圖辨識與斑馬線 LED 預警系統

## 🚀 一鍵部署 (Raspberry Pi 5)
在您的 Raspberry Pi 終端機執行以下指令即可自動安裝所有依賴環境：
```bash
curl -sSL https://raw.githubusercontent.com/Nono0325/pedestrian-safety/main/install.sh | bash
```

## 📊 監控儀表板 (Web Dashboard)
本專案包含一個現代化的 Web 儀表板，可同時監控多達 5 台 ESP32-CAM：
- **啟動方式**: `venv/bin/python3 pi/dashboard.py`
- **瀏覽網址**: `http://<樹莓派IP>:8000`
- **功能**: 即時影像串流、連線狀態顯示、玻璃擬態 UI。

本專案旨在參與「115年人本環境全國大專院校學生競賽」，透過邊緣運算 (Edge AI) 技術提升校園與社區行穿線的安全性。

## 系統架構
- **攝影機 (Source)**: ESP32-CAM 執行 MJPEG Server。
- **邊緣運算 (Inference)**: Raspberry Pi 5 進行人形偵測 (YOLOv8n) 與多目標追蹤 (ByteTrack)。
- **邏輯判斷 (Logic)**: 透過 Homography 座標轉換與速度向量分析，預測行人穿越意圖。
- **預警輸出 (Output)**: 透過 GPIO 點亮斑馬線側邊 LED。

## 目錄說明
- `/docs`: 競賽提案報告與 A0 版面構想。
- `/esp32`: ESP32-CAM 原始碼。
- `/pi`: Raspberry Pi 核心 AI 串流處理程式。

## 快速開始

### 1. ESP32-CAM 部署
1. 使用 Arduino IDE 開啟 `esp32/camera_stream.ino`。
2. 修改 `ssid` 與 `password` 為您的 Wi-Fi 資訊。
3. 上傳至 ESP32-CAM，並記錄下串流網址 (如 `http://192.168.1.10/stream`)。

### 2. Raspberry Pi 5 部署
1. 安裝環境依賴：
   ```bash
   pip install -r pi/requirements.txt
   ```
2. 修改 `pi/main.py` 中的 `STREAM_URL` 為您的 ESP32 IP。
3. 執行主程式：
   ```bash
   python pi/main.py
   ```

## 技術關鍵點
- **Homography**: 將 2D 影像座標精準映射至現實地面尺寸，實現精確的速度與距離計算。
- **Intent Recognition**: 非單純偵測「人」，而是判斷「意圖」。透過停留時間與朝向向量過濾掉平行經過的行人，減少 LED 誤觸。

## 聯絡資訊
[團隊名稱] - 2026年人本環境競賽 B組創意構想組
