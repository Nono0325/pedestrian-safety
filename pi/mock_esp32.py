"""
mock_esp32.py — 軟體模擬 ESP32-CAM 節點

功能：
  - 模擬 MJPEG 影像串流 (/stream)
  - 模擬感測器狀態 (/status)
  - 模擬警示燈控制 (/alarm)
  - 模擬霍爾感測器有車/無車切換 (/mock_vehicle?active=1)
  - 支援 API Key 驗證 (auth query 參數)

使用方式：
  python pi/mock_esp32.py

設定說明：
  - 預設 Port：8080
  - 預設 API Key：從 pi/config.json 讀取；若不存在則使用 "nono_safety_sec_2026"
  - 在 dashboard 的「系統設定」中新增攝影機：IP 填 127.0.0.1:8080

霍爾感測器模擬：
  - 預設：每 15 秒自動模擬一次車輛通過（前 2 秒電壓升至 2.5V）
  - 手動切換：GET /mock_vehicle?active=1  → 強制有車
              GET /mock_vehicle?active=0  → 恢復自動模擬
"""

import cv2
import time
import json
import os
import asyncio
import numpy as np
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
import uvicorn

# ── 載入 API Key ──
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")
try:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        _cfg = json.load(f)
    API_KEY = _cfg.get("security", {}).get("api_key", "nono_safety_sec_2026")
except Exception:
    API_KEY = "nono_safety_sec_2026"

print(f"[MOCK] 使用 API Key: {API_KEY!r}")

app = FastAPI(title="Mock ESP32-CAM Simulator")

# ── 狀態 ──
alarm_active       = 0
start_time         = time.time()
mock_vehicle_active: int = -1  # -1=自動模擬, 0=強制無車, 1=強制有車


def verify_auth(request: Request):
    """驗證 auth query 參數，與真實 ESP32 行為一致"""
    auth = request.query_params.get("auth", "")
    if auth != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized: invalid auth key")


def create_frame(t: float, alarm: bool, vehicle: bool) -> np.ndarray:
    """產生帶有動態模擬行人和車輛狀態的畫面"""
    img = np.zeros((480, 640, 3), dtype=np.uint8)

    # 背景網格（模擬道路）
    for y in range(0, 480, 40):
        cv2.line(img, (0, y), (640, y), (30, 30, 30), 1)
    for x in range(0, 640, 40):
        cv2.line(img, (x, 0), (x, 480), (30, 30, 30), 1)

    # 行穿線
    cv2.rectangle(img, (0, 380), (640, 480), (50, 50, 50), -1)
    for x in range(0, 640, 80):
        cv2.rectangle(img, (x, 385), (x + 50, 475), (220, 220, 220), -1)

    # 模擬行人（來回移動）
    bx = int(320 + 200 * np.sin(t * 0.5))
    by = int(300 + 60 * np.cos(t * 0.3))
    color = (0, 80, 255) if alarm else (0, 220, 80)

    # 身體
    cv2.ellipse(img, (bx, by - 45), (12, 18), 0, 0, 360, color, -1)  # 頭
    cv2.line(img, (bx, by - 27), (bx, by + 10), color, 4)            # 軀幹
    cv2.line(img, (bx - 15, by - 15), (bx + 15, by - 15), color, 3)  # 手臂
    cv2.line(img, (bx, by + 10), (bx - 10, by + 35), color, 3)       # 左腿
    cv2.line(img, (bx, by + 10), (bx + 10, by + 35), color, 3)       # 右腿

    cv2.putText(img, "SIMULATED PEDESTRIAN", (bx - 90, by - 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    # 車輛狀態指示（右上角）
    v_color = (0, 200, 255) if vehicle else (60, 60, 60)
    v_label = "VEHICLE: ON" if vehicle else "VEHICLE: OFF"
    cv2.circle(img, (600, 65), 14, v_color, -1)
    cv2.putText(img, v_label, (480, 92),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, v_color, 1)

    # 標題
    cv2.putText(img, "MOCK ESP32-CAM", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)
    cv2.putText(img, f"t={t:.1f}s  port=8080", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)

    # 警報提示
    if alarm:
        cv2.rectangle(img, (0, 0), (640, 480), (0, 0, 200), 4)
        cv2.circle(img, (600, 30), 18, (0, 80, 255), -1)
        cv2.putText(img, "ALARM ACTIVE", (440, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 80, 255), 2)

    return img


@app.get("/status")
async def get_status(request: Request):
    verify_auth(request)
    uptime = int(time.time() - start_time)
    rssi   = -50 - (uptime % 10)

    # 霍爾感測器模擬
    # mock_vehicle_active == 1  → 強制有車（sensor=2.5V）
    # mock_vehicle_active == 0  → 強制無車（sensor=0.3V）
    # mock_vehicle_active == -1 → 自動：每 15 秒模擬一次車輛通過（前 2 秒）
    if mock_vehicle_active == 1:
        hall_voltage = 2.5
    elif mock_vehicle_active == 0:
        hall_voltage = 0.3
    else:  # -1: 自動模擬
        cycle = uptime % 15
        if cycle < 2:
            hall_voltage = 2.5   # 車輛通過的磁場峰值（持續約 2 秒）
        else:
            hall_voltage = 0.3 + (cycle % 3) * 0.05  # 背景磁場

    return {
        "rssi":    rssi,
        "uptime":  uptime,
        "sensor":  round(hall_voltage, 2),
        "alarm":   alarm_active,
        "vehicle": 1 if hall_voltage >= 2.0 else 0,  # 方便 dashboard 直接讀取
    }



@app.get("/alarm")
async def set_alarm(state: str, request: Request):
    global alarm_active
    verify_auth(request)
    if state == "on":
        alarm_active = 1
        print(">>> [MOCK ESP32] ALARM ON: Crosswalk LED Lit!")
    else:
        alarm_active = 0
        print(">>> [MOCK ESP32] ALARM OFF: Crosswalk LED Dark.")
    return "OK"


@app.get("/mock_vehicle")
async def mock_vehicle(active: int, request: Request):
    """
    手動切換霍爾感測器模擬狀態（測試用，無需 auth）
      /mock_vehicle?active=1  → 強制有車（sensor=2.5V）
      /mock_vehicle?active=0  → 強制無車（sensor=0.3V）
      /mock_vehicle?active=-1 → 恢復自動模擬（每 15 秒一次）
    """
    global mock_vehicle_active
    mock_vehicle_active = active
    state_str = {1: "強制有車", 0: "強制無車", -1: "自動模擬"}.get(active, f"未知({active})")
    print(f">>> [MOCK ESP32] 霍爾感測器模式切換：{state_str}")
    return {"ok": True, "mock_vehicle_active": mock_vehicle_active, "mode": state_str}


async def gen_frames():
    while True:
        t       = time.time() - start_time
        vehicle = mock_vehicle_active == 1 or (
            mock_vehicle_active == -1 and int(t) % 15 < 2
        )
        frame = create_frame(t, bool(alarm_active), vehicle)
        _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
        await asyncio.sleep(0.1)  # ~10 fps


@app.get("/stream")
async def stream(request: Request):
    verify_auth(request)
    return StreamingResponse(gen_frames(), media_type="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    print("=" * 52)
    print("  先行一步 — Mock ESP32-CAM 模擬器")
    print("=" * 52)
    print(f"  Port    : 8080")
    print(f"  API Key : {API_KEY}")
    print(f"  Stream  : http://127.0.0.1:8080/stream?auth={API_KEY}")
    print(f"  Status  : http://127.0.0.1:8080/status?auth={API_KEY}")
    print()
    print("  霍爾感測器模擬控制（瀏覽器直接開啟）：")
    print("    強制有車 → http://127.0.0.1:8080/mock_vehicle?active=1")
    print("    強制無車 → http://127.0.0.1:8080/mock_vehicle?active=0")
    print("    自動模擬 → http://127.0.0.1:8080/mock_vehicle?active=-1")
    print()
    print("  在 dashboard 設定頁面新增攝影機：")
    print("    IP → 127.0.0.1:8080")
    print("=" * 52)
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="warning")
