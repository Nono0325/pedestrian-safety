import os
import sys

# ── Windows UTF-8 修復：強制 stdout/stderr 使用 UTF-8，避免 cp950 錯誤 ──
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import json
import subprocess
import asyncio
import time
import threading
import cv2
import httpx
import socket
import numpy as np
from collections import deque
from datetime import datetime
from fastapi import FastAPI, Request, BackgroundTasks, Form
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List, Optional
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import concurrent.futures

# zeroconf
try:
    from zeroconf import ServiceBrowser, Zeroconf, ServiceListener
    ZEROCONF_AVAILABLE = True
except ImportError:
    ZEROCONF_AVAILABLE = False

# psutil
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# ultralytics YOLOv8
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

# ===================
# CONFIG & INIT
# ===================
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")
MODEL_PATH  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "yolov8n.pt")

def load_config():
    if not os.path.exists(CONFIG_PATH):
        return {"cameras": [], "security": {"api_key": "nono_safety_sec_2026"}}
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[CONFIG] 無法讀取設定檔: {e}")
        return {"cameras": [], "security": {"api_key": "nono_safety_sec_2026"}}

config   = load_config()
API_KEY  = config.get("security", {}).get("api_key", "")
is_shutting_down = False

# ===================
# AI MODELS & LOGIC
# ===================

# Homography Calibration (場域校正坐標) — 可在 config 中覆寫
SRC_PTS = [[0, 480], [640, 480], [640, 0], [0, 0]]
DST_PTS = [[0, 5],   [5, 5],     [5, 0],   [0, 0]]

# Thresholds
INTENT_THRESHOLD_DIST = 1.2
INTENT_THRESHOLD_VEL  = 0.4
INTENT_MIN_FRAMES     = 4
ALARM_DURATION        = 5
CURB_Y_THRESHOLD      = 2.5

class HomographyTransformer:
    def __init__(self, src_pts, dst_pts):
        self.H, _ = cv2.findHomography(np.float32(src_pts), np.float32(dst_pts))

    def transform(self, px, py):
        point = np.float32([px, py, 1.0])
        transformed = np.dot(self.H, point)
        transformed /= transformed[2]
        return transformed[0], transformed[1]


class PedestrianTracker:
    def __init__(self, cam_id: int, alarm_url: str):
        self.cam_id = cam_id
        self.alarm_url = alarm_url
        self.tracks: dict = {}
        self.last_alarm_time: float = 0
        self._alarm_lock: bool = False

    def _send_alarm_request(self, state: str):
        try:
            import requests as req
            url = f"{self.alarm_url}?state={state}&auth={API_KEY}"
            req.get(url, timeout=1.5)
        except Exception as e:
            print(f"[ALARM CAM{self.cam_id}] 請求失敗: {e}")
        finally:
            self._alarm_lock = False

    def trigger_alarm(self):
        now = time.time()
        if self._alarm_lock or (self.last_alarm_time > 0 and now - self.last_alarm_time < ALARM_DURATION):
            return
        print(f">>> AI TRIGGER CAM{self.cam_id}: Alarm ON")
        self._alarm_lock = True
        self.last_alarm_time = now
        threading.Thread(target=self._send_alarm_request, args=("on",), daemon=True).start()

    def reset_alarm_if_needed(self):
        now = time.time()
        if self.last_alarm_time > 0 and now - self.last_alarm_time >= ALARM_DURATION:
            print(f">>> AI TRIGGER CAM{self.cam_id}: Alarm OFF")
            self.last_alarm_time = 0
            threading.Thread(target=self._send_alarm_request, args=("off",), daemon=True).start()

    @property
    def alarm_active(self) -> bool:
        return self.last_alarm_time > 0 and (time.time() - self.last_alarm_time < ALARM_DURATION)

    def cleanup_tracks(self):
        now = time.time()
        expired = [tid for tid, info in self.tracks.items() if now - info['last_time'] > 60]
        for tid in expired:
            del self.tracks[tid]

    def update(self, track_id, pos_ground):
        now = time.time()
        if track_id not in self.tracks:
            self.tracks[track_id] = {'last_pos': pos_ground, 'last_time': now, 'intent_count': 0}
            return False
        info = self.tracks[track_id]
        dt = now - info['last_time']
        if dt <= 0:
            return False

        dx = pos_ground[0] - info['last_pos'][0]
        dy = pos_ground[1] - info['last_pos'][1]
        dist = np.sqrt(dx**2 + dy**2)
        velocity = dist / dt
        vx, vy = dx / dt, dy / dt

        # is_approaching_curb
        approaching = False
        d_to_curb = abs(pos_ground[1] - CURB_Y_THRESHOLD)
        if pos_ground[1] < CURB_Y_THRESHOLD and vy > 0.2:
            approaching = True
        elif pos_ground[1] > CURB_Y_THRESHOLD and vy < -0.2:
            approaching = True

        is_intent = False
        if approaching and d_to_curb < INTENT_THRESHOLD_DIST and velocity > INTENT_THRESHOLD_VEL:
            info['intent_count'] += 1
        else:
            info['intent_count'] = max(0, info['intent_count'] - 1)

        if info['intent_count'] >= INTENT_MIN_FRAMES:
            is_intent = True
            self.trigger_alarm()

        info['last_pos'] = pos_ground
        info['last_time'] = now
        return is_intent


# ── 全域 AI 資源 ──
_yolo_model    = None
_yolo_lock     = threading.Lock()
_transformer   = HomographyTransformer(SRC_PTS, DST_PTS)
_trackers: dict[int, PedestrianTracker] = {}   # cam_id -> tracker
_thread_pool   = concurrent.futures.ThreadPoolExecutor(max_workers=2)

def get_yolo_model():
    """惰性載入 YOLO 模型（全域單例）"""
    global _yolo_model
    if _yolo_model is not None:
        return _yolo_model
    with _yolo_lock:
        if _yolo_model is None:
            if YOLO_AVAILABLE and os.path.exists(MODEL_PATH):
                print(f"[AI] 正在載入 YOLOv8 模型: {MODEL_PATH}")
                _yolo_model = YOLO(MODEL_PATH)
                print("[AI] 模型載入完成")
            else:
                print(f"[AI] YOLOv8 不可用 (YOLO_AVAILABLE={YOLO_AVAILABLE}, model_path={MODEL_PATH})")
                _yolo_model = None
    return _yolo_model


def _sync_ai_process(cam_id: int, jpg_bytes: bytes) -> bytes:
    """
    同步 AI 推論 — 在 ThreadPoolExecutor 中執行，不阻塞 Event Loop。
    回傳帶有偵測框的 JPEG bytes。
    """
    model = get_yolo_model()
    if model is None:
        return jpg_bytes  # 無 AI 時直接回傳原始畫面

    frame = cv2.imdecode(np.frombuffer(jpg_bytes, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        return jpg_bytes

    tracker = _trackers.get(cam_id)
    if tracker is None:
        return jpg_bytes

    tracker.reset_alarm_if_needed()

    try:
        results = model.track(
            frame, persist=True, classes=[0],
            tracker="bytetrack.yaml", verbose=False, imgsz=320
        )
    except Exception as e:
        print(f"[AI CAM{cam_id}] 推論錯誤: {e}")
        return jpg_bytes

    person_count = 0
    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids   = results[0].boxes.id.cpu().numpy().astype(int)
        person_count = len(ids)

        for box, track_id in zip(boxes, ids):
            foot_x = (box[0] + box[2]) / 2
            foot_y = box[3]
            try:
                gx, gy = _transformer.transform(foot_x, foot_y)
                is_danger = tracker.update(track_id, (gx, gy))
            except Exception:
                is_danger = False

            color = (0, 0, 255) if is_danger else (0, 255, 0)
            # 畫偵測框
            cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
            # 畫腳部圓點
            cv2.circle(frame, (int(foot_x), int(foot_y)), 5, color, -1)
            # 標籤
            label = f"ID:{track_id}"
            if is_danger:
                label += " DANGER"
            cv2.putText(frame, label, (int(box[0]), int(box[1]) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    # 更新 AI 狀態快取
    ai_status_cache[cam_id] = {
        "person_count": person_count,
        "alarm": 1 if tracker.alarm_active else 0,
        "ai_enabled": True,
    }

    # 編碼回 JPEG
    success, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if success:
        return buf.tobytes()
    return jpg_bytes


# ===================
# GLOBAL STATE
# ===================
frame_cache:    dict[int, bytes] = {}   # 原始畫面 (備用)
ai_frame_cache: dict[int, bytes] = {}   # AI 標記後的畫面
status_cache:   dict[int, dict]  = {}
ai_status_cache: dict[int, dict] = {}
discovered_devices: list = []
system_logs = deque(maxlen=50)


def add_log(message: str, level: str = "INFO"):
    timestamp = datetime.now().strftime("%H:%M:%S")
    entry = {"time": timestamp, "msg": message, "level": level}
    system_logs.append(entry)
    try:
        print(f"[{timestamp}] {level}: {message}")
    except UnicodeEncodeError:
        safe_msg = message.encode(sys.stdout.encoding or 'utf-8', errors='replace').decode(sys.stdout.encoding or 'utf-8')
        print(f"[{timestamp}] {level}: {safe_msg}")


# ===================
# NETWORK UTILS
# ===================
async def check_tcp_port(host_port: str, default_port: int = 80, timeout: float = 3.0):
    try:
        if ":" in host_port:
            host, port_str = host_port.split(":")
            port = int(port_str)
        else:
            host = host_port
            port = default_port
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port), timeout=timeout
        )
        writer.close()
        await writer.wait_closed()
        return True, "OPEN"
    except asyncio.TimeoutError:
        return False, "Timeout"
    except ConnectionRefusedError:
        return False, "Refused (Busy)"
    except Exception as e:
        return False, f"Err: {str(e)[:20]}"


def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "127.0.0.1"


# ===================
# PYDANTIC MODELS
# ===================
class CameraConfig(BaseModel):
    id: int
    name: str
    ip: str

class SettingsUpdate(BaseModel):
    cameras: List[CameraConfig]


# ===================
# MDNS DISCOVERY
# ===================
if ZEROCONF_AVAILABLE:
    from zeroconf import ServiceListener

    class MDNSListener(ServiceListener):
        def add_service(self, zc: Zeroconf, type_: str, name: str) -> None:
            info = zc.get_service_info(type_, name)
            if info:
                addresses = [".".join(map(str, addr)) for addr in info.addresses]
                if addresses:
                    ip = addresses[0]
                    if "esp32-safety" in name:
                        device = {"name": name.split(".")[0], "ip": ip}
                        if device not in discovered_devices:
                            discovered_devices.append(device)
                            print(f"Discovered via mDNS: {device}")

        def update_service(self, zc, type_, name): pass
        def remove_service(self, zc, type_, name): pass

    def start_mdns_discovery():
        zeroconf = Zeroconf()
        listener = MDNSListener()
        ServiceBrowser(zeroconf, "_http._tcp.local.", listener)
        return zeroconf
else:
    def start_mdns_discovery():
        return None


# ===================
# BACKGROUND TASKS
# ===================
async def fetch_camera_data(cam_id: int, ip: str):
    stream_url = f"http://{ip}/stream?auth={API_KEY}"
    status_url = f"http://{ip}/status?auth={API_KEY}"

    async def poll_status():
        async with httpx.AsyncClient() as client:
            while not is_shutting_down:
                start_time = asyncio.get_event_loop().time()
                try:
                    resp = await client.get(status_url, timeout=3.0)
                    latency = int((asyncio.get_event_loop().time() - start_time) * 1000)
                    if resp.status_code == 200:
                        data = resp.json()
                        data["latency"] = latency
                        data["tcp"] = "OPEN"
                        status_cache[cam_id] = data
                    else:
                        add_log(f"[CAM {cam_id}] 認證失敗: HTTP {resp.status_code}", "ERROR")
                        status_cache[cam_id] = {"error": f"HTTP {resp.status_code}", "tcp": "OPEN"}
                except Exception:
                    status_cache[cam_id] = {"error": "Timeout", "tcp": "CLOSED"}
                await asyncio.sleep(5)

    async def poll_video():
        loop = asyncio.get_event_loop()
        reconnect_delay = 5
        frame_count = 0
        while not is_shutting_down:
            try:
                is_port_open, msg = await check_tcp_port(ip, 80)
                if not is_port_open:
                    add_log(f"[CAM {cam_id}] TCP 連線異常 ({msg})", "WARN")
                    await asyncio.sleep(min(reconnect_delay, 60))
                    reconnect_delay *= 2
                    continue

                reconnect_delay = 5

                async with httpx.AsyncClient() as client:
                    async with client.stream("GET", stream_url, timeout=None) as response:
                        if response.status_code != 200:
                            if response.status_code == 401:
                                add_log(f"[CAM {cam_id}] 認證失敗 (API Key 錯誤)", "ERROR")
                            else:
                                add_log(f"[CAM {cam_id}] 影像流回應異常 (HTTP {response.status_code})", "WARN")
                            await asyncio.sleep(10)
                            continue

                        add_log(f"[CAM {cam_id}] 串流建立成功", "INFO")

                        buffer = b""
                        async for chunk in response.aiter_bytes():
                            if is_shutting_down:
                                break
                            buffer += chunk

                            while True:
                                a = buffer.find(b'\xff\xd8')
                                b_end = buffer.find(b'\xff\xd9')
                                if a != -1 and b_end != -1 and b_end > a:
                                    jpg = buffer[a:b_end + 2]
                                    buffer = buffer[b_end + 2:]

                                    # 存原始 frame 備用
                                    frame_cache[cam_id] = jpg
                                    frame_count += 1

                                    # ── AI 推論（每 2 幀做一次，節省 CPU）──
                                    if frame_count % 2 == 0:
                                        try:
                                            annotated = await loop.run_in_executor(
                                                _thread_pool,
                                                _sync_ai_process,
                                                cam_id,
                                                jpg
                                            )
                                            ai_frame_cache[cam_id] = annotated
                                        except Exception as e:
                                            ai_frame_cache[cam_id] = jpg
                                    else:
                                        # 非推論幀直接沿用上一張 AI 結果
                                        if cam_id not in ai_frame_cache:
                                            ai_frame_cache[cam_id] = jpg

                                    # 定期清理 tracker
                                    if frame_count % 200 == 0:
                                        t = _trackers.get(cam_id)
                                        if t:
                                            t.cleanup_tracks()

                                    await asyncio.sleep(0.001)
                                else:
                                    break

                            if len(buffer) > 500000:
                                buffer = b""

            except asyncio.CancelledError:
                break
            except Exception as e:
                add_log(f"[CAM {cam_id}] 串流中斷: {str(e)[:40]}", "ERROR")
                await asyncio.sleep(5)

    await asyncio.gather(poll_status(), poll_video())


active_tasks = []
zc_instance  = None


def setup_trackers():
    """為每台攝影機建立獨立的 PedestrianTracker"""
    for cam in config["cameras"]:
        cam_id = cam["id"]
        alarm_url = f"http://{cam['ip']}/alarm"
        _trackers[cam_id] = PedestrianTracker(cam_id, alarm_url)
        ai_status_cache[cam_id] = {"person_count": 0, "alarm": 0, "ai_enabled": YOLO_AVAILABLE}


def start_all_fetchers():
    global active_tasks
    for task in active_tasks:
        task.cancel()
    active_tasks = []
    setup_trackers()
    for cam in config["cameras"]:
        task = asyncio.create_task(fetch_camera_data(cam["id"], cam["ip"]))
        active_tasks.append(task)


# ===================
# LIFESPAN
# ===================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global zc_instance, is_shutting_down
    # 預先載入 YOLO（在事件迴圈外的執行緒裡）
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(_thread_pool, get_yolo_model)
    start_all_fetchers()
    zc_instance = start_mdns_discovery()
    yield
    # Shutdown
    is_shutting_down = True
    add_log("系統正在關閉背景任務...", "WARN")
    if zc_instance:
        zc_instance.close()
    for task in active_tasks:
        task.cancel()
    if active_tasks:
        await asyncio.gather(*active_tasks, return_exceptions=True)
    _thread_pool.shutdown(wait=False)
    print("資源已釋放。")


# ===================
# FASTAPI APP
# ===================
app = FastAPI(
    title="One Step Ahead Dashboard",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))


# ===================
# ENDPOINTS
# ===================
@app.get("/favicon.ico")
async def favicon():
    return StreamingResponse(iter([]), status_code=204)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "cameras": config["cameras"],
            "API_KEY": API_KEY,
            "ai_enabled": YOLO_AVAILABLE,
        }
    )


@app.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="settings.html",
        context={"cameras": config["cameras"]}
    )


@app.post("/settings")
async def update_settings(data: SettingsUpdate):
    global config
    config["cameras"] = [cam.model_dump() for cam in data.cameras]
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    start_all_fetchers()
    return {"status": "ok"}


@app.get("/scan")
async def scan_devices():
    return discovered_devices


@app.post("/control/{cam_id}/{state}")
async def control_led(cam_id: int, state: str):
    cam = next((c for c in config["cameras"] if c["id"] == cam_id), None)
    if not cam:
        return {"status": "error"}
    target_url = f"http://{cam['ip']}/alarm?state={state}&auth={API_KEY}"
    async with httpx.AsyncClient() as client:
        try:
            await client.get(target_url, timeout=2.0)
            return {"status": "ok"}
        except:
            return {"status": "error"}


async def gen_frames(cam_id: int):
    """優先輸出 AI 標記畫面，降級回原始畫面"""
    while True:
        frame = ai_frame_cache.get(cam_id) or frame_cache.get(cam_id)
        if frame:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        await asyncio.sleep(1 / 20)


@app.get("/video_feed/{cam_id}")
async def video_feed(cam_id: int):
    return StreamingResponse(gen_frames(cam_id), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/api/logs")
async def get_logs():
    return list(system_logs)


@app.get("/api/net_info")
async def get_net_info():
    return {
        "local_ip": get_local_ip(),
        "hostname": socket.gethostname()
    }


@app.get("/api/wifi_status")
async def get_wifi_status():
    if os.name == 'nt':
        return {"mode": "Simulation", "ssid": "OneStepAhead_AP (MOCK)", "clients": 0}
    try:
        res = subprocess.check_output(["nmcli", "-t", "-f", "ACTIVE,SSID,MODE", "dev", "wifi"],
                                      text=True, stderr=subprocess.DEVNULL)
        for line in res.splitlines():
            if line.startswith("yes"):
                parts = line.split(":")
                return {"mode": parts[2], "ssid": parts[1], "clients": "N/A"}
        return {"mode": "Disconnected", "ssid": "None", "clients": 0}
    except:
        return {"mode": "Unknown", "ssid": "None", "clients": 0}


@app.get("/api/ping/{target}")
async def ping_target(target: str):
    success, msg = await check_tcp_port(target, 80, timeout=3.0)
    return {"status": "success" if success else "failed", "message": msg, "target": target}


@app.post("/api/restart")
async def restart_fetchers():
    add_log("系統正在重啟所有攝影機連線任務...")
    start_all_fetchers()
    return {"status": "restarting"}


@app.get("/api/ai_status")
async def get_ai_status():
    """回傳每台攝影機的 AI 辨識狀態（人數、警報）"""
    result = []
    for cam in config["cameras"]:
        cam_id = cam["id"]
        ai_info = ai_status_cache.get(cam_id, {"person_count": 0, "alarm": 0, "ai_enabled": YOLO_AVAILABLE})
        result.append({
            "id": cam_id,
            "name": cam["name"],
            **ai_info,
        })
    return result


@app.get("/api/system_resources")
async def get_system_resources():
    """回傳真實系統資源 (psutil)"""
    if not PSUTIL_AVAILABLE:
        return {"cpu": 0, "ram": 0, "temp": 0, "available": False}

    cpu  = psutil.cpu_percent(interval=None)
    ram  = psutil.virtual_memory().percent

    # CPU 溫度（Linux Pi 有；Windows 回傳 0）
    temp = 0
    try:
        temps = psutil.sensors_temperatures()
        if temps:
            for name, entries in temps.items():
                if entries:
                    temp = round(entries[0].current, 1)
                    break
    except (AttributeError, Exception):
        temp = 0

    return {
        "cpu": round(cpu, 1),
        "ram": round(ram, 1),
        "temp": temp,
        "available": True,
    }


@app.get("/status")
async def get_status():
    combined_status = []
    for cam in config["cameras"]:
        cam_id = cam["id"]
        online = cam_id in ai_frame_cache or cam_id in frame_cache
        data = status_cache.get(cam_id, {"rssi": 0, "uptime": 0, "sensor": 0.0, "alarm": 0, "tcp": "UNKNOWN", "latency": -1})
        ai_info = ai_status_cache.get(cam_id, {"alarm": 0})
        # AI 觸發的警報優先
        alarm = 1 if ai_info.get("alarm") == 1 else data.get("alarm", 0)
        combined_status.append({
            "id": cam_id,
            "ip": cam["ip"],
            "online": online,
            "rssi": data.get("rssi", 0),
            "uptime": data.get("uptime", 0),
            "sensor": data.get("sensor", 0.0),
            "alarm": alarm,
            "tcp": data.get("tcp", "UNKNOWN"),
            "latency": data.get("latency", -1),
        })
    return combined_status


if __name__ == "__main__":
    import uvicorn
    import signal

    port = config.get("server", {}).get("port", 8000)
    uv_config = uvicorn.Config(app, host="0.0.0.0", port=port, log_level="info")
    server = uvicorn.Server(uv_config)

    shutdown_calls = 0

    def handle_exit(sig, frame):
        global shutdown_calls, is_shutting_down
        shutdown_calls += 1
        is_shutting_down = True
        if shutdown_calls > 1:
            print("\n[FORCE] 強制退出進程...")
            os._exit(1)
        print("\n[INFO] 正在關閉系統資源 (再按一次 Ctrl+C 可強制退出)...")
        asyncio.create_task(server.shutdown())

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, handle_exit)

    try:
        asyncio.run(server.serve())
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        os._exit(0)
