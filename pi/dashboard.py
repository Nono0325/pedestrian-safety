import os
import json
import asyncio
import cv2
import httpx
import socket
from collections import deque
from datetime import datetime
from fastapi import FastAPI, Request, BackgroundTasks, Form
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List
from zeroconf import ServiceBrowser, Zeroconf, ServiceListener
from fastapi.middleware.cors import CORSMiddleware

# ===================
# CONFIG & INIT
# ===================
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")

def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

config = load_config()
API_KEY = config.get("security", {}).get("api_key", "")

app = FastAPI(title="先行一步 AI 監控儀表板")

# Tighten CORS: Allow local network and the server itself
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://127.0.0.1:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))

# Global State
frame_cache = {}
status_cache = {}
discovered_devices = [] # List of {"name": x, "ip": x}
system_logs = deque(maxlen=50)

def add_log(message: str, level: str = "INFO"):
    timestamp = datetime.now().strftime("%H:%M:%S")
    entry = {"time": timestamp, "msg": message, "level": level}
    system_logs.append(entry)
    print(f"[{timestamp}] {level}: {message}")

async def check_tcp_port(ip: str, port: int, timeout: float = 1.0) -> bool:
    """底層 TCP 探測，確認通訊埠是否開放"""
    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(ip, port), timeout=timeout
        )
        writer.close()
        await writer.wait_closed()
        return True
    except:
        return False

class CameraConfig(BaseModel):
    id: int
    name: str
    ip: str

class SettingsUpdate(BaseModel):
    cameras: List[CameraConfig]

# ===================
# MDNS DISCOVERY
# ===================
class MDNSListener(ServiceListener):
    def add_service(self, zc: Zeroconf, type_: str, name: str) -> None:
        info = zc.get_service_info(type_, name)
        if info:
            addresses = [".".join(map(str, addr)) for addr in info.addresses]
            if addresses:
                ip = addresses[0]
                # Only add if it looks like our safety cam
                if "esp32-safety" in name:
                    device = {"name": name.split(".")[0], "ip": ip}
                    if device not in discovered_devices:
                        discovered_devices.append(device)
                        print(f"Discovered via mDNS: {device}")

    def update_service(self, zc: Zeroconf, type_: str, name: str) -> None:
        pass

    def remove_service(self, zc: Zeroconf, type_: str, name: str) -> None:
        pass

def start_mdns_discovery():
    zeroconf = Zeroconf()
    listener = MDNSListener()
    browser = ServiceBrowser(zeroconf, "_http._tcp.local.", listener)
    return zeroconf

# ===================
# BACKGROUND TASKS
# ===================
async def fetch_camera_data(cam_id: int, ip: str):
    stream_url = f"http://{ip}/stream?auth={API_KEY}"
    status_url = f"http://{ip}/status?auth={API_KEY}"
    
    async def poll_status():
        async with httpx.AsyncClient() as client:
            while True:
                start_time = asyncio.get_event_loop().time()
                try:
                    resp = await client.get(status_url, timeout=2.0)
                    latency = int((asyncio.get_event_loop().time() - start_time) * 1000)
                    if resp.status_code == 200:
                        data = resp.json()
                        data["latency"] = latency
                        data["tcp"] = "OPEN"
                        status_cache[cam_id] = data
                    else:
                        add_log(f"🔥 [CAM {cam_id}] 認證失敗或回應異常: HTTP {resp.status_code}", "ERROR")
                        status_cache[cam_id] = {"error": f"HTTP {resp.status_code}", "tcp": "OPEN"}
                except Exception as e:
                    status_cache[cam_id] = {"error": "Timeout", "tcp": "CLOSED"}
                await asyncio.sleep(3)

    async def poll_video():
        first_frame = True
        while True:
            try:
                # 1. 第一步：先進行極細 TCP 探測
                is_port_open = await check_tcp_port(ip, 80)
                if not is_port_open:
                    add_log(f"⚠️ [CAM {cam_id}] TCP 握手失敗 (Port 80 被防火牆或路徑阻斷)", "WARN")
                    await asyncio.sleep(5)
                    continue

                # 2. 第二步：開啟串流
                cap = await asyncio.to_thread(cv2.VideoCapture, stream_url)
                await asyncio.to_thread(cap.set, cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
                
                if not await asyncio.to_thread(cap.isOpened):
                    # 深度探測原因
                    async with httpx.AsyncClient() as client:
                        try:
                            probe = await client.get(stream_url, timeout=3.0)
                            if probe.status_code == 401:
                                add_log(f"🔥 [CAM {cam_id}] 認證失敗：API Key 不正確！請檢查設定。", "ERROR")
                            elif probe.status_code == 503:
                                add_log(f"🔥 [CAM {cam_id}] 設備忙碌：可能有其他裝置正在監看。", "WARN")
                            else:
                                add_log(f"⚠️ [CAM {cam_id}] 影像流拒絕連線 (HTTP {probe.status_code})", "WARN")
                        except:
                            add_log(f"⚠️ [CAM {cam_id}] 影像流開啟超時 (可能是網路被過濾)", "WARN")
                    
                    await asyncio.sleep(5)
                    continue

                while await asyncio.to_thread(cap.isOpened):
                    ret, frame = await asyncio.to_thread(cap.read)
                    if not ret: 
                        add_log(f"❌ [CAM {cam_id}] 串流訊號中斷 (可能是網路不穩或 ESP 重啟)", "ERROR")
                        break
                    
                    if first_frame:
                        add_log(f"✅ [CAM {cam_id}] 影像解碼成功！通訊穩定。", "INFO")
                        first_frame = False

                    frame = cv2.resize(frame, (640, 480))
                    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    frame_cache[cam_id] = buffer.tobytes()
                    await asyncio.sleep(1/15)
                
                await asyncio.to_thread(cap.release)
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"🔥 [CAM {cam_id}] 發生錯誤: {e}")
                frame_cache.pop(cam_id, None)
                await asyncio.sleep(5)
    
    await asyncio.gather(poll_status(), poll_video())

active_tasks = []
zc_instance = None

def start_all_fetchers():
    global active_tasks
    for task in active_tasks: task.cancel()
    active_tasks = []
    for cam in config["cameras"]:
        task = asyncio.create_task(fetch_camera_data(cam["id"], cam["ip"]))
        active_tasks.append(task)

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    global zc_instance
    start_all_fetchers()
    zc_instance = start_mdns_discovery()
    yield
    # Shutdown
    add_log("🛑 系統正在關閉...", "WARN")
    if zc_instance:
        zc_instance.close()
    
    for task in active_tasks:
        task.cancel()
    
    if active_tasks:
        # 給予 1 秒的時間嘗試優雅關閉，否則直接進到主程序的 os._exit
        await asyncio.wait(active_tasks, timeout=1.0)
    
    print("✅ 資源已釋放。")

app = FastAPI(
    title="One Step Ahead Dashboard",
    lifespan=lifespan
)

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
        context={"cameras": config["cameras"]}
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
    """Returns the list of discovered devices."""
    return discovered_devices

@app.post("/control/{cam_id}/{state}")
async def control_led(cam_id: int, state: str):
    cam = next((c for c in config["cameras"] if c["id"] == cam_id), None)
    if not cam: return {"status": "error"}
    target_url = f"http://{cam['ip']}/alarm?state={state}&auth={API_KEY}"
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(target_url, timeout=2.0)
            return {"status": "ok"}
        except: return {"status": "error"}

async def gen_frames(cam_id: int):
    while True:
        if cam_id in frame_cache:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_cache[cam_id] + b'\r\n')
        await asyncio.sleep(1/20)

@app.get("/video_feed/{cam_id}")
async def video_feed(cam_id: int):
    return StreamingResponse(gen_frames(cam_id), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/api/logs")
async def get_logs():
    return list(system_logs)

@app.post("/api/restart")
async def restart_fetchers():
    add_log("🔄 系統正在重啟所有攝影機連線任務...")
    start_all_fetchers()
    return {"status": "restarting"}

@app.get("/status")
async def get_status():
    combined_status = []
    for cam in config["cameras"]:
        cam_id = cam["id"]
        online = cam_id in frame_cache
        data = status_cache.get(cam_id, {"rssi": 0, "uptime": 0, "sensor": 0.0, "alarm": 0, "tcp": "UNKNOWN", "latency": -1})
        combined_status.append({
            "id": cam_id, 
            "ip": cam["ip"],
            "online": online, 
            "rssi": data.get("rssi", 0), 
            "uptime": data.get("uptime", 0),
            "sensor": data.get("sensor", 0.0), 
            "alarm": data.get("alarm", 0),
            "tcp": data.get("tcp", "UNKNOWN"),
            "latency": data.get("latency", -1)
        })
    return combined_status

if __name__ == "__main__":
    import os
    try:
        uvicorn.run(app, host="0.0.0.0", port=config.get("server", {}).get("port", 8000), log_level="info")
    except KeyboardInterrupt:
        print("\n\n[FORCE] 偵測到使用者中斷 (Ctrl+C)，正在強制關閉系統...")
        # 直接結束進程，不再等候卡住的執行緒
        os._exit(0)
