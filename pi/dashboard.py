import os
import json
import asyncio
import cv2
import httpx
import socket
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
                try:
                    resp = await client.get(status_url, timeout=2.0)
                    if resp.status_code == 200:
                        status_cache[cam_id] = resp.json()
                except:
                    status_cache.pop(cam_id, None)
                await asyncio.sleep(3)

    async def poll_video():
        first_frame = True
        while True:
            try:
                # 使用執行緒開啟捕捉
                cap = await asyncio.to_thread(cv2.VideoCapture, stream_url)
                await asyncio.to_thread(cap.set, cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
                
                if not await asyncio.to_thread(cap.isOpened):
                    print(f"⚠️ [CAM {cam_id}] 無法連上串流，將在 5 秒後重試...")
                    await asyncio.sleep(5)
                    continue

                while await asyncio.to_thread(cap.isOpened):
                    ret, frame = await asyncio.to_thread(cap.read)
                    if not ret: 
                        print(f"❌ [CAM {cam_id}] 影像讀取中斷...")
                        break
                    
                    if first_frame:
                        print(f"✅ [CAM {cam_id}] 成功抓取到首幀影像！路徑: {ip}")
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
    # 徹底清除後台任務，確保 Ctrl+C 能瞬間關閉
    if zc_instance:
        zc_instance.close()
    for task in active_tasks:
        task.cancel()
    print("\n系統正在安全關閉...")

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

@app.get("/status")
async def get_status():
    combined_status = []
    for cam in config["cameras"]:
        cam_id = cam["id"]
        online = cam_id in frame_cache
        data = status_cache.get(cam_id, {"rssi": 0, "uptime": 0, "sensor": 0.0, "alarm": 0})
        combined_status.append({
            "id": cam_id, "online": online, "rssi": data["rssi"], "uptime": data["uptime"],
            "sensor": data["sensor"], "alarm": data["alarm"]
        })
    return combined_status

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=config["server"]["port"])
