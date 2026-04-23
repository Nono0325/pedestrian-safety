import os
import json
import subprocess
import asyncio
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
is_shutting_down = False

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

async def check_tcp_port(host_port: str, default_port: int = 80, timeout: float = 3.0):
    """底層 TCP 探測，返回 (success, message)"""
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
    """獲取當前設備在區域網路中的 IP"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "127.0.0.1"

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
                        add_log(f"🔥 [CAM {cam_id}] 認證失敗: HTTP {resp.status_code}", "ERROR")
                        status_cache[cam_id] = {"error": f"HTTP {resp.status_code}", "tcp": "OPEN"}
                except Exception:
                    status_cache[cam_id] = {"error": "Timeout", "tcp": "CLOSED"}
                await asyncio.sleep(5) # 降低頻率從 3s -> 5s

    async def poll_video():
        reconnect_delay = 5
        while not is_shutting_down:
            try:
                # 1. 第一步：先進行極細 TCP 探測
                is_port_open, msg = await check_tcp_port(ip, 80)
                if not is_port_open:
                    add_log(f"⚠️ [CAM {cam_id}] TCP 連線異常 ({msg})", "WARN")
                    await asyncio.sleep(min(reconnect_delay, 60))
                    reconnect_delay *= 2
                    continue
                
                reconnect_delay = 5 # 重置
                
                # 2. 第二步：使用 httpx 手動解析 MJPEG
                async with httpx.AsyncClient() as client:
                    async with client.stream("GET", stream_url, timeout=None) as response:
                        if response.status_code != 200:
                            if response.status_code == 401:
                                add_log(f"🔥 [CAM {cam_id}] 認證失敗 (API Key 錯誤)", "ERROR")
                            else:
                                add_log(f"⚠️ [CAM {cam_id}] 影像流回應異常 (HTTP {response.status_code})", "WARN")
                            await asyncio.sleep(10)
                            continue
                        
                        add_log(f"✅ [CAM {cam_id}] 串流建立成功", "INFO")
                        
                        buffer = b""
                        async for chunk in response.aiter_bytes():
                            if is_shutting_down: break
                            buffer += chunk
                            
                            while True:
                                a = buffer.find(b'\xff\xd8') # JPEG Start
                                b = buffer.find(b'\xff\xd9') # JPEG End
                                if a != -1 and b != -1 and b > a:
                                    jpg = buffer[a:b+2]
                                    buffer = buffer[b+2:]
                                    
                                    # 解碼與縮放
                                    nparr = np.frombuffer(jpg, np.uint8)
                                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                                    if frame is not None:
                                        frame = cv2.resize(frame, (640, 480))
                                        _, buffer_jpg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                                        frame_cache[cam_id] = buffer_jpg.tobytes()
                                    
                                    await asyncio.sleep(0.001) # 給予系統喘息空間
                                else:
                                    break
                                    
                            if len(buffer) > 1000000: # 緩衝區保護
                                buffer = b""
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                add_log(f"🔥 [CAM {cam_id}] 串流中斷: {str(e)[:30]}", "ERROR")
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
    global zc_instance, is_shutting_down
    start_all_fetchers()
    zc_instance = start_mdns_discovery()
    yield
    # Shutdown
    is_shutting_down = True
    add_log("🛑 系統正在關閉背景任務...", "WARN")
    if zc_instance:
        zc_instance.close()
    
    for task in active_tasks:
        task.cancel()
    
    if active_tasks:
        # 使用 return_exceptions=True 避免在 gather 時拋出 CancelledError 導致中斷
        await asyncio.gather(*active_tasks, return_exceptions=True)
    
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

@app.get("/api/net_info")
async def get_net_info():
    """返回 Pi 的當前網路狀態"""
    return {
        "local_ip": get_local_ip(),
        "hostname": socket.gethostname()
    }

@app.get("/api/wifi_status")
async def get_wifi_status():
    """獲取 Pi 的 Wi-Fi AP 狀態 (僅限 Linux)"""
    if os.name == 'nt':
        return {"mode": "Simulation", "ssid": "OneStepAhead_AP (MOCK)", "clients": 0}
        
    try:
        # 獲取當前啟用的 SSID
        res = subprocess.check_output(["nmcli", "-t", "-f", "ACTIVE,SSID,MODE", "dev", "wifi"], text=True, stderr=subprocess.DEVNULL)
        for line in res.splitlines():
            if line.startswith("yes"):
                parts = line.split(":")
                return {"mode": parts[2], "ssid": parts[1], "clients": "N/A"}
        return {"mode": "Disconnected", "ssid": "None", "clients": 0}
    except:
        return {"mode": "Unknown", "ssid": "None", "clients": 0}

@app.get("/api/ping/{target}")
async def ping_target(target: str):
    """測試到目標 IP 的 TCP 連通性"""
    success, msg = await check_tcp_port(target, 80, timeout=3.0)
    return {"status": "success" if success else "failed", "message": msg, "target": target}

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
    import uvicorn
    import signal

    port = config.get("server", {}).get("port", 8000)
    
    # 建立 Uvicorn 配置
    uv_config = uvicorn.Config(app, host="0.0.0.0", port=port, log_level="info")
    server = uvicorn.Server(uv_config)

    # 覆寫信號處理，避免顯示雜亂的 Traceback
    shutdown_calls = 0
    def handle_exit(sig, frame):
        global shutdown_calls, is_shutting_down
        shutdown_calls += 1
        is_shutting_down = True
        if shutdown_calls > 1:
            print("\n[FORCE] 偵測到多次中斷，立即強制退出進程...")
            os._exit(1)
        
        print("\n\n[INFO] 正在優雅關閉系統資源 (再按一次 Ctrl+C 可強制退出)...")
        # 建立一個 Task 來關閉 Server
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(server.shutdown())
            else:
                os._exit(0)
        except:
            os._exit(0)

    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)

    try:
        asyncio.run(server.serve())
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        print("✅ 系統已結束。")
        os._exit(0)
