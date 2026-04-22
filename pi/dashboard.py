import os
import json
import asyncio
import cv2
import httpx
from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

# ===================
# CONFIG & INIT
# ===================
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = json.load(f)

app = FastAPI(title="先行一步 AI 監控儀表板")
templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))

# Global Frame Cache
# camera_id -> b'jpeg_bytes'
frame_cache = {}

# ===================
# BACKGROUND TASKS
# ===================
async def fetch_frames(cam_id: int, stream_url: str):
    """Background task to fetch frames from an ESP32-CAM MJPEG stream."""
    print(f"Starting fetcher for Cam {cam_id} at {stream_url}")
    
    # We use OpenCV or HTTPX to pull the stream. 
    # For a relay, HTTPX is often more manageable for MJPEG headers.
    async with httpx.AsyncClient() as client:
        while True:
            try:
                # Direct relay or frame capture. 
                # Simplest for relay is using OpenCV in a thread if needed, 
                # but let's try a direct request if IP is MJPEG.
                async with client.stream("GET", stream_url, timeout=None) as response:
                    # Note: Parsings multipart MJPEG manually is tricky.
                    # As a prototype, we'll use OpenCV's backend for simplicity 
                    # in a separate thread-executor if needed.
                    cap = cv2.VideoCapture(stream_url)
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret: break
                        
                        # Downsample for dashboard efficiency
                        frame = cv2.resize(frame, (640, 480))
                        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                        frame_cache[cam_id] = buffer.tobytes()
                        
                        await asyncio.sleep(1/15) # Cap at 15 FPS
                    cap.release()
            except Exception as e:
                print(f"Error fetching Cam {cam_id}: {e}")
                await asyncio.sleep(5) # Retry delay

@app.on_event("startup")
async def startup_event():
    for cam in config["cameras"]:
        stream_url = f"http://{cam['ip']}/stream"
        asyncio.create_task(fetch_frames(cam["id"], stream_url))

# ===================
# ENDPOINTS
# ===================

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "cameras": config["cameras"]
    })

async def gen_frames(cam_id: int):
    while True:
        if cam_id in frame_cache:
            frame = frame_cache[cam_id]
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        else:
            # Placeholder "offline" frame logic could go here
            await asyncio.sleep(1)
        await asyncio.sleep(1/20)

@app.get("/video_feed/{cam_id}")
async def video_feed(cam_id: int):
    return StreamingResponse(gen_frames(cam_id), 
                             media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/status")
async def get_status():
    status_list = []
    for cam in config["cameras"]:
        online = cam["id"] in frame_cache
        status_list.append({
            "id": cam["id"],
            "name": cam["name"],
            "online": online,
            "ip": cam["ip"]
        })
    return status_list

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=config["server"]["port"])
