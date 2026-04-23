import cv2
import time
import requests
import json
import threading
import numpy as np
from ultralytics import YOLO
from utils import HomographyTransformer, calculate_velocity, is_approaching_curb

# ===================
# CONFIGURATION
# ===================
# Load configuration
try:
    with open("pi/config.json", "r") as f:
        config_data = json.load(f)
        API_KEY = config_data.get("security", {}).get("api_key", "")
        DEFAULT_TARGET_IP = config_data["cameras"][0]["ip"] if config_data.get("cameras") else "127.0.0.1:8080"
except Exception as e:
    print(f"⚠️ [CONFIG] 無法載入設定檔，使用預設值: {e}")
    API_KEY = "nono_safety_sec_2026"
    DEFAULT_TARGET_IP = "127.0.0.1:8080"

# Derived URLs
STREAM_URL = f"http://{DEFAULT_TARGET_IP}/stream?auth={API_KEY}"
ALARM_URL = f"http://{DEFAULT_TARGET_IP}/alarm"
MODEL_PATH = "yolov8n.pt"

# Homography Calibration (場域校正坐標)
# 建議：在現場找一個 1m x 1m 的地磚區域，記錄其在影像中的四個角點作為 SRC_PTS
SRC_PTS = [[0, 480], [640, 480], [640, 0], [0, 0]]
DST_PTS = [[0, 5],   [5, 5],     [5, 0],   [0, 0]]

# Thresholds (辨識門檻)
INTENT_THRESHOLD_DIST = 1.2 # 距離門檻 (米)
INTENT_THRESHOLD_VEL  = 0.4 # 速度門檻 (m/s)
INTENT_MIN_FRAMES     = 4   # 最少持續偵測幀數
ALARM_DURATION        = 5   # 警報持續時間 (秒)
CURB_Y_THRESHOLD      = 2.5 # 馬路邊緣界線 (Y 坐標)

class PedestrianTracker:
    def __init__(self):
        self.tracks = {} 
        self.last_alarm_time = 0
        self._alarm_lock = False # 確保執行緒不會重複啟動

    def _send_alarm_request(self, state):
        """背景發送請求，避免阻塞 AI 辨識迴圈"""
        try:
            url = f"{ALARM_URL}?state={state}&auth={API_KEY}"
            requests.get(url, timeout=1.5)
        except Exception as e:
            print(f"⚠️ [ALARM] 請求發送失敗: {e}")
        finally:
            self._alarm_lock = False

    def trigger_alarm(self):
        """觸發警示燈 (非同步)"""
        now = time.time()
        # 檢查冷卻時間與執行緒鎖
        if self._alarm_lock or (self.last_alarm_time > 0 and now - self.last_alarm_time < ALARM_DURATION):
            return 
            
        print(">>> AI TRIGGER: ACTivating Crosswalk Alarm")
        self._alarm_lock = True
        self.last_alarm_time = now
        # 開啟獨立執行緒發送請求，防止畫面凍結
        threading.Thread(target=self._send_alarm_request, args=("on",), daemon=True).start()

    def reset_alarm_if_needed(self):
        """重置警示燈 (非同步)"""
        now = time.time()
        if self.last_alarm_time > 0 and now - self.last_alarm_time >= ALARM_DURATION:
            print(">>> AI TRIGGER: DEactivating Crosswalk Alarm")
            self.last_alarm_time = 0
            threading.Thread(target=self._send_alarm_request, args=("off",), daemon=True).start()

    def cleanup_tracks(self):
        """定期清理過期的追蹤資料，避免記憶體溢位"""
        now = time.time()
        expired = [tid for tid, info in self.tracks.items() if now - info['last_time'] > 60]
        for tid in expired:
            del self.tracks[tid]
        if expired:
            print(f"🧹 [GC] 已清理 {len(expired)} 個過期追蹤 ID")

    def update(self, track_id, pos_ground):
        now = time.time()
        if track_id not in self.tracks:
            self.tracks[track_id] = {'last_pos': pos_ground, 'last_time': now, 'intent_count': 0}
            return False
        
        info = self.tracks[track_id]
        dt = now - info['last_time']
        if dt <= 0: return False
        
        velocity, vector = calculate_velocity(info['last_pos'], pos_ground, dt)
        
        approaching, dist = is_approaching_curb(pos_ground, vector, CURB_Y_THRESHOLD)
        
        is_intent = False
        if approaching and dist < INTENT_THRESHOLD_DIST and velocity > INTENT_THRESHOLD_VEL:
            info['intent_count'] += 1
        else:
            info['intent_count'] = max(0, info['intent_count'] - 1)
        
        if info['intent_count'] >= INTENT_MIN_FRAMES:
            is_intent = True
            self.trigger_alarm()
            
        info['last_pos'] = pos_ground
        info['last_time'] = now
        return is_intent

def main():
    print("🚀 [INIT] 正在載入 YOLOv8 模型...")
    model = YOLO(MODEL_PATH)
    transformer = HomographyTransformer(SRC_PTS, DST_PTS)
    tracker_logic = PedestrianTracker()
    frame_count = 0
    
    # 診斷連線
    print(f"🔍 [DIAG] 正在測試連線: {DEFAULT_TARGET_IP}...")
    try:
        resp = requests.get(f"http://{DEFAULT_TARGET_IP}/status?auth={API_KEY}", timeout=3)
        if resp.status_code == 200:
            print("✅ [OK] ESP32 連線正常")
    except:
        print("⚠️ [WARN] 無法連接 ESP32，請檢查網路。系統將繼續嘗試啟動串流...")

    cap = cv2.VideoCapture(STREAM_URL)
    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
    
    print(f"📹 [LIVE] 正在連接影像流: {STREAM_URL}")

    while True:
        tracker_logic.reset_alarm_if_needed()
        ret, frame = cap.read()
        if not ret:
            print("❌ [ERR] 影像流中斷，嘗試重連...")
            time.sleep(2)
            cap = cv2.VideoCapture(STREAM_URL)
            continue

        frame_count += 1
        # 定期清理記憶體
        if frame_count % 100 == 0:
            tracker_logic.cleanup_tracks()

        # 推論 (imgsz=320 以提升 Pi 5 效能)
        results = model.track(frame, persist=True, classes=[0], tracker="bytetrack.yaml", verbose=False, imgsz=320)
        
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            
            for box, track_id in zip(boxes, ids):
                # 以腳部位置作為地面坐標參考點
                foot_x, foot_y = (box[0] + box[2]) / 2, box[3]
                gx, gy = transformer.transform(foot_x, foot_y)
                is_danger = tracker_logic.update(track_id, (gx, gy))
                
                # 繪製視覺反饋
                color = (0, 0, 255) if is_danger else (0, 255, 0)
                cv2.circle(frame, (int(foot_x), int(foot_y)), 6, color, -1)
                cv2.putText(frame, f"ID:{track_id} {gy:.1f}m", (int(box[0]), int(box[1]-5)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imshow("先行一步: AI 核心處理終端", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
