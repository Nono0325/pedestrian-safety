import cv2
import time
import requests
import json
import numpy as np
from ultralytics import YOLO
from utils import HomographyTransformer, calculate_velocity, is_approaching_curb

# ===================
# CONFIGURATION
# ===================
# 預設指向第一個攝影機或本地模擬器位址 (127.0.0.1:8080)
# Load configuration
with open("pi/config.json", "r") as f:
    config_data = json.load(f)
    API_KEY = config_data.get("security", {}).get("api_key", "")
    # Use the first configured camera IP, or fallback to mock address
    DEFAULT_TARGET_IP = config_data["cameras"][0]["ip"] if config_data.get("cameras") else "127.0.0.1:8080"

# Derived URLs
STREAM_URL = f"http://{DEFAULT_TARGET_IP}/stream?auth={API_KEY}"
ALARM_URL = f"http://{DEFAULT_TARGET_IP}/alarm"
MODEL_PATH = "yolov8n.pt"

# Homography Calibration (場域校正坐標)
SRC_PTS = [[0, 480], [640, 480], [640, 0], [0, 0]]
DST_PTS = [[0, 5],   [5, 5],     [5, 0],   [0, 0]]

# Thresholds (辨識門檻)
INTENT_THRESHOLD_DIST = 1.0 # 距離門檻 (米)
INTENT_THRESHOLD_VEL  = 0.5 # 速度門檻 (m/s)
INTENT_MIN_FRAMES     = 5   # 最少持續偵測幀數

# Derived URLs
STREAM_URL = f"http://{DEFAULT_TARGET_IP}/stream?auth={API_KEY}"
ALARM_URL = f"http://{DEFAULT_TARGET_IP}/alarm"

class PedestrianTracker:
    def __init__(self):
        self.tracks = {} 
        self.last_alarm_time = 0

    def trigger_alarm(self):
        """Send network command to ESP32 to light up the crosswalk LED."""
        now = time.time()
        if now - self.last_alarm_time < ALARM_DURATION:
            return # Already active
            
        print(">>> AI TRIGGER: ACTivating Crosswalk Alarm")
        try:
            requests.get(f"{ALARM_URL}?state=on&auth={API_KEY}", timeout=1.0)
            self.last_alarm_time = now
            # In a production environment, we'd use a timer to send 'off' 
            # or handle it on the ESP32 side. For MVP, we send 'off' later.
        except:
            print("Failed to reach ESP32 for alarm")

    def reset_alarm_if_needed(self):
        now = time.time()
        if self.last_alarm_time > 0 and now - self.last_alarm_time >= ALARM_DURATION:
            print(">>> AI TRIGGER: DEactivating Crosswalk Alarm")
            try:
                requests.get(f"{ALARM_URL}?state=off&auth={API_KEY}", timeout=1.0)
                self.last_alarm_time = 0
            except:
                pass

    def update(self, track_id, pos_ground):
        now = time.time()
        if track_id not in self.tracks:
            self.tracks[track_id] = {'last_pos': pos_ground, 'last_time': now, 'intent_count': 0}
            return False
        
        info = self.tracks[track_id]
        dt = now - info['last_time']
        velocity, vector = calculate_velocity(info['last_pos'], pos_ground, dt)
        
        curb_y = 2.5
        approaching, dist = is_approaching_curb(pos_ground, vector, curb_y)
        
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
    model = YOLO(MODEL_PATH)
    transformer = HomographyTransformer(SRC_PTS, DST_PTS)
    tracker_logic = PedestrianTracker()
    
    cap = cv2.VideoCapture(STREAM_URL)
    print("AI Integrated System Running...")

    while True:
        tracker_logic.reset_alarm_if_needed()
        ret, frame = cap.read()
        if not ret:
            time.sleep(1); cap = cv2.VideoCapture(STREAM_URL); continue

        results = model.track(frame, persist=True, classes=[0], tracker="bytetrack.yaml", verbose=False)
        
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            
            for box, track_id in zip(boxes, ids):
                foot_x, foot_y = (box[0] + box[2]) / 2, box[3]
                gx, gy = transformer.transform(foot_x, foot_y)
                tracker_logic.update(track_id, (gx, gy))
                
                cv2.circle(frame, (int(foot_x), int(foot_y)), 5, (0, 255, 0), -1)

        cv2.imshow("先行一步: Unified AI & Alarm Demo", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
