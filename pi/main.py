import cv2
import time
import numpy as np
from ultralytics import YOLO
from utils import HomographyTransformer, calculate_velocity, is_approaching_curb

# ===================
# CONFIGURATION
# ===================
STREAM_URL = "http://ESP32_IP_ADDRESS/stream"
MODEL_PATH = "yolov8n.pt" # Or custom ONNX model
LED_GPIO_PIN = 18

# Homography Calibration (Example: VGA 640x480 to 5x5m ground)
# In real scenario, use 4 corners of a known rectangle on floor
SRC_PTS = [[0, 480], [640, 480], [640, 0], [0, 0]]
DST_PTS = [[0, 5],   [5, 5],     [5, 0],   [0, 0]]

# Thresholds
INTENT_THRESHOLD_DIST = 1.0  # meters
INTENT_THRESHOLD_VEL  = 0.5  # m/s
INTENT_MIN_FRAMES     = 5    # Smoothing

class PedestrianTracker:
    def __init__(self):
        self.tracks = {} # track_id: {'last_pos': (X,Y), 'last_time': T, 'intent_count': 0}

    def update(self, track_id, pos_ground):
        now = time.time()
        if track_id not in self.tracks:
            self.tracks[track_id] = {'last_pos': pos_ground, 'last_time': now, 'intent_count': 0}
            return False
        
        info = self.tracks[track_id]
        dt = now - info['last_time']
        velocity, vector = calculate_velocity(info['last_pos'], pos_ground, dt)
        
        # Intent Logic
        curb_y = 2.5 # Arbitrary curb line at Y=2.5m
        approaching, dist = is_approaching_curb(pos_ground, vector, curb_y)
        
        is_intent = False
        if approaching and dist < INTENT_THRESHOLD_DIST and velocity > INTENT_THRESHOLD_VEL:
            info['intent_count'] += 1
        else:
            info['intent_count'] = max(0, info['intent_count'] - 1)
        
        if info['intent_count'] >= INTENT_MIN_FRAMES:
            is_intent = True
            
        # Update state
        info['last_pos'] = pos_ground
        info['last_time'] = now
        return is_intent

def main():
    # Initialize components
    model = YOLO(MODEL_PATH)
    transformer = HomographyTransformer(SRC_PTS, DST_PTS)
    tracker_logic = PedestrianTracker()
    
    # Try opening stream
    cap = cv2.VideoCapture(STREAM_URL)
    
    # GPIO Init (Mock if not on Pi)
    try:
        import lgpio
        h = lgpio.gpiochip_open(0)
        lgpio.gpio_claim_output(h, LED_GPIO_PIN)
    except:
        print("GPIO not available, running in simulation mode")
        h = None

    print("System Running...")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Stream lost, retrying...")
            time.sleep(1)
            cap = cv2.VideoCapture(STREAM_URL)
            continue

        # Inference with ByteTrack
        results = model.track(frame, persist=True, classes=[0], tracker="bytetrack.yaml", verbose=False)
        
        any_intent = False
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            
            for box, track_id in zip(boxes, ids):
                # Foot point (bottom center of bbox)
                foot_x = (box[0] + box[2]) / 2
                foot_y = box[3]
                
                # Ground coordinate
                gx, gy = transformer.transform(foot_x, foot_y)
                
                # Update logic
                if tracker_logic.update(track_id, (gx, gy)):
                    any_intent = True
                
                # Draw for debug
                cv2.circle(frame, (int(foot_x), int(foot_y)), 5, (0, 255, 0), -1)
                cv2.putText(frame, f"ID:{track_id} G:({gx:.1f},{gy:.1f})", 
                            (int(box[0]), int(box[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)

        # Output Control
        if any_intent:
            print(">>> INTENT DETECTED! LED ON")
            if h: lgpio.gpio_write(h, LED_GPIO_PIN, 1)
            cv2.putText(frame, "WARNING: PEDESTRIAN CROSSING", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        else:
            if h: lgpio.gpio_write(h, LED_GPIO_PIN, 0)

        # Display
        cv2.imshow("先行一步: AEI Pedestrian Intent Demo", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    if h: lgpio.gpiochip_close(h)

if __name__ == "__main__":
    main()
