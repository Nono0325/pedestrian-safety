import numpy as np
import cv2

class HomographyTransformer:
    def __init__(self, src_pts, dst_pts):
        """
        src_pts: 4 points in image coordinates (pixels)
        dst_pts: 4 points in ground coordinates (meters)
        """
        self.H, _ = cv2.findHomography(np.float32(src_pts), np.float32(dst_pts))

    def transform(self, px, py):
        """Transform (x, y) pixels to (X, Y) meters"""
        point = np.float32([px, py, 1.0])
        transformed = np.dot(self.H, point)
        transformed /= transformed[2]
        return transformed[0], transformed[1]

def calculate_velocity(prev_pos, curr_pos, dt):
    """
    prev_pos, curr_pos: (X, Y) in meters
    dt: time difference in seconds
    """
    if prev_pos is None or curr_pos is None or dt <= 0:
        return 0, (0, 0)
    
    dx = curr_pos[0] - prev_pos[0]
    dy = curr_pos[1] - prev_pos[1]
    dist = np.sqrt(dx**2 + dy**2)
    velocity = dist / dt
    vector = (dx / dt, dy / dt)
    return velocity, vector

def is_approaching_curb(pos, vector, curb_line_y, threshold=0.2):
    """
    Simple rule: if velocity vector points towards the curb 
    and distance is decreasing.
    curb_line_y: Y coordinate of the curb in meters.
    """
    # Assuming vertical approach for simplicity
    dist = abs(pos[1] - curb_line_y)
    approaching = False
    if pos[1] < curb_line_y and vector[1] > threshold: # Moving from top to curb
        approaching = True
    elif pos[1] > curb_line_y and vector[1] < -threshold: # Moving from bottom to curb
        approaching = True
        
    return approaching, dist
