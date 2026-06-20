import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time

class SimpleLandmark:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

class VideoCamera(object):
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        
        model_path = "hand_landmarker.task"
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            running_mode=vision.RunningMode.VIDEO
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        
        self.prev_landmarks = None
        self.alpha = 0.5 
        
        # Landmark IDs
        self.tip_ids = [4, 8, 12, 16, 20]
        self.pip_ids = [3, 6, 10, 14, 18]
        self.mcp_ids = [2, 5, 9, 13, 17]
        self.finger_names = ["Thumb", "Index", "Middle", "Ring", "Little"]

        # Shared state for API
        self.current_status = {
            "hand_detected": False,
            "hand_label": "None",
            "finger_count": 0,
            "fingers": []
        }

    def __del__(self):
        self.cap.release()
        self.detector.close()
        
    def calculate_angle(self, a, b, c):
        a = np.array([a.x, a.y, a.z])
        b = np.array([b.x, b.y, b.z])
        c = np.array([c.x, c.y, c.z])
    
        ba = a - b
        bc = c - b
    
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    
        angle = np.degrees(np.arccos(cosine_angle))
        return angle

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
            
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    
        result = self.detector.detect_for_video(mp_image, int(time.time() * 1000))
        
        status_update = {
            "hand_detected": False,
            "hand_label": "None",
            "finger_count": 0,
            "fingers": []
        }

        if result.hand_landmarks:
            status_update["hand_detected"] = True
            raw_landmarks = result.hand_landmarks[0]
    
            if self.prev_landmarks is None:
                self.prev_landmarks = [SimpleLandmark(l.x, l.y, l.z) for l in raw_landmarks]
                hand_landmarks = raw_landmarks
            else:
                smoothed_landmarks = []
                for i, curr in enumerate(raw_landmarks):
                    prev = self.prev_landmarks[i]
                    new_x = self.alpha * curr.x + (1 - self.alpha) * prev.x
                    new_y = self.alpha * curr.y + (1 - self.alpha) * prev.y
                    new_z = self.alpha * curr.z + (1 - self.alpha) * prev.z
                    smoothed_landmarks.append(SimpleLandmark(new_x, new_y, new_z))
                
                self.prev_landmarks = smoothed_landmarks
                hand_landmarks = smoothed_landmarks
    
            hand_label = result.handedness[0][0].category_name
            status_update["hand_label"] = hand_label
        
            h, w, _ = frame.shape
            finger_count = 0
    
            for i in range(5):
                if i == 0:
                    tip = hand_landmarks[4] 
                    pip = hand_landmarks[3] 
                    mcp = hand_landmarks[2] 
                else:
                    tip = hand_landmarks[self.tip_ids[i]]
                    pip = hand_landmarks[self.pip_ids[i]]
                    mcp = hand_landmarks[self.mcp_ids[i]]
    
                # Visualize (draw only skeleton, no text)
                tip_px = (int(tip.x * w), int(tip.y * h))
                pip_px = (int(pip.x * w), int(pip.y * h))
                mcp_px = (int(mcp.x * w), int(mcp.y * h))
                
                cv2.line(frame, tip_px, pip_px, (0, 255, 0), 2)
                cv2.line(frame, pip_px, mcp_px, (0, 255, 0), 2)
                cv2.circle(frame, tip_px, 5, (0, 255, 255), -1)
                cv2.circle(frame, pip_px, 5, (0, 255, 255), -1)
                cv2.circle(frame, mcp_px, 5, (0, 255, 255), -1)
    
                angle = self.calculate_angle(tip, pip, mcp)
    
                if i == 0:
                    fold_percent = np.interp(angle, [110, 170], [100, 0])
                else:
                    fold_percent = np.interp(angle, [50, 170], [100, 0])
    
                fold_percent = int(fold_percent)
    
                if fold_percent < 40:
                    state = "Extended"
                    finger_count += 1
                else:
                    state = "Folded"
                
                status_update["fingers"].append({
                    "name": self.finger_names[i],
                    "fold_percent": fold_percent,
                    "state": state
                })
            
            status_update["finger_count"] = finger_count

        else:
            self.prev_landmarks = None

        self.current_status = status_update
        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()
