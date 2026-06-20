import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
from collections import deque, Counter

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
            num_hands=2,
            running_mode=vision.RunningMode.VIDEO
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        
        self.prev_landmarks = {} # Dict to store prev landmarks by hand index
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
        
        # Smoothing buffer for sign detection
        # Key: Hand Index, Value: Deque
        self.sign_history = {}
        
        # Sentence Construction
        self.sentence = []
        self.last_word_time = 0
        self.current_sentence_str = ""

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

    def detect_sign(self, finger_states):
        # finger_states: [Thumb, Index, Middle, Ring, Little]
        # 1 = Extended, 0 = Folded
        
        # Dictionary mapping tuple of finger states (Thumb, Index, Middle, Ring, Little) to Sign Name
        # 1 = Extended, 0 = Folded
        gestures = {
            (0, 0, 0, 0, 0): "No / Stop",        # Fist
            (1, 1, 1, 1, 1): "Hello / Help",     # Open Hand
            (0, 1, 1, 0, 0): "Peace / Victory",
            (1, 0, 0, 0, 0): "Yes / Good",       # Thumbs Up
            (0, 1, 0, 0, 0): "You / Look",       # Pointing
            (1, 1, 0, 0, 1): "I Love You",
            (1, 1, 0, 0, 0): "L / Later",
            (0, 1, 0, 0, 1): "Rock / Fun",
            (0, 0, 0, 0, 1): "Toilet / Excuse Me", 
            (0, 0, 1, 0, 0): "Middle Finger",    
            (1, 0, 0, 0, 1): "Phone / Call Me",
            (0, 1, 1, 1, 0): "Water / Three",    
            (0, 1, 1, 1, 1): "Four",
            (1, 1, 1, 0, 0): "Food / Eat",     
            (0, 0, 1, 1, 1): "Okay / Perfect",             
            (1, 0, 0, 1, 0): "Rock'n'Roll",
            (1, 1, 1, 1, 0): "Wait",             # 4 fingers (no pinky)
            (1, 0, 0, 1, 1): "Rare / Special",   # Thumb + Ring + Pinky
            (0, 1, 0, 1, 1): "Rest / Break",     # Index + Ring + Pinky
            (0, 1, 1, 0, 1): "Shock / Surprise", # Index + Mid + Pinky
            (1, 1, 0, 0, 1): "Family",           # ILY shape used for Family context here
            (1, 0, 1, 0, 1): "Medicine / Sick",  # Thumb + Mid + Pinky
            (0, 1, 0, 1, 0): "Unknown",
            (1, 1, 1, 0, 1): "Unknown",
        }
        
        return gestures.get(tuple(finger_states), "Unknown")

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
            
            # Reset fingers list in status
            status_update["fingers"] = []
            
            current_frame_signs = []

            for idx, (raw_landmarks, handedness) in enumerate(zip(result.hand_landmarks, result.handedness)):
                hand_label = handedness[0].category_name
                
                # Smoothing
                if idx not in self.prev_landmarks:
                    self.prev_landmarks[idx] = [SimpleLandmark(l.x, l.y, l.z) for l in raw_landmarks]
                    hand_landmarks = raw_landmarks
                else:
                    smoothed_landmarks = []
                    for i, curr in enumerate(raw_landmarks):
                        prev = self.prev_landmarks[idx][i]
                        new_x = self.alpha * curr.x + (1 - self.alpha) * prev.x
                        new_y = self.alpha * curr.y + (1 - self.alpha) * prev.y
                        new_z = self.alpha * curr.z + (1 - self.alpha) * prev.z
                        smoothed_landmarks.append(SimpleLandmark(new_x, new_y, new_z))
                    
                    self.prev_landmarks[idx] = smoothed_landmarks
                    hand_landmarks = smoothed_landmarks
                
                h, w, _ = frame.shape
                
                finger_states = []
                
                # Draw Landmarks & Calculate Angles
                for i in range(5):
                    if i == 0:
                        tip = hand_landmarks[4] 
                        pip = hand_landmarks[3] 
                        mcp = hand_landmarks[2] 
                    else:
                        tip = hand_landmarks[self.tip_ids[i]]
                        pip = hand_landmarks[self.pip_ids[i]]
                        mcp = hand_landmarks[self.mcp_ids[i]]
        
                    # Visualize
                    tip_px = (int(tip.x * w), int(tip.y * h))
                    pip_px = (int(pip.x * w), int(pip.y * h))
                    mcp_px = (int(mcp.x * w), int(mcp.y * h))
                    
                    color = (0, 255, 0) if hand_label == "Right" else (255, 0, 0) # Green for Right, Blue for Left
                    
                    cv2.line(frame, tip_px, pip_px, color, 2)
                    cv2.line(frame, pip_px, mcp_px, color, 2)
                    cv2.circle(frame, tip_px, 5, (0, 255, 255), -1)
        
                    angle = self.calculate_angle(tip, pip, mcp)
        
                    if i == 0:
                        fold_percent = np.interp(angle, [120, 160], [100, 0])
                    else:
                        fold_percent = np.interp(angle, [60, 160], [100, 0])
        
                    fold_percent = int(fold_percent)
        
                    if fold_percent < 40:
                        state = "Extended"
                        finger_states.append(1) 
                    else:
                        state = "Folded"
                        finger_states.append(0)
                
                # Detect Sign
                raw_sign = self.detect_sign(finger_states)
                
                # History per hand
                if idx not in self.sign_history:
                    self.sign_history[idx] = deque(maxlen=10)
                
                self.sign_history[idx].append(raw_sign)
                
                most_common = Counter(self.sign_history[idx]).most_common(1)
                detected_sign = most_common[0][0] if most_common else "Unknown"
                
                if detected_sign != "Unknown":
                     current_frame_signs.append(detected_sign)

                # Display Sign Label near hand
                # Calculate centroid of hand for label position
                cx = int(hand_landmarks[9].x * w)
                cy = int(hand_landmarks[9].y * h)
                
                cv2.putText(frame, f"{hand_label}: {detected_sign}", (cx - 50, cy - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

            # Sentence Logic
            current_time = time.time()
            if current_frame_signs:
                # If multiple signs detected, pick unique ones or just the first stable one
                # For simplicity, add unique signs to sentence if enough time passed
                unique_signs = sorted(list(set(current_frame_signs)))
                
                for sign in unique_signs:
                    if sign == "No / Stop": # Use Stop to clear
                         self.sentence = []
                         self.current_sentence_str = ""
                    elif current_time - self.last_word_time > 2.0: # 2 seconds delay between words
                        if not self.sentence or self.sentence[-1] != sign:
                            self.sentence.append(sign)
                            self.last_word_time = current_time
                            # Keep sentence length manageable
                            if len(self.sentence) > 5:
                                self.sentence.pop(0)
                            self.current_sentence_str = " ".join(self.sentence)
            
            # Display Sentence Bar
            # Draw semi-transparent background at bottom
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, h - 60), (w, h), (0, 0, 0), -1)
            alpha = 0.7
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
            
            cv2.putText(frame, f"Says: {self.current_sentence_str}", (20, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Help Text
            cv2.putText(frame, "Hold 2s to add word. 'Stop' gesture to clear.", (w - 450, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

        else:
            self.prev_landmarks = {} # Reset if no hands

        self.current_status = status_update
        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()
