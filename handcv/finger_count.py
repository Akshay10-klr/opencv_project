import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time

model_path = "hand_landmarker.task"

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    running_mode=vision.RunningMode.VIDEO
)

detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

# Function to calculate angle between three points
def calculate_angle(a, b, c):
    a = np.array([a.x, a.y, a.z])
    b = np.array([b.x, b.y, b.z])
    c = np.array([c.x, c.y, c.z])

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)

    angle = np.degrees(np.arccos(cosine_angle))

    return angle

class SimpleLandmark:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

prev_landmarks = None
alpha = 0.5  # Smoothing factor (0.0 - 1.0). Lower = smoother but more lag.

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

    result = detector.detect_for_video(mp_image, int(time.time() * 1000))

    finger_count = 0

    if result.hand_landmarks:
        raw_landmarks = result.hand_landmarks[0]

        if prev_landmarks is None:
            # Initialize previous landmarks
            prev_landmarks = [SimpleLandmark(l.x, l.y, l.z) for l in raw_landmarks]
            hand_landmarks = raw_landmarks
        else:
            # Apply exponential smoothing
            smoothed_landmarks = []
            for i, curr in enumerate(raw_landmarks):
                prev = prev_landmarks[i]
                new_x = alpha * curr.x + (1 - alpha) * prev.x
                new_y = alpha * curr.y + (1 - alpha) * prev.y
                new_z = alpha * curr.z + (1 - alpha) * prev.z
                smoothed_landmarks.append(SimpleLandmark(new_x, new_y, new_z))
            
            prev_landmarks = smoothed_landmarks
            hand_landmarks = smoothed_landmarks

        hand_label = result.handedness[0][0].category_name

        # Landmark IDs
        tip_ids = [4, 8, 12, 16, 20]
        pip_ids = [3, 6, 10, 14, 18]
        mcp_ids = [2, 5, 9, 13, 17]
        finger_names = ["Thumb", "Index", "Middle", "Ring", "Little"]
    
        h, w, _ = frame.shape

        for i in range(5):
            if i == 0:
                # Thumb: Use Tip, IP, MCP (Landmarks 4, 3, 2)
                tip = hand_landmarks[4] # Tip
                pip = hand_landmarks[3] # IP
                mcp = hand_landmarks[2] # MCP
            else:
                # Fingers: Use Tip, PIP, MCP
                tip = hand_landmarks[tip_ids[i]]
                pip = hand_landmarks[pip_ids[i]]
                mcp = hand_landmarks[mcp_ids[i]]

            # Visualize the bones being measured
            # Convert normalized coordinates to pixel coordinates for drawing
            tip_px = (int(tip.x * w), int(tip.y * h))
            pip_px = (int(pip.x * w), int(pip.y * h))
            mcp_px = (int(mcp.x * w), int(mcp.y * h))
            
            # Draw the lines that form the angle
            cv2.line(frame, tip_px, pip_px, (0, 255, 0), 2)
            cv2.line(frame, pip_px, mcp_px, (0, 255, 0), 2)

            # Draw the landmarks (Yellow dots)
            cv2.circle(frame, tip_px, 5, (0, 255, 255), -1)
            cv2.circle(frame, pip_px, 5, (0, 255, 255), -1)
            cv2.circle(frame, mcp_px, 5, (0, 255, 255), -1)

            angle = calculate_angle(tip, pip, mcp)

            # Convert angle (approx 30°–180°) → 0–100%
            if i == 0:
                # Thumb: Adjust range so ~110 deg is 100% folded
                fold_percent = np.interp(angle, [110, 170], [100, 0])
            else:
                fold_percent = np.interp(angle, [50, 170], [100, 0])

            fold_percent = int(fold_percent)

            # Finger state
            if fold_percent < 40:
                state = "Extended"
                finger_count += 1
            else:
                state = "Folded"
            
            # Display angle and percent
            cv2.putText(frame,
                        f"{finger_names[i]}: {fold_percent}% ({state})",
                        (10, 40 + i*30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2)

        # Show total finger count
        cv2.putText(frame,
                    f"Fingers: {finger_count}",
                    (400, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 0, 0),
                    3)

        # Show Hand Side
        cv2.putText(frame,
                    f"Side: {hand_label}",
                    (400, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 0, 0),
                    3)
    else:
        prev_landmarks = None

    cv2.imshow("Real-Time Hand Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
detector.close()