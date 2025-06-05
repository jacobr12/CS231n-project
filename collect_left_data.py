# collect_left_hand_data.py
import cv2
import mediapipe as mp
import os

# Setup
mp_hands = mp.solutions.hands
hand_detector = mp_hands.Hands(model_complexity=1, max_num_hands=2, min_detection_confidence=0.7)

# Output folders
output_dir = 'left_hand_images/test'
os.makedirs(os.path.join(output_dir, 'open'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'closed'), exist_ok=True)

# Determine the next available index
open_files = os.listdir(os.path.join(output_dir, 'open'))
closed_files = os.listdir(os.path.join(output_dir, 'closed'))
open_indices = [int(f.split('_')[1].split('.')[0]) for f in open_files if f.startswith("frame_") and f.endswith(".jpg")]
closed_indices = [int(f.split('_')[1].split('.')[0]) for f in closed_files if f.startswith("frame_") and f.endswith(".jpg")]

max_index = max(open_indices + closed_indices) + 1 if (open_indices or closed_indices) else 0
counter = max_index

# Init video
cap = cv2.VideoCapture(0)
print("Press 'o' to save frame as OPEN, 'c' as CLOSED. Press ESC to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    # Flip the frame horizontally
    frame = cv2.flip(frame, 1)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hand_detector.process(image_rgb)

    label = None
    key = cv2.waitKey(1) & 0xFF
    if key == ord('o'):
        label = 'open'
    elif key == ord('c'):
        label = 'closed'
    elif key == 27:
        break

    if results.multi_hand_landmarks and results.multi_handedness:
        for i, hand_info in enumerate(results.multi_handedness):
            if hand_info.classification[0].label == 'Left':
                hand_landmarks = results.multi_hand_landmarks[i]

                h, w, _ = frame.shape
                xs = [pt.x * w for pt in hand_landmarks.landmark]
                ys = [pt.y * h for pt in hand_landmarks.landmark]
                x1, y1 = int(min(xs)) - 20, int(min(ys)) - 20
                x2, y2 = int(max(xs)) + 20, int(max(ys)) + 20

                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)

                crop = frame[y1:y2, x1:x2]

                if label is not None and crop.size > 0:
                    filename = os.path.join(output_dir, label, f"frame_{counter}.jpg")
                    cv2.imwrite(filename, crop)
                    print(f"Saved frame_{counter}.jpg as {label.upper()}")
                    counter += 1

                mp.solutions.drawing_utils.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow('Collecting Left Hand Images', frame)

cap.release()
cv2.destroyAllWindows()
