import cv2
import mediapipe as mp
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms, models

# ===== MediaPipe Setup =====
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# ===== Model Setup =====
IMG_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load('resnet18_left_hand.pth', map_location=DEVICE))
model.to(DEVICE)
model.eval()

# ===== Image Preprocessing =====
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ===== Helper Functions =====
def is_left_hand(handedness):
    return handedness.classification[0].label == 'Left'

def get_left_hand_bbox(landmarks, frame_shape):
    h, w, _ = frame_shape
    x_coords = [lm.x * w for lm in landmarks.landmark]
    y_coords = [lm.y * h for lm in landmarks.landmark]
    x_min, x_max = int(min(x_coords)), int(max(x_coords))
    y_min, y_max = int(min(y_coords)), int(max(y_coords))
    padding = 20
    return max(x_min - padding, 0), max(y_min - padding, 0), min(x_max + padding, w), min(y_max + padding, h)

def get_octave_from_position(landmarks):
    return 0 if landmarks.landmark[0].y > 0.5 else 1

# ===== Main Loop =====
cap = cv2.VideoCapture(0)

with mp_hands.Hands(model_complexity=1, max_num_hands=2, min_detection_confidence=0.7) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        if not ret:
            continue

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks and results.multi_handedness:
            for i, hand_handedness in enumerate(results.multi_handedness):
                if is_left_hand(hand_handedness):
                    landmarks = results.multi_hand_landmarks[i]
                    x1, y1, x2, y2 = get_left_hand_bbox(landmarks, frame.shape)
                    cropped = frame[y1:y2, x1:x2]
                    if cropped.size == 0:
                        continue
                    input_tensor = transform(cropped).unsqueeze(0).to(DEVICE)

                    with torch.no_grad():
                        logits = model(input_tensor)
                        pred_class = torch.argmax(logits, dim=1).item()

                    hand_state = "OPEN" if pred_class == 1 else "CLOSED"
                    octave = get_octave_from_position(landmarks)
                    print(f"Left hand: {hand_state}, Octave: {octave}")

                    mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{hand_state}, Octave {octave}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow('Left Hand Tracking', frame)
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
