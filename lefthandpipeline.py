# left_hand_tracker.py
import cv2
import mediapipe as mp
import torch
import torch.nn as nn
import numpy as np

# ===== MediaPipe Setup =====
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# ===== Model Definition =====
class HandOpenNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(63, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)  # 0: closed, 1: open
        )

    def forward(self, x):
        return self.net(x)

# ===== Helper Functions =====
def is_left_hand(handedness):
    return handedness.classification[0].label == 'Left'

def get_left_hand_landmarks(results):
    for i, hand_handedness in enumerate(results.multi_handedness):
        if is_left_hand(hand_handedness):
            return results.multi_hand_landmarks[i]
    return None

def extract_landmark_features(landmarks):
    features = []
    for lm in landmarks.landmark:
        features.extend([lm.x, lm.y, lm.z])
    return features

def get_octave_from_position(landmarks):
    wrist_y = landmarks.landmark[0].y
    return 0 if wrist_y > 0.5 else 1

# ===== Load Model and Setup Camera =====
model = HandOpenNet()
model.load_state_dict(torch.load('left_hand_model.pth', map_location=torch.device('cpu')))
model.eval()

cap = cv2.VideoCapture(0)

with mp_hands.Hands(model_complexity=1, max_num_hands=2, min_detection_confidence=0.7) as hands:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        if results.multi_hand_landmarks and results.multi_handedness:
            landmarks = get_left_hand_landmarks(results)
            if landmarks:
                features = extract_landmark_features(landmarks)
                input_tensor = torch.tensor(features).float().unsqueeze(0)
                with torch.no_grad():
                    logits = model(input_tensor)
                    pred_class = torch.argmax(logits, dim=1).item()

                hand_state = "OPEN" if pred_class == 1 else "CLOSED"
                octave = get_octave_from_position(landmarks)
                print(f"Left hand: {hand_state}, Octave: {octave}")

                mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.imshow('Left Hand Tracking', frame)
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
