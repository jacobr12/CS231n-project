import cv2
import mediapipe as mp
import pygame
import time
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import numpy as np
import os

os.makedirs("strum_captures", exist_ok=True)

# ========== Model Setup ==========
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "checkpoints/finger_mobilenet_v2_with_aug.pt"

if "resnet18" in MODEL_PATH:
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 5)
elif "efficientnet_b0" in MODEL_PATH:
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 5)
else:
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, 5)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval().to(DEVICE)

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ========== MediaPipe + Sound Setup ==========
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

pygame.mixer.init()
sound = pygame.mixer.Sound("strum.wav")

# ========== Webcam and Strum Logic ==========
cap = cv2.VideoCapture(0)
prev_y = None
last_strum_time = 0
cooldown = 0.4  # seconds

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    prediction_text = ""

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, hand_handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            label = hand_handedness.classification[0].label
            if label == "Right":
                h, w, _ = frame.shape
                xs = [pt.x * w for pt in hand_landmarks.landmark]
                ys = [pt.y * h for pt in hand_landmarks.landmark]
                x1, y1 = int(min(xs)) - 20, int(min(ys)) - 20
                x2, y2 = int(max(xs)) + 20, int(max(ys)) + 20
                crop = frame[y1:y2, x1:x2]

                # ===== Predict finger count every frame =====
                if crop.size > 0:
                    img_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                    img_tensor = transform(img_pil).unsqueeze(0).to(DEVICE)

                    with torch.no_grad():
                        output = model(img_tensor)
                        pred = torch.argmax(output, dim=1).item() + 1

                    prediction_text = f"{pred} fingers"

                # ===== Detect strum and play sound =====
                wrist_y = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y
                now = time.time()
                if prev_y is not None:
                    dy = wrist_y - prev_y
                    if dy > 0.08 and (now - last_strum_time > cooldown):
                        last_strum_time = now
                        sound.play()
                        print("🎸 Strum detected! Prediction:", prediction_text)
                        timestamp = int(time.time() * 1000)
                        save_path = f"strum_captures/strum_{timestamp}.jpg"
                        cv2.imwrite(save_path, crop)
                        print(f"🖼️  Saved strum crop to {save_path}")
                prev_y = wrist_y

                # Draw hand landmarks
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display live frame with prediction
    if prediction_text:
        cv2.putText(frame, prediction_text, (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    cv2.imshow("Strum & Predict", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
