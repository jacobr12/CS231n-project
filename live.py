import cv2
import mediapipe as mp
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import numpy as np


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "checkpoints/finger_mobilenetv2.pt"

# Load model
model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, 5)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval().to(DEVICE)

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# Mediapipe hand detector
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Could not open webcam.")
    exit()
else:
    print("✅ Webcam successfully opened.")

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    prediction_text = ""

    if results.multi_hand_landmarks and results.multi_handedness:
        for lm, hand_type in zip(results.multi_hand_landmarks, results.multi_handedness):
            if hand_type.classification[0].label == "Right":
                h, w, _ = frame.shape
                xs = [pt.x * w for pt in lm.landmark]
                ys = [pt.y * h for pt in lm.landmark]
                x1, y1 = int(min(xs)) - 20, int(min(ys)) - 20
                x2, y2 = int(max(xs)) + 20, int(max(ys)) + 20

                crop = frame[y1:y2, x1:x2]
                if crop.size > 0:
                    # Convert to tensor
                    img_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                    img_tensor = transform(img_pil).unsqueeze(0).to(DEVICE)

                    with torch.no_grad():
                        output = model(img_tensor)
                        pred = torch.argmax(output, dim=1).item() + 1  # labels are 1–5

                    prediction_text = f"{pred} fingers"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                    cv2.putText(frame, prediction_text, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

                mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Finger Count Demo", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
