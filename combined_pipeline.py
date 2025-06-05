import cv2
import mediapipe as mp
import torch
import time
import pygame
import mido
import rtmidi
from torchvision import models, transforms
from PIL import Image

# ====== Config & Models ======
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 128
COOLDOWN = 0.4

# MIDI Configuration
MIDI_PORT_NAME = "AirGuitar"
midi_out = mido.open_output(MIDI_PORT_NAME, virtual=True)  # macOS & Linux

CHORDS = {
    1: [48, 52, 55],      # C major triad
    2: [50, 55, 59],      # D minor
    3: [52, 57, 60],      # E minor
    4: [53, 57, 60, 64],  # F major add-6
    5: [55, 59, 62]       # G major
}

# A minor pentatonic scale notes (A3, C4, D4, E4, G4)
NOTES = {
    1: 57,  # A3
    2: 60,  # C4
    3: 62,  # D4
    4: 64,  # E4
    5: 67   # G4
}
NOTE_LEN = 0.4  # seconds

pending_off = []

left_hand_status = None
left_hand_octave = None


# Left hand (open/closed)
left_model = models.resnet18(weights=None)
left_model.fc = torch.nn.Linear(left_model.fc.in_features, 2)
left_model.load_state_dict(torch.load('resnet18_left_hand.pth', map_location=DEVICE))
left_model.to(DEVICE).eval()

# Right hand (finger count)
right_model = models.mobilenet_v2(weights=None)
right_model.classifier[1] = torch.nn.Linear(right_model.last_channel, 5)
right_model.load_state_dict(torch.load('checkpoints/finger_mobilenet_v2_with_aug.pt', map_location=DEVICE))
right_model.to(DEVICE).eval()

# Image transform
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# MediaPipe & Audio
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5)

# Removed pygame.mixer.init() and strum_sound

# ====== Helper Functions ======
def get_bbox(landmarks, frame_shape):
    h, w, _ = frame_shape
    xs = [lm.x * w for lm in landmarks.landmark]
    ys = [lm.y * h for lm in landmarks.landmark]
    x1, y1 = int(min(xs)) - 20, int(min(ys)) - 20
    x2, y2 = int(max(xs)) + 20, int(max(ys)) + 20
    return max(0, x1), max(0, y1), min(w, x2), min(h, y2)

def get_octave(landmarks):
    return 0 if landmarks.landmark[0].y > 0.5 else 1

# ====== Main Loop ======
# List available cameras
cap = cv2.VideoCapture(0)  # 0 is typically the built-in camera
if not cap.isOpened():
    print("Error: Could not open camera")
    exit()

# Set camera properties for better performance
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

prev_y = None
last_strum_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if 'latest_left_result' not in locals():
        latest_left_result = "Unknown"
    right_result = ""

    if results.multi_hand_landmarks and results.multi_handedness:
        for i, hand_handedness in enumerate(results.multi_handedness):
            label = hand_handedness.classification[0].label
            landmarks = results.multi_hand_landmarks[i]
            x1, y1, x2, y2 = get_bbox(landmarks, frame.shape)
            crop = frame[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            img_tensor = transform(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(DEVICE)

            if label == "Left":
                with torch.no_grad():
                    pred = torch.argmax(left_model(img_tensor), dim=1).item()
                state = "OPEN" if pred == 1 else "CLOSED"
                octave = get_octave(landmarks)
                left_result = f"{state}, Octave {octave}"
                mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)
                latest_left_result = f"{state}, Octave {octave}"

                

            elif label == "Right":
                with torch.no_grad():
                    pred = torch.argmax(right_model(img_tensor), dim=1).item() + 1
                right_result = f"{pred} fingers"

                wrist_y = landmarks.landmark[mp_hands.HandLandmark.WRIST].y
                now = time.time()
                if prev_y is not None and (wrist_y - prev_y > 0.08) and (now - last_strum_time > COOLDOWN):
                    last_strum_time = now
                    # Removed strum_sound.play()
                    print("🎸 Strum Detected!")
                    print("Right Hand:", right_result)
                    print("Left Hand:", latest_left_result)

                    # Clear old notes
                    for note in pending_off:
                        midi_out.send(mido.Message('note_off', note=note, velocity=0))
                    pending_off.clear()

                    # Parse left hand state
                    left_state = latest_left_result.split(", ")[0]
                    octave = int(latest_left_result.split("Octave ")[1])
                    chord_mode = left_state == "OPEN"

                    # Calculate velocity based on strum speed
                    velocity = min(127, int(abs(wrist_y - prev_y) * 1000))

                    if chord_mode:   # play chord
                        notes = [n + octave*12 for n in CHORDS.get(pred, [])]
                    else:            # play single note
                        notes = [NOTES.get(pred, 47) + octave*12]

                    for n in notes:
                        midi_out.send(mido.Message('note_on', note=n, velocity=velocity))
                        print(f"Sent MIDI note_on: {n}, velocity: {velocity}")
                    pending_off = notes

                prev_y = wrist_y
                mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

    # Overlay results
    if latest_left_result:
        cv2.putText(frame, latest_left_result, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255), 2)
    if right_result:
        cv2.putText(frame, right_result, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    cv2.imshow("AirGtr Combined", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
