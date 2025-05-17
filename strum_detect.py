import cv2
import mediapipe as mp
import pygame
import time

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7, # think that this could be a good medium
    min_tracking_confidence=0.5
)

mp_draw = mp.solutions.drawing_utils

pygame.mixer.init()
sound = pygame.mixer.Sound("strum.wav")

cap = cv2.VideoCapture(0)

prev_y = None
last_strum_time = 0
cooldown = 0.4  


while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1) 
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, hand_handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            label = hand_handedness.classification[0].label  
            if label == "Right": #checks for right hand, will be useful when rest of logic is implemented
                wrist_y = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y
                if prev_y is not None: 
                    dy = wrist_y - prev_y #computes the change of the hands distance vertically
                    now = time.time()
                    if dy > 0.08 and (now - last_strum_time > cooldown): #measures a strum based on the hand moving down
                        sound.play()
                        last_strum_time = now
                prev_y = wrist_y
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Strum Detector", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()