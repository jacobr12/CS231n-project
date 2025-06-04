import cv2
import mediapipe as mp
import os

label = input("Enter finger count (1–5): ")
save_dir = f"./data_diff/{label}"
os.makedirs(save_dir, exist_ok=True)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2)
mp_draw = mp.solutions.drawing_utils

#start webcam capture video
cap = cv2.VideoCapture(0)
existing_files = os.listdir(save_dir)
img_count = len([f for f in existing_files if f.endswith(".jpg")])

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1) # mirrors view so that user can see
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks and results.multi_handedness:
        for lm, hand_type in zip(results.multi_hand_landmarks, results.multi_handedness):
            if hand_type.classification[0].label == "Right":
                #calculates box for right hand
                h, w, _ = frame.shape
                xs = [pt.x * w for pt in lm.landmark]
                ys = [pt.y * h for pt in lm.landmark]
                x1, y1 = int(min(xs)) - 20, int(min(ys)) - 20
                x2, y2 = int(max(xs)) + 20, int(max(ys)) + 20
                #just get the hand region
                crop = frame[y1:y2, x1:x2]
                if crop.size > 0:
                    cv2.imshow("Crop", crop)
                # save image when s is ressed
                if cv2.waitKey(1) & 0xFF == ord('s'):
                    path = os.path.join(save_dir, f"{label}_{img_count}.jpg")
                    cv2.imwrite(path, crop)
                    print(f"Saved {path}")
                    img_count += 1
                #draw the landmarks of a hand as deinfed by mp
                mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()