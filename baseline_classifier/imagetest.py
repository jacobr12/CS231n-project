import cv2
import mediapipe as mp
import os
from rule import count_extended_fingers

#initialize static mp hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)
#for tracking accuracy
correct = 0
total = 0
root = "./data/"
#loop over all examples
for label in os.listdir(root):
    folder = os.path.join(root, label)
    #loop over all images
    for fname in os.listdir(folder):
        img_path = os.path.join(folder, fname)
        img = cv2.imread(os.path.join(folder, fname))
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)
        #only if a hand is detected, count extended fingers
        if result.multi_hand_landmarks:
            pred = count_extended_fingers(result.multi_hand_landmarks[0])
            #update accuracy
            if int(label) == pred:
                print(f"Now evaluating: {img_path}")
                correct += 1
            total += 1
            #print ground truth and prediction and accuracy stats
            print(f"GT: {label}, Pred: {pred}")
            print(f"Accuracy: {correct}/{total} = {correct / total:.2%}")
        else:
            print("No hand detected in image:", fname)
