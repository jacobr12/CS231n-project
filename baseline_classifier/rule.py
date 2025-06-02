import mediapipe as mp
import math

mp_hands = mp.solutions.hands

def euclidean(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def is_finger_up(hand, tip_id, pip_id):
    return hand.landmark[tip_id].y < hand.landmark[pip_id].y - 0.02  # add margin

def is_thumb_up(hand):
    tip = hand.landmark[mp_hands.HandLandmark.THUMB_TIP]
    mcp = hand.landmark[mp_hands.HandLandmark.THUMB_MCP]
    index_mcp = hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
    wrist = hand.landmark[mp_hands.HandLandmark.WRIST]

    # normalize thumb distance by hand width
    hand_width = euclidean(wrist, index_mcp)
    extension = euclidean(tip, mcp)

    return extension > 0.6 * hand_width  # tuned threshold

def count_extended_fingers(hand):
    count = 0

    # Thumb (special case — sideways)
    if is_thumb_up(hand):
        count += 1

    # Index
    if is_finger_up(hand, mp_hands.HandLandmark.INDEX_FINGER_TIP,
                          mp_hands.HandLandmark.INDEX_FINGER_PIP):
        count += 1

    # Middle
    if is_finger_up(hand, mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                          mp_hands.HandLandmark.MIDDLE_FINGER_PIP):
        count += 1

    # Ring
    if is_finger_up(hand, mp_hands.HandLandmark.RING_FINGER_TIP,
                          mp_hands.HandLandmark.RING_FINGER_PIP):
        count += 1

    # Pinky
    if is_finger_up(hand, mp_hands.HandLandmark.PINKY_TIP,
                          mp_hands.HandLandmark.PINKY_PIP):
        count += 1

    return count
