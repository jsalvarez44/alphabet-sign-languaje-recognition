import cv2
import mediapipe as mp 

cap = cv2.VideoCapture(0)

cap.set(3, 1080)
cap.set(4, 640)

hands_class = mp.solutions.hands
hands = hands_class.Hands()

mpDraw = mp.solutions.drawing_utils

while True:
    success, img = cap.read()
    if not success:
        break
    
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(imgRGB)
    
    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, hand, hands_class.HAND_CONNECTIONS)
    
    cv2.imshow('Lectura', img)
    
    if cv2.waitKey(1) == 27:
        break
