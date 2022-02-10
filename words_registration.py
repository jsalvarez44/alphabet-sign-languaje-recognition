import cv2
import mediapipe as mp
import os

word = 'E'
#dataPath = 'Data/training/' + word
dataPath = 'Data/validation/' + word

if not os.path.exists(dataPath):
    print("Carpeta creada: ",dataPath)
    os.makedirs(dataPath)
    count = 0
    breakCondition = 300   

    capture = cv2.VideoCapture(0)

    hands_class = mp.solutions.hands
    hands = hands_class.Hands()

    draw = mp.solutions.drawing_utils

    while(True):
        ret, frame = capture.read()
        color = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        copy = frame.copy()
        result = hands.process(color)
        positions = []
        
        if result.multi_hand_landmarks:
            for hand in result.multi_hand_landmarks:
                for id, lm in enumerate(hand.landmark):
                    width, heigth, c = frame.shape
                    corx, cory = int(lm.x * heigth), int(lm.y * width)
                    positions.append([id, corx, cory])
                    draw.draw_landmarks(frame, hand, hands_class.HAND_CONNECTIONS)
                
                if len(positions) != 0:
                    central_point = positions[9]
                    
                    x1, y1 = (central_point[1]-120), (central_point[2]-120)
                    heigth, width = (x1 + 200), (y1 + 200)
                    x2, y2 = x1 + heigth, y1 + width
                    
                    fingers_reg = copy[y1:y2, x1:x2]
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    cv2.putText(frame, f"Posiciones detectadas: {count}", (x1, y1-20), 2,
                            0.8, (0, 0, 255), 1, cv2.LINE_AA)
                
                fingers_reg = cv2.resize(fingers_reg, (200, 200), interpolation=cv2.INTER_CUBIC)
                cv2.imwrite(dataPath + f"/{word}_{count}.jpg", fingers_reg)
                count += 1
        
        cv2.imshow("Lectura", frame)
        k =  cv2.waitKey(1)
        if k == 27 or count >= breakCondition:
            break

    capture.release()
    cv2.destroyAllWindows()

else:
    print(f"{dataPath} ya existe y tiene las imagenes necesarias!")
