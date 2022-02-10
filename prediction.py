import cv2
import mediapipe as mp
import numpy as np
from keras.preprocessing.image import img_to_array
from keras.models import load_model

model = 'Model/Model.h5'
weights = 'Model/Weights.h5'
CNN = load_model(model)
CNN.load_weights(weights)

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
                
                x1, y1 = (central_point[1]-100), (central_point[2]-100)
                heigth, width = (x1 + 200), (y1 + 200)
                x2, y2 = x1 + heigth, y1 + width
                
                fingers_reg = copy[y1:y2, x1:x2]
                fingers_reg = cv2.resize(fingers_reg, (64,64))
                #gray_img = cv2.cvtColor(fingers_reg, cv2.COLOR_BGR2GRAY)
                img_roi = cv2.threshold(fingers_reg, 128, 255, cv2.THRESH_BINARY)[1]

                x = img_to_array(img_roi)
                x = np.expand_dims(x, axis=0)
                
                vector = CNN.predict(x)
                result = vector[0]
                response = np.argmax(result)
                
                if response == 0:
                    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0),3)
                    cv2.putText(frame, 'A', (x1,y1-5),2,1.3,(0,0,255),1,cv2.LINE_AA)
                elif response == 1:
                    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0),3)
                    cv2.putText(frame, 'B', (x1,y1-5),2,1.3,(0,0,255),1,cv2.LINE_AA)
                elif response == 2:
                    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0),3)
                    cv2.putText(frame, 'C', (x1,y1-5),2,1.3,(0,0,255),1,cv2.LINE_AA)
                elif response == 3:
                    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0),3)
                    cv2.putText(frame, 'D', (x1,y1-5),2,1.3,(0,0,255),1,cv2.LINE_AA)
                elif response == 4:
                    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0),3)
                    cv2.putText(frame, 'E', (x1,y1-5),2,1.3,(0,0,255),1,cv2.LINE_AA)
                elif response == 5:
                    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0),3)
                    cv2.putText(frame, 'F', (x1,y1-5),2,1.3,(0,0,255),1,cv2.LINE_AA)
                elif response == 6:
                    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0),3)
                    cv2.putText(frame, 'G', (x1,y1-5),2,1.3,(0,0,255),1,cv2.LINE_AA)
                elif response == 7:
                    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0),3)
                    cv2.putText(frame, 'H', (x1,y1-5),2,1.3,(0,0,255),1,cv2.LINE_AA)
                elif response == 8:
                    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0),3)
                    cv2.putText(frame, 'I', (x1,y1-5),2,1.3,(0,0,255),1,cv2.LINE_AA)
                elif response == 9:
                    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0),3)
                    cv2.putText(frame, 'J', (x1,y1-5),2,1.3,(0,0,255),1,cv2.LINE_AA)
                elif response == 10:
                    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0),3)
                    cv2.putText(frame, 'K', (x1,y1-5),2,1.3,(0,0,255),1,cv2.LINE_AA)
                elif response == 11:
                    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0),3)
                    cv2.putText(frame, 'L', (x1,y1-5),2,1.3,(0,0,255),1,cv2.LINE_AA)
                elif response == 12:
                    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0),3)
                    cv2.putText(frame, 'M', (x1,y1-5),2,1.3,(0,0,255),1,cv2.LINE_AA)
                elif response == 13:
                    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0),3)
                    cv2.putText(frame, 'N', (x1,y1-5),2,1.3,(0,0,255),1,cv2.LINE_AA)
                elif response == 14:
                    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0),3)
                    cv2.putText(frame, 'O', (x1,y1-5),2,1.3,(0,0,255),1,cv2.LINE_AA)
                elif response == 15:
                    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0),3)
                    cv2.putText(frame, 'P', (x1,y1-5),2,1.3,(0,0,255),1,cv2.LINE_AA)
                elif response == 16:
                    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0),3)
                    cv2.putText(frame, 'Q', (x1,y1-5),2,1.3,(0,0,255),1,cv2.LINE_AA)
                elif response == 17:
                    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0),3)
                    cv2.putText(frame, 'R', (x1,y1-5),2,1.3,(0,0,255),1,cv2.LINE_AA)
                elif response == 18:
                    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0),3)
                    cv2.putText(frame, 'S', (x1,y1-5),2,1.3,(0,0,255),1,cv2.LINE_AA)
                elif response == 19:
                    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0),3)
                    cv2.putText(frame, 'T', (x1,y1-5),2,1.3,(0,0,255),1,cv2.LINE_AA)
                elif response == 20:
                    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0),3)
                    cv2.putText(frame, 'U', (x1,y1-5),2,1.3,(0,0,255),1,cv2.LINE_AA)
                elif response == 21:
                    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0),3)
                    cv2.putText(frame, 'V', (x1,y1-5),2,1.3,(0,0,255),1,cv2.LINE_AA)
                elif response == 22:
                    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0),3)
                    cv2.putText(frame, 'W', (x1,y1-5),2,1.3,(0,0,255),1,cv2.LINE_AA)
                elif response == 23:
                    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0),3)
                    cv2.putText(frame, 'X', (x1,y1-5),2,1.3,(0,0,255),1,cv2.LINE_AA)
                elif response == 24:
                    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0),3)
                    cv2.putText(frame, 'Y', (x1,y1-5),2,1.3,(0,0,255),1,cv2.LINE_AA)
                elif response == 25:
                    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0),3)
                    cv2.putText(frame, 'Z', (x1,y1-5),2,1.3,(0,0,255),1,cv2.LINE_AA)
                else:
                    cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0),3)
                    cv2.putText(frame, 'Letra desconocida', (x1,y1-5),1,1.3,(255,0,0),1,cv2.LINE_AA)

    cv2.imshow("Prediction", frame)
    k =  cv2.waitKey(1)
    if k == 27:
        break

capture.release()
cv2.destroyAllWindows()
