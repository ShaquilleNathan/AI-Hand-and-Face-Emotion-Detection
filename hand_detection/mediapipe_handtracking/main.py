import cv2 
import time
import HandDetection as hd
handDetect = hd.handDetector(detection_confident=0.8)
cap = cv2.VideoCapture(0)
top_idx = [4,8,12,16,20]
previous_time = 0
current_time = 0
while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame = handDetect.findHands(frame)
    lmList = handDetect.getHandLocation(frame, draw=True)
    #print(lmList)
    try:
        if len(lmList) != 0:
            fingers = []
            if lmList[top_idx[0]][1] < lmList[top_idx[0]-1][1]:
                fingers.append(1)
            else:
                fingers.append(0)
            for idx in range(1,5):
                if lmList[top_idx[idx]][2] < lmList[top_idx[idx]-2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)
            print(fingers)
            openFingers = fingers.count(1)
            cv2.rectangle(frame, (20,20), (200,200), (255,255,255), cv2.FILLED)
            cv2.putText(frame, str(int(openFingers)), (50,170), cv2.FONT_HERSHEY_PLAIN, 10, (255,0,0), 25)
            if fingers[1] and fingers[2] and fingers.count(1) == 2:
                cv2.putText(frame, "Peace", (50, 270), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0), 3)
            elif fingers[0] and fingers.count(1) == 1:
                cv2.putText(frame, "Nice", (50, 270), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0), 3 )
    except:
        pass
    current_time = time.time()
    fps = 1 / (current_time - previous_time)
    previous_time = current_time
    cv2.putText(frame, "frame rate: "+str(int(fps)), (350, 70), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
