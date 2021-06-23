import cv2
import time
import os
import Hand_Tracking_Module as htm

wCam, hCam = 640, 480
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
folderPath = "FingerImages"
myList = os.listdir(folderPath)
overlayList = []
pTime = 0
font = cv2.FONT_HERSHEY_SIMPLEX

detector = htm.handDetector(detection_confidence=0.75)

tipIDs = [4, 8, 12, 16, 20]

for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    image = cv2.resize(image, (200, 200))
    overlayList.append(image)

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False, handNumber=0)
    if len(lmList) != 0:
        fingers = []
        # Thumb
        if lmList[tipIDs[0]][1] > lmList[tipIDs[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # Other fingers
        for id in range(1, 5):
            if lmList[tipIDs[id]][2] < lmList[tipIDs[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        total_fingers = fingers.count(1)
        print(total_fingers)

        h, w, c = overlayList[0].shape
        img[:h, :w] = overlayList[total_fingers]
        cv2.rectangle(img, (int(0.05 * wCam), int(hCam * 0.6)), (int(0.2 * wCam), int(hCam * 0.9)), (0, 0, 0),
                      cv2.FILLED)
        cv2.putText(img, str(total_fingers), (int(0.1 * wCam), int(hCam * 0.8)), font, 2, (255, 255, 255), 3)

    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}', (380, 70), font, 2, (255, 255, 255), 2)


    cv2.imshow("Camera", img)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()