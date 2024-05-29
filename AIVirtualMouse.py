import autopy
import cv2
import numpy as np
import HandTrackingModulemouse as htm
import time

######################
wCam, hCam = 640, 480
frameR = 100  # Frame Reduction
smoothening = 7  # Random value for smoothening
######################

pTime = 0
plocX, plocY = 0, 0
clocX, clocY = 0, 0

# Initialize video capture
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

# Initialize hand detector
detector = htm.HandDetector()

# Get screen size
wScr, hScr = autopy.screen.size()

while True:
    # Step 1: Find the landmarks
    success, img = cap.read()
    if not success:
        continue

    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)

    # Step 2: Get the tip of the index and middle finger
    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        # Step 3: Check which fingers are up
        fingers = detector.fingersUp()

        # Draw frame rectangle for tracking area
        cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR),
                      (255, 0, 255), 2)

        # Step 4: Only Index Finger: Moving Mode
        if fingers[1] == 1 and fingers[2] == 0:
            # Step 5: Convert the coordinates
            x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
            y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))

            # Step 6: Smooth Values
            clocX = plocX + (x3 - plocX) / smoothening
            clocY = plocY + (y3 - plocY) / smoothening

            # Step 7: Move Mouse
            autopy.mouse.move(wScr - clocX, clocY)
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            plocX, plocY = clocX, clocY

        # Step 8: Both Index and Middle Fingers are Up: Clicking Mode
        if fingers[1] == 1 and fingers[2] == 1:
            # Step 9: Find distance between fingers
            length, img, lineInfo = detector.findDistance(8, 12, img)

            # Step 10: Click mouse if distance is short
            if length < 40:
                cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)
                autopy.mouse.click()

    # Step 11: Frame rate
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (28, 58), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    # Step 12: Display
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
