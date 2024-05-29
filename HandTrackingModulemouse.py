# Import necessary libraries
import cv2
import mediapipe as mp
import time
import math
import numpy as np

# Define a class for hand detection and tracking
class HandDetector:
    # Initialize the hand detector with default settings
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode  # Mode for static or dynamic images
        self.maxHands = maxHands  # Maximum number of hands to detect
        self.detectionCon = detectionCon  # Minimum detection confidence
        self.trackCon = trackCon  # Minimum tracking confidence

        # Initialize MediaPipe hands solution
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=self.mode,
                                        max_num_hands=self.maxHands,
                                        min_detection_confidence=self.detectionCon,
                                        min_tracking_confidence=self.trackCon)
        # Drawing utility for landmarks
        self.mpDraw = mp.solutions.drawing_utils
        # IDs for fingertips
        self.tipIds = [4, 8, 12, 16, 20]

    # Method to find hands in an image
    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert the image to RGB
        self.results = self.hands.process(imgRGB)  # Process the image to find hands

        # If hands are detected
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    # Draw hand landmarks on the image
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img  # Return the image with drawn landmarks if draw is True

    # Method to find the position of hand landmarks
    def findPosition(self, img, handNo=0, draw=True):
        lmlist = []
        xList = []  # List to store x-coordinates of landmarks
        yList = []  # List to store y-coordinates of landmarks
        bbox = []  # Bounding box for the hand
        self.lmList = []  # List to store landmark positions
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]  # Get the landmarks of the specified hand
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape  # Get image dimensions
                cx, cy = int(lm.x * w), int(lm.y * h)  # Convert normalized coordinates to pixel coordinates
                lmlist.append((id, cx, cy))
                xList.append(cx)
                yList.append(cy)
                self.lmList.append([id, cx, cy])  # Append landmark id and position to lmList
                if draw:
                    # Draw circles on landmarks
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

            # Calculate bounding box
            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bbox = xmin, ymin, xmax, ymax

            if draw:
                # Draw the bounding box
                cv2.rectangle(img, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20), (0, 255, 0), 2)

        return self.lmList, bbox  # Return landmark list and bounding box

    # Method to determine which fingers are up
    def fingersUp(self):
        fingers = []  # List to store the state of each finger
        # Thumb
        if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0] - 1][1]:
            fingers.append(1)  # Thumb is up
        else:
            fingers.append(0)  # Thumb is down

        # Other fingers
        for id in range(1, 5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                fingers.append(1)  # Finger is up
            else:
                fingers.append(0)  # Finger is down

        return fingers  # Return the state of all fingers

    # Method to find the distance between two landmarks
    def findDistance(self, p1, p2, img, draw=True, r=15, t=3):
        x1, y1 = self.lmList[p1][1:]  # Coordinates of the first point
        x2, y2 = self.lmList[p2][1:]  # Coordinates of the second point
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2  # Midpoint between the two points

        if draw:
            # Draw line and circles between the points
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)
        length = math.hypot(x2 - x1, y2 - y1)  # Calculate the distance between the points

        return length, img, [x1, y1, x2, y2, cx, cy]  # Return the distance, modified image, and coordinates

# Main function to capture video and apply hand detection
def main():
    pTime = 0  # Previous time for calculating FPS
    cap = cv2.VideoCapture(1)  # Capture video from webcam
    detector = HandDetector()  # Initialize the hand detector

    while True:
        success, img = cap.read()  # Read a frame from the webcam
        if not success:
            break

        img = detector.findHands(img)  # Detect hands in the frame
        lmList, bbox = detector.findPosition(img)  # Find positions of hand landmarks

        if len(lmList) != 0:
            print(lmList[4])  # Print the position of the thumb tip

        cTime = time.time()  # Current time
        fps = 1 / (cTime - pTime)  # Calculate FPS
        pTime = cTime  # Update previous time

        # Display FPS on the image
        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        cv2.imshow("Image", img)  # Show the image
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit the loop if 'q' is pressed
            break

    cap.release()  # Release the webcam
    cv2.destroyAllWindows()  # Destroy all OpenCV windows

# Run the main function if this script is executed
if __name__ == "__main__":
    main()
