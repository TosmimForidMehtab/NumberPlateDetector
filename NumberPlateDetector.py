import cv2
import numpy as np

numberPlateCascade = cv2.CascadeClassifier(
    "Resources/haarcascade_russian_plate_number.xml")
minArea = 500
COLOR = (255, 10, 250)
WIDTH, HEIGHT = 640, 480
cap = cv2.VideoCapture(1)
cap.set(3, WIDTH)
cap.set(4, HEIGHT)
cap.set(10, 150)

while True:
    success, img = cap.read()

    imgGray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    numberPlates = numberPlateCascade.detectMultiScale(imgGray, 1.1, 4)
    for (x, y, w, h) in numberPlates:
        area = w*h
        if area > minArea:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(img, "Number Plate", (x, y-5),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, COLOR, 2)
            imgROI = img[y:y+h, x:x+w]
            cv2.imshow("ROI", imgROI)

    cv2.imshow("Result", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
