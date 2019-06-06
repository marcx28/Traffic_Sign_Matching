import cv2
import numpy as np
from matplotlib import pyplot as plt

capture = cv2.VideoCapture('Autofahrt.mp4')
print("hello world")
while cv2.waitKey(1) != ord('q'):
    _, frame = capture.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_red = np.array([30, 150, 50])
    upper_red = np.array([255, 255, 180])

    mask = cv2.inRange(hsv, lower_red, upper_red)
    res = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.imshow('Original', frame)
    edges = cv2.Canny(frame, 100, 200)
    cv2.imshow('Edges', edges)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
capture.release()

cv2.destroyAllWindows()
cap.release()