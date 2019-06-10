import cv2
import numpy as np
from matplotlib import pyplot as plt

capture = cv2.VideoCapture(0)
print("hello world")
k = cv2.waitKey(1)
while k != ord('q'):
    _, frame = capture.read()

    cv2.imshow('Original', frame)
    edges = cv2.Canny(frame, 100, 200)
    cv2.imshow('Edges', edges)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
capture.release()
