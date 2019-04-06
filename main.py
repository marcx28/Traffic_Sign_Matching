import cv2
import numpy as np


def find_outer_contour(sign_contour):
    print(sign_contours[0])
    return sign_contour[0]


stopsign = cv2.imread('stopsign.jpg')
signs = [stopsign]
sign_contours = [cv2.findContours(cv2.Canny(sign, 100, 200), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0] for sign in signs]

cv2.waitKey(0)

cap = cv2.VideoCapture(0)
while cv2.waitKey(1) != ord('q'):
    _, frame = cap.read()
    edges = cv2.Canny(frame, 100, 200)

    contoursimg = frame
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    stopsign_contour = sign_contours[0]
    stopsign_outer_contour = find_outer_contour(stopsign_contour)

    cv2.drawContours(stopsign, [stopsign_outer_contour], -1, (255, 0, 255), 5)
    cv2.imshow('stop sign', stopsign)

    for contour in contours:
        similarity = cv2.matchShapes(contour, stopsign_outer_contour, cv2.CONTOURS_MATCH_I2, 0.0)
        if similarity > .01:
            similarity = .01
        cv2.drawContours(contoursimg, [contour], -1, (255-similarity * 25500, 0, 0), 2)



    cv2.imshow('contours', contoursimg)


cv2.destroyAllWindows()
cap.release()