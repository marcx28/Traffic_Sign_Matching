import cv2
import numpy as np


def find_outer_contour(sign_contour):
#    print(sign_contours[0])
    return sign_contour[0]


stopsign = cv2.imread('stopsign.jpg')
vorfahrtsign = cv2.imread('vorfahrt.jpg')
vorrangstrasse = cv2.imread('vorrangstrasse.png')
durchfahrtverboten = cv2.imread('durchfahrtverboten.jpg')
names = ["stop sign", "vorrang geben", "vorrang straße", "durchfahrt verboten"]
signs = [stopsign, vorfahrtsign, vorrangstrasse, durchfahrtverboten]
limit = [.005, .05, 0.005, 0.005]
corners = [8, 3, 4, 0]
highlightColors = [(255, 0, 0), (255, 0, 255), (0, 255, 0), (0, 255, 255)]
sign_contours = [cv2.findContours(cv2.Canny(sign, 100, 200), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0] for sign in signs]

minContourArea = 50

cv2.waitKey(0)

cap = cv2.VideoCapture(0)

showAllContours = True

for i in range(len(signs)):
    contour = sign_contours[i]
    outer_contour = find_outer_contour(contour)
    cv2.drawContours(signs[i], [outer_contour], -1, highlightColors[i], 5)
    cv2.imshow(names[i], signs[i])

key = cv2.waitKey(1)
while key != ord('q'):

    key = cv2.waitKey(1)
    if key == ord('c'):
        showAllContours = not showAllContours
    if key == ord('0'):
        cap = cv2.VideoCapture(0)
    if key == ord('1'):
        cap = cv2.VideoCapture(1)
    if key == ord('5'):
        cap = cv2.VideoCapture('Autofahrt.mp4')

    _, frame = cap.read()

    if frame is None:
        continue

    edges = cv2.Canny(frame, 100, 200)

    contoursimg = frame
#    contoursimg[:] = (255, 255, 255)
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for i, contour in enumerate(contours):
        if cv2.contourArea(contour) < minContourArea:
            continue
        if showAllContours:
            cv2.drawContours(contoursimg, [contour], -1, (0, 0, 0), 2)

        bestSim = 1000
        bestSign = 0
        for signIdx in range(len(signs)):
            approxCorners = len(cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True))
            similarity = cv2.matchShapes(contour, find_outer_contour(sign_contours[signIdx]), cv2.CONTOURS_MATCH_I2, 0.0)
            if similarity < bestSim and (corners[signIdx] == 0 or corners[signIdx] == approxCorners):
                bestSim = similarity
                bestSign = signIdx
        if bestSim < limit[bestSign]:
            cv2.drawContours(contoursimg, [contour], -1, highlightColors[bestSign], 2)
            print(names[bestSign], ": ", similarity)

    cv2.imshow('contours', contoursimg)

cv2.destroyAllWindows()
cap.release()