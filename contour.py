import cv2
import sys
import pytesseract
import imutils
import argparse
import numpy as np


def find_outer_contour(sign_contour):
    return sign_contour[0]


stopsign = cv2.imread('stopsign.jpg')
vorfahrtsign = cv2.imread('vorfahrt.jpg')
vorrangstrasse = cv2.imread('vorrangstrasse.png')
durchfahrtverboten = cv2.imread('durchfahrtverboten.jpg')
names = ["stop sign", "vorrang geben", "vorrang straße", "durchfahrt verboten"]
signs = [stopsign, vorfahrtsign, vorrangstrasse]#, durchfahrtverboten]
limit = [.05, .05, 0.005, 0.005]
#templateThreshold = [0.4, 0.15, 0.1, 1]
templateThreshold = [0, 0.15, 0.15, 1]
corners = [8, 3, 4, 0]
highlightColors = [(255, 0, 0), (255, 0, 255), (0, 255, 0), (0, 255, 255)]
# degrees offset of the sign from its best fit bounding box
rotationoffset = [0, 0, 45, 0]
boundingbox_symmetry = [8, 6, 1, 1]
sign_contours = [cv2.findContours(cv2.Canny(sign, 100, 200), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0] for sign in signs]
minAreaRect = [cv2.minAreaRect(find_outer_contour(sign_contours[i])) for i in range(len(signs))]

minContourArea = 200

cv2.waitKey(0)


showAllContours = True
displaySourceImage = True
templateMatching = True
paused = False
cap = cv2.VideoCapture(0)#"Autofahrt.mp4")

for i in range(len(signs)):
    contour = sign_contours[i]
    outer_contour = find_outer_contour(contour)
    cv2.drawContours(signs[i], [outer_contour], -1, highlightColors[i], 5)
#    cv2.imshow(names[i], signs[i])

key = cv2.waitKey(1)
frame = None
while key != ord('q'):

    # automatically, while loop only has to run while paused if a setting has been changed
    key = cv2.waitKey(1 if not paused else 0)

    # options
    if key == ord('c'):
        showAllContours = not showAllContours
    if key == ord('d'):
        displaySourceImage = not displaySourceImage
    if key == ord('t'):
        templateMatching = not templateMatching
    if key == ord('p'):
        paused = not paused

    # change image source
    if key == ord('0'):
        cap = cv2.VideoCapture(0)
        paused = False
    if key == ord('1'):
        cap = cv2.VideoCapture(1)
        paused = False
    if key == ord('5'):
        cap = cv2.VideoCapture('Autofahrt.mp4')
        paused = False

    if not paused:
        _, frame = cap.read()

    if frame is None:
        continue

    edges = cv2.Canny(frame, 100, 200)

    contoursimg = frame.copy()
    origFrame = frame.copy()
    if not displaySourceImage:
        contoursimg[:] = (100, 100, 100)
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for i, contour in enumerate(contours):
        if cv2.contourArea(contour) < minContourArea:
            continue
        if showAllContours:
            cv2.drawContours(contoursimg, [contour], -1, (0, 0, 0), 2)

        bestSim = 1000
        bestSign = 0
        # match the contour with all known sign contours and save the best match
        for signIdx in range(len(signs)):
            approxCorners = len(cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True))
            similarity = cv2.matchShapes(contour, find_outer_contour(sign_contours[signIdx]), cv2.CONTOURS_MATCH_I2, 0.0)
            if similarity < bestSim and (corners[signIdx] == 0 or corners[signIdx] == approxCorners):
                bestSim = similarity
                bestSign = signIdx

        if bestSim < limit[bestSign]:
            if not templateMatching:
                cv2.drawContours(contoursimg, [contour], -1, highlightColors[bestSign], 2)
            else:
                # now confirm detection with template matching
                rect = cv2.minAreaRect(contour)
                x, y, boundingWidth, boundingHeight = cv2.boundingRect(contour)
                center = rect[0]
                width, height = rect[1]
                angle = -(rect[2] - rotationoffset[bestSign])

                signRect = minAreaRect[bestSign]
                signCenter = signRect[0]
                signWidth, signHeight = signRect[1]
                boundingRect = cv2.boundingRect(contour)

                scale = min(width / signWidth, height / signHeight)
                template = cv2.cvtColor(signs[bestSign], cv2.COLOR_BGR2GRAY)

                margin = 5
#                print(y-margin, y + height + margin, x-margin, x + width + margin)

                search_top = max(0, int(y - margin))
                search_bot = min(origFrame.shape[1], int(y + height + margin))
                search_left = max(0, int(x - margin))
                search_right = min(origFrame.shape[1], int(x + width + margin))
                search_image = cv2.cvtColor(origFrame[search_top:search_bot, search_left: search_right], cv2.COLOR_BGR2GRAY)
                mask = np.ones(search_image.shape).astype(search_image.dtype)
                cv2.fillPoly(mask, [contour], (0, 0, 0), offset=(-search_left, -search_top))
                search_image = cv2.bitwise_or(search_image, mask)


                best_min_val = 1
                best_template = template

                for alpha in range(int(angle), int(360 + angle), int(360 / boundingbox_symmetry[bestSign])):
#                    alpha = angle
                    rot = cv2.getRotationMatrix2D(signCenter, alpha, 1)
                    curtemplate = cv2.warpAffine(template, rot, (template.shape[0], template.shape[1]), borderValue=(255, 255, 255))
                    curtemplate = cv2.resize(curtemplate, (int(signWidth * scale), int(signHeight * scale)))

                    res = cv2.matchTemplate(search_image, curtemplate, cv2.TM_SQDIFF_NORMED)
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                    if min_val < best_min_val:
                        best_min_val = min_val
                        best_template = curtemplate

                print(best_min_val)
                if best_min_val < templateThreshold[bestSign]:
                    cv2.drawContours(contoursimg, [contour], -1, highlightColors[bestSign], 2)
                    cv2.imshow("search" + names[bestSign], search_image)
                    cv2.imshow(names[bestSign], curtemplate)

    cv2.imshow('contours', contoursimg)

cv2.destroyAllWindows()
cap.release()
