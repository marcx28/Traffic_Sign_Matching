import cv2
import numpy as np
import speedrec
import threading


class Detection:
    def __init__(self, contour, frame, speed):
        self.contour = contour
        self.speed = speed
        self.emptyFrames = 0
        self.tracker = cv2.TrackerCSRT_create()
        self.tracker.init(frame, cv2.boundingRect(contour))

    def found(self, contour):
        self.contour = contour
        self.emptyFrames = 0

    def notfound(self):
        self.emptyFrames += 1


def find_outer_contour(sign_contour):
    return sign_contour[0]


stopsign = cv2.imread('stopsign.jpg')
vorfahrtsign = cv2.imread('vorfahrt.jpg')
vorrangstrasse = cv2.imread('vorrangstrasse.png')
durchfahrtverboten = cv2.imread('durchfahrtverboten.jpg')

names = ["Stopschild", "Vorrang Geben", "Vorrangstraße", "Durchfahrt Verboten"]
signs = [stopsign, vorfahrtsign, vorrangstrasse, durchfahrtverboten]
limit = [.05, .05, 0.005, 0.005]
#templateThreshold = [0.25, 0.15, 0.15, .15]
templateThreshold = [0.25, 0.25, 0.25, .35]
corners = [8, 3, 4, -12]
highlightColors = [(255, 0, 0), (255, 0, 255), (0, 255, 0), (0, 255, 255)]
# degrees offset of the sign from its best fit bounding box
rotationoffset = [0, 0, 45, 0]
boundingbox_symmetry = [8, 6, 1, 1]
sign_contours = [cv2.findContours(cv2.Canny(sign, 100, 200), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0] for sign in signs]
minAreaRect = [cv2.minAreaRect(find_outer_contour(sign_contours[i])) for i in range(len(signs))]
detections = [None for i in range(len(signs))]

# How long an object has to be not found in order to discard its last location
maxEmptyFrames = 500

minContourArea = 200
videoScale = 1.0


showAllContours = True
displaySourceImage = True
templateMatching = True
paused = False
cap = cv2.VideoCapture(0)#"Autofahrt.mp4")


def detectSpeed(image, i):
    speed = speedrec.get_speed_limit(image)
    if detections[i] is not None:
        detections[i].speed = speed


for i in range(len(signs)):
    contour = sign_contours[i]
    outer_contour = find_outer_contour(contour)
    cv2.drawContours(signs[i], [outer_contour], -1, highlightColors[i], 5)
#    cv2.imshow(names[i], signs[i])


def clearDetections():
    for i in range(len(detections)):
        detections[i] = None


key = cv2.waitKey(1)
frame = None
while key != ord('q'):

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
    if key == ord('r'):
        for i in range(len(detections)):
            detections[i] = None

    # zoom
    if key == ord('+'):
        videoScale += 0.1
    if key == ord('-'):
        videoScale -= 0.1

    # change image source
    if key == ord('0'):
        cap = cv2.VideoCapture(0)
        paused = False
        clearDetections()
    if key == ord('1'):
        cap = cv2.VideoCapture(1)
        clearDetections()
        paused = False
    if key == ord('2'):
        cap = cv2.VideoCapture('videos/Tempo70.mp4')
        clearDetections()
        paused = False
    if key == ord('3'):
        cap = cv2.VideoCapture('videos/VorrangGeben.mp4')
        clearDetections()
        paused = False
    if key == ord('4'):
        cap = cv2.VideoCapture('videos/Vorrangstrasse.mp4')
        clearDetections()
        paused = False
    if key == ord('5'):
        cap = cv2.VideoCapture('videos/Autofahrt.mp4')
        clearDetections()
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

    for detection in detections:
        if detection is not None:
            detection.notfound()

    for i, contour in enumerate(contours):
        if cv2.contourArea(contour) < minContourArea:
            continue
        if showAllContours:
            cv2.drawContours(contoursimg, [contour], -1, (0, 0, 0), 2)

        bestSim = 1000
        bestSign = 0
        # match the contour with all known sign contours and save the best match
        for signIdx in range(len(signs)):
            # if we're already tracking one of these signs, no need to check for another
            if detections[bestSign] is not None:
                continue
            # check if the similarity is close enough and the number of corners on the contour are about right
            approxCorners = len(cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True))
            similarity = cv2.matchShapes(contour, find_outer_contour(sign_contours[signIdx]), cv2.CONTOURS_MATCH_I2, 0.0)
            if similarity < bestSim and ((corners[signIdx] <= 0 and approxCorners >= -corners[signIdx]) or abs(corners[signIdx] - approxCorners) <= 2):
                bestSim = similarity
                bestSign = signIdx
        # check if the best match is good enough to justify template matching
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

                margin = 15

                search_top = max(0, int(y - margin))
                search_bot = min(origFrame.shape[1], int(y + height + margin))
                search_left = max(0, int(x - margin))
                search_right = min(origFrame.shape[1], int(x + width + margin))
                search_image_color = origFrame[search_top:search_bot, search_left: search_right]
                search_image = cv2.cvtColor(search_image_color, cv2.COLOR_BGR2GRAY)
                mask = np.ones(search_image.shape).astype(search_image.dtype)
                cv2.fillPoly(mask, [contour], (0, 0, 0), offset=(-search_left, -search_top))
                search_image = cv2.bitwise_or(search_image, mask)

                best_min_val = 1
                best_template = template

                # for some signs orientation matters
                for alpha in range(int(angle), int(360 + angle), int(360 / boundingbox_symmetry[bestSign])):
                    rot = cv2.getRotationMatrix2D(signCenter, alpha, 1)
                    curtemplate = cv2.warpAffine(template, rot, (template.shape[0], template.shape[1]), borderValue=(255, 255, 255))
                    curtemplate = cv2.resize(curtemplate, (int(signWidth * scale), int(signHeight * scale)))

                    res = cv2.matchTemplate(search_image, curtemplate, cv2.TM_SQDIFF_NORMED)
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                    if min_val < best_min_val:
                        best_min_val = min_val
                        best_template = curtemplate

#                print(best_min_val)
                if best_min_val < templateThreshold[bestSign]:
                    if detections[bestSign] is None:
                        speed = 0
                        if bestSign == 3:
                            speed = -1000
                            t = threading.Thread(target=detectSpeed, args=(search_image_color, bestSign))
                            t.start()
                        detections[bestSign] = Detection(contour, origFrame, speed)
                    else:
                        detections[bestSign].found(contour)

    for i, detection in enumerate(detections):
        if detection is None:
            continue
        (success, box) = detection.tracker.update(origFrame)
        if not success:
            detections[i] = None
            continue
        if i == 3 and detection.speed == 0:
            detections[i] = None
            continue
        x, y, w, h = [int(v) for v in box]
        cv2.rectangle(contoursimg, (x, y), (x+w, y+h), highlightColors[i], 2)
        text = names[i]
        if detection.speed == -1000:
            text = "Tempo erkennen..."
        if detection.speed > 0:
            text = "Tempo: " + str(detection.speed)

        cv2.putText(contoursimg, text, (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, highlightColors[i], 2)
#        cv2.drawContours(contoursimg, [detection.contour], -1, highlightColors[i], 2)

    cv2.imshow('contours', contoursimg)

cv2.destroyAllWindows()
cap.release()
