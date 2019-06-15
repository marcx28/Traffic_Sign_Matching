# grab the number of rows and columns from the scores volume, then
import pytesseract
import cv2
from imutils.object_detection import non_max_suppression
import numpy as np
import shutil

tesseractAvailable = True

if shutil.which('tesseract') is None:
    print("Tesseract not installed or not in path, speed recognition won't be available")
    tesseractAvailable = False
else:
    pytesseract.pytesseract.tesseract_cmd = shutil.which('tesseract')


def decode_predictions(scores, geometry):
    min_confidence = 0.5
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []

    for y in range(0, numRows):
        # extract the scores (probabilities), followed by the
        # geometrical data used to derive potential bounding box
        # coordinates that surround text
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        for x in range(0, numCols):
            if scoresData[x] < min_confidence:
                continue

            (offsetX, offsetY) = (x * 4.0, y * 4.0)

            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            # get width and height based on the volume of the result
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            # get text bounding box
            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])

    # return a tuple of the bounding boxes and associated confidences
    return rects, confidences


def get_speed_limit(image):
    if not tesseractAvailable:
        return -1

    orig = image.copy()
    (origH, origW) = image.shape[:2]

    # W and H have to be multiples of 32
    (newW, newH) = (origW - origW % 32, origH - origH % 32)
    rW = origW / float(newW)
    rH = origH / float(newH)

    image = cv2.resize(image, (newW, newH))
    (H, W) = image.shape[:2]

    layer_names = [
        "feature_fusion/Conv_7/Sigmoid",
        "feature_fusion/concat_3"]
    config = '-l eng --oem 1 --psm 7'

    net = cv2.dnn.readNet("./east.pb")
    blob = cv2.dnn.blobFromImage(image, 1.0, (W, H), (123.68, 116.78, 103.94), swapRB=True, crop=False)
    net.setInput(blob)

    (scores, geometry) = net.forward(layer_names)
    (rects, confidences) = decode_predictions(scores, geometry)
    boxes = non_max_suppression(np.array(rects), probs=confidences)

    cv2.imshow("source", image)

    results = []
    for (startX, startY, endX, endY) in boxes:
        # revert stretching in beginning (cause of % 32)
        startX = int(startX * rW)
        startY = int(startY * rH)
        endX = int(endX * rW)
        endY = int(endY * rH)

        padding = 0.01
        dX = int((endX - startX) * padding)
        dY = int((endY - startY) * padding)

        startX = max(0, startX - dX)
        startY = max(0, startY - dY)
        endX = min(origW, endX + (dX * 2))
        endY = min(origH, endY + (dY * 2))

        roi = orig[startY:endY, startX:endX]
        cv2.imshow("text", roi)
        text = pytesseract.image_to_string(roi, config=config)
        results.append(text)

    for result in results:
        result = result.lower()
        result = result.replace("i", "1")
        result = result.replace("l", "1")
        result = result.replace("o", "0")
        try:
            return int(result)
        except:
            continue
    return 0
