import cv2

capture = cv2.VideoCapture(0)
print("hello world")
while cv2.waitKey(1) != ord('q'):
    ret, frame = capture.read()
    cv2.imshow('frame', frame)

