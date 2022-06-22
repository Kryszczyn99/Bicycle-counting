from include.centroidtracker import CentroidTracker
from include.trackableobject import TrackableObject
from imageai.Detection import ObjectDetection
import os
import numpy as np
import imutils
import dlib
import cv2

PROTOTXT = 'bicycle_counting/mobilenet_ssd/MobileNetSSD_deploy.prototxt'
MODEL = 'bicycle_counting/mobilenet_ssd/MobileNetSSD_deploy.caffemodel'
INPUT = 'bicycle_counting/videos/video_1.mp4'
CONFIDENCE = 0.4
SKIP_FRAMES = 30

# inicjowanie listy nazw klas ktore MobileNet SSD wykrywa
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]


net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)
vs = cv2.VideoCapture(INPUT)

W = None
H = None

ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
trackers = []
trackableObjects = {}

totalFrames = 0
counter = 0

# przechodzimy po kazdej klatce filmu
while True:
    frame = vs.read()
    frame = frame[1]

    if frame is None:
        break

    frame = imutils.resize(frame, width=500)

    if W is None or H is None:
        (H, W) = frame.shape[:2]

    rects = []

    if totalFrames % SKIP_FRAMES == 0:
        trackers = []

        blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
        net.setInput(blob)
        detections = net.forward()

        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > CONFIDENCE:
                index = int(detections[0, 0, i, 1])

                if CLASSES[index] != "bicycle":
                    continue

                startX = int(detections[0, 0, i, 3] * W)
                startY = int(detections[0, 0, i, 4] * H)
                endX = int(detections[0, 0, i, 5] * W)
                endY = int(detections[0, 0, i, 6] * W)

                tracker = dlib.correlation_tracker()
                rect = dlib.rectangle(startX, startY, endX, endY)
                tracker.start_track(frame, rect)

                trackers.append(tracker)

    else:

        for tracker in trackers:

            tracker.update(frame)
            pos = tracker.get_position()

            startX = int(pos.left())
            startY = int(pos.top())
            endX = int(pos.right())
            endY = int(pos.bottom())

            rects.append((startX, startY, endX, endY))

    # uzyj "centroid tracker" zeby powiazac stare "object centroids"
    # z nowym "object centroids"
    objects = ct.update(rects)

    # przejdz po sledzonych objektach
    for (objectID, centroid) in objects.items():

        to = trackableObjects.get(objectID, None)

        if to is None:
            to = TrackableObject(objectID, centroid)
        else:
            to.centroids.append(centroid)

            if not to.counted:
                counter += 1
                to.counted = True

        trackableObjects[objectID] = to

    text = "Counter: {}".format(counter)
    cv2.putText(frame, text, (10, H - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    totalFrames += 1

vs.release()
cv2.destroyAllWindows()
