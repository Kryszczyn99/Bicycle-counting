
from include.centroidtracker import CentroidTracker
from include.trackableobject import TrackableObject
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

# Ladowanie modelu z dysku
net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)

vs = cv2.VideoCapture(INPUT)

# inicjalizowanie wysokosci i szerokosci klatki
W = None
H = None

# inicjalizowanie zmieniej klasy centroid tracker, nastepnie
# inicjalizacja listy do przechowywania "correlation trackers"
# i stworzenie slownika przechowojacego kazdy sledzony objekt z wlasnym ID
ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
trackers = []
trackableObjects = {}

# inicjalizacja zmiennej zliczajacej calkowita ilosc klatek
# oraz zmiennej zliczajacej ilosc rowerow
totalFrames = 0
counter = 0

# przechodzimy po kazdej klatce filmu
while True:
    # pobierz kolejna klatke filmu
    frame = vs.read()
    frame = frame[1]

    # jesli nie udalo sie pobrac klatki to film sie skonczyl
    if frame is None:
        break

    # zmienic rozmiar klatki to maksymalnej szerokosci 500 pikseli
    # i konwertuj z BGR do RGM dla dlib
    frame = imutils.resize(frame, width=500)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # jezeli wysokosc i szerokosc klatki jest pusta przypisz im wartosci
    if W is None or H is None:
        (H, W) = frame.shape[:2]

    # inicjalizuj liste "bounding box'ow" ktore beda zwracane
    # przez "object detector" lub "correlation trackers"
    rects = []

    # sprawdz czy musimy uzyc dokladniejszej metody
    # mtody detekcji objektow
    if totalFrames % SKIP_FRAMES == 0:
        # inicjalizuj zestaw "object tracker'ow"
        trackers = []

        # konwertuj klatke do "blob'a" i przekarz ja do sizeci by
        # otrzymać wykryte objekty
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
        net.setInput(blob)
        detections = net.forward()

        # przejdz po wszystkich wykrytych objektach
        for i in np.arange(0, detections.shape[2]):
            # pobierz  prawdopodobienstwo poprawnego wykrycia objektu
            confidence = detections[0, 0, i, 2]

            # jesli prawdopodobienstwo wykrycia jest za male odrzuc objekt
            if confidence > CONFIDENCE:
                # pobierz nazwe klasy z listy wykrytych objektow
                idx = int(detections[0, 0, i, 1])

                # jesli nazwa klasy nie jest rowerem pomin
                if CLASSES[idx] != "bicycle":
                    continue

                # oblicz wspolrzedne (x, y) dla "bounding box'a" la objektu
                box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                (startX, startY, endX, endY) = box.astype("int")

                # stworz prostokatny objekt "dlib" z wspolrzednych
                # "bounding box'a" i uruchom "dlib correlation tracker"
                tracker = dlib.correlation_tracker()
                rect = dlib.rectangle(startX, startY, endX, endY)
                tracker.start_track(rgb, rect)

                # dodaj "tracker" do listy "tracker'ow" by go uzyc
                # kiedy bedziemy pomijac klatki
                trackers.append(tracker)

    # w innym przypadku pomnismy uzyc sledzenia objektow
    else:

        # przejdz po wszystkich "trackers"
        for tracker in trackers:
            # zaktualizuj "tracker" i pobierz jego pozycje
            tracker.update(rgb)
            pos = tracker.get_position()

            # rozpakuj objekt przechowujacy pozycje
            startX = int(pos.left())
            startY = int(pos.top())
            endX = int(pos.right())
            endY = int(pos.bottom())

            # dodaj wspolrzedne "bounding box'u" do listy prostokatow
            rects.append((startX, startY, endX, endY))

    # uzyj "centroid tracker" zeby powiazac stare "object centroids"
    # z nowym "object centroids"
    objects = ct.update(rects)

    # przejdz po sledzonych objektach
    for (objectID, centroid) in objects.items():

        # sprawdz czy dla ID obecnego objektu istnieje już jakiś
        # sledzony objekt
        to = trackableObjects.get(objectID, None)

        # jesli nie istnieje sledzony objekt to go tworzymy
        if to is None:
            to = TrackableObject(objectID, centroid)

        # w innym wypadku, sledzony objekt istnieje
        # wiec uzywamy go by wykryc kierunek ruchu
        else:

            to.centroids.append(centroid)

            # sprawdz czy objekt nie zostal wczesniej zliczony
            if not to.counted:
                counter += 1
                to.counted = True

        # przechowaj sledzony objekt w slowniku
        trackableObjects[objectID] = to

    text = "Counter: {}".format(counter)
    cv2.putText(frame, text, (10, H - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # wyswietl klatke
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # po wcisnieciu 'q' program zostaje zakonczony
    if key == ord("q"):
        break

    # zwieksz calkowita liczbe klatek
    totalFrames += 1

# zwolni wskaznik na film
vs.release()

# zamknij wszystkie okna
cv2.destroyAllWindows()
