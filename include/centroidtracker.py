from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np


class CentroidTracker:
    def __init__(self, maxDisappeared=50, maxDistance=50):
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()

        self.maxDisappeared = maxDisappeared
        self.maxDistance = maxDistance

    def register(self, centroid):
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]

    """Funkcja update przyjmuje liste prostokatow (bounding box)
    (StartX, StartY, EndX, EndY)"""

    def update(self, rects):

        # jesli lista "rects" jest pusta zwiekszamy licznik "disappeared"
        # i jesli przekroczy on maksymalna wartosc kasujemy centroid
        if len(rects) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)

            return self.objects

        # obliczamy pozycje centroidow i zapisujemy je do "inputCentroids"
        inputCentroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)

        # Jeśli nie ma żadnych centroidów w (self.object)
        # zarejestruj nowe centroidy z (inputCentroids)
        if len(self.objects) == 0:
            for inputCentroid in inputCentroids:
                self.register(inputCentroid)

        # Jesli centroidy w (self.object) istnieja porównujemy
        # pozycję nowych i starych centroidów za pomocą 
        # obliczania odległości euklidesowych
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())

            # obliczamy odleglosci miedzy zarejestrowanymi
            # centroidami (objectCentroids) a
            # nowymi centroidami (inputCentroids)
            D = dist.cdist(np.array(objectCentroids), inputCentroids)

            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            usedRows = set()
            usedCols = set()

            # dopasowujemy do istniejących centroidów (self.object)
            # nowe centroidy (inputCentroids) które są w najmniejszej
            # odległości od istniejących centroidów
            for (row, col) in zip(rows, cols):

                if row in usedRows or col in usedCols:
                    continue

                if D[row, col] > self.maxDistance:
                    continue

                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0

                usedRows.add(row)
                usedCols.add(col)

            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            # Jeśli jest więcej (self.object) niż (inputCentroids)
            # zwiekszamy licznik "disappeared"
            # i jesli przekroczy on maksymalna wartosc kasujemy centroid
            if D.shape[0] >= D.shape[1]:
                for row in unusedRows:
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1

                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)

            # W przeciwnym wypadku musimy zarejestrować nowe centroidy
            else:
                for col in unusedCols:
                    self.register(inputCentroids[col])

        return self.objects
