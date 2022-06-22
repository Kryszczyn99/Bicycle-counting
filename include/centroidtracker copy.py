from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np


class CentroidTracker:
    def __init__(self, maxDisappeared=50, maxDistance=50):
        # inicjalizacja kolejnego unikalnego identyfikatora
        # oobiektu z dwoma słownikami
        # używanymi do ledzenia mapowania danego obiektu
        # Identyfikator jego srodka ciezkosci i liczba kolejnych klatek
        # oznaczenie "zniknął"
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()

        # zapisuje liczbę max kolejnych klatek danego obiektu
        # dozwolone jest oznaczenie obiektu jako "zniknął",
        # dopóki nie skończymy
        # (wyrejestrujemy obiekt ze sledzenia)

        self.maxDisappeared = maxDisappeared

        # zapisuje maksymalna odleglosc miedzy centroidami
        # do skojarzenia obiektu
        # jesli odleglosc ta jest wieksza niz to maksimum, bedziemy oznaczac
        # obiekt jako "zniknął"
        self.maxDistance = maxDistance

    def register(self, centroid):
        # przy rejestracji obiektu używamy nastepnego dostepnego ID obiektu
        # w ktorym przechowujemy centroid
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        # aby wyrejestrowac ID obiektu usuwamy ID
        # obiektu z obu naszych slownikow
        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self, rects):
        # sprawdz czy lista "bouding boxow" jest pusta
        if len(rects) == 0:
            # petla dla istniejacych sledzonych obiektow
            # i oznaczenie ich jako "zniknal"
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                # jesli osiagnelismy maksimum liczbę kolejnych
                # klatek w ktorych dany obiekt
                # zostal oznaczony jako "zniknal", wyrejestrujemy go
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)

            # zwracamy wczesniej bo nie ma centroidow lub informacji
            # o sledzeniu do aktualizacji
            return self.objects

        # inicjalizacja tablicy wejsciowych centroidow dla obecnej klatki
        inputCentroids = np.zeros((len(rects), 2), dtype="int")

        # petla dla "bounding boxow"
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            # uzyj wspolrzednych "bounding boxow" aby uzyskac centroid
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)
        # jesli obecnie nie sledzimy zadnych obiektow, wez input centroidow
        # i zarejestruj kazde z nich
        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i])
        # inaczej, jesli obecnie sledzimy jakies obiekty
        # to musimy sprobowac doposowac
        # wejsciowe centroidy do centroidow istniejacego obiektu
        else:
            # złap zestaw ID obiektow i odpowiadajacym im centroidy
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())

            # oblicz odleglosc miedzy kazda para centroidów
            # obiektu i centroidów werjsciowych
            # naszym celem bedzie dopasowanie centroidu werjsciowego
            # do istniejacego centroidu
            D = dist.cdist(np.array(objectCentroids), inputCentroids)

            # w celu wykonania tego doposowania musimy znalezc
            # najmniejsza wartosc w kazdym
            # rzedzie a nastepnie posortujemy wiersze
            # inedksowane na podstawie ich
            # minimalnych wartosci aby wiersz o najmniejszej
            # wartosci byl na przodzie listy
            rows = D.min(axis=1).argsort()

            # nastepnie wykonujemy podobny proces na kolumnach
            # przez znajdowanie najmniejszej
            # wartosci w kazdej kolumnie a nastepnie
            # sortujemy przy uzyciu wczesniej
            # obliczonej listy indeksow wierszy
            cols = D.argmin(axis=1)[rows]

            # w celu ustalenia czy musimy zaktualizowac,
            # zarejestrowac lub wyrejestrowac obiekt
            # ktory musimy sledzic z indeksow wieszy
            # i kolumn, ktore juz sprawdzilismy
            usedRows = set()
            usedCols = set()

            # zapetlic kombinacje indeksu(wiersz,kolumna),tuples
            for (row, col) in zip(rows, cols):
                # jesli juz sprawdzilismy wiersz lub kolumne (wartosc)
                # wczesniej to ignorujemy to
                if row in usedRows or col in usedCols:
                    continue

                # jesli dystans miedzy centroidami jest wiekszy
                # niz maksymalna odleglosc.
                # nie łącz dwóch centroidow do tego samego obiektu
                if D[row, col] > self.maxDistance:
                    continue

                # w przeciwnym razie, wez ID obiektu dla biezacego wiersza
                # ustaw nowy centroid i zresetuj licznik "zniknietych"
                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0

                # zwieksz ilosc zbadanych kolum i wierszy
                usedRows.add(row)
                usedCols.add(col)

            # ustaw kolumny i wiersze ktore nie sa jeszcze zbadane
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            # gdy liczba zarejestrowanych objektow "centroid"
            # jest >= liczbie wprowadzonych objektow musimy
            # sprawdzic czy jakies nie zniknely
            if D.shape[0] >= D.shape[1]:
                # przejdz po nieuzytych wierszach
                for row in unusedRows:
                    # pobierz ID objektu w danym indeksie wiersza
                    # i zwieksz licznik znikniec
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1

                    # sprawdz czy objekt zostal oznaczony jako znikniety
                    # i odrejestruj objekt
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)

            # w innym wypadku kiedy liczba wprowadzonych punktow
            # jest wieksza od liczby istniejacych punktow
            # musimy zarejestrowac kazdy nowy punkt jako
            # sledzony objekt
            else:
                for col in unusedCols:
                    self.register(inputCentroids[col])

        # zwroc zestaw sledzonych objektow
        return self.objects
