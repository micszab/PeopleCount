from scipy.spatial import distance
#korzystamy z orderdict poniewaz utrzymuje ona kolejnosc wstawiania elementow
from collections import OrderedDict
import numpy as np


#Konstruktor akceptuje jeden parametr,
#maksymalna liczba klatek przez ktory obiekt jest zgubiony musi minac aby,
#zostal usuniety z trackera
class CentroidTracker():
	def __init__(self, maxDisappeared=50, maxDistance=50):
		self.nextID = 0
		#przechowuje ID jako klucz oraz wspolrzedne centroidy
		self.obj = OrderedDict()
		#przechowuje ID jako klucz oraz liczbe klatek przez ktore ID jest zgubione
		self.disappeared = OrderedDict()
		#liczba klatek przez ktory obiekt moze zostac zgubiony przed wyrejestrowaniem
		self.maxDisappeared = maxDisappeared
		#jezeli przekroczy jej wartosc bedzie jako disappeared
		self.maxDistance = maxDistance

        #dodaje do slownika objekt uzywajac kolejnego dostepnego ID
	def register(self, centroid):
		self.obj[self.nextID] = centroid
		self.disappeared[self.nextID] = 0
		self.nextID += 1

        #usuwa ze slownika objekt oraz jego zmienna disappeared 
	def deregister(self, ID):
		del self.obj[ID]
		del self.disappeared[ID]

        #jako parametr podajemy wymiary w formacie(startX, startY, endX, endY)
	def update(self, rects):
		# jezeli parametr jest pusty(brak wykrytych objektow) to:
		if len(rects) == 0:
			#oznacza wszystkie sledzony obiekty jako disappeared
			for ID in list(self.disappeared.keys()):
				self.disappeared[ID] += 1

				# jezeli zostala osiagnieta maksymalna liczba klatek
				# przez ktore obiekt byl zaginiony, wyrejestruj go
				if self.disappeared[ID] > self.maxDisappeared:
					self.deregister(ID)

			return self.obj

		# stworz wektor do przechowywania nowych wspolrzednych
		newCentroids = np.zeros((len(rects), 2), dtype="int")

		# petla po wspolrzednych obwiedni
		for (i, (startX, startY, endX, endY)) in enumerate(rects):
			# wyznaczamy srodek symetrii
			x = int((startX + endX) / 2)
			y = int((startY + endY) / 2)
			newCentroids[i] = (x, y)

		# jezeli nie sledzimy zadnych obiektow zarejestruj kazdy nowy obiekt
		if len(self.obj) == 0:
			for i in range(0, len(newCentroids)):
				self.register(newCentroids[i])

		# w przeciwnym razie musimy zaktualizowac koordynaty objektu
		else:
			# zaladuj aktualnie sledzone obiekty i ich wartosci centroid do nowych zmiennych
			objIDs = list(self.obj.keys())
			objCentroids = list(self.obj.values())

			# oblicz odleglosc miedzy kazda para istniejacych centroid a nowymi
			# output - numpy array
			D = distance.cdist(np.array(objCentroids), newCentroids, 'euclidean')

			# w celu dopasowania znajdujemy najmniejsza wartość każdego wiersza
			# oraz sortujemy od najmniejszej do najwiekszej
			rows = D.min(axis=1).argsort()

			# wykonujemy podobny proces na kolumnach, sortujac jest na podstawie wierszy
			cols = D.argmin(axis=1)[rows]

			# aby ustalic co mamy zrobic z obiektem tworzymy dwa zestawy
			# w celu okreslenia ktore kolumny i wiersze wykorzystalismy
			# set - podobne do listy ale tylko unikalne wartosci
			usedRows = set()
			usedCols = set()

			# petla po kombinacjach (row , col)
			# zip - mapowanie
			for (row, col) in zip(rows, cols):
				# jezeli indeks wiersza lub kolumny zostal uzyty kontynuuj
				if row in usedRows or col in usedCols:
					continue

				if D[row, col] > self.maxDistance:
					continue

				# w przeciwnym razie znalezlismy nowa centroide, 
				# ktora ma najmniejsza odleglosc euklidesowa od istniejacej i
				# nie zostala dopasowana do innego obiektu
				# aktualizujemy nasze slowniki
				objID = objIDs[row]
				self.obj[objID] = newCentroids[col]
				self.disappeared[objID] = 0

				# dodajemy wiersz i kolumne do odpowiednich zestawow
				usedRows.add(row)
				usedCols.add(col)

			# zestawy okreslajace ktorych indeksow wiersz i kolumn jeszcze nie uzylismy
			unusedRows = set(range(0, D.shape[0])).difference(usedRows)
			unusedCols = set(range(0, D.shape[1])).difference(usedCols)

			# sprawdzamy czy jakis obiekt sie nie zagubil lub zniknal
			# jezeli liczba centroid jest wieksza lub rowna liczbie nowych centroid to
			if D.shape[0] >= D.shape[1]:
				# petla po nieuzytych wierszach
				for row in unusedRows:
					# zwieksz wartosc disappeared dla ID o 1
					objID = objIDs[row]
					self.disappeared[objID] += 1

					# jezeli wartosc disappeared przekroczyla wartosc maksymalna, wyrejestruj
					if self.disappeared[objID] > self.maxDisappeared:
						self.deregister(objID)

			# w przeciwnym razie mamy nowe obiekty do zarejestrowania i sledzenia
			else:
				for col in unusedCols:
					self.register(newCentroids[col])

		# zwroc obiekty do sledzenia
		return self.obj
