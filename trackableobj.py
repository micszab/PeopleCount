#klasa do przechowywania ID sledzonego obiektu,
#poprzednie wspolrzedne centroidy oraz czy byl juz policzony
class TrackableObject():
	def __init__(self, objID, centroid):
		self.objID = objID
		self.centroids = [centroid]
		self.counted = False
