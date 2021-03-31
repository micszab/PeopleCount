#!/usr/bin/python

# uzycie...
# live kamera z urzadzenia
#peoplecounter.py

# zewnetrzne wideo oraz zapis
#peoplecounter.py -i videos/example_01.mp4 -o output/output_01.avi

# przydatne biblioteki
from centroidtrack import CentroidTracker
from trackableobj import TrackableObject
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import sys

#obsluga
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str,
	help="sciezka do wejsciowego pliku wideo")
ap.add_argument("-o", "--output", type=str,
	help="sciezka do wyjsciowego pliku wideo")
ap.add_argument("-c", "--confidence", type=float, default=0.4,
	help="minimalne prawdopodobienstwo do odfiltrowania slabych detekcji")
ap.add_argument("-s", "--skip-frames", type=int, default=30,
	help="# liczba pominietych klatek pomiedzy detekcja")
args = vars(ap.parse_args())

# inicjalizacja listy etykiet klas, ktore MobileNet SSD bylo wytrenowane by wykrywac
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]

# zaladuj model
net = cv2.dnn.readNetFromCaffe("mobilenet_ssd/MobileNetSSD_deploy.prototxt", "mobilenet_ssd/MobileNetSSD_deploy.caffemodel")

# jezeli nie wprowadzono sciezki do video, uzywaj kamerki internetowej
if not args.get("input", False):
	print("[INFO] starting video stream...")
	vs = VideoStream(src=0).start()
	time.sleep(2.0)

# w przeciwnym razie uzywaj video
else:
	print("[INFO] opening video file...")
	vs = cv2.VideoCapture(args["input"])

# inicjalizacja zapisu video
writer = None

# inicjalizacja wymiarow video
W = None
H = None

# utworzenie instacji modulu sledzacego centroidy 
ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
# lista do przechowywania modulow sledzenia korelacji dlib
trackers = []
# slownik mapujacy ID obiektu do TrackableObject
trackableObjects = {}

# Liczba przetworzonych klatek
totalFrames = 0
# Zmienne przechowujace liczbe osob przekraczajacych linie
totalDown = 0
totalUp = 0

# rozpocznij zliczanie klatek na sekunde
fps = FPS().start()

# petla po klatkach video
while True:
	# czyta kolejna klatke z filmu lub kamerki
	frame = vs.read()
	frame = frame[1] if args.get("input", False) else frame

	# jezeli czytamy z video i nie przechwycilismy kolejnej ramki
	# wyjdz z petli - koniec filmu
	if args["input"] is not None and frame is None:
		break

        # zmien szerokosc klatki tak aby miala maksymalna szerokosc 500 pixeli
	frame = imutils.resize(frame, width=500)
	# zmien kanal obrazu z BGR na RGB
	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

	# pobiera rozmiary klatki video
	if W is None or H is None:
		(H, W) = frame.shape[:2]

	# jezeli mamy zapisywac plik wynikowy na dysk inicjulizuj zapis video
	if args["output"] is not None and writer is None:
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"], fourcc, 30,
			(W, H), True)

        # ustaw status jako waiting
	status = "Waiting"
	# lista ta zostanie wypelniona poprzez wykrywanie lub sledzenie
	rects = []

	# rozpoznawania obiektow w kazdej klatce jest bardzo kosztowne obliczeniowo
	# dlatego bedziemy rozpoznawac tylko co n klatke 
	if totalFrames % args["skip_frames"] == 0:
		# zmien status na detecting
		status = "Detecting"
		trackers = []

		# stworz blob z ramki
		# blob - 4 wymiarowa tablica przechowywana w sposob ciagly
		# zapewnia synchronizacje miedzy CPU i GPU
		blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
		# nastepnie przepusc przez siec w celu wykrycia obiektow
		net.setInput(blob)
		detections = net.forward()

		# petla po wykryciach
		for i in np.arange(0, detections.shape[2]):
			# wyciagnij prawdopodobienstwo rozpoznania obiektu
			confidence = detections[0, 0, i, 2]

			# odfiltruj slabe rozpoznania
			if confidence > args["confidence"]:
				# wyciagnij indeks etykiety klasy
				idx = int(detections[0, 0, i, 1])

				# jezeli etykieta klasy jest inna niz person, zignoruj
				if CLASSES[idx] != "person":
					continue

				# obliczanie koordynatow obwiedni
				box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
				(startX, startY, endX, endY) = box.astype("int")

				# tworzymy instancje naszego modulu sledzenia
				tracker = dlib.correlation_tracker()
				# przekazujemy wspolrzedne obwiedni obiektu do dlib.rectangle
				rect = dlib.rectangle(startX, startY, endX, endY)
				# uruchamiamy modul
				tracker.start_track(rgb, rect)

				# dodaj tracker do naszej listy
				trackers.append(tracker)

	# w przeciwnym razie bedziemy uzywac naszych wykrytych obiektow(trackers)
	# zamiast stosowac wykrywanie co zwiekszy wydajnosc
	else:
		# petla po wykrytych obiektach
		for tracker in trackers:
			# status ustawiamy na tracking
			status = "Tracking"

			# aktualizuj nasz tracker i zapisz aktualizowana pozycje
			tracker.update(rgb)
			pos = tracker.get_position()

			# wyodrebnij wspolrzedne
			startX = int(pos.left())
			startY = int(pos.top())
			endX = int(pos.right())
			endY = int(pos.bottom())

			# dodaj wspolrzedne obwiedni do listy prostokatow(rects)
			rects.append((startX, startY, endX, endY))

	# narysuj linie, gdy obiekt ja przekroczy ustalamy czy
	# poruszal sie w gory czy w dol
	cv2.line(frame, (0, H // 2), (W, H // 2), (0, 255, 255), 2)

	# skojarz stary obiekt z nowo obliczonymi centroidami obiektu
	objects = ct.update(rects)

	# petla po sledzonych obiektach - sprawdza czy obiekt porusza sie w dol czy w gore
	for (objectID, centroid) in objects.items():
		# sprawdz czy istnieje mozliwy do sledzenia obiekt dla biezacego ID
		to = trackableObjects.get(objectID, None)

		# jezeli nie ma istniejacego obiektu mozliwego do sledzenia, utworz go
		if to is None:
			to = TrackableObject(objectID, centroid)

		# w przeciwnym razie jest on juz utworzony i musimy dowiedziec sie gdzie sie porusza
		else:
			# roznica wspolrzednych "y" poprzednich centroidow pokaze
			# nam w ktorym kierunku porusza sie obiekt
			# ujemny dla w gore, dodatni dla w dol
			y = [c[1] for c in to.centroids]
			direction = centroid[1] - np.mean(y)
			to.centroids.append(centroid)

			# sprawdz czy obiekt byl juz policzony
			if not to.counted:
				# jezeli kierunek jest negatywny(ujemny) i centroida jest ponad linia
				# w takim razie dodajemy do totalUp
				if direction < 0 and centroid[1] < H // 2:
					totalUp += 1
					to.counted = True

                                # jezeli kierunek jest dodatni i centroida jest ponizej linii
				# w takim razie dodajemy do totalDown
				elif direction > 0 and centroid[1] > H // 2:
					totalDown += 1
					to.counted = True

		# przechowujemy sledzony obiekt w slowniku
		trackableObjects[objectID] = to

		# wyswietl ID obiektu oraz centroide
		text = "ID {}".format(objectID)
		cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
		cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

	# stworz tuple ktora bedziemy wyswietlac
	info = [
		("Up", totalUp),
		("Down", totalDown),
		("Status", status),
	]

	# petla po utworzonej tuple w celu wyswietlenia informacji
	for (i, (k, v)) in enumerate(info):
		text = "{}: {}".format(k, v)
		cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
			cv2.FONT_HERSHEY_PLAIN, 1.2, (255, 0, 0), 2)

	# sprawdz czy mamy zapisywac video na dysk
	if writer is not None:
		writer.write(frame)

	# pokaz video
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# jezeli wcisnieto "q" zakoncz
	if key == ord("q"):
		break

	# zwieksz liczbe przetworzonych ramek i zaktualizuj fps
	totalFrames += 1
	fps.update()

# zatrzymaj timer i pokaz informacje o fps
fps.stop()
print("Elapsed time: {:.2f}".format(fps.elapsed()))
print("Approximate FPS: {:.2f}".format(fps.fps()))

# sprawdz czy musimy zwolnic zapis video
if writer is not None:
	writer.release()

# jezeli nie uzywamy video zatrzymaj przechwytywanie obrazu z kamerki
if not args.get("input", False):
	vs.stop()

# w przeciwnym razie zwolnij zapisany plik video
else:
	vs.release()

# zamknij wszystkie otwarte okna aplikacji
cv2.destroyAllWindows()

sys.exit()
