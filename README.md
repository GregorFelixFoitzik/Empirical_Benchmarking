# EmpiricalBenchmarking

## Aufbau des Repository
Für jeden Datensatz, für den ein neuronales Netz entwickelt wurde, existiert ein Ordner. 
In diesem Ordner befindet sich ein zip-Datei, welche die verwendeten csv-Dateien beinhaltet.
Dazu zählen zwei aufbereitete Datensätzen. 
Ein Datensatz für das Trainieren und Hypertuning der Architektur (data_train.csv) und ein Datensatz, um die Güte des Netzwerks zu evaluieren (data_val.csv).


### Erklärung der eval_ann_(...).py Dateien und Hyperparameter Tuning:
Die Datei  wurde für das Hypertuning der Architektur verwendet. Aus dem Code kann entnommen werden, welche Einstellungen und Parameter getestet wurden.
Festgelegte Parameter:
- Criterion: Mean Squared Error
- Optimizer: Adam (Adaptive Moment Estimation)
- Activation function: Rectified Linear Unit (ReLU)
- batch_size = 256
- learning_rate = 0.001
- epochs = 250


Hypertuning der Paramter (Parameterwerte):
1. Anzahl der hidden layer (2, 5, 10)

2. Anzahl der Neuronen N im ersten hidden layer (#feature*1, #feature*2, #feature*3)

3. Aufbau weiterer hidden layer:
	- architec_equal
		- N Neuronen in jedem hidden layer 
	- architec_decre
		- für jedes hidden layer gilt: N * 1/i
			- i := eins indiziertes hidden layer
				- für das erste hidden layer gilt i=1; für das zweite i=2; ...

4. Dropout layer (True, False)
	- True
		- Dropout nach input layer mit Wahrscheinlichkeit von 20%
		- Dropout nach jedem hidden layer mit Wahrscheinlichkeit von 50%
	- False: Keine dropout layer


### Erklärung der final_nn_(...).py Dateien und Bestimmung der finalen Konfiguration:
Die berechneten Verlustfunktionen und RMSE logPR10 runtimes für jede getesete Architektur und jede Epoche wurden in einer .csv-Datei gespeichert (hpt.csv), welche sich ebenfalls im zip-Ordner befindet.

In den Dateien final_nn_(...).py wurden mit den ausgewählten Parametern die neuronalen Netze trainiert und in .pth-Dateien abgespeichert.


### Grafiken generieren
Mit der Datei hpt_outcome.py kann die Bestimmung der final verwendeten Konfiguration nachvollzogen werden. 
Zudem werden zwei Grafiken generiert, welche die logarithmierte Verlustfunktion über die Epochen darstellt.

Um die Grafiken zu generieren müssen die zip-Ordner entpackt werden. 
Nur wenn der entpackte Ordner nicht in dem Ordner des Algorithmus abgelegt wird muss der Pfad der Datei angepasst werden.


###
Funktionen für das Einlesen sowie Aufbereiten der Daten befinden sich im Ordner "preparation".
