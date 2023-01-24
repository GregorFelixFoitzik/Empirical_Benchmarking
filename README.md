# EmpiricalBenchmarking

### Aufbau des Repository
Für jeden Datensatz, für den ein neuronales Netz entwickelt wurde, existiert ein Ordner. 
In diesem Ordner befindet sich ein zip-Datei, welche die verwendeten csv-Dateien beinhaltet.
Dazu zählen zwei aufbereitete Datensätzen. 
Ein Datensatz für das Trainieren und Hypertuning der Architektur und ein Datensatz, um die Güte des Netzwerks zu evaluieren.

#### Erklärung der eval_ann_(...).py Datein:
Die Datei  wurde für das Hypertuning der Architektur verwendet. Aus dem Code kann entnommen werden, welche Einstellungen und Parameter getestet wurden.
Festgelegte Parameter:
- Criterion: Mean Squared Error
- Optimizer: Adam (Adaptive Moment Estimation)
- Activation function: Rectified Linear Unit (ReLU)
- batch_size = 256
- learning_rate = 0.001
- epochs = 250

Hypertuning der Paramter (Parameterwerte):
- Anzahl der hidden layer (2, 5, 10)
- Anzahl der Neuronen pro hidden layer (#feature*1, #feature*2, #feature*3)

- Dropout layer (True, False)
	- True
		- Dropout nach input layer mit Wahrscheinlichkeit von 20%
		- Dropout nach jedem hidden layer mit Wahrscheinlichkeit von 50%
	- False: Keine dropout layer



Mit der Datei hpt_outcome.py kann die Bestimmung der final verwendeten Konfiguration nachvollzogen werden. 
Zudem werden Bilder generiert, die die logarithmierte Verlustfunktion über die Epochen darstellt.

Die Date final_ann_(...).py beinhaltet die beste Konfiguration unter den getesten. 
	Dazu speichert die Datei das trainierte NN, sodass es unmittelbar eingesetzt werden kann. 


Funktionen für das Einlesen sowie Aufbereiten der Daten befinden sich im Ordner "preparation".
