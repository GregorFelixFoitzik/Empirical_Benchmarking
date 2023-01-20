# EmpiricalBenchmarking

### Aufbau des Repository
Für jeden Datensatz, für den ein neuronales Netz entwickelt wurde, existiert ein Ordner. 
In diesem Ordner befindet sich ein zip-Datei, welche die verwendeten csv-Dateien beinhaltet.
Dazu zählen zwei aufbereitete Datensätzen. 
Ein Datensatz für das Trainieren und Bestimmen der Architektur und ein Datensatz, um die Güte des Netzwerks zu evaluieren.

Die Datei eval_ann_(...).py beinhaltet die alle Konfigurationen, die getestet wurden.

Mit der Datei hpt_outcome.py kann die Bestimmung der final verwendeten Konfiguration nachvollzogen werden. 
Zudem werden Bilder generiert, die die logarithmierte Verlustfunktion über die Epochen darstellt.

Die Date final_ann_(...).py beinhaltet die beste Konfiguration unter den getesten. 
	Dazu speichert die Datei das trainierte NN, sodass es unmittelbar eingesetzt werden kann. 


Funktionen für das Einlesen sowie Aufbereiten der Daten befinden sich im Ordner "preparation".
