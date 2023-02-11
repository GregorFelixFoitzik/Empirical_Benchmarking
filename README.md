# EmpiricalBenchmarking 

## Aufbau des Repository
	Für jeden Datensatz existiert ein Ordner. (bspw.: Clasp-Rooks) In diesen Ordner befinden zwei Sub-Ordner. 
		1. Daten
			- helper_{algoritm}.csv
				- vereinfacht das Einlesen der Daten
		2. Models
			- trainierte neuronale Netzwerke als .pth

	In der Datei neural_network.py sind die Funktionen für die neuronalen Netzwerke, den Trainingprozess und die Validierung enthalten.

	Mit Hilfe der generate_graphics.py wurden die Abbildungen für die Bachelorarbeit generiert.

	Mit Hilfe der calcualte_metrics.py wurden die Metriken der trainierten neuronalen Netzwerk berechnet. 

	In dem Ordner graphics sind alle Abbildungen, die in der Bachelorarbeit verwendet wurden abgespeichert.

	In dem Ordner preparation befinden sich Python-Dateien, die für das Einlesen der JSON-Dateien verwendet werden. 



### Erklärung der eval_ann_{algoritm}.py Dateien:
Die Datei wurde für das Hypertuning der Architektur verwendet. Aus dem Code kann entnommen werden, welche Einstellungen und Parameter getestet wurden.
Für alle getesteten Konfigurationen wurden die Werte der Verlustfunktion des Training- und Testdatensatz in jeder Epoche in die Datei HPTuning.csv abgespeichert.
(nicht für den CPLEX-Regions, siehe Bachelorarbeit )

Vordefinierte Einstellungen:
- Criterion: Mean Squared Error
- Optimizer: Adam (Adaptive Moment Estimation)
- Activation function: Rectified Linear Unit (ReLU)
- batch_size = 256
- learning_rate = 0.001
- epochs = 200

### Erklärung der final_nn_{algoritm}.py Dateien:
Mit Hilfe dieser Dateien wurden 10 neuronale Netzwerke mit einem random_state trainiert. 
Die trainierten Modell werden in dem Ordner Models abgespeichert. 

### Beschreibung hpt_outcome.py
Diese Datei greift auf die abgespeicherten Informationen aus dem Evaluationsprozess zu und sortiert die Konfiguration aufsteigend nach dem Wert der Verlustfunktion des Testdatensatzes.
Dadurch können einfach und schnell die besten Konfigurationen nachgesehen werden.

