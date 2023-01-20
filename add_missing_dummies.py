import pandas as pd
import numpy as np

def fill_missing_dummies(d1, d2, d3, path, helper):
    for data, purpose in zip([d1, d2, d3], ["data_train", "data_traintest", "data_val"]):
        dummies = [col for col in data.columns if col not in helper["Spalte"] and col != "time"]
        d1[dummies] = data[dummies].fillna(0)
        d2[dummies] = data[dummies].fillna(0)
        d3[dummies] = data[dummies].fillna(0)

        data[dummies] = data[dummies].fillna(0)
        data.to_csv(f"{path}\\{purpose}.csv", sep=";", index=False)
        print(f"{purpose}: {data.isna().sum().sum()} & shape: {data.shape}")


###### minisat
#path = "D:\\_Bachelorarbeit_Daten_\\minisat_randomk3"
#helper = pd.read_csv(f"{path}\\helper_minisat.csv", sep=";")
#train     = pd.read_csv("D:\\_Bachelorarbeit_Daten_\\minisat_randomk3\\data_train.csv", sep=";")
#traintest = pd.read_csv("D:\\_Bachelorarbeit_Daten_\\minisat_randomk3\\data_traintest.csv", sep=";")
#val       = pd.read_csv("D:\\_Bachelorarbeit_Daten_\\minisat_randomk3\\data_val.csv", sep=";")
#fill_missing_dummies(train, traintest, val, path, helper)


###### probsat
path = "./ProbSAT-7SAT/Daten"
helper = pd.read_csv(f"{path}/helper_probsat.csv", sep=";")
train     = pd.read_csv("./ProbSAT-7SAT/Daten/data_train.csv", sep=";")
traintest = pd.read_csv("./ProbSAT-7SAT/Daten/data_traintest.csv", sep=";")
val       = pd.read_csv("./ProbSAT-7SAT/Daten/data_val.csv", sep=";")
fill_missing_dummies(train, traintest, val, path, helper)

###### claps-rooks
#path = "D:\\_Bachelorarbeit_Daten_\\clasp-rooks"
#helper = pd.read_csv(f"{path}\\helper_clasp-rooks.csv", sep=";")
#train     = pd.read_csv("D:\\_Bachelorarbeit_Daten_\\clasp-rooks\\data_train.csv", sep=";")
#traintest = pd.read_csv("D:\\_Bachelorarbeit_Daten_\\clasp-rooks\\data_traintest.csv", sep=";")
#val       = pd.read_csv("D:\\_Bachelorarbeit_Daten_\\clasp-rooks\\data_val.csv", sep=";")
#fill_missing_dummies(train, traintest, val, path, helper)

##### lingeling
#path = "D:\\_Bachelorarbeit_Daten_\\lingeling_circuitfuzz"
#helper = pd.read_csv(f"{path}\\helper_lingeling_circuitfuzz.csv", sep=";")
#train     = pd.read_csv("D:\\_Bachelorarbeit_Daten_\\lingeling_circuitfuzz\\data_train.csv", sep=";")
#traintest = pd.read_csv("D:\\_Bachelorarbeit_Daten_\\lingeling_circuitfuzz\\data_traintest.csv", sep=";")
#val       = pd.read_csv("D:\\_Bachelorarbeit_Daten_\\lingeling_circuitfuzz\\data_val.csv", sep=";")
#fill_missing_dummies(train, traintest, val, path, helper)