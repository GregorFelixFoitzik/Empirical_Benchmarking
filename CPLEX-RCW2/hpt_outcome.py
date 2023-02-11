import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

headers = [ 
    "epoch", "train_loss", "test_loss",                           
    "Dropout", "architec", "num_hidden_layers", "size_hidden_layers", "Lasso(L1)", "Ridge(L2)"
]

outcome = pd.read_csv("./CPLEX-RCW2/Daten/HPTuning.csv", sep=";", header=0, names=headers)

min_loss = outcome.sort_values(by='test_loss')
min_loss["train_loss"] = np.log(min_loss["train_loss"])
min_loss["test_loss"] = np.log(min_loss["test_loss"])
print(min_loss[:15])