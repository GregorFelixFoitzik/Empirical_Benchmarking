import pandas as pd
import matplotlib.pyplot as plt

headers = ["epoch","loss", "batch_size", 
           "batches", "num_epochs","num_hidden_layers","learning_rate",
           "criterion","optimizer","activation", "val_loss_mean", "val_rmse_pr10"]
df = pd.read_csv("C:/Users/gregf/Desktop/Bachelorarbeit/cplex_rcw/hpt_first.csv", sep=";", header=None, names=headers)

test = df.groupby(["num_epochs", "batches", "num_hidden_layers", "learning_rate"]).last()

min3 = test.sort_values(['val_rmse_pr10','val_loss_mean'],ascending=False).groupby(["num_epochs", "batches", "num_hidden_layers", "learning_rate"]).head(3)

min_loss = test.loc[test["val_rmse_pr10"] == min(test["val_rmse_pr10"])]
print(min_loss)