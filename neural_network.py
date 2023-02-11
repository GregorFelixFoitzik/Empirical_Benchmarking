import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error as mse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from time import time
import sys


class EPM_Dataset(Dataset):
   def __init__(self, dataframe):
      super(EPM_Dataset, self).__init__()
      self.x = torch.from_numpy(dataframe.to_numpy(dtype=np.float32)[:, 1:])
      self.y = torch.from_numpy(dataframe.to_numpy(dtype=np.float32)[:, [0]])
      self.n_samples, self.n_features = dataframe.shape

   def __getitem__(self, index):
      return self.x[index] , self.y[index]

   def __len__(self):
      return self.n_samples

class ANN(torch.nn.Module):
    def __init__(self, input_size, architecture:list, drop_out:bool=False):
        super(ANN, self).__init__()
        self.drop_out = drop_out
        self.input_size = input_size
        self.layers = nn.ModuleList()
        self.activation = nn.ReLU()
        
        for idx, hl_size in enumerate(architecture):
            if idx == 0:
                self.layers.append( nn.Linear(self.input_size, architecture[idx]) )
            else:
                self.layers.append( nn.Linear(architecture[idx-1], architecture[idx]) )
        
        self.output_layer = nn.Linear(architecture[-1], 1)
        self.layers.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight,  mode="fan_out", nonlinearity="relu")


    def forward(self, x):
        self.dropout_input = nn.Dropout(p=0.2)
        self.dropout_hidden = nn.Dropout(p=0.5)

        ##### drop-out layer in the input layer and hidden layer like "G.E.Hinton"
        if self.drop_out:
            for i, layer in enumerate(self.layers):
                if i == 0:
                    x = layer(x)
                    x = self.activation(x)
                    x = self.dropout_input(x)    
                else:
                    x = layer(x)
                    x = self.activation(x)
                    x = self.dropout_hidden(x)

        ##### without dropout layers
        else:
            for i, layer in enumerate(self.layers):
                x = layer(x)
                x = self.activation(x)

        x = self.output_layer(x)
        return x



def training_loop(model, training_dataset, val_dataset, device, criterion, lr, epochs, batch_size, 
                  patience, l1, l2):
    data_loader = DataLoader(dataset=training_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    train_loss = torch.tensor([], device=device)
    data_runs = pd.DataFrame(columns=["epoch", "train_loss", "test_loss"])
    
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_loss = np.inf
    breakpoint = 0

    start = time()

    for ii in range(epochs):
        model.train()    
        infos_epoch = []
        print(f"Epoch: {ii+1}/{epochs} | Early Stopping: {breakpoint}", end="       \r")
        for batch_idx, (inputs, actual) in enumerate(data_loader):
            inputs = inputs.to(device)
            actual = actual.to(device).log()

            optimizer.zero_grad(set_to_none=True)
            preds = model(inputs)
            loss = criterion(actual, preds)
            train_loss = torch.cat((train_loss, loss.unsqueeze(0)))

            l1_penalty = l1 * sum([p.abs().sum() for p in model.parameters()])
            l2_penalty = l2 * sum([(p**2).sum() for p in model.parameters()])
            loss_with_penalty = loss + l1_penalty + l2_penalty

            loss_with_penalty.backward()
            optimizer.step()

        with torch.no_grad():
            test_loss, true_value, predictions = validate(model, val_dataset, criterion, device)
            mean_trainloss = np.mean(train_loss.tolist())
            mean_testloss = np.mean(test_loss.tolist())
            losses = pd.DataFrame({"epoch":[ii+1], "train_loss":[mean_trainloss], "test_loss":[mean_testloss]})
            data_runs = pd.concat([data_runs, losses], ignore_index=True)

            if mean_testloss < best_loss:
                best_loss = mean_testloss
                breakpoint = 0
            else:
                breakpoint += 1

            if breakpoint >= patience:
                print(f'\nEarly stopping at epoch {ii+1}')
                break
    
    x = true_value.tolist()
    y = predictions.tolist()

    rmse_log = np.sqrt( mse(x, y) )

    print(f"Finished: {time() - start:.1f} sec. | RMSE logPAR10: {rmse_log:.2f}", end="     \r\n")
    print(f"Actual: {np.min(x):.5f} | {np.median(x):.5f} | {np.mean(x):.5f} | {np.max(x):.5f}" )
    print(f"Preds: {np.min(y):.5f} | {np.median(y):.5f} | {np.mean(y):.5f} | {np.max(y):.5f}" )
    return data_runs


def validate(model, val_dataset, criterion, device, batch_size=100):
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    model.eval()

    total_loss  = torch.tensor([], device=device)
    true_values = torch.tensor([], device=device)
    predictions = torch.tensor([], device=device)

    with torch.no_grad():
        for inputs, actual in val_loader:
            inputs = inputs.to(device)
            actual = actual.to(device).log()

            preds = model(inputs)
            loss = criterion(actual, preds)

            preds = torch.flatten(preds)
            actual = torch.flatten(actual)

            predictions = torch.cat((predictions, preds))
            true_values = torch.cat((true_values, actual))
            total_loss = torch.cat((total_loss, loss.unsqueeze(0)))

    return total_loss, true_values, predictions