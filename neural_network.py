import pandas as pd
import numpy as np
from scipy.stats import spearmanr

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from time import time
import sys


class EPM_Dataset(Dataset):
   def __init__(self, dataframe):
      super(EPM_Dataset, self).__init__()
      #xy = pd.read_csv(file_path, sep=";").drop("instance", axis=1)
      #xy     = np.loadtxt(file_path, delimiter=";", dtype=np.float32, skiprows=1)
      self.x = torch.from_numpy(dataframe.to_numpy(dtype=np.float32)[:, 1:])
      self.y = torch.from_numpy(dataframe.to_numpy(dtype=np.float32)[:, [0]])  # runtimes PR10 as Eggensperger (PR10:= penalized runtimes by factor 10)
      self.n_samples, self.n_features = dataframe.shape

   def __getitem__(self, index):
      return self.x[index] , self.y[index]

   def __len__(self):
      return self.n_samples

class ANN(torch.nn.Module):
    def __init__(self, input_size, architecture:list, output_size=1, activation_func="relu", drop_out:bool=False):
        super(ANN, self).__init__()
        """
        This is a docstring
        """
        self.drop_out = drop_out
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layers = nn.ModuleList()
        self.activation_func = activation_func
        
        for idx, hl_size in enumerate(architecture):
            if idx == 0:
                self.hidden_layers.append(nn.Linear(input_size, architecture[idx]))
            else:
                self.hidden_layers.append(nn.Linear(architecture[idx-1], architecture[idx]) )
        
        self.output_layer = nn.Linear(architecture[-1], self.output_size)
    
    def forward(self, x):
        activation_functions = {"sigmoid": torch.sigmoid, "relu": torch.relu, "tanh":torch.tanh}
        self.dropout_input = nn.Dropout(p=0.2)
        self.dropout_hidden = nn.Dropout(p=0.5)

        ##### drop-out layer in the input layer and hidden layer like "G.E.Hinton"
        if self.drop_out:
            x = self.dropout_input(x)
            for i, layer in enumerate(self.hidden_layers):
                x = layer(x)
                x = self.dropout_hidden(x)
                ##### apply the activation function
                if i < (len(self.hidden_layers) - 1):
                    x = activation_functions[self.activation_func](x)

        else:
            for i, layer in enumerate(self.hidden_layers):
                x = layer(x)
            ##### apply the activation function
                if i < (len(self.hidden_layers) - 1):
                    x = activation_functions[self.activation_func](x)

        x = self.output_layer(x)
        return x



def training_loop(model, training_dataset, val_dataset, device, criterion, optimizer_str, lr, epochs, batch_size, shuffle, patience):
    data_loader = DataLoader(dataset=training_dataset, batch_size=batch_size, shuffle=shuffle)
    train_loss = []
    test_loss = []
    
    dict_optimizer = {"SGD": optim.SGD, 
                      "Adam": optim.Adam, 
                      "RMSprop": optim.RMSprop}
    optimizer = dict_optimizer[optimizer_str](model.parameters(), lr=lr)
   
    ##### Setting for early stopping
    best_loss = np.inf
    breakpoint = 0

    start = time()

    for ii in range(epochs):
        model.train()
        infos_epoch = []
        for batch_idx, (inputs, actual) in enumerate(data_loader):
            print(f"Epoch: {ii+1}/{epochs} := Mini-Batch: {batch_idx+1}/{len(data_loader)} | Early Stopping: {breakpoint}", end="       \r")
            inputs = inputs.to(device) # tensor with algo-parameters
            actual = actual.to(device) # tensor with runtime

            optimizer.zero_grad()
            preds = model(inputs)
            loss = criterion(preds, actual)
            loss.backward()
            optimizer.step()

        # Saving progress
        with torch.no_grad():
            loss_test, rmse = validate(model, val_dataset, criterion, device)
            test_loss.append([ii, loss.item(), loss_test, rmse])

            if loss_test < best_loss:
                best_loss = loss_test
                breakpoint = 0
            else:
                breakpoint += 1

            if breakpoint >= patience:
                print(f'\nEarly stopping at epoch {ii+1}')
                break


    print(f"\nFinished training_loop: {time() - start:.1f} sec. | last RMSE logPAR10: {rmse:.2f}", end="     \r\n")
    return test_loss


def validate(model, val_dataset, criterion, device):
    val_loader = DataLoader(dataset=val_dataset, batch_size=100, shuffle=False)
    model.eval() # set model to evaluation mode

    total_samples = 0
    total_loss = 0.0

    true_values = []
    predictions = []

    # disable gradient calculation
    with torch.no_grad():
        for inputs, actual in val_loader:
            # move data to device
            inputs = inputs.to(device)
            actual = actual.to(device)

            # forward pass
            preds = model(inputs)
            loss = criterion(preds, actual)

            # update total loss
            actual_cpu_value = actual.cpu().numpy().flatten()
            preds_cpu_value  = preds.cpu().numpy().flatten()
            #print(actual_cpu_value, preds_cpu_value)

            for ii, vv in enumerate(preds_cpu_value):
                if vv <= 0:
                    preds_cpu_value[ii] = 1e-4

            total_loss += loss.item()
            total_samples += 1

            true_values.append( np.mean( np.log(actual_cpu_value) ) )
            predictions.append( np.mean( np.log(preds_cpu_value) ) )

            #print(true_values, predictions)


        avg_true_runtime = np.mean( true_values )
        avg_pred_runtime = np.mean( predictions )
        rmse = np.sqrt( (avg_true_runtime - avg_pred_runtime)**2 )
        #corr, _ = spearmanr(predictions, true_values)
            
    return total_loss / total_samples, np.round(rmse, 4)#, np.round(corr, 2)