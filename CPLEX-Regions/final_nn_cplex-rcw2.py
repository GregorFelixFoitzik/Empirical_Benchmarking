import sys
import pandas as pd
sys.path.append("C:\\Users\\gregf\\Desktop\\Bachelorarbeit")
from neural_network import *
torch.manual_seed(232437)

data_train = pd.read_csv("./CPLEX-Regions/Daten/data_train.csv", sep=";").drop("instance", axis=1)
data_train = EPM_Dataset(data_train)
features = data_train.n_features - 1

### Setting the device to gpu if it is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nThe ANN will run on the: {device}")

##################################################
### Set: Loss-Function and Optimizer and Activation Function
criterion = nn.MSELoss()   
optimizer_str = "Adam"
acti_func = "relu"

### hyperparameters evaluated manually (tested hyperparameter in ./hpt_(...).csv; steps in hpt_outcome.py)
num_hidden_layers = 2
size_hidden_layers = 2
Dropout = False
epochs = 245
architec = "decre"

##################################################
architecture_decre = [np.int16(np.ceil( (features*size_hidden_layers) / i )) for i in range(1, num_hidden_layers+1)]
architecture_equal = [features*size_hidden_layers for i in range(num_hidden_layers)]
architec_dict = {"equal":architecture_equal, "decre":architecture_decre}
dict_optimizer = {"SGD": optim.SGD, "Adam": optim.Adam, "RMSprop": optim.RMSprop}

train_loader = DataLoader(dataset = data_train, batch_size = 256, shuffle = True)
model = ANN(input_size = features, architecture = architec_dict[architec], output_size = 1, activation_func = acti_func, drop_out = Dropout).to(device)
optimizer = dict_optimizer[optimizer_str](model.parameters(), lr=.001)
                     
for ii in range(epochs):
    model.train()

    for batch_idx, (inputs, actual) in enumerate(train_loader):
        print(f"Epoch: {ii+1}/{epochs} := Mini-Batch: {batch_idx+1}/{len(train_loader)}", end="      \r")
        inputs = inputs.to(device) # tensor with algo-parameters
        actual = actual.to(device) # tensor with runtime

        optimizer.zero_grad()
        preds = model(inputs)
        loss = criterion(preds, actual)
        loss.backward()
        optimizer.step()

file = "ann_cplex_regions.pth"
torch.save(model, file)
print("Finished training and saving!\n\n")