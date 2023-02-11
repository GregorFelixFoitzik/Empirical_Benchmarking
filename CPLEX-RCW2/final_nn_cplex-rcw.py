import sys
import pandas as pd
sys.path.append("C:\\Bachelorarbeit")
from neural_network import *

data_train = pd.read_csv("./CPLEX-RCW2/Daten/data_train.csv", sep=";").drop("instance", axis=1)
df_train = EPM_Dataset(data_train)
features = df_train.n_features - 1

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
size_hidden_layers = 1
Dropout = False
epochs = 46
architec = "equal"
l1 = 0
l2 = 0

architecture_equal = [features*size_hidden_layers for i in range(num_hidden_layers)]
architecture_decre = [np.int16(np.ceil(features*size_hidden_layers*(i/num_hidden_layers))) for i in range(num_hidden_layers, 0, -1)]
architec_dict = {"equal":architecture_equal, "decre":architecture_decre}

##################################################
random_seeds = [232437, 312449, 95618, 623721, 499768, 118521, 90023, 813940, 757462, 513896]
for ii_model, rn in enumerate(random_seeds):
    torch.manual_seed(rn)

    train_loader = DataLoader(dataset = df_train, batch_size = 256, shuffle = True)
    model = ANN(input_size = features, architecture = architec_dict[architec], drop_out = Dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=.001)

    for ii in range(epochs):
        model.train()

        for batch_idx, (inputs, actual) in enumerate(train_loader):
            print(f"Epoch: {ii+1}/{epochs} := Mini-Batch: {batch_idx+1}/{len(train_loader)}", end="      \r")
            inputs = inputs.to(device) # tensor with algo-parameters
            actual = actual.to(device).log() # tensor with runtime

            optimizer.zero_grad()
            preds = model(inputs)
            loss = criterion(preds, actual)

            # compute penalty only for net.hidden parameters
            l1_penalty = l1 * sum([p.abs().sum() for p in model.parameters()])
            l2_penalty = l2 * sum([(p**2).sum() for p in model.parameters()])
            loss_with_penalty = loss + l1_penalty + l2_penalty
                
            loss_with_penalty.backward()
            optimizer.step()

    file = f"./CPLEX-RCW2/Models/NN_cplex_rcw2-{ii_model}.pth"
    torch.save(model, file)
    print("\nFinished training and saving!\n")