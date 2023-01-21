import sys
sys.path.append("C:\\Users\\gregf\\Desktop\\Bachelorarbeit")
from neural_network import *
from sklearn.model_selection import train_test_split

torch.cuda.empty_cache()
torch.manual_seed(232437)

data_train = pd.read_csv("./Lingeling-CF/Daten/data_train.csv", sep=";").drop("instance", axis=1)
df_train, df_test = train_test_split(data_train, test_size=0.2, random_state=483858)
trainset = EPM_Dataset(df_train)
testset  = EPM_Dataset(df_test)

features = trainset.n_features - 1
count = 0

##### Setting the device to gpu if it is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nThe ANN will run on the: {device}.")

##### Set: Loss-Function and Optimizer and Activation Function
criterion = nn.MSELoss()   
optimizer = "Adam"          # SGD | Adam | RMSprop
acti_func = "relu"          # sigmoid | relu | tanh
batch_size = 256
learning_rate = 0.001
epochs = 250

##### stops after n epochs without improving the loss (can be turned off with: patience = epochs)
patience = 50


##### nested for loop to evaluate good hyperparameters
for num_hidden_layers in [2, 5, 10]:
    for size_hl in [1, 2, 3]:
        architecture_equal = [features*size_hl for i in range(num_hidden_layers)]
        architecture_decre = [np.int16(np.ceil( (features*size_hl) / i )) for i in range(1, num_hidden_layers+1)] # 1/1 | 1/2 | 1/3 | ...
        architec_dict = {"equal":architecture_equal, "decre":architecture_decre}

        for architec in ["equal", "decre"]:
            for do in [True, False]:
                count += 1
                print(f"\nprobSAT Epochs: {epochs} | batch size: {batch_size} | #HL.: {num_hidden_layers} | Size HL.: {size_hl} | Dropout: {do} | {count}/36")

                ##### Initialize model and move to device
                model = ANN(input_size=features, architecture=architec_dict[architec], output_size=1, activation_func=acti_func, drop_out=do).to(device)
                        
                test_loss = training_loop(  model, trainset, testset, epochs = epochs, lr=learning_rate,
                                                        batch_size = batch_size, shuffle = True,
                                                        criterion=criterion, optimizer_str=optimizer, device=device,
                                                        patience=patience)

                ##### Preparing and saving dataset as csv
                ##### The csv file will be appended
                train_preds = pd.DataFrame(test_loss, columns=["epoch", "train_loss", "test_loss", "rmse_logPAR10"])

                add_cols = {"Dropout":do, "architec":architec, "num_hidden_layers":num_hidden_layers, "size_hidden_layers":size_hl}

                for cc, vv in add_cols.items():
                    train_preds[cc] = vv

                ##### headers = ["epoch","adj. loss","loss","rmse","batch_size","batches","num_epochs","num_hidden_layers","learning_rate",
                ######           "criterion","optimizer","activation","val_loss_mean","val_rmse_mean"]
                train_preds.to_csv("./Lingeling-CF/Daten/hpt.csv", sep=";", header=False, mode="a", index=False)