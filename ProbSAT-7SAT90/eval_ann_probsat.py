import sys
sys.path.append("C:\\Bachelorarbeit")
from neural_network import *
from sklearn.model_selection import train_test_split

torch.cuda.empty_cache()
torch.manual_seed(232437)

data_train = pd.read_csv("./ProbSAT-7SAT90/Daten/data_train.csv", sep=";").drop("instance", axis=1)
df_train, df_test = train_test_split(data_train, test_size=0.2, random_state=2383948)

trainset = EPM_Dataset(df_train)
testset  = EPM_Dataset(df_test)

features = trainset.n_features - 1
count = 0

##### Setting the device to gpu if it is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nThe ANN will run on the: {device}.")

##### Set: Loss-Function and Optimizer and Activation Function
criterion = nn.MSELoss()
batch_size = 256
learning_rate = 0.001
epochs = 200

##### stops after n epochs without improving the loss (can be turned off with: patience = epochs)
patience = 10

##### nested for loop to evaluate good hyperparameters
l1_weights = [0, 5e-1, 1e-2, 1e-3]
l2_weights = [0, 5e-1, 1e-2, 1e-3]
architec = "decre"   # equal | decre 
do = False           # True  | False


for l1 in l1_weights:
    for l2 in l2_weights:
        for num_hidden_layers in [1, 2, 4]:
            for size_hl in [1, 2, 4]:
                architecture_equal = [features*size_hl for i in range(num_hidden_layers)]
                architecture_decre = [np.int16(np.ceil(features*size_hl*(i/num_hidden_layers))) for i in range(num_hidden_layers, 0, -1)]
                architec_dict = {"equal":architecture_equal, "decre":architecture_decre}

                count += 1
                print(f"\nprobSAT L1:{l1} | L2:{l2} | #HL.:{num_hidden_layers} | Size HL.:{size_hl} | Arch.:{architec} | Dropout:{do} | {count}/81 | MSELoss")

                model = ANN(input_size=features, architecture=architec_dict[architec], drop_out=do).to(device, non_blocking=True)

                losses = training_loop( model, trainset, testset, epochs = epochs, lr=learning_rate,
                                        batch_size = batch_size, criterion=criterion, device=device,
                                        patience=patience, l1=l1, l2=l2)
                        
                add_cols = {"Dropout":do, "architec":architec, "num_hidden_layers":num_hidden_layers, "size_hidden_layers":size_hl, 
                            "Lasso(L1)":l1, "Ridge(L2)":l2}
                        
                for cc, vv in add_cols.items():
                    losses[cc] = vv

                losses.to_csv("./ProbSAT-7SAT90/Daten/HPTuning.csv", sep=";", header=False, mode="a", index=False)