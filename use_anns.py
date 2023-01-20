import sys
import pandas as pd
sys.path.append("C:\\Users\\gregf\\Desktop\\Bachelorarbeit")
from neural_network import *
from scipy.stats import spearmanr

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nThe ANN will run on the: {device}")

### here is code example to test the saved network
loaded_model = torch.load("ann_probSAT.pth")
data_val = pd.read_csv("./ProbSAT-7SAT/Daten/data_val.csv", sep=";").drop("instance", axis=1)
data_val = EPM_Dataset(data_val)
val_loader = DataLoader(data_val, batch_size=1, shuffle=False)  # LOCO is performed by setting the batch-size to 1

actuals = []
predictions = []
test = []

for ii, (inputs, actual) in enumerate(val_loader):
    print(f"Batch: {ii+1}/{len(val_loader)}", end="      \r")
    inputs = inputs.to(device)
    actual = actual.to(device)

    loaded_model.eval()
    with torch.no_grad():
        outputs = loaded_model(inputs)
        actual_cpu_value = actual.cpu().numpy().item()
        preds_cpu_value = outputs.cpu().numpy().item()
        
        actuals.append(np.log(actual_cpu_value))
        predictions.append(np.log(preds_cpu_value))

        test.append( ( np.log(actual_cpu_value) - np.log(preds_cpu_value) )**2 )

print()
print(np.sqrt( np.mean(test) ) )
#print(spearmanr(actuals, predictions))