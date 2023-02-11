import sys
import pandas as pd
import numpy as np
sys.path.append("C:/Bachelorarbeit")
from neural_network import *
from scipy.stats import spearmanr

import matplotlib.pyplot as plt
import seaborn as sns


criterion = nn.MSELoss()  
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nThe ANN will run on the: {device}")

algo_dic = {
    #"probSAT":          ["ProbSAT-7SAT90", "probsat"]
    #,"miniSAT":         ["Minisat", "minisat"]
    #,"lingeling":       ["Lingeling-CF", "lingeling"]
    "clasp":          ["Clasp-Rooks", "clasp"]
    #,"cplex_rcw":       ["CPLEX-RCW2", "cplex_rcw2"]
    #,"cplex_regions":   ["CPLEX-Regions", "cplex_regions"]
}

for algo in algo_dic:   
    all_predictions = torch.tensor([], device=device)
    data_val = pd.read_csv(f"./{algo_dic[algo][0]}/Daten/data_val.csv", sep=";").drop("instance", axis=1)
    data_val = EPM_Dataset(data_val)
    for ii_model in range(2):
        print(f"./{algo_dic[algo][0]}/Models/NN_{algo_dic[algo][1]}-{ii_model}.pth")
        loaded_model = torch.load(f"./{algo_dic[algo][0]}/Models/NN_{algo_dic[algo][1]}-{ii_model}.pth")
        loaded_model.eval()

        loss, y, y_hat = validate(loaded_model, data_val, criterion, device=device, batch_size=100)

        all_predictions = torch.cat((all_predictions, y_hat.unsqueeze(0)))

        y_test = y.tolist()
        y_hat_test = y_hat.tolist()

        print(f"Actual: {np.min(y_test):.5f} | {np.median(y_test):.5f} | {np.mean(y_test):.5f} | {np.max(y_test):.5f}")
        print(f"ØPreds: {np.min(y_hat_test):.5f} | {np.median(y_hat_test):.5f} | {np.mean(y_hat_test):.5f} | {np.max(y_hat_test):.5f}")


    all_predictions_cpu = all_predictions.tolist()
    true_value = y.tolist()

    all_predictionsT = list(zip(*all_predictions_cpu))
    mean_predictions_over_NNs = [np.mean(row) for row in all_predictionsT]

    rmse_log = np.sqrt( mse( true_value, mean_predictions_over_NNs ) )
    spmr = spearmanr(true_value, mean_predictions_over_NNs)
    
    print()
    print(f"Finished: RMSE: {rmse_log:.2f} | CC: {spmr}")
    print(f"Actual: {np.min(true_value):.5f} | {np.median(true_value):.5f} | {np.mean(true_value):.5f} | {np.max(true_value):.5f}")
    print(f"ØPreds: {np.min(mean_predictions_over_NNs):.5f} | {np.median(mean_predictions_over_NNs):.5f} | {np.mean(mean_predictions_over_NNs):.5f} | {np.max(mean_predictions_over_NNs):.5f}")
    print()

    np.savetxt(f"./{algo_dic[algo][0]}/Daten/Model_Preds.csv", np.array([true_value, mean_predictions_over_NNs]).T, delimiter=";")