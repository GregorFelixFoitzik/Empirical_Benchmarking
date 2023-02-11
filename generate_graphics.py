import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

pathes = {".\CPLEX-Regions\Daten"   : {"Dropout":False, "architec":"equal", "HL":2, "rho":1, "L1":0, "L2":0}
         ,".\CPLEX-RCW2\Daten"      : {"Dropout":False, "architec":"equal", "HL":2, "rho":1, "L1":0, "L2":0} 
         ,".\ProbSAT-7SAT90\Daten"  : {"Dropout":False, "architec":"equal", "HL":2, "rho":1, "L1":0.001, "L2":0.001}
         ,".\Minisat\Daten"         : {"Dropout":False, "architec":"decre", "HL":2, "rho":2, "L1":0, "L2":0}
         ,".\Clasp-Rooks\Daten"     : {"Dropout":False, "architec":"equal", "HL":2, "rho":2, "L1":0, "L2":0}
         ,".\Lingeling-CF\Daten"    : {"Dropout":False, "architec":"equal", "HL":4, "rho":2, "L1":0.001, "L2":0}
}
Dropout_pathes = ["./ProbSAT-7SAT90/Daten/HPTuning.csv" ,"./Minisat/Daten/HPTuning.csv"]
folders = ["Minisat", "ProbSAT-7SAT90", "CPLEX-RCW2", "CPLEX-Regions", "Lingeling-CF", "Clasp-Rooks"]


### Histograms and ECDF of runtimes
def Histograms_and_ECDF(pathes=pathes):
    for path in pathes:
        fig, ((train_hist, test_hist, val_hist), (train_ecdf, test_ecdf, val_ecdf)) = plt.subplots(nrows=2, ncols=3, sharex=True, figsize=(16,6))
        name = path.split("\\")[-2]

        trainset = pd.read_csv(f"{path}\\data_train.csv", sep=";", usecols=["instance", "time"])
        _, testset = train_test_split(trainset, test_size=0.2, random_state=2383948)
        validation = pd.read_csv(f"{path}\\data_val.csv", sep=";", usecols=["instance", "time"])

        sns.histplot(x = trainset["time"],    ax=train_hist,   color="#545454", bins=50, element="step", log_scale=10)
        sns.histplot(x = testset["time"],     ax=test_hist,    color="#545454", bins=50, element="step", log_scale=10)
        sns.histplot(x = validation["time"],  ax=val_hist,     color="#545454", bins=50, element="step", log_scale=10)
                     
        sns.ecdfplot(x = trainset["time"],    ax=train_ecdf,   color="#545454", linewidth=3, log_scale=10)
        sns.ecdfplot(x = testset["time"],     ax=test_ecdf,    color="#545454", linewidth=3, log_scale=10)
        sns.ecdfplot(x = validation["time"],  ax=val_ecdf,     color="#545454", linewidth=3, log_scale=10)
        
        #fig.suptitle(name, fontsize=16, y=1.02)
        train_hist.set(title="Complete Trainset", ylabel="Anzahl", xlabel="log10(runtime)")
        test_hist.set(title="Splitted Testset", ylabel="", xlabel="log10(runtime)")
        val_hist.set(title="Validation", ylabel="", xlabel="log10(runtime)")

        train_ecdf.set(title="", yticks=[], ylabel="", xlabel="log10(runtime)")
        test_ecdf.set(title="", yticks=[], ylabel="", xlabel="log10(runtime)")
        val_ecdf.set(title="", yticks=[], ylabel="", xlabel="log10(runtime)")


        fig.savefig(f"./graphics/3_runtime_dist-{name}.pdf", dpi=300, bbox_inches='tight', pad_inches=0.1)

### HPT_Dropout
def HPT_Dropout(Dropout_pathes=Dropout_pathes):
    headers = [ 
    "epoch", "train_loss", "test_loss",# "test_true", "test_pred",                               ### Information about training loop 
    "Dropout", "architec", "num_hidden_layers", "size_hidden_layers", "Lasso(L1)", "Ridge(L2)"  ### Information about neural network
    ]

    for path in Dropout_pathes:
        outcome = pd.read_csv(path, sep=";", header=0, names=headers)

        g = sns.FacetGrid(outcome, row="Dropout", col="architec", height=5, aspect=25.0/29.7)
        g.map_dataframe(sns.lineplot, x="epoch", y="train_loss", errorbar=('ci', 95), color="red", label="Train")
        g.map_dataframe(sns.lineplot, x="epoch", y="test_loss", errorbar=('ci', 95), color="blue", label="Test")
        g.set(yscale="log")
        g.set(xlim=(1,201), ylim=(None,None), ylabel="")
        g.add_legend()

        sns.move_legend(g, "upper center", frameon=True, ncol=2, title="Loss function on", bbox_to_anchor=(.85,1))
        g.fig.subplots_adjust(top=.9, hspace=.15)
        plt.subplots_adjust(hspace=.15, wspace=.1)

        title = f"log(MSELoss) by Dropout and Architecture"
        g.fig.suptitle(f"{title}", fontsize=14)
        g.set_titles(template = "Dropout {row_name} | {col_name} Architecture")

        for ax in g.axes.flat:
            ax.grid(True, linestyle='--')

        name = path.split("/")[1]
        plt.savefig(f"./graphics/2_{name}_HPT_Dropout.pdf", dpi=300, format="pdf")

### Histograms True vs. Pred runtimes
def HistScatter_Runtimes(folders=folders):
    for folder in folders:
        yy_hat = np.loadtxt(f"./{folder}/Daten/Model_Preds.csv", delimiter=";").T
        y = yy_hat[0]
        y_hat = yy_hat[1]

        fig, (hist, scatter) = plt.subplots(nrows=2, ncols=1, sharex=False, figsize=(20,26))
        sns.histplot(x=y,     ax=hist, element="step", fill=False, color="red",  linewidth=3, bins=50, label="Wahre Laufzeiten")
        sns.histplot(x=y_hat, ax=hist, element="step", fill=False, color="blue", linewidth=3, bins=50, label="Prognose")
        hist.set(title="Verteilung der wahren und prognostizierte Laufzeiten")
    
        hist.set_title("Verteilung der wahren Laufzeiten und logPAR10", fontsize=26)
        hist.set_ylabel("Anzahl", fontsize=20)
        hist.set_xlabel("y | logPAR10", fontsize=20)
        hist.tick_params(axis='both', labelsize=16)

        
        scatter.plot(y, y, color="red", alpha=.3, label="y = logPAR10")
        scatter = sns.histplot(x=y_hat, y=y, bins=100, cbar=True, cbar_kws=dict(location="bottom", pad=0.1, aspect=40, fraction=0.025), cmap=plt.get_cmap('Blues',100), 
                     ax=scatter)
        cbar = scatter.collections[-1].colorbar
        cbar.ax.tick_params(labelsize=16)

        scatter.set_title("Prognose vs. wahre Werte", fontsize=26)
        scatter.set_ylabel("y", fontsize=20)
        scatter.set_xlabel("logPAR10", fontsize=20)
        scatter.tick_params(axis='both', labelsize=16)

        plt.subplots_adjust(hspace=.2, wspace=.1)
        hist.legend(loc='upper left', fontsize="xx-large")
        scatter.legend(loc='upper left', fontsize="xx-large")
        fig.savefig(f"./graphics/1_{folder}_HistScatter-Runtimes.pdf", bbox_inches='tight', pad_inches=0.1)

### Verlustfunktionen ausgewählter Parametereinstellungen
def Verlustfunktionen_Beste_Konfi(pathes=pathes):
    headers = ["epoch", "train_loss", "test_loss", "Dropout", "architec", "HL", "rho", "L1", "L2"]

    for path in pathes: 
        if path != ".\CPLEX-Regions\Daten":
            outcome = pd.read_csv(f"{path}\HPTuning.csv", sep=";", header=0, names=headers)
            confi = pathes[path]
            best_confi = outcome.loc[(outcome[list(confi)] == pd.Series(confi)).all(axis=1)][:-10]

            fig, ax = plt.subplots(1,1, figsize=(20,13))
            sns.lineplot(best_confi, x="epoch", y="train_loss", color="red", label="Training", ax=ax)
            sns.lineplot(best_confi, x="epoch", y="test_loss", color="blue", label="Validation", ax=ax)

            ax.legend(loc="upper right", fontsize="xx-large")
            ax.set_title("Verlustfunktionen über Epochen", fontsize=26)
            ax.set_xlabel("Epoche", fontsize=20)
            ax.set_ylabel("Verlustfunktionswert (MSE)", fontsize=20)
            ax.tick_params(axis='both', labelsize=16)
            plt.yscale("log")

            name = path.split("\\")[1]
            fig.savefig(f"./graphics/4_{name}_BestConfi_Loss.pdf",bbox_inches='tight', pad_inches=0.1)


#Histograms_and_ECDF()
#HPT_Dropout()
HistScatter_Runtimes()
Verlustfunktionen_Beste_Konfi()