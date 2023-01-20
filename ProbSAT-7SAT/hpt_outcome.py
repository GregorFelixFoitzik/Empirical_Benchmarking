import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

headers = ["epoch", "train_loss", "test_loss",                               ### Information about training loop 
           "rmse_PR10",     # only for fifth
           "Dropout", "architec", "num_hidden_layers", "size_hidden_layers"] ### information about architecture

outcome = pd.read_csv("./ProbSAT-7SAT/Daten/hpt.csv", sep=";", header=0, names=headers)


##### creating new columns so that the visuals are easier to read
outcome["log(mse)_train_loss"] = np.log( outcome["train_loss"] )
outcome["log(mse)_test_loss"] = np.log( outcome["test_loss"] )

min_loss = outcome.loc[(outcome["epoch"] > 60)]
min_loss = min_loss.loc[(min_loss["log(mse)_test_loss"] == min(min_loss["log(mse)_test_loss"]))]
print(min_loss)



##### Plotting (nicht löschen!)
#outcome["Dropout | Architecture"] = outcome["Dropout"].astype(str) + " | " + outcome["architec"]

for arc in ["equal", "decre"]:
    df_viz = outcome.loc[outcome["architec"]==arc]
    
    g = sns.FacetGrid(df_viz, row="num_hidden_layers", col="size_hidden_layers", hue="Dropout", height=3, aspect=16/9) #"Dropout"
    g.map(sns.lineplot, "epoch", "log(mse)_test_loss")
    g.set(xlim=(None,250), ylim=(0,None))
    g.add_legend()

    sns.move_legend(g, "upper center", frameon=True, ncol=2, title="Dropout", bbox_to_anchor=(.9,1))
    g.fig.subplots_adjust(top=.9, hspace=.2)
    
    title = f"MSELoss by epoch with {arc} Architecture"
    subtitle = "with information about the hidden layers"

#    g.fig.set_title("123", fontsize=200)
    g.fig.suptitle(f"{title}\n{subtitle}", fontsize=14)
    g.set_titles(template = "count {row_name} | size {col_name}")

    for ax in g.axes.flat:
        ax.grid(True, linestyle='--')
    
    #plt.show()
    #g.map(sns.scatterplot, "epoch", "rmse_test_loss", marker="o", s=5, color="red", data=min_test_loss)
    plt.savefig(f"./ProbSAT-7SAT/hpt_{arc}_architecture.pdf", dpi=300, format="pdf")