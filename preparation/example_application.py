from read_data_with_helpers import *

drive = "D:\\_Bachelorarbeit_Daten_\\" ### this has to be updated for personal use

infos = { 
    "dirs":   ["minisat_randomk3",   "probsat_7sat90",
               "clasp-rooks",        "lingeling_circuitfuzz",
               "cplex_rcw",          "cplex_regions200"],

    "hpaths": ["helper_minisat",     "helper_probsat",
               "helper_clasp-rooks", "helper_lingeling_circuitfuzz",
               "helper_cplex_rcw",   "helper_cplex_regions200"]
}

for dir, hpath in zip(infos["dirs"][:], infos["hpaths"][:]):
    helper = pd.read_csv(f"{drive}{dir}/{hpath}.csv", sep=";")
    save_jsons_as_csv(f"{drive}{dir}", "data_train",     helper)
    save_jsons_as_csv(f"{drive}{dir}", "data_traintest", helper)
    save_jsons_as_csv(f"{drive}{dir}", "data_val",       helper)

    add_missing_dummies(dir)

    helper    = pd.read_csv(f"{drive}{dir}/{hpaths}.csv",       sep=";")
    train     = pd.read_csv(f"{drive}{dir}/data_train.csv",     sep=";")
    traintest = pd.read_csv(f"{drive}{dir}/data_traintest.csv", sep=";")
    val       = pd.read_csv(f"{drive}{dir}/data_val.csv",       sep=";")
    fill_missing_dummies(train, traintest, val, dir, helper)