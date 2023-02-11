from functions_read_jsons import *

drive = "D:/_Bachelorarbeit_Daten_/" ### this has to be updated for personal use

infos = [
    ["minisat_randomk3",      "helper_minisat",                 300]
    ,["probsat_7sat90",        "helper_probsat",                 300]
    ,["clasp-rooks",           "helper_clasp-rooks",             300]
    ,["lingeling_circuitfuzz", "helper_lingeling_circuitfuzz",   300]
    ,["cplex_rcw",             "helper_cplex_rcw",               10_000]
    ,["cplex_regions200",      "helper_cplex_regions200",        10_000]
]


for dir, hpath, cutoff in infos:
    helper = pd.read_csv(f"{drive}{dir}/{hpath}.csv",   sep=";")
    save_jsons_as_csv(f"{drive}{dir}", "data_train",     helper, cutoff)
    save_jsons_as_csv(f"{drive}{dir}", "data_traintest", helper, cutoff)
    save_jsons_as_csv(f"{drive}{dir}", "data_val",       helper, cutoff)
    
    add_missing_dummies(f"{drive}{dir}")
    
    train     = pd.read_csv(f"{drive}{dir}/data_train.csv",     sep=";", low_memory=False)
    traintest = pd.read_csv(f"{drive}{dir}/data_traintest.csv", sep=";", low_memory=False)
    val       = pd.read_csv(f"{drive}{dir}/data_val.csv",       sep=";", low_memory=False)
    fill_missing_dummies(train, traintest, val, f"{drive}{dir}", helper)

    train     = pd.read_csv(f"{drive}{dir}/data_train.csv",     sep=";", low_memory=False)
    traintest = pd.read_csv(f"{drive}{dir}/data_traintest.csv", sep=";", low_memory=False)
    val       = pd.read_csv(f"{drive}{dir}/data_val.csv",       sep=";", low_memory=False)


    ### reorder columns
    train_cols = train.columns
    traintest = traintest[train_cols]
    val = val[train_cols]

    train       = reorder_columns(dataframe=train,      col_name='instance',    position=0)
    traintest   = reorder_columns(dataframe=traintest,  col_name='instance',    position=0)
    val         = reorder_columns(dataframe=val,        col_name='instance',    position=0)

    train       = reorder_columns(dataframe=train,      col_name='time',        position=1)
    traintest   = reorder_columns(dataframe=traintest,  col_name='time',        position=1)
    val         = reorder_columns(dataframe=val,        col_name='time',        position=1)

    train.to_csv(f"{drive}{dir}/data_train.csv", sep=";", index=False)
    traintest.to_csv(f"{drive}{dir}/data_traintest.csv", sep=";", index=False)
    val.to_csv(f"{drive}{dir}/data_val.csv", sep=";", index=False)