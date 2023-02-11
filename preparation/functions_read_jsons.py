import pandas as pd
import numpy as np
import json, os

##### all parameters are marked as a set() => OneHotEncodig
def encoding_parameters(dataframe, helper):
    for col, dt, lg, default in zip(helper["Spalte"], helper["DataType"], helper["LogScale"], helper["Default"]):
        if col not in dataframe.columns:
            dataframe[col] = default
        if dt == "Float" or dt == "Integer":
            dataframe[col] = dataframe[col].astype(np.float32)
        if lg:
            dataframe.loc[dataframe[col]>0, col] = np.log(dataframe.loc[dataframe[col]>0, col])

    onehots = helper.loc[helper["DataType"]=="OneHot","Spalte"]
    dataframe = pd.get_dummies(dataframe,columns=onehots,drop_first=True, dtype=np.int8, prefix=onehots, prefix_sep='__')
    return dataframe


def setting_conditional_columns(dataframe, helper):
    for col, condition in zip(helper["Spalte"], helper["Condition"]):
        if "#" in condition:
            cond1, cond2 = condition.split("#")
            for cond in [cond1, cond2]:
                cond_col, _, vals_str = cond.split("__")
                vals = vals_str.replace("{","").replace("}","").replace(" ","").split(",")
                
                if len(vals) > 1 and (cond_col in dataframe.columns):
                    mask = [x in vals for x in dataframe[cond_col]]
                    if f"{cond_col}__{vals_str}" not in dataframe.columns:
                        dataframe[f"{cond_col}__{vals_str}"] = 0
                    dataframe.loc[mask, f"{cond_col}__{vals_str}"] += 1
        
        ##### single conditions get already caught by one hot encoding 
        elif condition != "?":
            cond_col, _, vals_str = condition.split("__")
            vals = vals_str.replace("{","").replace("}","").replace(" ","").split(",")

            if (len(vals) > 1) and (cond_col in dataframe.columns):
                mask = [x in vals for x in dataframe[cond_col]]
                if f"{cond_col}__{vals_str}" not in dataframe.columns:
                    dataframe[f"{cond_col}__{vals_str}"] = 0
                dataframe.loc[mask, f"{cond_col}__{vals_str}"] += 1
    return dataframe


##### set missing values to default (as Eggensperger: Section 3.1) and encoding parameters
def missing_to_default(dataframe, helper):    
    existing_cols = [col for col in helper["Spalte"] if col in dataframe.columns]    
    helper = helper.loc[helper["Spalte"].isin([*existing_cols])]
    for col, default in zip(helper["Spalte"], helper["Default"]):
        dataframe[col] = dataframe[col].fillna(default)
    for col in dataframe.columns:
        if "__" in col:
            dataframe[col] = dataframe[col].fillna(0)
    return dataframe


def load_json_file(path, cutoff):
    dataframe = pd.DataFrame()

    ##### read all lines from .json into a pandas.DataFrame
    with open(path, 'r') as f:
        for idx, line in enumerate(f):
            print(f"{path} | {round(idx, -2)}", end=f"              \r")
            data = json.loads(line)
            dataframe = pd.concat([dataframe, pd.json_normalize(data)], ignore_index=True)
        
    ##### due to normalization the headers has to be cleaned
    dataframe.columns = [col.replace('config.-','') for col in dataframe.columns]

    ##### Eggensperger: Section 3.1
    print(dataframe["status"].unique())
    dataframe = dataframe.loc[dataframe["status"]!="CRASHED"]
    dataframe.loc[dataframe["status"]=="TIMEOUT", "time"] = cutoff*10
    print(dataframe["status"].unique())

    for col in ["misc","seed", "status"]:
        dataframe.drop(col, axis=1, inplace=True)
    return dataframe


def save_jsons_as_csv(path, data_purpose, helper, cutoff):
    """
    data_purpose := {"data_train", "data_traintest", "data_val"}
    """
    df = pd.DataFrame()
    file_paths = []
    data_purpose_dir = [path +"/"+ f for f in os.listdir(path) if "." not in f]
    features = pd.read_csv(f"{path}/features.txt", sep=",")

    for dpd in data_purpose_dir:
        if dpd.endswith(data_purpose):
            file_paths = [dpd +"/"+ f for f in os.listdir(dpd) if ".json" in f]

    for fp in file_paths:
        dataframe = load_json_file(fp, cutoff)

        ##### Eggensperger: Section 3.1
        dataframe = setting_conditional_columns(dataframe, helper)
        dataframe = encoding_parameters(dataframe, helper)
        df = pd.concat([df, dataframe])

        df = missing_to_default(df, helper)

    features = features.rename(columns={features.columns[0]: "instance"})
    dataset = df.merge(features, how="left", on="instance")

    dataset.to_csv(f"{path}/{data_purpose}.csv", sep=";", index=False)    
    print(f"\nLoading of {path} {data_purpose} finished! | Missing: {dataset.isna().sum().sum()}", end=f"                                                       \r")
    

def add_missing_dummies(direc):
    train = pd.read_csv(f"{direc}/data_train.csv", sep=";", low_memory=False)
    traintest = pd.read_csv(f"{direc}/data_traintest.csv", sep=";", low_memory=False)
    val = pd.read_csv(f"{direc}/data_val.csv", sep=";", low_memory=False)

    for data in [train, traintest, val]:
        for data2 in [train, traintest, val]:
            for col in data.columns:
                if col not in data2.columns:
                    data2[col] = 0

    train.to_csv(f"{direc}/data_train.csv", sep=";", index=False)
    traintest.to_csv(f"{direc}/data_traintest.csv", sep=";", index=False)
    val.to_csv(f"{direc}/data_val.csv", sep=";", index=False)


def fill_missing_dummies(d1, d2, d3, path, helper):
    for data, purpose in zip([d1, d2, d3], ["data_train", "data_traintest", "data_val"]):
        dummies = [col for col in data.columns if col not in helper["Spalte"] and col != "time"]
        d1[dummies] = data[dummies].fillna(0)
        d2[dummies] = data[dummies].fillna(0)
        d3[dummies] = data[dummies].fillna(0)

        data[dummies] = data[dummies].fillna(0)
        data.to_csv(f"{path}/{purpose}.csv", sep=";", index=False)
        print(f"{purpose}: {data.isna().sum().sum()} & shape: {data.shape}")


def reorder_columns(dataframe, col_name, position):
    """Reorder a dataframe's column.

    Args:
        dataframe (pd.DataFrame): dataframe to use
        col_name (string): column name to move
        position (0-indexed position): where to relocate column to

    Returns:
        pd.DataFrame: re-assigned dataframe
    """
    temp_col = dataframe[col_name]
    dataframe = dataframe.drop(columns=[col_name])
    dataframe.insert(loc=position, column=col_name, value=temp_col)

    return dataframe