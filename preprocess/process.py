from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
import argparse
import csv
import os

os.environ['MKL_THREADING_LAYER'] = 'GNU'

p = argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter)

p.add_argument('--root', default="/home/gayane/BartLM",
                    help="add your root path")

args = p.parse_args()
root = args.root
def scaff(dataset_name, type):

    path = f"{root}/chemical/checkpoints/evaluation_data/"
    file=open(path + f"{dataset_name}/{dataset_name}/{type}_{dataset_name}.csv")
    file = file.read().split('\n')
    sm = list()
    col = list()

    for i in range(1,len(file)-1):
        sm.append(file[i].split(',')[-1])
        col.append(file[i].split(',')[-2].strip("'"))
    bool_col = [float(i) for i in col]

    d = {"Classification": bool_col , "SMILES": sm}
    df = pd.DataFrame(d) 
    df = df.dropna()
    return df

def split_train_val_test(df, smiles_col_name, label_col_name):

    train_ratio = 0.8
    validation_ratio = 0.1
    test_ratio = 0.1

    x_train, x_test, y_train, y_test = train_test_split(df['SMILES'],df['Classification'], 
                                                        test_size=1 - train_ratio)

    x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=test_ratio/(test_ratio + validation_ratio)) 
    x_train = x_train.to_list()
    y_train = y_train.to_list()
    x_val = x_val.to_list()
    y_val = y_val.to_list()
    x_test = x_test.to_list()
    y_test = y_test.to_list()

    train_df = pd.DataFrame({smiles_col_name: x_train, label_col_name: y_train })    
    valid_df = pd.DataFrame({smiles_col_name: x_val, label_col_name: y_val })    
    test_df = pd.DataFrame({smiles_col_name: x_test, label_col_name: y_test })
    
    return train_df, valid_df, test_df

def cross_val(df, smiles_col_name, label_col_name, k_fold):
    # train_ratio = 0.8
    # validation_ratio = 0.1
    # test_ratio = 0.1

    # x_train, x_test, y_train, y_test = train_test_split(df['SMILES'],df['Classification'], 
    #                                                 test_size=1 - train_ratio -validation_ratio)
    # test_df = pd.DataFrame({smiles_col_name: x_test, label_col_name: y_test })
    x_train = pd.DataFrame({'ind':df['SMILES'].index, smiles_col_name: df['SMILES'].values})
    y_train = pd.DataFrame({'ind':df['Classification'].index, label_col_name: df['Classification'].values})


    kf = KFold(n_splits=k_fold, shuffle=True, random_state=42).split(x_train, y_train)
    kf_data = list()
    for train_index, val_index in kf:
        X_train_, X_val_ = x_train.iloc[train_index], x_train.iloc[val_index]
        y_train_, y_val_ = y_train.iloc[train_index], y_train.iloc[val_index]
        train_df = pd.merge(X_train_, y_train_, on="ind")   
        valid_df = pd.merge(X_val_, y_val_, on='ind') 
        kf_data.append([train_df,valid_df])   

    return kf_data #, test_df

def read_file(ind_path):

    ind_list = []
    with open(ind_path, "r") as f:
        reader = csv.reader(f, delimiter="\t")
        for i, line in enumerate(reader):
            ind_list.append(np.array(list(map(int, line[0].split(","))))-1)
    return ind_list
