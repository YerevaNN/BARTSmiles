import sys
from process import cross_val
import deepchem as dc
import pandas as pd
import numpy as np
import argparse
import json
import os
# os.environ['MKL_THREADING_LAYER'] = 'GNU'

EMPTY_INDEX = -1

p = argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter)

# p.add_argument("--input", help="input file", type=str, required=True)
p.add_argument("--dataset-name", 
                type=str, required=True)
p.add_argument("--delimiter", 
                type=str, default=",")
p.add_argument("--is-MoleculeNet", 
                help="MoleculeNet", 
                choices=('True','False'), 
                default=False)
p.add_argument('--root', 
                default="/home/gayane/BartLM",
                help="add your root path")

args = p.parse_args()
root = args.root
MolNet_flag = args.is_MoleculeNet == 'True'



sys.path.append(f"{root}/BARTSmiles/utils/")
from utils import tokenize, create_raw, fairseq_preprocess_cmd


np.random.seed(123)
with open(f'{root}/BARTSmiles/datasets.json') as f:
    datasets_json = json.load(f)

dataset = datasets_json[args.dataset_name]
si = dataset['smiles_index']
store_path = f"{root}/chemical/checkpoints/evaluation_data"

if len(dataset["class_index"]) > 1:
    path = list()
    for i in range(len(dataset["class_index"])):
        os.system(f'mkdir -p {store_path}/{args.dataset_name}_{i}')
        os.system(f'mkdir -p {store_path}/{args.dataset_name}_{i}/{args.dataset_name}')
        path.append(f"{store_path}/{args.dataset_name}_{i}/{args.dataset_name}/")

else:
    os.system(f'mkdir -p {store_path}/{args.dataset_name}')
    os.system(f'mkdir -p {store_path}/{args.dataset_name}/{args.dataset_name}')
    path = f"{store_path}/{args.dataset_name}/{args.dataset_name}/"

if MolNet_flag:
    # For MoleculeNet data
    
    v = eval(f"dc.molnet.load_{dataset['load_name']}")
    tasks, datasets, transformers = v(splitter=dataset['split_type'],
                                                featurizer = 'ECFP')
    train_data, valid_data, test_data = datasets

    train_df = train_data.to_dataframe()
    valid_df = valid_data.to_dataframe() 
    test_df = test_data.to_dataframe() 

    # Remove some not usefull comlumns
    remove_cols = [col for col in valid_df.columns if 'X' in col]
    valid_df.drop(remove_cols, axis='columns', inplace=True)
    train_df.drop(remove_cols, axis='columns', inplace=True)
    test_df.drop(remove_cols, axis='columns', inplace=True)
    remove_cols = [col for col in valid_df.columns if 'w' in col]
    valid_df.drop(remove_cols, axis='columns', inplace=True)
    train_df.drop(remove_cols, axis='columns', inplace=True)
    test_df.drop(remove_cols, axis='columns', inplace=True)


    if dataset['filter']:
        assert len(dataset['class_index']) == 1, "We do not want to filter multi-task datasets."
        ci = dataset['class_index'][0]

        nan_value = float("NaN")
        train_df.replace("", nan_value, inplace=True)
        train_df.dropna(subset=[train_df.columns[ci], train_df.columns[si]], inplace=True)

        valid_df.replace("", nan_value, inplace=True)
        valid_df.dropna(subset=[valid_df.columns[ci], valid_df.columns[si]], inplace=True)

        test_df.replace("", nan_value, inplace=True)
        test_df.dropna(subset=[test_df.columns[ci], test_df.columns[si]], inplace=True)
    else:
        train_df.replace("", EMPTY_INDEX, inplace=True)
        train_df.fillna(EMPTY_INDEX, inplace=True)
        valid_df.replace("", EMPTY_INDEX, inplace=True)
        valid_df.fillna(EMPTY_INDEX, inplace=True)
        test_df.replace("", EMPTY_INDEX, inplace=True)
        test_df.fillna(EMPTY_INDEX, inplace=True)

else: 
    v = eval(f"{args.dataset_name }")
    train_df, valid_df, test_df = v(args.dataset_name, path)

if len(dataset["class_index"]) >1:
    print("___________________")
    for i in range(len(dataset["class_index"])):
        test_df.to_csv(f"{store_path}/{args.dataset_name}_{i}/{args.dataset_name}/test_{args.dataset_name}.csv")
        train_df.to_csv(f"{store_path}/{args.dataset_name}_{i}/{args.dataset_name}/train_{args.dataset_name}.csv")
        valid_df.to_csv(f"{store_path}/{args.dataset_name}_{i}/{args.dataset_name}/valid_{args.dataset_name}.csv")
else:

    test_df.to_csv(f"{store_path}/{args.dataset_name}/{args.dataset_name}/test_{args.dataset_name}.csv")
    train_df.to_csv(f"{store_path}/{args.dataset_name}/{args.dataset_name}/train_{args.dataset_name}.csv")
    valid_df.to_csv(f"{store_path}/{args.dataset_name}/{args.dataset_name}/valid_{args.dataset_name}.csv")

loc = valid_df.columns.get_loc

if dataset["type"] == "regression":
    print(len(dataset['class_index']))
    assert len(dataset['class_index']) == 1, "Regression tasks are always single-task."
    ci = dataset['class_index'][0]
    mi = dataset['minimum']
    ma = dataset['maximum']

    class_train = [(i - mi)/(ma - mi) for i in list(train_df.iloc[:, ci].values.tolist())] 
    class_test = [(i - mi)/(ma - mi) for i in list(test_df.iloc[:, ci].values.tolist())]
    class_val = [(i - mi)/(ma - mi) for i in list(valid_df.iloc[:, ci].values.tolist())]
    
    print(f"Scale {args.dataset_name} (type={dataset['type']}) dataset [0,1] interval")
    print(f"Regression task target minimum value: {mi} and max value: {ma} ")

if dataset["type"] == "classification":
    if len(dataset["class_index"]) >1:
        print("___________________")
        class_dict = {}
        for cii in range(len(dataset["class_index"])):
            class_dict["class_train" +str(cii)] = list(map(int, train_df.iloc[:, dataset["class_index"][cii]].tolist()))
            class_dict["class_val" +str(cii)] =list(map(int, valid_df.iloc[:, dataset["class_index"][cii]].tolist()))
            class_dict["class_test" +str(cii)] =list(map(int, test_df.iloc[:, dataset["class_index"][cii]].tolist()))

    else:
        ci = dataset['class_index'][0]
        class_train = list(map(int, train_df.iloc[:, ci].tolist()))
        class_val = list(map(int, valid_df.iloc[:, ci].tolist()))
        class_test = list(map(int, test_df.iloc[:, ci].tolist()))

smiles_train = list(map(str, train_df.iloc[:, si].tolist()))
smiles_val = list(map(str, valid_df.iloc[:, si].tolist()))
smiles_test = list(map(str, test_df.iloc[:, si].tolist()))


if dataset["type"] =='regression':
    os.system(f'mkdir {store_path}/{args.dataset_name}/label')
    with open(f"{store_path}/{args.dataset_name}/label/train.label", "w") as f:
        for item in class_train:
            f.write("%s\n" % item)
    with open(f"{store_path}/{args.dataset_name}/label/valid.label", "w") as f:
        for item in class_val:
            f.write("%s\n" % item)
    with open(f"{store_path}/{args.dataset_name}/label/test.label", "w") as f:
        for item in class_test:
            f.write("%s\n" % item)


if len(dataset['class_index'])>1:
    for i in range(len(dataset['class_index'])):

        os.system(f"mkdir {store_path}/{args.dataset_name}_{i}/raw")
        os.system(f"mkdir {store_path}/{args.dataset_name}_{i}/tokenized")
        os.system(f"mkdir {store_path}/{args.dataset_name}_{i}/processed")
        os.system(f'mkdir {store_path}/{args.dataset_name}_{i}/processed/label')

        os.system(f"mkdir {store_path}/{args.dataset_name}_{i}/processed/input0")

        with open(f"{store_path}/{args.dataset_name}_{i}/processed/label/train.label", "w") as f:
            for item in  class_dict["class_train" +str(i)]:
                f.write("%s\n" % item)
        with open(f"{store_path}/{args.dataset_name}_{i}/processed/label/valid.label", "w") as f:
            for item in class_dict["class_val" +str(i)]:
                f.write("%s\n" % item)
        with open(f"{store_path}/{args.dataset_name}_{i}/processed/label/test.label", "w") as f:
            for item in class_dict["class_test" +str(i)]:
                f.write("%s\n" % item)
    
else:
    os.system(f"mkdir {store_path}/{args.dataset_name}/raw")
    os.system(f"mkdir {store_path}/{args.dataset_name}/tokenized")
    os.system(f"mkdir {store_path}/{args.dataset_name}/processed")
    os.system(f'mkdir {store_path}/{args.dataset_name}/processed/label')
    os.system(f"mkdir {store_path}/{args.dataset_name}/processed/input0")
    with open(f"{store_path}/{args.dataset_name}/processed/label/train.label", "w") as f:
        for item in class_train:
            f.write("%s\n" % item)
    with open(f"{store_path}/{args.dataset_name}/processed/label/valid.label", "w") as f:
        for item in class_val:
            f.write("%s\n" % item)
    with open(f"{store_path}/{args.dataset_name}/processed/label/test.label", "w") as f:
        for item in class_test:
            f.write("%s\n" % item)

print(f"{args.dataset_name} Train Length: {len(smiles_train)}")
print(f"{args.dataset_name} Valid Length: {len(smiles_val)}")
print(f"{args.dataset_name} Test Length: {len(smiles_test)}")

names = ["train", "valid", "test"]
X_splits = []
y_splits = []

# Write Raw Splits
print("Writing Input Splits")
print(args.dataset_name )
if len(dataset["class_index"]) >1:
    paths_ = []
    print("________________________")
    for i in range(len(dataset["class_index"])):
        paths_ = create_raw(f"{store_path}/{args.dataset_name}_{i}", names, smiles_train, smiles_val, smiles_test, file_output = ".input")
        X_splits.append(paths_)
else:
    X_splits = create_raw(f"{store_path}/{args.dataset_name}", names, smiles_train, smiles_val, smiles_test, file_output = ".input")

print("Writing Output Splits")
if len(dataset['class_index'])>1:
    # pass
    for i in range(len(dataset['class_index'])):
        y_splits_current = create_raw(f"{store_path}/{args.dataset_name}_{i}", names, 
                                class_dict["class_train" + str(i)], class_dict["class_val" +str(i)], 
                                class_dict["class_test" + str(i)], file_output = ".target")
        y_splits.append(y_splits_current)

else:
    y_splits = create_raw(f"{store_path}/{args.dataset_name}", names, class_train, class_val, class_test, file_output = ".target")

# Tokenize Texts

print("Tokenizing")

if len(dataset["class_index"]) > 1:
    X_splits[0] = tokenize(X_splits[0], root)
    X_splits[1] = tokenize(X_splits[1], root)
    X_splits[2] = tokenize(X_splits[2], root)

else:
    X_splits = tokenize(X_splits, root)


if len(dataset["class_index"]) > 1:
    for i in range(len(dataset['class_index'])):
        fairseq_preprocess_cmd(X_splits[i][0], X_splits[i][1], X_splits[i][2], "input0", store_path, f"{args.dataset_name}_{i}")
        fairseq_preprocess_cmd(y_splits[i][0], y_splits[i][1], y_splits[i][2], "label", store_path, f"{args.dataset_name}_{i}")
else:
    fairseq_preprocess_cmd(X_splits[0], X_splits[1], X_splits[2], "input0", store_path, args.dataset_name)
    fairseq_preprocess_cmd(y_splits[0], y_splits[1], y_splits[2], "label", store_path, args.dataset_name)