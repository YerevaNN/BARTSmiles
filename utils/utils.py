from sklearn.metrics import  roc_auc_score, classification_report, mean_squared_error
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
import torch.nn.functional as F 
from rdkit import Chem
import pandas as pd
import numpy as np
import torch
import os

def compute_rmse(y_pred, y, ma, mi):
    y_prd = [(ma -mi)*x +mi  for x in y_pred]
    y_l = [(ma -mi)*x + mi  for x in y]
    df = pd.DataFrame(data={"y_l": y, "y_pred": y_prd, "y_l_scale": y_l, "y_pred_scale": y_pred})
    rmse = np.sqrt(mean_squared_error([(ma -mi)*x + mi  for x in y], [(ma -mi)*x +mi  for x in y_pred]))
    print(f"RMSE: {rmse}")
    return rmse

def compute_auc(y_pred, y):
    auc = roc_auc_score(y, y_pred)
    print(f"ROC_AUC_SCORE: {auc}")
    
    return auc

def compute_conf_matrix(y_pred, y):
    print("Confusion matrix:")
    y_pred_binary = np.array(y_pred) > 0.5
    print(classification_report(y, y_pred_binary))

def multi_task_predict(self, head: str, tokens: torch.LongTensor, return_logits: bool = False, dataset_js: dict = {}):
    if tokens.dim() == 1:
        tokens = tokens.unsqueeze(0)
    features = self.extract_features(tokens.to(device=self.device))
    sentence_representation = features[
        tokens.eq(self.task.source_dictionary.eos()), :
    ].view(features.size(0), -1, features.size(-1))[:, -1, :]
    logits = list()
    for i in range(len(dataset_js["class_index"])>1): 
        logits.append(self.model.classification_heads[head+str(i)](sentence_representation))
    if return_logits:
        return logits
    probabies = list()
    for i in range(len(dataset_js["class_index"])>1): 
        probabies.append(F.log_softmax(logits[i], dim=-1))
    return probabies

def tokenize(X_splits, root):
    print("Tokenizing")
    splits = []
    for path_ in X_splits:
        cur_path = path_.replace('raw', 'tokenized')
        print(path_)
        print(cur_path)
        splits.append(cur_path)
        cmd = f"python {root}/BARTSmiles/preprocess/spm_parallel.py --input {path_} --outputs {cur_path} --model {root}/chemical/tokenizer/chem.model"
        print(cmd)
        os.system(cmd)
    return splits

def create_raw(path, names, _train, _val, _test, file_output = ".input0"):
    _splits = list()
    print(f"Writing {file_output} Splits")
    for name, inp_or_trg in zip(names, (_train, _val, _test)):
        print(name + file_output)
        new_path = f"{path}" + "/raw/" + name + file_output 
        _splits.append(new_path)
        with open(new_path, "w+") as f:
            for i_or_t in inp_or_trg:
                f.write(f"{i_or_t}\n")
    return _splits


def getMurcoScaffoldList(df: pd.DataFrame, column: str, include_chirality: bool = False):
    _scaff = list()
    for i in range(len(df)):
        mol = Chem.MolFromSmiles(df[column][i])
        _scaff.append(MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality))
    df["MurckoScaffold"] = _scaff
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    return df


def generateMurcoScaffold(dataset_name: str, root):

    path = f"{root}/chemical/checkpoints/evaluation_data/{dataset_name}/{dataset_name}/"
    train_df = pd.read_csv(f"{path}train_{dataset_name}.csv")
    valid_df = pd.read_csv(f"{path}valid_{dataset_name}.csv")
    test_df = pd.read_csv(f"{path}test_{dataset_name}.csv")

    include_chirality = False
    train_df = getMurcoScaffoldList(train_df, 'ids', include_chirality)
    valid_df = getMurcoScaffoldList(valid_df, 'ids', include_chirality)
    test_df = getMurcoScaffoldList(test_df, 'ids', include_chirality)

    print(train_df)
    return train_df, valid_df, test_df


def fairseq_preprocess_cmd(root, _train, _valid, _test, input0_or_label, store_path, dataset_name, src_dict):

    os.system(('fairseq-preprocess --only-source '
        f'--trainpref "{_train}" '
        f'--validpref "{_valid}" '
        f'--testpref "{_test}" '
        f'--destdir "{store_path}/{dataset_name}/processed/{input0_or_label}" --workers 60 '
        f'{src_dict}'))
