import sys
from fairseq.data.data_utils import load_indexed_dataset
from sklearn.metrics import roc_auc_score
from fairseq.models.bart import BARTModel
from fairseq.data import Dictionary
import torch.nn.functional as F 
from tqdm import tqdm
import pandas as pd
import numpy as np
import argparse
import torch
import json
import os
os.environ['MKL_THREADING_LAYER'] = 'GNU'


parser = argparse.ArgumentParser()
parser.add_argument('--dataset-name', required=True,
                        help='dataset name.')
parser.add_argument('--root', default="/home/gayane/BartLM",
                    help="add your root path")
parser.add_argument("--disk", default="/mnt/good/gayane/data/chkpt/")
parser.add_argument('--subtask',
                        help='subtask count')
parser.add_argument('--warmup-update',
                        help='warmup update', default=118)
parser.add_argument('--total-number-update',
                        help='total number update', default=739)
parser.add_argument('--lr', default="5e-6",
                        help='learning rate')
parser.add_argument('--dropout', default="0.1",
                        help='learning rate')
parser.add_argument('--r3f',
                        help='lambda param')
parser.add_argument('--noise_type',
                        help='normal or uniform')
parser.add_argument('--dataset-type', default="valid",
                        help='train, valid or test')
parser.add_argument('--checkpoint_name', default="checkpoint_best.pt")
args = parser.parse_args()
 
root = args.root
disk = args.disk
sys.path.append(f"{root}/BARTSmiles/utils/")
from utils import compute_rmse, compute_auc, compute_conf_matrix, multi_task_predict

dataset = args.dataset_name #if args.dataset_name in set(["esol", "freesolv", "lipo", "BBBP", "BACE", "HIV"]) else f"{args.dataset_name}_{args.subtask}"

store_path = f"{root}/chemical/checkpoints/evaluation_data"
model = f"{store_path}/{dataset}/processed"


with open(f'{root}/BARTSmiles/datasets.json') as f:
    datasets_json = json.load(f)
dataset_js = datasets_json[dataset]
task_type = dataset_js['type']
is_regression = dataset_js["return_logits"]
if task_type == "regression":
    mi = dataset_js['minimum']
    ma = dataset_js['maximum']

os.system(f"mkdir -p {store_path}/{dataset}/")
os.system(f"mkdir -p {store_path}/{dataset}/processed/")
os.system(f"mkdir -p {store_path}/{dataset}/processed/input0/")
os.system(f"mkdir -p {store_path}/{dataset}/processed/label/")

warmup = args.warmup_update
totNumUpdate = args.total_number_update
lr = args.lr
noise_type = args.noise_type
r3f_lambda = args.r3f
drout = args.dropout
dataset_type = args.dataset_type

noise_params = f"_noise_type_{noise_type}_r3f_lambda_{r3f_lambda}" if noise_type in ["uniform", "normal"] else ""

chkpt_path = f"{disk}{dataset}_bs_16_dropout_{drout}_lr_{lr}_totalNum_{totNumUpdate}_warmup_{warmup}{noise_params}/{args.checkpoint_name}"
print(f"checkpoint path: {chkpt_path}") 
bart = BARTModel.from_pretrained(model,  checkpoint_file = chkpt_path, 
                                 bpe="sentencepiece",
                                 sentencepiece_model=f"{root}/chemical/tokenizer/chem.model")
bart.eval()
bart.cuda(device=1)


input_dict = Dictionary.load(f"{store_path}/{dataset}/processed/input0/dict.txt")
smiles = list(load_indexed_dataset(
    f"{store_path}/{dataset}/processed/input0/{dataset_type}", input_dict))

if len(dataset_js["class_index"])>1:
    test_label_path = list()
    for i in range(len(dataset_js["class_index"])):
        test_label_path.append(f"{store_path}/{dataset}_{i}/processed/label/{dataset_type}")

else:
    test_label_path = f"{store_path}/{dataset}/processed/label/{dataset_type}"

if task_type == 'classification':
    if len(dataset_js["class_index"])>1:
        target_dict = list()
        targets_list = list()
        for i in range(len(dataset_js["class_index"])):
            target_dict.append(Dictionary.load(f"{store_path}/{dataset}_{i}/processed/label/dict.txt"))
            targets_list.append(list(load_indexed_dataset(test_label_path[i], target_dict[i])))
        
    else: 
        target_dict = Dictionary.load(f"{store_path}/{dataset}/processed/label/dict.txt")
        targets = list(load_indexed_dataset(test_label_path, target_dict))
elif task_type == 'regression':
    with open(f'{test_label_path}.label') as f:
        lines = f.readlines()
        targets = [float(x.strip()) for x in lines]

y_pred = []
y = []
sm = []
if len(dataset_js["class_index"])>1:
    y_pred_list = list()
    y_list = list()
    for j in range(len(dataset_js["class_index"])):
        y_pred = list()
        y = list()
        for i, (smile, target) in tqdm(list(enumerate(zip(smiles, targets_list[j])))):
            smile = torch.cat((torch.cat((torch.tensor([0]), smile[:126])), torch.tensor([2])))
            output = bart.predict(f'sentence_classification_head{j}', smile)
            target = target[0].item()
            y_pred.append(output[0][1].exp().item())
            if target_dict.__getitem__(4) == "1":
                y.append(-1 * target + 5)
            else:
                y.append(target - 4)
        y_pred_list.append(y_pred)
        y_list.append(y)
else:
    for i, (smile, target) in tqdm(list(enumerate(zip(smiles, targets)))):
        smile = torch.cat((torch.cat((torch.tensor([0]), smile[:126])), torch.tensor([2])))  
        output = bart.predict('sentence_classification_head', smile, return_logits=is_regression)
        if not is_regression:
            target = target[0].item()
            y_pred.append(output[0][1].exp().item())
            y.append(target - 4)
            
        elif is_regression: 
            y_pred.append(output[0][0].item())
            y.append(target)
        sm.append(bart.decode(smile)[0])
    d = {"SMILES": sm, "prediction": y_pred , "y_true": y }
    df = pd.DataFrame(d) 
    df = df.dropna()
    df.to_csv(f"{root}/chemical/checkpoints/evaluation_data/{args.dataset_name}/{args.dataset_name}_test_.csv")     

if task_type == 'classification':
    if len(dataset_js["class_index"]) >1:
        roc_auc_list = list()
        for i in range(len(dataset_js["class_index"])):
            roc_auc_list.append(roc_auc_score(y_list[i], y_pred_list[i]))
            compute_auc(y_pred_list[i], y_list[i])
            compute_conf_matrix(y_pred_list[i], y_list[i])
        print("ROC_AUC_SCORE_MEAN: ", np.mean(roc_auc_list))
        print(roc_auc_list)
    else: 
        compute_auc(y_pred, y)
        compute_conf_matrix(y_pred, y)
else:
    compute_rmse(y_pred=y_pred, y=y, ma= ma, mi=mi)
