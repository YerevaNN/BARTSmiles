import matplotlib.pyplot as plt
from itertools import chain
from pathlib import Path
from sklearn import metrics
import torch.nn as nn
from fairseq.data.data_utils import collate_tokens
from tqdm import tqdm
import pandas as pd
import numpy as np
import logging
import torch
import json
import os
import argparse



parser = argparse.ArgumentParser()
parser.add_argument('--dataset-name', type=str, required=True)
parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--output-path', type=str, default='/mnt/good/gayane/data/data_load_folder')
args = parser.parse_args()


dataset = args.dataset_name
np_filename = os.path.join(args.output_path, f"np_{dataset}.npy")
print(np_filename)

if os.path.exists(np_filename):
    print(f"The file {np_filename} already exists")
    exit()
    

from fairseq.data.data_utils import load_indexed_dataset
from fairseq.models.bart import BARTModel
from fairseq.data import Dictionary
import torch.nn.functional as F 

store_path = f"{path}/checkpoints/evaluation_data"
model = f"{store_path}/{dataset}/processed"


bart = BARTModel.from_pretrained(model,  checkpoint_file = f'{path}/checkpoints/pretrained.pt', 
                                bpe="sentencepiece",
                                sentencepiece_model=f"{path}/tokenizer/chem.model")

input_dict = Dictionary.load(f"{store_path}/{dataset}/processed/input0/dict.txt")

bart.eval()
bart.cuda()


data_type = 'train'
def get_data(data_type):
    input_dict = Dictionary.load(f"{store_path}/{dataset}/processed/input0/dict.txt")
    smiles = load_indexed_dataset(
        f"{store_path}/{dataset}/processed/input0/{data_type}", input_dict)
    return list(smiles)

sm_train = get_data("train") # , y_pred_train
sm_valid = get_data("valid") # , y_pred_valid
sm_test = get_data("test") # , y_pred_test


smi = []

def get_features(sm):
    X = []
    with torch.no_grad():
        count = len(sm)
        batch_count = int(np.ceil(count / args.batch_size))
        for i in tqdm(range(batch_count)):
            inputs = sm[i * args.batch_size : (i+1) * args.batch_size]
            batch = collate_tokens(
                inputs, pad_idx=1
            ).to(bart.device)[:, :128] # manually cropping to the max length

            last_layer_features = bart.extract_features(batch)
            
            assert len(inputs) == len(last_layer_features), "len(inputs) == len(last_layer_features)"
    
            for inp, feat in zip(inputs, last_layer_features.to("cpu")):
#                 print(inp.shape, feat[:len(inp)].mean(axis=0).shape)
                # manually cropping till the padding
                X.append(feat[:len(inp)].mean(axis=0).numpy())
    return X

print("starting extract train data")
X_train = get_features(sm_train)
print("starting extract validation data")
X_valid = get_features(sm_valid)
print("starting extract test data")
X_test = get_features(sm_test)





X = np.array(X_train + X_valid + X_test)
print("X.shape", X.shape)

print(f"Saving to {np_filename}")
np.save(np_filename, X)

    
    
