from fairseq.data.data_utils import load_indexed_dataset
from fairseq.models.bart import BARTModel
from fairseq.data import Dictionary
import torch.nn.functional as F 
from fairseq.data.data_utils import collate_tokens
from tqdm import tqdm
import numpy as np
import torch
import os
import argparse





parser = argparse.ArgumentParser()
parser.add_argument('--dataset-name', type=str, required=True)
parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--output-path', type=str, default='/mnt/good/gayane/data/data_load_folder')
parser.add_argument('--path', type=str, required=True, )
args = parser.parse_args()


dataset = args.dataset_name
np_filename = os.path.join(args.output_path, f"np_{dataset}.npy")
path = args.path
print(np_filename)

if os.path.exists(np_filename):
    print(f"The file {np_filename} already exists")
    exit()
path = "/home/gayane/BartLM/Bart-smiles_testing/chemical"
store_path = f"{path}/checkpoints/evaluation_data"
model = f"{store_path}/{dataset}/processed/input0"

bart = BARTModel.from_pretrained(model,  checkpoint_file = f'{path}/checkpoints/pretrained.pt', 
                                 bpe="sentencepiece",
                                 sentencepiece_model=f"{path}/tokenizer/chem.model")

input_dict = Dictionary.load(f"{store_path}/{dataset}/processed/input0/dict.txt")

bart.eval()
bart.cuda()

def get_data(data_type):
    input_dict = Dictionary.load(f"{store_path}/{dataset}/processed/input0/dict.txt")
    smiles = load_indexed_dataset(
        f"{store_path}/{dataset}/processed/input0/{data_type}", input_dict)
    return list(smiles)

sm_train = get_data("train") 



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
                # manually cropping till the padding
                X.append(feat[:len(inp)].mean(axis=0).numpy())
    return X

print("starting extract train data")
X_ = get_features(sm_train)
X = np.array(X_)
print("X.shape", X.shape)

print(f"Saving to {np_filename}")
np.save(np_filename, X)

    
    
