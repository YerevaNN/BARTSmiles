# BARTSmiles: Generative Masked Language Models for Molecular Representations

**BARTSmiles** is a chemical language model based on BART, trained on 1.7 billion SMILES strings from ZINC20 dataset.

**BARTSmiles** can be fine-tuned on chemical property prediction and generative tasks, including chemical reaction prediction and retrosynthesis. *BARTSmiles* allows to get multiple state-of-the-art results. 




Assuming you are in a root folder. In the rest of this readme, we will name this folder `root`. 

Clone BARTSmiles repo in the root directory:

```bash
git clone https://github.com/YerevaNN/BARTSmiles.git
```
## Setup 

Setup a conda environment:

```bash
cd ~/root/BARTSmiles
conda env create --file=environment.yml
conda activate bartsmiles
cd ~/root/
```
Clone and install Fairseq in the root directory:

```bash 
git clone https://github.com/facebookresearch/fairseq.git
cd ~/root/fairseq
pip install --editable ./
```
You need to add ```add_if_not_exist=False``` in this row:
```python
tokens = self.task.source_dictionary.encode_line(bpe_sentence, append_eos=False, add_if_not_exist=False)
```
of this file:
```~/root/fairseq/fairseq/models/bart/hub_interface.py```
```
cd ~/root/

```

Download BARTSmiles pre-trained model and the vocabulary:

```bash
mkdir -p ~/root/chemical/tokenizer
cd ~/root/chemical/tokenizer
wget http://public.storage.yerevann.com/BARTSmiles/chem.model
wget http://public.storage.yerevann.com/BARTSmiles/chem.vocab.fs

cd ..
mkdir ~/root/checkpoints
cd ~/root/checkpoints
wget http://public.storage.yerevann.com/BARTSmiles/pretrained.pt
cd ~/root/BARTSmiles/
```

## Fine-tuning on MoleculeNet tasks

1) Download and preprocess MoleculeNet datasets: 
Use the following command from the BARTSmiles folder:
```
python preprocess/process_datasets.py --dataset-name esol --is-MoleculeNet True --root [the path where locate your BARTSmiles folder]
```
This will create folders in `~/root/chemical/checkpoints/evaluation_data/esol` directory: 
```
    esol
    │
    ├───esol
    │      train_esol.csv
    │      valid_esol.csv
    │      test_esol.csv
    │
    │
    ├───processed
    │   │
    │   ├───input0
    │   │       dict.txt
    │   │       preprocess.log
    │   │       test.bin
    │   │       train.bin
    │   │       valid.bin
    │   │       test.idx
    │   │       valid.idx
    │   │       train.idx
    │   │
    │   └───label
    │          dict.txt
    │          preprocess.log
    │          test.bin
    │          valid.bin
    │          train.bin
    │          test.idx
    │          valid.idx
    │          train.idx 
    │          test.label
    │          valid.label
    │          train.label
    │
    │
    ├───raw
    |      test.input
    |      test.target
    |      valid.input
    |      valid.target
    |      train.input
    |      train.target
    |   
    |
    |
    └───tokenized
        test.input
        valid.input
        train.input
```

2) Generate the grid of training hyperparameters by running the script `~/root/BARTSmiles/fine-tuning/generate_grid_bartsmiles.py`. This will write grid search parameters in `~/root/BARTSmiles/fine-tuning/grid_search.csv` file.

Command for the regression tasks: 
```
python fine-tuning/generate_grid_bartsmiles.py --root [the path where locate your BARTSmiles folder] --dataset-name esol --single-task True --dataset-size 1128 --is-Regression True
```

Command for the classification tasks having a single subtask: 
```
python fine-tuning/generate_grid_bartsmiles.py --root [the path where locate your BARTSmiles folder] --dataset-name BBBP --single-task True --dataset-size 2039
```

Command for a specific subtask of a multilabel classification task: 
``` 
python fine-tuning/generate_grid_bartsmiles.py --root [the path where locate your BARTSmiles folder] --dataset-name Tox21 --subtasks 12 --single-task False --dataset-size 7831
```
All required parameters for training are in grid_search.csv and you can start the training.

3) Login to your wandb
    Befor start the training you have to login in wandb for tracking the trainings.
    For login you can follow: https://docs.wandb.ai/ref/cli/wandb-login 

4) Train the models using the following command:

```bash
python fine-tuning/train_grid_bartsmiles.py --root [the path where locate your BARTSmiles folder] --disk [the path where you want to store your checkpoints]  >> root/chemical/log/esol.log
```

This will produce a checkpoint in `disk/clintox_1_bs_16_dropout_0.1_lr_5e-6_totalNum_739_warmup_118/` folder.

5) You will write wandb url in `root/BARTSmiles/evaluation/wandb_url.csv` file 
example:

``` 
url

gayanec/Fine_Tune_clintox_0/6p76cyzr
```

6) Perform Stochastic Weight Averaging and evaluate from `~/root/BARTSmiles/evaluation` using the following command.

``` 
python evaluation/evaluate_swa_bartsmiles.py  --root [the path where locate your BARTSmiles folder] --disk [the path will your checkpoints be located] --dataset-type [dataset type: train, valid or test]
```

This will produce a log file with output and averaged checkpoints respectively in `~/root/chemical/log/`  and `disk/clintox_1_bs_16_dropout_0.1_lr_5e-6_totalNum_739_warmup_118/` folders.


## Note 
If you want to fine-tune another dataset you have to add deatails in datasets.json files and your preprocessing code in `~/root/preprocess/process_datasets.py` file in line 103. The key must not contain the '_' symbol unless the following symbols are numbers.

## Load the model 

The model path has to be processed data path and it's important that input0 directory contains dict.txt file. You need to provide the structure of directories mentioned above
```python
model = f"~/root/checkpoints/evaluation_data/data_directory_name/processed/input0"
bart = BARTModel.from_pretrained(model, checkpoint_file = f'~/root/checkpoints/pretrained.pt',
                                 bpe="sentencepiece",
                                 sentencepiece_model=f"~/root/tokenizer/chem.model")
```

## Extruct the features
Extract the last layer's features:

```python
last_layer_features = bart.extract_features(bart.encode(smiles)) 
```
or you can use this file for batches: ```python ~/root/BARTSmiles/utils/extract_features.py --dataset-name esol --batch-size 32 --output-path [where you want to locate the outputs]```

