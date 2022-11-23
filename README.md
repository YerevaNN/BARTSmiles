# BARTSmiles: Generative Masked Language Models for Molecular Representations

## Introduction
**BARTSmiles** is a chemical language model based on BART, which input is SMILES.
We present our pretrained **BARTSmiles** model and fine-tuning strategy on multiple chemical property prediction, chemical reaction prediction and retrosynthesis tasks, and set state-of-the-art results on 11 of them.

### Load BARTSmiles Pre-trained models
`wget link`

### Load Vocab
`wget link`

## Setup 
```bash 
git clone https://github.com/YerevaNN/fairseq.git
cd fairseq
pip install --editable ./
conda create --name <env_name> --file examples/requirements.txt
```

## Fine-tune MoleculeNet tasks

1) Download and preprocess MoleculeNet datasets: 
Use CMD from the `root/BARTSmiles/preprocess`:
```
python root/BARTSmiles/process/process_datasets.py --dataset-name esol --is-MoleculeNet True
```
This will create folders in `root/chemical/checkpoints/evaluation_data/esol` directory: 
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

2) Generate the grid of training hyperparameters `root/BARTSmiles/fine-tuning/generate_grid_bartsmiles.py`. This will create a csv.

CMD for regression task: 
```
python fine-tuning/generate_grid_bartsmiles.py --dataset-name esol --single-task True --dataset-size 1128 --is-Regression True
```

CMD for classification single task: 
```
python fine-tuning/generate_grid_bartsmiles.py --dataset-name BBBP --single-task True --dataset-size 2039
```

CMD for classification multilabel task: 
``` 
python fine-tuning/generate_grid_bartsmiles.py --dataset-name Tox21 --subtasks 12 --single-task False --dataset-size 7831
```

This will write grid search parameters in `root/BARTSmiles/fine-tuning/grid_search.csv` file.

3) Login to your wandb
    You have to login in wandb.
    You can follow: https://docs.wandb.ai/ref/cli/wandb-login 

4) Train the models 
CMD: 
```python root/BARTSmiles/fine-tuning/train_grid_bartsmiles.py  >> root/chemical/log/esol.log``` 
This will produce checkpoint in 
`disk/clintox_1_bs_16_dropout_0.1_lr_5e-6_totalNum_739_warmup_118/` folder.

5) You will write wandb url in `wandb_url.csv` file 
example:

``` 
url

gayanec/Fine_Tune_clintox_0/6p76cyzr
```

6) Perform SWA and evaluate from `root/BARTSmiles/evaluation`.
CMD: 
``` 
python evaluate_swa_bartsmiles.py 
```

This will produce a log file with output and averaged checkpoints respectivly in   `root/chemical/log/`  and `root_data/clintox_1_bs_16_dropout_0.1_lr_5e-6_totalNum_739_warmup_118/` folders.