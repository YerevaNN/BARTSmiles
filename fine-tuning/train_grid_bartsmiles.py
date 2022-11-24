#!/usr/bin/env python
# coding: utf-8
from tqdm import tqdm
import argparse
import csv
import os
# os.environ['MKL_THREADING_LAYER'] = 'GNU'



parser = argparse.ArgumentParser()

parser.add_argument('--root', default="/home/gayane/BartLM",
                    help="add your root path")
parser.add_argument("--disk", default="/mnt/good/gayane/data/chkpt/")

args = parser.parse_args()
disk = args.disk
root = args.root


path = f"{root}/BARTSmiles"

with open(f'{path}/fine-tuning/grid_search.csv') as f:
    r = csv.DictReader(f)
    lines = [row for row in r]



PATH1 = f"{root}/chemical/checkpoints/evaluation_data/"
PATH2 = "/processed"
BART_PATH = f"{root}/chemical/checkpoints/checkpoint_last.pt"



for task in tqdm(lines):
    task_name = task['']
    print(f"\nStarting {task_name}")
    
    TOTAL_NUM_UPDATES = int(float(task['# of steps']))
    WARMUP_UPDATES = int(0.16 * TOTAL_NUM_UPDATES)
    drout = task["dropout"]
    num_class = 2 if task['Type'] == 'Classification' and "AcuteOralToxicity" not in task_name else (1 if "AcuteOralToxicity" not in task_name else 3)
    bs = 16
    noise_type = task['noise_type'] 
    r3f_lambda = task['lambda']
    lr = task['lr']
    print(f"lr: -----> {lr}")
    subtask_count = int(task['# of subtasks'])
    skip_set = set([i for i in range(21)])
    save_check = True
    if save_check :
        keep_last_check = "--keep-last-epochs 10"
    else:
        keep_last_check = "--keep-last-epochs 5"
    if task['Type'] == 'Classification':
        
        p = 23 if 'SIDER-scaffold' in task_name else 0
        print(p)
        for subtask in range(subtask_count):
            # if lr == "1e-5" and drout == 0.1 and (subtask == 0 or subtask == 1 or subtask == 2) :
            #     print("skip")
            #     continue
            noise_params = "" if noise_type not in ["uniform", "normal"] else f"_r3f  --noise-type {noise_type} --r3f-lambda {r3f_lambda}"
            noise_params_name = "" if noise_type not in ["uniform", "normal"] else f"_noise_type_{noise_type}_r3f_lambda_{r3f_lambda}"
            name = f"{task_name}_{subtask}" if subtask_count > 1 else f"{task_name}"
            codename = f"{name}_bs_{bs}_dropout_{drout}_lr_{lr}_totalNum_{TOTAL_NUM_UPDATES}_warmup_{WARMUP_UPDATES}{noise_params_name}"
            
            directory = f"{disk}{codename}"
            
            cmd = f"mkdir -p {directory}"

            os.system(cmd)
            cmd = f"""CUDA_VISIBLE_DEVICES=0 fairseq-train {PATH1}{name}{PATH2} --update-freq {bs//2} --restore-file {BART_PATH} --wandb-project Fine_Tune_{name} --batch-size 2 --task sentence_prediction --num-workers 1 --add-prev-output-tokens --layernorm-embedding --share-all-embeddings --share-decoder-input-output-embed --reset-optimizer --reset-dataloader --reset-meters --required-batch-size-multiple 1 --arch bart_large --skip-invalid-size-inputs-valid-test --criterion sentence_prediction{noise_params} --max-target-positions 128 --max-source-positions 128 --dropout {drout} --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-08 --weight-decay 0.01 --attention-dropout 0.2 --relu-dropout 0.1 --clip-norm 0.1 --lr-scheduler polynomial_decay --lr {lr} --total-num-update {TOTAL_NUM_UPDATES} --max-update {TOTAL_NUM_UPDATES} --warmup-updates {WARMUP_UPDATES} --fp16 --keep-best-checkpoints 1 {keep_last_check} --num-classes {num_class} --save-dir {directory} >> {root}/chemical/log/{codename}.log"""
            print(cmd)
            os.system(cmd)

            cmd = f"""rm -rf {directory}/checkpoint_last.pt {directory}/checkpoint.best*.pt""" 
            print("best remove last and best2 checkpoints: ", cmd)
            os.system(cmd)
            print(f"\n   {subtask+1}/{subtask_count}: Running the following command:")
    else:
        
        codename = f"{task_name}_bs_{bs}_dropout_{drout}_lr_{lr}_totalNum_{TOTAL_NUM_UPDATES}_warmup_{WARMUP_UPDATES}"
        directory = f"{disk}{codename}"  

        cmd = f"mkdir -p {directory}"
            
        os.system(cmd) 
        cmd = f"""CUDA_VISIBLE_DEVICES=0 fairseq-train {PATH1}{task_name}{PATH2} --update-freq {bs//2} --restore-file {BART_PATH} --wandb-project Fine_Tune_{task_name} --batch-size 2 --task sentence_prediction --num-workers 1 --add-prev-output-tokens --layernorm-embedding --share-all-embeddings --share-decoder-input-output-embed --reset-optimizer --reset-dataloader --reset-meters --required-batch-size-multiple 1 --init-token 0 --arch bart_large --skip-invalid-size-inputs-valid-test --criterion sentence_prediction --max-target-positions 128 --max-source-positions 128 --dropout {drout} --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-08 --weight-decay 0.01 --attention-dropout 0.2 --relu-dropout 0.1 --clip-norm 0.1 --lr-scheduler polynomial_decay --lr {lr} --total-num-update {TOTAL_NUM_UPDATES} --max-update {TOTAL_NUM_UPDATES} --warmup-updates {WARMUP_UPDATES} {keep_last_check} --keep-best-checkpoints 1 --fp16 --threshold-loss-scale 1 --fp16-scale-window 128 --max-epoch 10 --best-checkpoint-metric loss --regression-target --num-classes {num_class} --save-dir {directory} >> {root}/chemical/log/{codename}.log"""
        print(f"\n   {1}/{subtask_count}: Running the following command:")
        print(cmd)
        os.system(cmd)

        cmd = f"""rm -rf {directory}/checkpoint_last.pt {directory}/checkpoint.best*.pt""" 
        print("best remove last and best2 checkpoints: ", cmd)
        os.system(cmd)
    
    print("\n\n")
