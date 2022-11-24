from tqdm import tqdm
import pandas as pd
import argparse
import wandb
import json
import csv
import os

os.environ['MKL_THREADING_LAYER'] = 'GNU'

parser = argparse.ArgumentParser()

parser.add_argument('--root', default="/home/gayane/BartLM",
                    help="add your root path")
parser.add_argument("--disk", default="/mnt/good/gayane/data/chkpt/")

args = parser.parse_args()
disk = args.disk
root = args.root

with open(f'{root}/BARTSmiles/evaluation/wandb_url.csv') as f:
    r = csv.DictReader(f)
    lines = [row for row in r]
    print(f"------------> {lines}")

COLUMN_NAMES = ["name", 'lr','dropout', 'total_num_update','warmup_updates','save_dir', 'batch_size']
df = pd.DataFrame(columns=COLUMN_NAMES)

d2 = {}

for i, task in zip(range(len(lines)),tqdm(lines)):
    api = wandb.Api()
    run = api.run(task['url'])
    params = json.loads(run.json_config)

    lr = str(params['args']['value']['lr'][0]).replace("0", "")
    dropout = params['args']['value']['dropout']
    total_num_update = params['args']['value']['total_num_update']
    warmup_updates = params['args']['value']['warmup_updates']
    batch_size = 2* params['args']['value']['update_freq'][0]
    save_dir = params['args']['value']['save_dir']

    if 'noise_type' not in params['args']['value'].keys():
         noise_type, r3f = "", 0
    else:
        noise_type, r3f = params['args']['value']['noise_type'], params['args']['value']['r3f_lambda']
    name = run.name
    is_regress = params['args']['value']['regression_target']
    # codename = f"{name}_bs_{bs}_dropout_{drout}_lr_{lr}_totalNum_{TOTAL_NUM_UPDATES}_warmup_{WARMUP_UPDATES}"

    dic = {"lr" : lr,
            "dropout" : dropout,
            "total_num_update" : total_num_update,
            "warmup_updates" : warmup_updates,
            "batch_size" : batch_size,
            "save_dir" : save_dir,
            "name" : name,
            "path_of_history": f'{root}/chemical/checkpoints/training_results_csv/{name}.csv',
            'wandb_id' : task['url'],
            'is_regression' : is_regress,
            "r3f": r3f,
            "noise_type": noise_type
    }
    df = df.append(dic, ignore_index=True)

    params = json.loads(run.json_config)
    if df['r3f'][i] == 0 :
        if params['args']['value']['regression_target']:
            df_summ = run.history()[['_step', 'train/loss_scale', 'train/loss', 
                'valid/loss', 'train/lr', 'valid/best_loss']]
        else:
            df_summ = run.history()[['_step', 'train/accuracy', 'train/loss_scale', 'train/loss', 
                'valid/loss', 'train/lr', 'valid/accuracy', 'valid/best_loss']]
    else: 
        df_summ = run.history()[['_step', 'train/accuracy', 'train/loss_scale', 'train/loss', 
                'valid/loss', 'train/lr', 'valid/accuracy', 'valid/best_loss', "train/symm_kl", "valid/symm_kl"]]


    print(run.name)


    df_summ = df_summ[df_summ['valid/loss'].notna()]
    df_summ.reset_index().to_csv(f'{root}/chemical/checkpoints/train_params_csv/{name}.csv')

df.to_csv(f'{root}/chemical/checkpoints/training_results_csv/wandb_path.csv')



names = []


for i in lines:
    for j in range(len(df)):

        if df['wandb_id'][j] == i['url']:
            names.append(df['name'][j])
            continue
    print(i['url'])

    
best_val_loss = []
best_val_accuracy = []
for name in names:
    params_df = pd.read_csv(f'{root}/chemical/checkpoints/train_params_csv/{name}.csv')

    best_val_loss.append(params_df['valid/loss'].argmin(skipna=True) +1 )
    best_val_accuracy.append(params_df['valid/loss'].argmax(skipna=True) + 1)


train_sum = pd.read_csv(f'{root}/chemical/checkpoints/training_results_csv/wandb_path.csv')

for i in range(len(names)):
    n = names[i]
    row = train_sum.loc[train_sum['name'] == n].to_dict()
    print(n)
    total_num_update = row['total_num_update'][i]
    warmup_updates = row['warmup_updates'][i]
    upper_bound_best_val_loss = best_val_loss[i]+2 if best_val_loss[i] > 1 else 4
    upper_bound_best_val_acc = best_val_accuracy[i] + 2 if best_val_accuracy[i] > 1 else 4
    upper_bound_last = 11
    bs = 16
    chkpt_count = 4
    lr = str(row['lr'][i]).replace('0', '')
    if row['r3f'][i] != 0 :
        r3f_lambda = row['r3f'][i]
        noise_type = row['noise_type'][i]

    drout = row["dropout"][i]
    chkpt_name_best_val_loss = f"chkpt_upper_bound_best_val_loss_{upper_bound_best_val_loss}_count_{chkpt_count}"
    chkpt_name_best_val_acc = f"chkpt_upper_bound_best_val_acc_{upper_bound_best_val_acc}_count_{chkpt_count}"
    chkpt_name_last = f"chkpt_upper_bound_last_{upper_bound_last}_count_{chkpt_count}"
    
    directory = f"{disk}{n}"
    task_name = n.split("_")[0] + '_' + n.split("_")[1] if n.split("_")[1].split('-')[0].isdigit() or n.split("_")[1].isdigit() else n.split("_")[0]
    is_regress = row['is_regression']
    noise_params = f" --noise_type {noise_type} --r3f {r3f_lambda}" if noise_type in ["uniform", "normal"] else ""
    cmd = f"""python {root}/BARTSmiles/evaluation/compute_score.py --lr {lr} --dropout {drout}{noise_params} --dataset-name {task_name} --subtask 1 --warmup-update {warmup_updates} --total-number-update {total_num_update} --checkpoint_name checkpoint_best.pt >> {root}/chemical/log/{task_name}.log""" 
    print("---------------------> chkpt_name_best_val_loss: ", cmd)
    os.system(cmd)
    cmd = f"""python {root}/fairseq/scripts/average_checkpoints.py --inputs {directory}/ --output {directory}/{chkpt_name_best_val_loss}.pt --checkpoint-upper-bound {upper_bound_best_val_loss} --num-epoch-checkpoints {chkpt_count}""" 
    print("---------------------> chkpt_name_best_val_loss SWA: ")
    print(cmd)
    os.system(cmd)

    
    cmd = f"""python {root}/BARTSmiles/evaluation/compute_score.py --lr {lr} --dropout {drout} {noise_params} --dataset-name {task_name} --subtask 1 --warmup-update {warmup_updates} --total-number-update {total_num_update} --checkpoint_name {chkpt_name_best_val_loss}.pt >> {root}/chemical/log/{task_name}.log""" 
    print("---------------------> chkpt_name_best_val_loss SWA score: ", cmd)
    os.system(cmd)



    if not is_regress:

        cmd = f"""python {root}/fairseq/scripts/average_checkpoints.py --inputs {directory}/ --output {directory}/{chkpt_name_best_val_acc}.pt --checkpoint-upper-bound {upper_bound_best_val_acc} --num-epoch-checkpoints {chkpt_count}""" 
        print("---------------------> chkpt_name_best_val_acc SWA: ", cmd)
        os.system(cmd)


        cmd = f"""python {root}/BARTSmiles/evaluation/compute_score.py --lr {lr} --dropout {drout} {noise_params} --dataset-name {task_name} --subtask 1 --warmup-update {warmup_updates} --total-number-update {total_num_update} --checkpoint_name {chkpt_name_best_val_acc}.pt >> {root}/chemical/log/{task_name}.log""" 
        print("---------------------> chkpt_name_best_val_acc SWA score: ", cmd)
        os.system(cmd)


        cmd = f"""python {root}/BARTSmiles/evaluation/compute_score.py --lr {lr} --dropout {drout} --dataset-name {task_name} {noise_params} --subtask 1 --warmup-update {warmup_updates} --total-number-update {total_num_update} --checkpoint_name checkpoint{best_val_accuracy[i]}.pt >> {root}/chemical/log/{task_name}.log""" 
        print("--------------------->  chkpt_name_best_val_acc score: ")
        print(cmd)
        os.system(cmd)


    cmd = f"""python {root}/fairseq/scripts/average_checkpoints.py --inputs {directory}/ --output {directory}/{chkpt_name_last}.pt --checkpoint-upper-bound 11 --num-epoch-checkpoints {chkpt_count}""" 
    print(" --------------------> last checkpoints SWA: ", cmd)
    os.system(cmd)


    cmd = f"""python {root}/BARTSmiles/evaluation/compute_score.py --lr {lr} --dropout {drout} --dataset-name {task_name} {noise_params} --subtask 1 --warmup-update {warmup_updates} --total-number-update {total_num_update} --checkpoint_name {chkpt_name_last}.pt >> {root}/chemical/log/{task_name}.log""" 
    print(" --------------------> last checkpoints SWA score: ")
    print(cmd)
    os.system(cmd)



    cmd = f"""rm -rf {directory}/checkpoint_last.pt {directory}/checkpoint.best*.pt""" 
    print("best remove last and best2 checkpoints: ", cmd)
    os.system(cmd)
