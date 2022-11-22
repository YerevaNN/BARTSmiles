import argparse


p = argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--dataset-name", 
                    type=str, 
                    required=True)
p.add_argument("--dataset-size", 
                    type=str, 
                    default="4200")
p.add_argument("--subtasks", 
                    type=int, 
                    default=1)
p.add_argument("--single-task", 
                    type=bool, 
                    default=False)
p.add_argument("--epoch", 
                    type=int, 
                    default=10)
p.add_argument("--batch-size", 
                    type=int, default=16)

p.add_argument("--is-Regression", 
                    default=False, 
                    help="Regrestion։ True or Classification: False")

p.add_argument("--add-noise", 
                    type=bool,
                    help="True or False")

p.add_argument('--root', default="/home/gayane/BartLM",
                    help="add your root path")

args = p.parse_args()
root = args.root
name = args.dataset_name
ep = args.epoch
bs = args.batch_size
single_task = bool(args.single_task)
dataset_size = int(args.dataset_size)
subtask = args.subtasks
print(single_task, type(args.single_task))
print(subtask)
print(0 if single_task else subtask)
# subtask = 0 if single_task else subtask
# print(subtask)
TOTAL_NUM_UPDATES = (dataset_size * 0.8) / (ep * bs)
# False
time_param = 5.2 / 7000

head = ["","Type","Experimental","Datasize","# of steps","# of subtasks","lr","Minutes to train 1 subtask","Hours to train all subtasks","dropout","noise_type","lambda"]

regr_or_class = "Regression" if args.is_Regression else "Classification"
minuts_1_task = dataset_size * 0.8 * time_param * 10
Hours_to_train_all_subtasks = minuts_1_task * (subtask) / 60 if args.single_task else minuts_1_task * time_param * subtask / 6

minuts_1_task = round(minuts_1_task, 1)
Hours_to_train_all_subtasks = round(Hours_to_train_all_subtasks, 1)
learning_rate = ["5e-6", "1e-5", "3e-5"]
dropouts = [ 0.1, 0.2, 0.3 ] 

lmb = [0.1, 0.5, 1.0, 5.0]
noise = ["uniform", "normal"]
path = f"{root}/BARTSmiles/fine-tuning/grid_search.csv"
with open(path, 'w') as f:
    head = ["","Type","Datasize","# of steps","# of subtasks","lr","Minutes to train 1 subtask","Hours to train all subtasks","dropout","noise_type","lambda\n"]
    f.write(",".join(head))
    task = subtask if single_task else 1
    print(task)
    # print (task, 1 if not single_task else subtask)
    for lr in learning_rate:
        for dropout in dropouts:
            if args.add_noise:
                for nt in noise:
                    for ld in lmb: 
                        print(subtask)
                        row = f"{name},{regr_or_class},{dataset_size},{int(TOTAL_NUM_UPDATES*100)},{task},{lr},{minuts_1_task * (subtask )},{Hours_to_train_all_subtasks},{dropout},{nt},{ld}"
                        f.write(row)
                        f.write('\n')
                        
            else:
                row = f"{name},{regr_or_class},{dataset_size},{int(TOTAL_NUM_UPDATES*100)},{task},{lr},{minuts_1_task * (subtask )},{Hours_to_train_all_subtasks},{dropout}"
                f.write(row)
                f.write('\n')




         