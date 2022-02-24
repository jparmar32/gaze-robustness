import json
import numpy as np
import os


train_set = 'cxr_p'
test_set = 'cxr_p'
ood_shift = None
subclass_eval = None
gaze_task = "segmentation_reg"
results_dir = f'/mnt/data/gaze_robustness_results/{gaze_task}/cw_0.01/train_set_{train_set}/test_set_{test_set}'
if ood_shift is not None:
    results_dir = f'/mnt/data/gaze_robustness_results/{gaze_task}/cw_1/train_set_{train_set}/test_set_{test_set}/ood_shift_{ood_shift}'

if subclass_eval:
    results_dir = results_dir + "_subclass_evaluation"


#results_dir = f'/mnt/gaze_robustness_results'

seeds = [x for x in range(10)] 
lrs = [.01, .001, .0001]
wds = [1, .1, .01, .001, .0001]

#no .0001


'''#for each lr, wd combo get all seeds, then give results of highest 
best_auroc = 0.0
best_lr = 0
best_wd = 0

for lr in lrs:
    for wd in wds:
        means = []
        for cv in seeds: 
        
            lr_string = str(lr)
            wd_string = str(wd)

            lr_string = lr_string.strip("0")
            wd_string = wd_string.strip("0")
           
            if wd == 0:
                res_file = os.path.join(results_dir, f"lr_{lr_string}/wd_0/gaze_data_augmentation/train_set_{train_set}/test_set_{test_set}/seed_{cv}/results.json")
            else:
                res_file = os.path.join(results_dir, f"lr_{lr_string}/wd_{wd_string}/gaze_data_augmentation/train_set_{train_set}/test_set_{test_set}/seed_{cv}/results.json")

            with open(res_file) as data_file:
                results = json.load(data_file)
                means.append(results['test_auroc'])



        mean = np.mean(means)
        print(f"lr: {lr} and wd: {wd} with mean auroc: {mean}")

        if mean >= best_auroc:
            best_auroc = mean
            best_lr = lr
            best_wd = wd

print(f"best lr: {best_lr} and best wd: {best_wd} with mean auroc: {best_auroc}")'''




means = []
for cv in seeds:
    res_file = os.path.join(results_dir, f"seed_{cv}/results.json")

    with open(res_file) as data_file:
        results = json.load(data_file)
        means.append(results['test_auroc'])


print(f"\nMean Auroc: {np.mean(means):.3f}")
print(f"\nStd: {np.std(means):.3f}")