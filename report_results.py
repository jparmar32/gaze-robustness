import json
import numpy as np
import os


train_set = 'cxr_p'
test_set = 'cxr_p'
ood_shift = None
val = False
metric = 'test_auroc'
subclass_eval = False
gaze_task = "actdiff_lungmask"
tuning_eval = False
lungmask_size = 224
segmentation_class = "positive"
augmentation_type = "color_jitter"
al = '1e-5'

#/mnt/data/gaze_robustness_results/actdiff_lungmask/segmentation_classes_positive/augmentation_type_gaussian_noise/lungmask_size_56/actdiff_lambda_1e-7/train_set_cxr_p

results_dir = f'/mnt/data/gaze_robustness_results/{gaze_task}/segmentation_classes_{segmentation_class}/augmentation_type_{augmentation_type}/lungmask_size_{lungmask_size}/actdiff_lambda_{al}/train_set_{train_set}/test_set_{test_set}'
use_top_seeds = False
if ood_shift is not None:
    results_dir = f'/mnt/data/gaze_robustness_results/{gaze_task}/positive_segmentations/cosine_similarity/lungmask_size_{lungmask_size}/train_set_{train_set}/test_set_{test_set}/ood_shift_{ood_shift}'

if val:
    results_dir = f'/mnt/data/gaze_robustness_results/{gaze_task}/positive_segmentations/cosine_similarity/lungmask_size_{lungmask_size}/train_set_{train_set}/val_set_{test_set}_masked'
    metric = 'val_auroc'

if subclass_eval:
    results_dir = results_dir + "_subclass_evaluation"

if tuning_eval:
    results_dir = f'/mnt/data/gaze_robustness_results/{gaze_task}/positive_segmentations/cosine_similarity/lungmask_size_{lungmask_size}/train_set_{train_set}/val_set_{test_set}'
    metric = 'val_auroc'
    


#results_dir = f'/mnt/gaze_robustness_results'

seeds = [x for x in range(4)] 
lrs = ["1e-5", "1e-4", "1e-3"]
wds = ["1e-5", "1e-4", "1e-3", "1e-2", "1e-1", "1e-0"]

#no .0001

#for each lr, wd combo get all seeds, then give results of highest 
'''best_auroc = 0.0
best_lr = 0
best_wd = 0

for lr in lrs:
    for wd in wds:
        means = []
        for cv in seeds: 
        
            lr_string = str(lr)
            wd_string = str(wd)

            #lr_string = lr_string.strip("0")
            #wd_string = wd_string.strip("0")
           
            if wd == 0:
                res_file = os.path.join(results_dir, f"lr_{lr_string}/wd_0/train_set_{train_set}/val_set/seed_{cv}/results.json")
            else:
                #lr_string = lr_string.replace("0", "")
                #wd_string = wd_string.replace("0", "")
                res_file = os.path.join(results_dir, f"lr_{lr_string}/actdifflamb_{wd_string}/train_set_{train_set}/val_set/seed_{cv}/results.json")

            with open(res_file) as data_file:
                results = json.load(data_file)
                means.append(results[metric])


        mean = np.mean(means)
        print(f"lr: {lr} and wd: {wd} with mean auroc: {mean} and std: {np.std(means):.3f}")

        if mean >= best_auroc:
            best_auroc = mean
            best_lr = lr
            best_wd = wd

print(f"best lr: {best_lr} and best wd: {best_wd} with mean auroc: {best_auroc} and and std: {np.std(means):.3f}")

'''

### iterate 

if use_top_seeds:
    top_seeds = []
    for cv in seeds:
        val_results_dir = f'/mnt/data/gaze_robustness_results/{gaze_task}/train_set_{train_set}/val_set_{train_set}'
        res_file = os.path.join(val_results_dir, f"seed_{cv}/results.json")
        
        with open(res_file) as data_file:
            results = json.load(data_file)
            top_seeds.append((cv,results["val_auroc"]))

    top_seeds = sorted(top_seeds, key=lambda tup: tup[1], reverse=True)

    seeds = [seed for seed,_ in top_seeds[:5]]
   

means = []
for cv in seeds:
    res_file = os.path.join(results_dir, f"seed_{cv}/results.json")

    with open(res_file) as data_file:
        results = json.load(data_file)
        means.append(results[metric])

print(f"\nMean Auroc: {np.mean(means):.3f}")
print(f"\nStd: {np.std(means):.3f}")