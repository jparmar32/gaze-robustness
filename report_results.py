import json
import numpy as np
import os

train_set = 'cxr_p'
test_set = 'mimic_cxr'
ood_shift = 'hospital'
val = False
metric = 'test_auroc'
subclass_eval = False
gaze_task = "rrr_lungmask"
tuning_eval = False
lungmask_size = 224
segmentation_class = "positive"
augmentation_type = "normal"
similarity_type = "l2"
al = '1e-2'
machine = 'gemini'
prepend_path = '/media/nvme_data/jupinder_cxr_robustness_results' if machine == 'gemini' else '/mnt/data/gaze_robustness_results'

results_dir = f'{prepend_path}/{gaze_task}/cxr_p_train_size_small/similarity_type_{similarity_type}/segmentation_classes_positive/augmentation_type_normal/lungmask_size_{lungmask_size}/saliency_lambda_{al}/train_set_{train_set}/test_set_{test_set}'
use_top_seeds = False

if ood_shift is not None:
    results_dir = f'{prepend_path}/{gaze_task}/cxr_p_train_size_small/similarity_type_{similarity_type}/segmentation_classes_positive/augmentation_type_normal/lungmask_size_{lungmask_size}/saliency_lambda_{al}/train_set_{train_set}/test_set_{test_set}/ood_shift_{ood_shift}'
if val:
    results_dir = f'{prepend_path}/{gaze_task}/cxr_p_train_size_small/similarity_type_{similarity_type}/segmentation_classes_positive/augmentation_type_normal/lungmask_size_{lungmask_size}/saliency_lambda_{al}/train_set_{train_set}/val_set_{train_set}'
    metric = 'val_auroc'

if subclass_eval:
    results_dir = results_dir + "_subclass_evaluation"

if tuning_eval:
    results_dir = f'{prepend_path}/{gaze_task}/similarity_type_{similarity_type}/segmentation_classes_{segmentation_class}/augmentation_type_{augmentation_type}/lungmask_size_{lungmask_size}/actdiff_lambda_{al}/train_set_{train_set}/val_set_{train_set}'
    metric = 'val_auroc'

seeds = [x for x in range(5)] 


if use_top_seeds:
    top_seeds = []
    for cv in seeds:
        val_results_dir = f'{prepend_path}/{gaze_task}/cxr_p_train_size_small/similarity_type_{similarity_type}/segmentation_classes_positive/augmentation_type_normal/lungmask_size_{lungmask_size}/saliency_lambda_{al}/train_set_{train_set}/val_set_{train_set}'
        res_file = os.path.join(val_results_dir, f"seed_{cv}/results.json")
        
        with open(res_file) as data_file:
            results = json.load(data_file)
            top_seeds.append((cv,results["val_auroc"]))

    top_seeds = sorted(top_seeds, key=lambda tup: tup[1], reverse=True)

    seeds = [seed for seed,_ in top_seeds[5:]]
    
means = []
for cv in seeds:
    res_file = os.path.join(results_dir, f"seed_{cv}/results.json")

    with open(res_file) as data_file:
        results = json.load(data_file)
        means.append(results[metric])

print(f"\nMean Auroc: {np.mean(means):.3f}")
print(f"\nStd: {np.std(means):.3f}")
