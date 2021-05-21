import json
import numpy as np
import os


train_set = 'cxr_p'
test_set = 'cxr_p'
results_dir = f'/mnt/gaze_robustness_results/resnet_only/train_set_{train_set}/test_set_{test_set}'

seeds = [x for x in range(10)] 
means = []
for cv in seeds:
    res_file = os.path.join(results_dir,f"seed_{cv}/result.json")

    with open(res_file) as data_file:
        results = json.loads(data_file)
        means.append(results['test_auroc'])


print(f"\nMean Auroc: {np.mean(means):.3f}")
print(f"\nStd: {np.std(means):.3f}")