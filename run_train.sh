#!/bin/sh
train="cxr_p"
test="mimic_cxr"

for seed in 0 1 #4 5 6 7 8 9
do 
   for lr in 1e-4 
   do
       for sl in 1e-3
       do
            python ./train.py \
            --epochs 100 \
            --min_lr 0 \
            --lr $lr \
            --wd 0 \
            --seed $seed \
            --batch_size 16 \
            --train_set $train \
            --test_set $test \
            --train_size "all" \
            --save_dir "/media/nvme_data/jupinder_cxr_robustness_results/rrr/cxr_p_train_size_all/saliency_lambda_$sl" \
            --checkpoint_dir "/media/nvme_data/jupinder_cxr_robustness_results/rrr/cxr_p_train_size_all/saliency_lambda_$sl/train_set_$train/seed_$seed/model.pt" \
            --ood_shift "hospital" \
            --gaze_task "rrr" \
            --saliency_segmentation_classes "positive" \
            --saliency_lambda $sl \
            --machine 'gemini' \
            --save_model \
        done
    done
done
