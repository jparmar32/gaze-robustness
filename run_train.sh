#!/bin/sh
train="cxr_p"
test="cxr_p"

for seed in 0 1 2 3 4 5 6 7 8 9 
do
    python ./train.py \
        --epochs 15 \
        --min_lr 0 \
        --lr .0001 \
        --wd .1 \
        --seed $seed \
        --batch_size 32 \
        --train_set $train \
        --test_set $test \
        --save_dir "gaze_robustness_results/gaze_data_augmentation" \
        --gaze_task "data_augment"
done