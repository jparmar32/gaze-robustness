#!/bin/sh
train="cxr_p"
test="cxr_p"

#for seed in 0 #1 2 3 4 5 6 7 8 9 
#do
  
    #python ./train.py \
        #--epochs 15 \
        #--min_lr 0 \
        #--lr .0001 \
        #--wd .1 \
        #--seed $seed \
        #--batch_size 32 \
        #--train_set $train \
        #--test_set $test \
        #--save_dir "/mnt/gaze_robustness_results/gaze_cam_reg_convex" \
        #--gaze_task "cam_reg_convex" \
        #--cam_weight 1 \
        #--cam_convex_alpha .2 \
        #--checkpoint_dir "/mnt/gaze_robustness_results/gaze_data_augmentation/train_set_$train/seed_$seed/model.pt" \
        #--subclass_eval \
        #--ood_shift "age" \
        
        
        
#done

#hyperparaam search
for wd in 1 .1 .01 .001 .0001 
do  
    for cw in 0.5 1 2
    do   
        for seed in 0 1 2 3 4 5 6 7 8 9 
        do
            python ./train.py \
            --epochs 15 \
            --min_lr 0 \
            --lr .0001 \
            --wd $wd \
            --seed $seed \
            --batch_size 32 \
            --train_set $train \
            --test_set $test \
            --save_dir "/mnt/gaze_robustness_results/wd_$wd/cw_$cw/gaze_cam_reg" \
            --save_model  \
            --gaze_task "cam_reg" \
            --cam_weight $cw \
        #--checkpoint_dir
        #--ood_shift \

        done 
    done 
done

#LR .1 .01 .001 .0001
#l2 .1 .01 .001 .0001