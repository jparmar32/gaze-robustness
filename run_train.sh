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
        --save_dir "/mnt/gaze_robustness_results/gaze_data_augmentation" \
        --checkpoint_dir "/mnt/gaze_robustness_results/gaze_data_augmentation/train_set_$train/seed_$seed/model.pt" \
        --subclass_eval \
        #--ood_shift "age" \
        #--gaze_task "data_augment" \
        
        
done

#hyperparaam search
#for lr in .0001 #.01 .001 .0001
#do
    #for wd in .1 #1 .1 .01 .001 .0001 0
    #do   
        #for seed in 2 #0 1 2 3 4 5 6 7 8 9 
        #do
            #python ./train.py \
            #--epochs 15 \
            #--min_lr 0 \
            #--lr $lr \
            #--wd $wd \
            #--seed $seed \
            #--batch_size 32 \
            #--train_set $train \
            #--test_set $test \
            #--save_dir "/mnt/gaze_robustness_results/lr_$lr/wd_$wd/gaze_data_augmentation" \
            #--save_model  \
            #--gaze_task "data_augment" \
        #--checkpoint_dir
        #--ood_shift \

        #done 
    #done 
#done

#LR .1 .01 .001 .0001
#l2 .1 .01 .001 .0001