#!/bin/sh
train="cxr_p"
test="cxr_p"

for seed in 0 1 2 3 4 5 6 7 8 9 
do
  
    python ./train.py \
        --epochs 15 \
        --min_lr 0 \
        --lr .0001 \
        --wd .01 \
        --seed $seed \
        --batch_size 32 \
        --train_set $train \
        --test_set $test \
        --save_dir "/mnt/gaze_robustness_results/resnet_only" \
        --checkpoint_dir "/mnt/gaze_robustness_results/resnet_only/train_set_$train/seed_$seed/model.pt" \
        --subclass_eval \
        #--ood_shift "hospital_age" \
   

done

#hyperparaam search

#for seed in 0 1 2 3 4 5 6 7 8 9 
#do
    #for lr in .0001 #.1 .01 .001 .0001
    #do 

        #for wd in 0 #1 .1 .01 .001 .0001 0
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
            #--save_dir "/mnt/gaze_robustness_results/resnet_only" \
            #--save_model 

        #--checkpoint_dir
    
        #--ood_shift \

        #done 

    #done 

#done

#LR .1 .01 .001 .0001
#l2 .1 .01 .001 .0001