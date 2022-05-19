#!/bin/sh
train="cxr_p"
test="cxr_p"

#for seed in 0 1 2 3 4 #5 6 7 8 9
#do 
#
#    python ./train.py \
#    --epochs 100 \
#    --min_lr 0 \
#    --lr 3.11e-4 \
#   --wd 0 \
#    --seed $seed \
#   --batch_size 16 \
#   --train_set $train \
#   --tes#t_set $test \
#   --save_dir "/mnt/data/gaze_robustness_results/actdiff" \
#   --save_model \
#   --gaze_task "actdiff" \
#   --actdiff_lambda 1.15e-1 
#done

for seed in 0 1 2 #4 5 6 7 8 9
do 
   for lr in 1e-4 
   do
       for sl in 1e-4 1e-3 1e-2 1e-1 1 
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
            --train_size "small" \
            --save_dir "/mnt/data/gaze_robustness_results/grad_mask/cxr_p_train_size_small/saliency_lambda_$sl" \
            --gaze_task "grad_mask" \
            --saliency_lambda $sl \
            --machine 'meteor' \
            --save_model \
            #--checkpoint_dir "/media/nvme_data/jupinder_cxr_robustness_results/actdiff_lungmask/similarity_type_cosine/segmentation_classes_$segmentation_classes/augmentation_type_$augmentation_type/lungmask_size_$lungmask_size/actdiff_lambda_$al/train_set_$train/seed_$seed/model.pt" \
            #--ood_shift "hospital_age" \
            #--checkpoint_dir "/mnt/data/gaze_robustness_results/actdiff_lungmask/lungmask_size_$lungmask_size/train_set_$train/seed_$seed/model.pt" \
            #--save_model \
            #--checkpoint_dir "/mnt/data/gaze_robustness_results/erm_resnet_baseline/wd_0/train_set_$train/seed_$seed/model.pt" \
            #--ood_shift "hospital_age" \
            #--gaze_task "actdiff" \
            #--actdiff_segmask_size $seg_size \
            #--ood_shift "age" \
            #--checkpoint_dir "/mnt/data/gaze_robustness_results/actdiff/seg_size_$seg_size/train_set_$train/seed_$seed/model.pt" \
            #--save_model \
            #--checkpoint_dir "/mnt/data/gaze_robustness_results/resnet_only/train_set_$train/seed_$seed/model.pt" \
            #--subclass_eval \
            #--gaze_task "resnet_only" \
            #--save_model \
            #--save_model \
            #--gan_positive_model "/home/jsparmar/gaze-robustness/gan/positive_class" \
            #--gan_negative_model "/home/jsparmar/gaze-robustness/gan/negative_class" \
            #--gan_type "gan" \
            #--ood_shift "hospital" \
            #--checkpoint_dir "/mnt/data/gaze_robustness_results/acgan_generation/train_set_$train/seed_$seed/model.pt" \
            #--cam_weight 0.5 \
            #--cam_convex_alpha 0.5 \
            #--subclass_eval \
            #--save_model
         
        done
    done
done


#hyperparaam search
#for wd in .1 .01 .001 #.0001 #1
#do  
    #for cw in 0.5 1 2
    #do   
        #for seed in 0 1 2 #3 4 5 6 7 8 9 
        #do
            #python ./train.py \
            #--epochs 15 \
            #--min_lr 0 \
            #--lr .0001 \
            #--wd $wd \
            #--seed $seed \
            #--batch_size 32 \
            #--train_set $train \
            #--test_set $test \
            #--save_dir "/mnt/gaze_robustness_results/wd_$wd/cw_$cw/gaze_cam_reg" \
            #--save_model  \
            #--gaze_task "cam_reg" \
            #--cam_weight $cw \
        #--checkpoint_dir
        #--ood_shift \

        #done 
    #done 
#done
#LR .1 .01 .001 .0001
#l2 .1 .01 .001 .0001