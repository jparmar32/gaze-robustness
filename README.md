# The Importance of Background Information for OOD Generalization

We explore the utility of incoporating background information, areas which are not contained in the ground-truth segmentation maps, in creating more robust models. Specifically, we posit that incorporating such information will reduce reliance on spurious corelates. This repo contains the code to run all gaze relate experiments. Specifically, it has the ability to train a gaze model and then test it on o.o.d datasets (chexpert, mimic cxr, chestxray8) or evaluate the worst case subgroup performance in the test set. 

### Running Experiments 

To train and evaluate models with this repo, one only needs to call the `./run_train.sh` script. Specifically, in this set one must specificy a train set, typically it is `cxr_p` which has gaze data associaetd with it, and additionally specify the test set. The test set can be the same as the train set if one wishes to evaluate i.i.d test set performance or to evaluate worst case subgroup performance. It can be different from the train set, either `mimic_cxr`, `chestxray8`, or `chexpert`, to evaluatet o.o.d performance. Specific configurations/args of `./run_train.sh` for given architectures are specified below. In addition, based on the machine that you are running the repo on you will need to add data specific paths and hence must change the `machine` flag appropriately.

#### Image Only Architecture (Resnet 50):

To train a model:

```
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
        --save_dir "/mnt/gaze_robustness_results/resnet_only" \
        --machine 'gemini' \
        --save_model            
done

```

To evaluate on an o.o.d set:

Simply add the flag `--checpkpoint_dir` with the specified location of the saved model from training to the script above while changing the test set variable to the desired o.o.d test set. Make sure to also include the `--ood_shift` flag with one of the following: `hospital`, `hospital_age`, `age` depending on the given test set.

To evaluate worst subclass performance:

Simply add the flag `--checpkpoint_dir` with the specified location of the saved model from training to the script above while also adding in the flag `subclass_eval`. Make sure that the train and test set are both `cxr_p` in this case. 

#### Gaze Data Augmentation:
To train a model: 
```
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
        --machine 'gemini' \
        --gaze_task "data_augmentation" \
      
done
```

To evaluate on an o.o.d set:

Simply add the flag `--checpkpoint_dir` with the specified location of the saved model from training to the script above while changing the test set variable to the desired o.o.d test set. Make sure to also include the `--ood_shift` flag with one of the following: `hospital`, `hospital_age`, `age` depending on the given test set.

To evaluate worst subclass performance:

Simply add the flag `--checpkpoint_dir` with the specified location of the saved model from training to the script above while also adding in the flag `subclass_eval`. Make sure that the train and test set are both `cxr_p` in this case. 


#### CAM Regularization with Gaze Heatmap: 

To train a model: 
```
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
        --save_dir "/mnt/gaze_robustness_results/gaze_cam_reg" \
        --gaze_task "cam_reg" \
        --machine 'gemini' \
        --cam_weight 1 \
      
done
```

To evaluate on an o.o.d set:

Simply add the flag `--checpkpoint_dir` with the specified location of the saved model from training to the script above while changing the test set variable to the desired o.o.d test set. Make sure to also include the `--ood_shift` flag with one of the following: `hospital`, `hospital_age`, `age` depending on the given test set.

To evaluate worst subclass performance:

Simply add the flag `--checpkpoint_dir` with the specified location of the saved model from training to the script above while also adding in the flag `subclass_eval`. Make sure that the train and test set are both `cxr_p` in this case. 

#### CAM Regularization with Convex Combinatiton of Image Specific Gaze Heatmap and Average Gaze Heatmap:

To train a model: 

```
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
        --save_dir "/mnt/gaze_robustness_results/gaze_cam_reg_convex" \
        --gaze_task "cam_reg_convex" \
        --cam_weight 1 \
        --machine 'gemini' \
        --cam_convex_alpha 0.5 \
      
done
```

To evaluate on an o.o.d set:

Simply add the flag `--checpkpoint_dir` with the specified location of the saved model from training to the script above while changing the test set variable to the desired o.o.d test set. Make sure to also include the `--ood_shift` flag with one of the following: `hospital`, `hospital_age`, `age` depending on the given test set.

To evaluate worst subclass performance:

Simply add the flag `--checpkpoint_dir` with the specified location of the saved model from training to the script above while also adding in the flag `subclass_eval`. Make sure that the train and test set are both `cxr_p` in this case. 


### GAN Data Augmentation 

To run a model such that the dataset is augmented by GAN images, run:

```
train="cxr_p"
test="mimic_cxr"

for seed in 8 #0 1 2 3 4 5 6 7 8 9
do

    python ./train.py \
        --epochs 15 \
        --min_lr 0 \
        --lr .0001 \
        --wd .01 \
        --seed $seed \
        --batch_size 8 \
        --train_set $train \
        --test_set $test \
        --save_dir "/mnt/data/gaze_robustness_results/acgan_generation" \
        --checkpoint_dir "/mnt/data/gaze_robustness_results/acgan_generation/train_set_$train/seed_$seed/model.pt" \
        --ood_shift "hospital" \
        --gan_positive_model "/home/jsparmar/gaze-robustness/gan/positive_class" \
        --gan_negative_model "/home/jsparmar/gaze-robustness/gan/negative_class" \
        --machine 'gemini' \
        --gan_type "gan" \
```

This requires that a GAN generator be trained before hand which is done by running: 

```
python gan_training/train_gan.py
```

### Right for The Right Reasons (RRR)

To train a model using RRR with the default settings simply use:

```
for seed in 0 1 2
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
            --save_dir "/media/nvme_data/jupinder_cxr_robustness_results/rrr/saliency_lambda_$sl" \
            --gaze_task "rrr" \
            --saliency_lambda $sl \
            --machine 'gemini' \
            --save_model \
        done
    done
done
```

The above will use ground-truth segmentation masks and use the small dataset. In order to evaluate using the lung-periphery segmentation masks simply change the `gaze_task` to `rrr_lungmask`. Additionally, we can change the resolution of the segmentation masks by either adding the flags `saliency_segmask_size` or `saliency_lungmask_size` with the desired resolution. Lastly, in order to use the larger training set add the flag `train_size` with the value `all`.

To evaluate on an o.o.d set:

Simply add the flag `--checpkpoint_dir` with the specified location of the saved model from training to the script above while changing the test set variable to the desired o.o.d test set. Make sure to also include the `--ood_shift` flag with one of the following: `hospital`, `hospital_age`, `age` depending on the given test set.


### GradMask
To train a model using GradMask with the default settings simply use:

```
for seed in 0 1 2
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
            --save_dir "/media/nvme_data/jupinder_cxr_robustness_results/grad_mask/saliency_lambda_$sl" \
            --gaze_task "grad_mask" \
            --saliency_lambda $sl \
            --machine 'gemini' \
            --save_model \
        done
    done
done
```

The above will use ground-truth segmentation masks and use the small dataset. In order to evaluate using the lung-periphery segmentation masks simply change the `gaze_task` to `grad_mask_lungmask`. Additionally, we can change the resolution of the segmentation masks by either adding the flags `saliency_segmask_size` or `saliency_lungmask_size` with the desired resolution. Lastly, in order to use the larger training set add the flag `train_size` with the value `all`.

To evaluate on an o.o.d set:

Simply add the flag `--checpkpoint_dir` with the specified location of the saved model from training to the script above while changing the test set variable to the desired o.o.d test set. Make sure to also include the `--ood_shift` flag with one of the following: `hospital`, `hospital_age`, `age` depending on the given test set.

### ActDiff
To train a model using ActDiff with the default settings simply use:

```
for seed in 0 1 2
do 
   for lr in 1e-4 
   do
       for al in 1e-3
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
            --save_dir "/media/nvme_data/jupinder_cxr_robustness_results/actdiff/saliency_lambda_$sl" \
            --gaze_task "actdiff" \
            --actdiff_lambda $al \
            --machine 'gemini' \
            --save_model \
        done
    done
done
```

The above will use ground-truth segmentation masks and use the small dataset. In order to evaluate using the lung-periphery segmentation masks simply change the `gaze_task` to `actdiff_lungmask`. Additionally, we can change the resolution of the segmentation masks by either adding the flags `actdiff_segmask_size` or `actdiff_lungmask_size` with the desired resolution. Lastly, in order to use the larger training set add the flag `train_size` with the value `all`. Additionally, we can change the type of similarity metric used to constrain the feature representations in ActDiff by using the flag 
`actdiff_similarity_type`, teh augmentation of the background by using `actdiff_augmentation_type`, and which features layers to constrain using `actdiff_features`.


To evaluate on an o.o.d set:

Simply add the flag `--checpkpoint_dir` with the specified location of the saved model from training to the script above while changing the test set variable to the desired o.o.d test set. Make sure to also include the `--ood_shift` flag with one of the following: `hospital`, `hospital_age`, `age` depending on the given test set.


