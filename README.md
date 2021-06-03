# Improving Model Generalization With Gaze

We explore the utility of human gaze sequeences in creating more robust models. Specifically, we posit that incorporating gaze information will reduce reliance on spurious corelates. This repo contains the code to run all gaze relate experiments. Specifically, it has the ability to train a gaze model and then test it on o.o.d datasets (chexpert, mimic cxr, chestxray8) or evaluate the worst case subgroup performance in the test set. 

### Running Experiments 

To train and evaluate models with this repo, one only needs to call the `./run_train.sh` script. Specifically, in this set one must specificy a train set, typically it is `cxr_p` which has gaze data associaetd with it, and additionally specify the test set. The test set can be the same as the train set if one wishes to evaluate i.i.d test set performance or to evaluate worst case subgroup performance. It can be different from the train set, either `mimic_cxr`, `chestxray8`, or `chexpert`, to evaluatet o.o.d performance. Specific configurations/args of `./run_train.sh` for given architectures are specified below.

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
        --cam_convex_alpha 0.5 \
      
done
```

To evaluate on an o.o.d set:

Simply add the flag `--checpkpoint_dir` with the specified location of the saved model from training to the script above while changing the test set variable to the desired o.o.d test set. Make sure to also include the `--ood_shift` flag with one of the following: `hospital`, `hospital_age`, `age` depending on the given test set.

To evaluate worst subclass performance:

Simply add the flag `--checpkpoint_dir` with the specified location of the saved model from training to the script above while also adding in the flag `subclass_eval`. Make sure that the train and test set are both `cxr_p` in this case. 


### DANN

To run the DANN model, first one needs to switch to the DANN branch of this repository. Then, to train a DANN model on CXR-P with the target distribution being a given o.o.d set they simply need to update the parameters of the function `dann_run` inside `experiments/dann_gaze.py` accordingly. For example, if one wanted to use the target domain of CheXpert with the hospital shif they would set the parameters to be:

```
dann_run(src_name=”cxr_p", tgt_name=”chexpert_hospital", ood_set=”chexpert", ood_shift=”hospital", dl_seed=0)
```

One can then train and evlauate the model by calling `python experiments/dann_gaze.py` from the command line. 
