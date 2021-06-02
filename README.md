# Improving Model Generalization With Gaze

We explore the utility of human gaze sequeences in creating more robust models. Specifically, we posit that incorporating gaze information will reduce reliance on spurious correlates. This repo contains the code to run all gaze relate experiments. Specifically, it has the ability to train a gaze model and then test it on o.o.d datasets (chexpert, mimic cxr, chestxray8) or evaluate the worst case subgroup performance in the test set. 

### Running Experiments 

To train and evaluate models with this repo, one only needs to call the `./run_train.sh` script. Specifically, in this set one must specificy a train set, typically it is `cxr_p` which has gaze data associaetd with it, and additionally specify the test set. The test set can be the same as the train set if one wishes to evaluate i.i.d test set performance or to evaluate worst case subgroup performance. It can be different from the train set, either `mimic_cxr`, `chestxray8`, or `chexpert`, to evaluatet o.o.d performance. Specific commands for given architectures are specified below.

#### Image Only Architecture (Resnet 50)

#### Gaze Data Augmentation 

#### CAM Regularization with Gaze Heatmap 

#### CAM Regularization with Convex Combinatiton of Image Specific Gaze Heatmap and Average Gaze Heatmap


