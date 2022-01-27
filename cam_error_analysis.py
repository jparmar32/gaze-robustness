### experiments to run: use ground truth segmentation to train a model on CXR-P using cam reg apporach, eval on CXR-P test and Chexpert 
#change above to dataloader and train to test that

#This one:
### using cam convex comb model, get CAM for images, get ground truth segmentation, get loss, find correlation beteween dice score of cam and ground truth and los

### using cam convex comb model, get CAM for images, ground truth segmentation, get positive and negative classification, find scores for grouping in posiitve and goruping in negative

import argparse
import glob
import json
import os
import pickle
import sys
import warnings
import sklearn.metrics as skl
import json
import math
from tqdm import tqdm
from skimage.transform import resize


import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
from dataloader import fetch_dataloaders

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="Seed to use in dataloader")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")

    parser.add_argument("--train_set", type=str, choices=['cxr_a','cxr_p'], required=True, help="Set to train on")
    parser.add_argument("--test_set", type=str, choices=['cxr_a','cxr_p', 'mimic_cxr', 'chexpert', 'chestxray8'], required=True, help="Test set to evaluate on")
    parser.add_argument("--ood_shift", type=str, choices=['hospital','hospital_age', 'age', None], default=None, help="Distribution shift to experiment with")

    parser.add_argument("--num_classes", type=int, default=2, help="Number of classes in the training set")
    parser.add_argument("--gaze_task", type=str, choices=['cam_error_analysis', None], default='cam_error_analysis', help="Type of gaze to analyze error of")

    parser.add_argument("--correlation_type", type=str, choices=['loss','class', None], default=None, help="Decide how to compute correlation between CAM score and model quality")

    args = parser.parse_args()
    return args


def compute_dice(mask1, mask2):
    area_overlap = np.sum(np.logical_and(mask1, mask2))
    total_pix = np.sum(mask1) + np.sum(mask2)
    return 2.0 * area_overlap / float(total_pix)


def rle2mask(rle, width, height):
    mask = np.zeros(width * height)
    array = np.asarray([int(x) for x in rle.split()])
    starts = array[0::2]
    lengths = array[1::2]

    current_position = 0
    for index, start in enumerate(starts):
        current_position += start
        mask[current_position : current_position + lengths[index]] = 1
        current_position += lengths[index]

    return mask.reshape(width, height)


def return_CAM(feature_conv, weight, class_idx):
    # generate the class-activation map
    nc, h, w = feature_conv.shape

    beforeDot = feature_conv.reshape((nc, h * w))
    # cam = np.matmul(weight[class_idx], beforeDot)
    cam = beforeDot.mean(0)
    cam = cam.reshape(h, w)
    cam = cam - np.min(cam)
    cam_img = cam / np.max(cam)
    return cam_img


def main():
    args = parse_args()

    # Load dataloader
    if args.ood_shift is not None:
        loaders = fetch_dataloaders(args.train_set,"/media",0.2,args.seed,args.batch_size,4, gaze_task=args.gaze_task, ood_set= args.test_set, ood_shift = args.ood_shift)
    else:
        loaders = fetch_dataloaders(args.train_set,"/media",0.2,args.seed,args.batch_size,4, gaze_task=args.gaze_task)
    
    avg_correct = []
    avg_incorrect = []
    for seed in range(10):
        
        checkpoint_dir = f"/mnt/data/gaze_robustness_results/gaze_cam_reg_convex/train_set_cxr_p/seed_{seed}/model.pt"

        # Load model
        num_classes = args.num_classes
        model = models.resnet50(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        model.load_state_dict(torch.load(checkpoint_dir))
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        print(f"evaluating on {args.test_set}:")


        model.cuda()
        model.eval()
        # remove global average pooling and fc
        cnn_only = nn.Sequential(*list(model.children())[:-2])
       
        # load seg dict
        seg_dict_pth = "/media/pneumothorax/rle_dict.pkl"
        with open(seg_dict_pth, "rb") as pkl_f:
            rle_dict = pickle.load(pkl_f)

        dice_correct = []
        dice_incorrect = []
        
        #loop through test set
        test_dataloader = loaders['test']
    
        for batch_idx, batch in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
            
            with torch.no_grad():
                inputs, targets, img_id = batch
                #print(img_id)
                rle = rle_dict[img_id[0].split("/")[-1].split(".dcm")[0]]
                y_true = rle != " -1"

                inputs = inputs.cuda()
                targets = targets.cuda()
                logits = model(inputs)

                output_probs = F.softmax(logits.data, dim=1)
                output_label = torch.argmax(output_probs).item()
                features = cnn_only(inputs).squeeze().cpu().detach().numpy()
   
                if y_true:

                    # calculate cam
                    cam = return_CAM(features, None, None)

                    # extract segmask
                    segmask = np.zeros(inputs.shape[2:4])
                    segmask_org = rle2mask(rle, 1024, 1024).T
                    segmask = resize(segmask_org, inputs.shape[2:4])

                    cam = resize(cam, inputs.shape[2:4], order=0)
                    
                    cam_org = resize(cam, segmask_org.shape)
                    


                    # Dice score
                    k = int(np.sum(segmask_org))
                    kth_maxvalue = np.sort(cam_org.flatten())[-k]
                    cam_binarized = cam_org >= kth_maxvalue
                    cam_dice = compute_dice(cam_binarized, segmask_org)
                    
                    ## correct 
                    if output_label == targets.item():
                        dice_correct.append(cam_dice)
                    ## incorrect
                    else:
                        dice_incorrect.append(cam_dice)

                    
        avg_correct.append(np.mean(dice_correct))
        avg_incorrect.append(np.mean(dice_incorrect))

    print(f"Avg dice of correct samples: {np.mean(avg_correct)} wth std: {np.std(avg_correct)}")
    print(f"Avg dice of incorrect samples: {np.mean(avg_incorrect)} wth std: {np.std(avg_incorrect)}")

        

if __name__ == "__main__":
    main()