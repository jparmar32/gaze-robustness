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


import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
import copy
from dataloader import fetch_dataloaders, GanDataset
from train import evaluate

from utils import AverageMeter, accuracy, compute_roc_auc, build_scheduler, get_lrs
from models.extract_CAM import get_CAM_from_img
import gan_training.gan.generator as gan_generator
from torch.utils.data import Dataset, DataLoader



def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--subclass_eval", action='store_true', help="Whether to report subclass performance metrics on the test set")
    parser.add_argument("--num_classes", type=int, default=2, help="Number of classes in the training set")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    aurocs = []

    for seed in range(10):

        erm_model = f"/mnt/data/gaze_robustness_results/resnet_only/train_set_cxr_p/seed_{seed}/model.pt"
        positive_gan = "/home/jsparmar/gaze-robustness/gan/positive_class/generator_best_ckpt.pt"
        negative_gan = "/home/jsparmar/gaze-robustness/gan/positive_class/generator_best_ckpt.pt"


        pos_generator = gan_generator.Generator_Advanced_224().cuda()
        neg_generator = gan_generator.Generator_Advanced_224().cuda()
        noise_size = 100

        pos_generator.load_state_dict(torch.load(positive_gan))
        neg_generator.load_state_dict(torch.load(negative_gan))


        class_amounts = [3598, 1009]

        neg_noise = torch.randn(class_amounts[0], noise_size, 1, 1).cuda()
        pos_noise = torch.randn(class_amounts[1], noise_size, 1, 1).cuda()

        # Feed noise into the generator to create new images
        neg_images = []
        for i in range(class_amounts[0]):
            neg_images.append(neg_generator(neg_noise[i].unsqueeze(dim=0)).detach().cpu())
        neg_images = torch.cat(neg_images)

        pos_images = []
        for i in range(class_amounts[1]):
            pos_images.append(pos_generator(pos_noise[i].unsqueeze(dim=0)).detach().cpu())
        pos_images = torch.cat(pos_images)

        #neg_images = neg_generator(neg_noise).detach()
        #pos_images = pos_generator(pos_noise).detach()
                    
        neg_labels = torch.zeros(neg_images.shape[0]).cpu().numpy()
        pos_labels = torch.ones(pos_images.shape[0]).cpu().numpy()

        neg_gaze_attr = torch.zeros(neg_images.shape[0]).cpu().numpy()
        pos_gaze_attr = torch.zeros(neg_images.shape[0]).cpu().numpy()

        positive_fake_data = GanDataset(images=pos_images, labels=pos_labels, gaze_attr=pos_gaze_attr)
        negative_fake_data = GanDataset(images=neg_images, labels=neg_labels, gaze_attr=neg_gaze_attr)

        dataset = torch.utils.data.ConcatDataset([positive_fake_data, negative_fake_data])

        generated_dataloader = DataLoader(
            dataset=dataset,
            shuffle=False,
            batch_size=32,
            num_workers=4)

        num_classes = 2
        print("Using checkpointed model...")
        model = models.resnet50(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        model.load_state_dict(torch.load(erm_model))
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        test_loss, test_acc, test_auroc, _, _ = evaluate(model, generated_dataloader, args, loss_fn=nn.CrossEntropyLoss())

        
        print(f"Best Test Auroc {test_auroc}")
        aurocs.append(test_auroc)

    print(f"\nMean Auroc: {np.mean(aurocs):.3f}")
    print(f"\nStd: {np.std(aurocs):.3f}")
