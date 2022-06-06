from audioop import avg
from cgitb import small
from logging.config import valid_ident
from multiprocessing.sharedctypes import Value
from re import M
from sys import int_info
import numpy as np
import torch
import os
import pickle 
from tqdm import tqdm
import math
import pandas as pd 
import json
from PIL import Image
import pydicom
from torchvision import transforms
import scipy.ndimage as ndimage
import cv2
import torch.nn.functional as F
import torch

import matplotlib.pyplot as plt
from skimage.transform import resize
from utils import load_file_markers, get_data_transforms, load_gaze_attribute_labels, rle2mask, load_gaze_data, compute_avg_heatmap
from collections import defaultdict

def get_gaze_sequences():
    train_seqs, train_labels, train_gaze_ids = load_gaze_data("cxr_p", "train", 1, 0.2, False, 0)
    val_seqs, val_labels, val_gaze_ids = load_gaze_data("cxr_p", "val", 1, 0.2, False, 0)

    train_gaze_sequences = {gaze: seq for seq, gaze in zip(train_seqs, train_gaze_ids)}
    val_gaze_sequences = {gaze: seq for seq, gaze in zip(val_seqs, val_gaze_ids)}

    train_gaze_sequences.update(val_gaze_sequences)

    return train_gaze_sequences


### average from sequences
gaze_seqs_dict = get_gaze_sequences()
gaze_seqs = list(gaze_seqs_dict.values())
avg_gaze_heatmap = compute_avg_heatmap(gaze_seqs, grid_width=16)
avg_gaze_heatmap = avg_gaze_heatmap.reshape((16,16))
avg_gaze_heatmap = resize(avg_gaze_heatmap, (224, 224))
avg_gaze_heatmap[avg_gaze_heatmap < 0.2] = 0

avg_gaze_heatmap_full = compute_avg_heatmap(gaze_seqs, grid_width=224)
avg_gaze_heatmap_full = avg_gaze_heatmap_full.reshape((224,224))
avg_gaze_heatmap_full[avg_gaze_heatmap_full < 0.008] = 0

print(avg_gaze_heatmap.shape)
print(avg_gaze_heatmap_full.shape)

'''def get_gaze_heatmaps(task):
    train_gaze_attribute_labels_dict = load_gaze_attribute_labels("cxr_p", "train", task, 0)
    val_gaze_attribute_labels_dict = load_gaze_attribute_labels("cxr_p", "val", task, 0)
    
    train_gaze_attribute_labels_dict.update(val_gaze_attribute_labels_dict)

    return train_gaze_attribute_labels_dict 

gaze_heatmap = get_gaze_heatmaps("heatmap2")
test = list(gaze_heatmap.values())
for idx in range(len(test)):
    val = test[idx][0]
    val[val < 0.05] = 0
    test[idx] = val

average_heatmap = np.mean(test, axis=0).squeeze()

average_heatmap = average_heatmap.reshape((7,7))
average_heatmap = resize(average_heatmap, (224, 224))
'''

### image average
with open("./filemarkers/cxr_p/trainval_list_gold.pkl", "rb") as f:
    img_file_markers = pickle.load(f)

tforms = transforms.Compose([transforms.Resize([224, 224]), transforms.ToTensor()])

average_imgs = []
for img_id, label in tqdm(img_file_markers):
    img_pth = os.path.join("/media", f"pneumothorax/dicom_images/{img_id}")
    ds = pydicom.dcmread(img_pth)
    img = ds.pixel_array
    img = Image.fromarray(np.uint8(img))
    img = tforms(img)
    average_imgs.append(img)

average_img = torch.mean(torch.stack(average_imgs), dim=0)
average_img = average_img.transpose(0,2)
average_img = average_img.transpose(0,1)
average_img = average_img.squeeze()
plt.imshow(average_img, cmap="gray")
plt.imshow(avg_gaze_heatmap_full, alpha=0.3, cmap="hot")
plt.savefig("gaze_test.png")
