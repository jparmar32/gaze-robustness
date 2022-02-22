
from cgitb import small
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
from utils import load_file_markers, get_data_transforms, load_gaze_attribute_labels, rle2mask, load_gaze_data
from collections import defaultdict

def compute_heatmap_iou(mask_1, mask_2):
    area_overlap = np.sum(np.logical_and(mask_1, mask_2))
    area_union = np.sum(mask_1) + np.sum(mask_2) - area_overlap
    return (area_overlap / area_union)


def compute_intersection_over_seg(gaze_mask, seg_mask):
    area_overlap = np.sum(np.logical_and(gaze_mask, seg_mask))
    return (area_overlap/np.sum(seg_mask))

def gaze_heatmap_segmentation_mask(gaze_maps, segmentation_masks, threshold, map_size, inverse_segmentation = False ,iou = True):
    
    metric = []

    for gaze_id, gaze_map in gaze_maps.items():
        gaze_map = gaze_map[0].reshape(map_size,map_size)
        img_path = gaze_id.split("/")[-1].split(".dcm")[0]
        rle_binary = segmentation_masks[img_path]

        if rle_binary != " -1":
            gaze_binary = gaze_map > threshold
            
            if inverse_segmentation:
                rle_binary = 1 - rle_binary
         
            if iou:
                metric.append(compute_heatmap_iou(gaze_binary, rle_binary))
            else:
                metric.append(compute_intersection_over_seg(gaze_binary, rle_binary))

    return np.mean(metric)

def gaze_heatmap_segmentation_iou_plots(gaze_seqs, segmentation_masks, gaze_maps, threshold, map_size, num_unique_patches = True, inverse=False):
    
    gaze_seg_iou = []
    time_spent = []
    time_spent_norm = []

    max_gaze_length = 0
    for gaze_id, gaze_seq in gaze_seqs.items():
        if len(gaze_seq) > max_gaze_length:
            max_gaze_length = len(gaze_seq)

    for gaze_id, gaze_map in gaze_maps.items():
        gaze_map = gaze_map[0].reshape(map_size,map_size)
        gaze_seq = gaze_seqs[gaze_id]

        img_path = gaze_id.split("/")[-1].split(".dcm")[0]
        rle_binary = segmentation_masks[img_path]

        if rle_binary != " -1":
            gaze_binary = gaze_map > threshold

            if inverse:
                rle_binary = 1 - rle_binary
            if num_unique_patches:
                gaze_seg_iou.append(np.sum(gaze_binary))
            else:
                gaze_seg_iou.append(compute_heatmap_iou(gaze_binary, rle_binary))
            time_spent.append(len(gaze_seq))
            time_spent_norm.append(len(gaze_seq)/max_gaze_length)

    
    plt.scatter(time_spent, gaze_seg_iou)
    plt.xlabel("Time Spent on Image")
    
    if inverse:
        plt.ylabel("IoU between Gaze Heatmap and Inverse Ground Truth Seg")
        plt.savefig(f"./gaze_analysis_results/gaze_inv_seg_iou/map_size_{map_size}_threshold_{threshold}.png")
    elif num_unique_patches:
        plt.ylabel("Number of Unique Patches in Gaze Heatmap")
        plt.savefig(f"./gaze_analysis_results/gaze_unique_num/map_size_{map_size}_threshold_{threshold}.png")
    else:
        plt.ylabel("IoU between Gaze Heatmap and Ground Truth Seg")
        plt.savefig(f"./gaze_analysis_results/gaze_seg_iou/map_size_{map_size}_threshold_{threshold}.png")
    plt.close()


    plt.scatter(time_spent_norm, gaze_seg_iou)
    plt.xlabel("Time Spent on Image Normalized")
    plt.ylabel("IoU between Gaze Heatmap and Ground Truth Seg")
    if inverse:
        plt.ylabel("IoU between Gaze Heatmap and Inverse Ground Truth Seg")
        plt.savefig(f"./gaze_analysis_results/gaze_inv_seg_iou_norm/map_size_{map_size}_threshold_{threshold}.png")
    elif num_unique_patches:
        plt.ylabel("Number of Unique Patches in Gaze Heatmap")
    else:
        plt.ylabel("IoU between Gaze Heatmap and Ground Truth Seg")
        plt.savefig(f"./gaze_analysis_results/gaze_seg_iou_norm/map_size_{map_size}_threshold_{threshold}.png")
    plt.close()


def gaze_time_spent_segmentation(gaze_seqs, segmentation_masks, map_size):
    time_in_seg = defaultdict(list) # 0 is false, 1 is true 
    time_unnorm_in_seg = defaultdict(list)

    for gaze_id, gaze_seq in gaze_seqs.items():
        img_path = gaze_id.split("/")[-1].split(".dcm")[0]
        rle_binary = segmentation_masks[img_path]

        if rle_binary != " -1":

            max_spent = 0
            for gaze_info in gaze_seq:
                x,y, spent = gaze_info
                if spent > max_spent:
                    max_spent = spent 

            for time_step, gaze_info in enumerate(gaze_seq):
                x,y, spent = gaze_info 
                x, y = np.clip([x, y], 0., 0.999)

                row = math.floor(y*map_size)
                col = math.floor(x*map_size)
                time_in_seg[spent/max_spent].append(rle_binary[row][col])
                time_unnorm_in_seg[spent].append(rle_binary[row][col])

    time = []
    in_seg = []
    counts = []

    for key, val in time_unnorm_in_seg.items():
        time.append(key)
        in_seg.append(sum(val)/len(val))
        print(f"Time spent of {key} has a total count of {len(val)}")
        for i in range(len(val)):
            counts.append(key)
                
    time = np.array(time)
    in_seg = np.array(in_seg)

    plt.hist(counts, bins=7)
    plt.xlabel("Time Spent")
    plt.ylabel("Counts of Val")
    plt.savefig("./gaze_analysis_results/time_spent_hist.png")
    plt.close()

    plt.scatter(time, in_seg)
    plt.xlabel("Time Spent")
    plt.ylabel("Proportion of Being In Seg Mask")
    plt.savefig("./gaze_analysis_results/time_spent_unnorm_vs_seg.png")

    plt.close()

    time = []
    in_seg = []

    for key, val in time_in_seg.items():
        time.append(key)
        in_seg.append(sum(val)/len(val))
                
    time = np.array(time)
    in_seg = np.array(in_seg)

    plt.scatter(time, in_seg)
    plt.xlabel("Time Spent")
    plt.ylabel("Proportion of Being In Seg Mask")
    plt.savefig("./gaze_analysis_results/time_spent_vs_seg.png")

    plt.close()

    return np.corrcoef(time, in_seg)[0,1]


def heatmap_segmentation_visual_analysis(gaze_maps, segmentation_masks, threshold, map_size, gaze_sequences = None, full_img_size = 224):

    ### this relies upon seg masks being 7 x7 but we want them to be 224 x 224
    iou_vals = []
    
    segmentation_masks_small = get_segmentation_masks(map_size, gaze_sequences)
   
    for gaze_id, gaze_map in gaze_maps.items():
        gaze_map = gaze_map[0].reshape(map_size,map_size)

        img_path = gaze_id.split("/")[-1].split(".dcm")[0]
        rle_binary = segmentation_masks_small[img_path]

        if rle_binary != " -1":
            gaze_binary = gaze_map > threshold
            iou_vals.append((compute_heatmap_iou(gaze_binary, rle_binary), gaze_id))

    ### sort iou_vals and get top and bottom 20
    iou_sorted = sorted(iou_vals, key=lambda tup: tup[0], reverse=True)
    iou_top = iou_sorted[:50]
    iou_worst = iou_sorted[-50:]

    
    #print(iou_top)
    #print(iou_worst)

    ### thes lists were obtained from the above
    #iou_top = [(1.0, '1.2.276.0.7230010.3.1.2.8323329.1533.1517875168.262191/1.2.276.0.7230010.3.1.3.8323329.1533.1517875168.262190/1.2.276.0.7230010.3.1.4.8323329.1533.1517875168.262192.dcm'), (1.0, '1.2.276.0.7230010.3.1.2.8323329.1823.1517875169.745070/1.2.276.0.7230010.3.1.3.8323329.1823.1517875169.745069/1.2.276.0.7230010.3.1.4.8323329.1823.1517875169.745071.dcm'), (1.0, '1.2.276.0.7230010.3.1.2.8323329.2123.1517875171.270460/1.2.276.0.7230010.3.1.3.8323329.2123.1517875171.270459/1.2.276.0.7230010.3.1.4.8323329.2123.1517875171.270461.dcm'), (0.8, '1.2.276.0.7230010.3.1.2.8323329.1824.1517875169.753286/1.2.276.0.7230010.3.1.3.8323329.1824.1517875169.753285/1.2.276.0.7230010.3.1.4.8323329.1824.1517875169.753287.dcm'), (0.75, '1.2.276.0.7230010.3.1.2.8323329.1339.1517875167.304185/1.2.276.0.7230010.3.1.3.8323329.1339.1517875167.304184/1.2.276.0.7230010.3.1.4.8323329.1339.1517875167.304186.dcm'), (0.75, '1.2.276.0.7230010.3.1.2.8323329.2236.1517875171.758937/1.2.276.0.7230010.3.1.3.8323329.2236.1517875171.758936/1.2.276.0.7230010.3.1.4.8323329.2236.1517875171.758938.dcm'), (0.75, '1.2.276.0.7230010.3.1.2.8323329.1755.1517875169.266541/1.2.276.0.7230010.3.1.3.8323329.1755.1517875169.266540/1.2.276.0.7230010.3.1.4.8323329.1755.1517875169.266542.dcm'), (0.6666666666666666, '1.2.276.0.7230010.3.1.2.8323329.1999.1517875170.634379/1.2.276.0.7230010.3.1.3.8323329.1999.1517875170.634378/1.2.276.0.7230010.3.1.4.8323329.1999.1517875170.634380.dcm'), (0.6666666666666666, '1.2.276.0.7230010.3.1.2.8323329.2209.1517875171.648279/1.2.276.0.7230010.3.1.3.8323329.2209.1517875171.648278/1.2.276.0.7230010.3.1.4.8323329.2209.1517875171.648280.dcm'), (0.6666666666666666, '1.2.276.0.7230010.3.1.2.8323329.1648.1517875168.738107/1.2.276.0.7230010.3.1.3.8323329.1648.1517875168.738106/1.2.276.0.7230010.3.1.4.8323329.1648.1517875168.738108.dcm'), (0.6666666666666666, '1.2.276.0.7230010.3.1.2.8323329.1557.1517875168.394767/1.2.276.0.7230010.3.1.3.8323329.1557.1517875168.394766/1.2.276.0.7230010.3.1.4.8323329.1557.1517875168.394768.dcm'), (0.6666666666666666, '1.2.276.0.7230010.3.1.2.8323329.1207.1517875166.781090/1.2.276.0.7230010.3.1.3.8323329.1207.1517875166.781089/1.2.276.0.7230010.3.1.4.8323329.1207.1517875166.781091.dcm'), (0.6666666666666666, '1.2.276.0.7230010.3.1.2.8323329.1527.1517875168.236419/1.2.276.0.7230010.3.1.3.8323329.1527.1517875168.236418/1.2.276.0.7230010.3.1.4.8323329.1527.1517875168.236420.dcm'), (0.6666666666666666, '1.2.276.0.7230010.3.1.2.8323329.1952.1517875170.398573/1.2.276.0.7230010.3.1.3.8323329.1952.1517875170.398572/1.2.276.0.7230010.3.1.4.8323329.1952.1517875170.398574.dcm'), (0.6666666666666666, '1.2.276.0.7230010.3.1.2.8323329.2259.1517875171.842593/1.2.276.0.7230010.3.1.3.8323329.2259.1517875171.842592/1.2.276.0.7230010.3.1.4.8323329.2259.1517875171.842594.dcm'), (0.6666666666666666, '1.2.276.0.7230010.3.1.2.8323329.2145.1517875171.363122/1.2.276.0.7230010.3.1.3.8323329.2145.1517875171.363121/1.2.276.0.7230010.3.1.4.8323329.2145.1517875171.363123.dcm'), (0.6666666666666666, '1.2.276.0.7230010.3.1.2.8323329.2060.1517875170.960325/1.2.276.0.7230010.3.1.3.8323329.2060.1517875170.960324/1.2.276.0.7230010.3.1.4.8323329.2060.1517875170.960326.dcm'), (0.6666666666666666, '1.2.276.0.7230010.3.1.2.8323329.1241.1517875166.927801/1.2.276.0.7230010.3.1.3.8323329.1241.1517875166.927800/1.2.276.0.7230010.3.1.4.8323329.1241.1517875166.927802.dcm'), (0.6666666666666666, '1.2.276.0.7230010.3.1.2.8323329.2138.1517875171.323419/1.2.276.0.7230010.3.1.3.8323329.2138.1517875171.323418/1.2.276.0.7230010.3.1.4.8323329.2138.1517875171.323420.dcm'), (0.6, '1.2.276.0.7230010.3.1.2.8323329.1839.1517875169.871072/1.2.276.0.7230010.3.1.3.8323329.1839.1517875169.871071/1.2.276.0.7230010.3.1.4.8323329.1839.1517875169.871073.dcm')]
    #iou_worst = [(0.0, '1.2.276.0.7230010.3.1.2.8323329.1556.1517875168.390346/1.2.276.0.7230010.3.1.3.8323329.1556.1517875168.390345/1.2.276.0.7230010.3.1.4.8323329.1556.1517875168.390347.dcm'), (0.0, '1.2.276.0.7230010.3.1.2.8323329.1754.1517875169.260174/1.2.276.0.7230010.3.1.3.8323329.1754.1517875169.260173/1.2.276.0.7230010.3.1.4.8323329.1754.1517875169.260175.dcm'), (0.0, '1.2.276.0.7230010.3.1.2.8323329.1700.1517875169.6933/1.2.276.0.7230010.3.1.3.8323329.1700.1517875169.6932/1.2.276.0.7230010.3.1.4.8323329.1700.1517875169.6934.dcm'), (0.0, '1.2.276.0.7230010.3.1.2.8323329.1512.1517875168.175003/1.2.276.0.7230010.3.1.3.8323329.1512.1517875168.175002/1.2.276.0.7230010.3.1.4.8323329.1512.1517875168.175004.dcm'), (0.0, '1.2.276.0.7230010.3.1.2.8323329.1616.1517875168.608626/1.2.276.0.7230010.3.1.3.8323329.1616.1517875168.608625/1.2.276.0.7230010.3.1.4.8323329.1616.1517875168.608627.dcm'), (0.0, '1.2.276.0.7230010.3.1.2.8323329.1841.1517875169.876271/1.2.276.0.7230010.3.1.3.8323329.1841.1517875169.876270/1.2.276.0.7230010.3.1.4.8323329.1841.1517875169.876272.dcm'), (0.0, '1.2.276.0.7230010.3.1.2.8323329.1909.1517875170.188917/1.2.276.0.7230010.3.1.3.8323329.1909.1517875170.188916/1.2.276.0.7230010.3.1.4.8323329.1909.1517875170.188918.dcm'), (0.0, '1.2.276.0.7230010.3.1.2.8323329.1374.1517875167.438881/1.2.276.0.7230010.3.1.3.8323329.1374.1517875167.438880/1.2.276.0.7230010.3.1.4.8323329.1374.1517875167.438882.dcm'), (0.0, '1.2.276.0.7230010.3.1.2.8323329.1667.1517875168.835417/1.2.276.0.7230010.3.1.3.8323329.1667.1517875168.835416/1.2.276.0.7230010.3.1.4.8323329.1667.1517875168.835418.dcm'), (0.0, '1.2.276.0.7230010.3.1.2.8323329.2229.1517875171.736201/1.2.276.0.7230010.3.1.3.8323329.2229.1517875171.736200/1.2.276.0.7230010.3.1.4.8323329.2229.1517875171.736202.dcm'), (0.0, '1.2.276.0.7230010.3.1.2.8323329.1361.1517875167.387852/1.2.276.0.7230010.3.1.3.8323329.1361.1517875167.387851/1.2.276.0.7230010.3.1.4.8323329.1361.1517875167.387853.dcm'), (0.0, '1.2.276.0.7230010.3.1.2.8323329.1820.1517875169.733764/1.2.276.0.7230010.3.1.3.8323329.1820.1517875169.733763/1.2.276.0.7230010.3.1.4.8323329.1820.1517875169.733765.dcm'), (0.0, '1.2.276.0.7230010.3.1.2.8323329.1956.1517875170.423046/1.2.276.0.7230010.3.1.3.8323329.1956.1517875170.423045/1.2.276.0.7230010.3.1.4.8323329.1956.1517875170.423047.dcm'), (0.0, '1.2.276.0.7230010.3.1.2.8323329.1889.1517875170.126228/1.2.276.0.7230010.3.1.3.8323329.1889.1517875170.126227/1.2.276.0.7230010.3.1.4.8323329.1889.1517875170.126229.dcm'), (0.0, '1.2.276.0.7230010.3.1.2.8323329.2034.1517875170.825135/1.2.276.0.7230010.3.1.3.8323329.2034.1517875170.825134/1.2.276.0.7230010.3.1.4.8323329.2034.1517875170.825136.dcm'), (0.0, '1.2.276.0.7230010.3.1.2.8323329.1830.1517875169.795470/1.2.276.0.7230010.3.1.3.8323329.1830.1517875169.795469/1.2.276.0.7230010.3.1.4.8323329.1830.1517875169.795471.dcm'), (0.0, '1.2.276.0.7230010.3.1.2.8323329.1577.1517875168.477893/1.2.276.0.7230010.3.1.3.8323329.1577.1517875168.477892/1.2.276.0.7230010.3.1.4.8323329.1577.1517875168.477894.dcm'), (0.0, '1.2.276.0.7230010.3.1.2.8323329.1522.1517875168.226235/1.2.276.0.7230010.3.1.3.8323329.1522.1517875168.226234/1.2.276.0.7230010.3.1.4.8323329.1522.1517875168.226236.dcm'), (0.0, '1.2.276.0.7230010.3.1.2.8323329.1245.1517875166.958339/1.2.276.0.7230010.3.1.3.8323329.1245.1517875166.958338/1.2.276.0.7230010.3.1.4.8323329.1245.1517875166.958340.dcm'), (0.0, '1.2.276.0.7230010.3.1.2.8323329.1356.1517875167.366220/1.2.276.0.7230010.3.1.3.8323329.1356.1517875167.366219/1.2.276.0.7230010.3.1.4.8323329.1356.1517875167.366221.dcm')]
    ### can use gaze_id to open image via image path 

    tforms = transforms.Compose([transforms.Resize([full_img_size, full_img_size]), transforms.ToTensor()])


    ### print qudrant of top 20 and qudrant of bottom 20 
    top_quadrants = []
    for top in iou_top:
        seg = segmentation_masks[top[1].split("/")[-1].split(".dcm")[0]]
        seg = torch.tensor(seg).unsqueeze(0)
        pooled = F.avg_pool2d(seg, int(224/2)).squeeze()
        top_quadrants.append(torch.argmax(pooled.flatten()).item())


    print(f"Top quadrants are: {top_quadrants}")
    plt.hist(top_quadrants, bins=4)
    plt.xlabel("Quadrant")
    plt.ylabel("Abnormality Count via Segmentation Mask")
    plt.savefig("./gaze_analysis_results/best_quadrants.png")
    plt.close()

    bottom_qudrants = []
    for bot in iou_worst:
        seg = segmentation_masks[bot[1].split("/")[-1].split(".dcm")[0]]
        seg = torch.tensor(seg).unsqueeze(0)
        pooled = F.avg_pool2d(seg, int(224/2)).squeeze()
        bottom_qudrants.append(torch.argmax(pooled.flatten()).item())


    print(f"Worst quadrants are: {bottom_qudrants}")
    plt.hist(bottom_qudrants, bins=4)
    plt.xlabel("Qudrant")
    plt.ylabel("Abnormality Count via Segmentation Mask")
    plt.savefig("./gaze_analysis_results/worst_quadrants.png")
    plt.close()

    for j in range(4):
        fig = plt.figure(figsize=(12, 12))
        columns = 4
        rows = 5
    
        for i in range(1, columns*rows +1,4):

            img_pth = os.path.join("/media", f"pneumothorax/dicom_images/{iou_top[5*j + int((i - 1)/4)][1]}")
            ds = pydicom.dcmread(img_pth)
            img = ds.pixel_array
            img = Image.fromarray(np.uint8(img))
            img = tforms(img)
            #img = torch.cat([img, img, img])
            img = img.transpose(0,2)
            img = img.transpose(0,1)
            img = img.squeeze()
            fig.add_subplot(rows, columns, i)
            plt.imshow(img, cmap="gray")
            #plt.close()

            fig.add_subplot(rows, columns, i + 1)
            plt.imshow(img, cmap="gray")
            seg = segmentation_masks[iou_top[5*j + int((i - 1)/4)][1].split("/")[-1].split(".dcm")[0]]*125 + np.ones((224,224))*30
            seg = seg.astype(int)        
            plt.imshow(seg, alpha=0.5, cmap="coolwarm")
            #plt.close()

            fig.add_subplot(rows, columns, i + 2)
            gaze_map = gaze_maps[iou_top[5*j + int((i - 1)/4)][1]][0].reshape(7,7)
            plt.imshow(gaze_map, cmap="coolwarm")
            #plt.close()

            fig.add_subplot(rows, columns, i + 3)
            gaze_map = gaze_maps[iou_top[5*j + int((i - 1)/4)][1]][0].reshape(7,7)
            gaze_binary = gaze_map > threshold
            plt.imshow(gaze_binary, cmap="coolwarm")
            #plt.close()

        #plt.title(f"Gaze Heatmap vs Ground Truth Segmentation Comparison, threshold of {threshold}")
        plt.savefig(f"./gaze_analysis_results/best_iou_images/top_{j}_threshold_{threshold}.png")

    for j in range(4):
        fig = plt.figure(figsize=(12, 12))
        columns = 4
        rows = 5
    
        for i in range(1, columns*rows +1,4):

            img_pth = os.path.join("/media", f"pneumothorax/dicom_images/{iou_worst[5*j + int((i - 1)/4)][1]}")
            ds = pydicom.dcmread(img_pth)
            img = ds.pixel_array
            img = Image.fromarray(np.uint8(img))
            img = tforms(img)
            #img = torch.cat([img, img, img])
            img = img.transpose(0,2)
            img = img.transpose(0,1)
            img = img.squeeze()
            fig.add_subplot(rows, columns, i)
            plt.imshow(img, cmap="gray")
            #plt.close()

            fig.add_subplot(rows, columns, i + 1)
            plt.imshow(img, cmap="gray")
            seg = segmentation_masks[iou_worst[5*j + int((i - 1)/4)][1].split("/")[-1].split(".dcm")[0]]*125 + np.ones((224,224))*30
            seg = seg.astype(int)        
            plt.imshow(seg, alpha=0.5, cmap="coolwarm")
            #plt.close()

            fig.add_subplot(rows, columns, i + 2)
            gaze_map = gaze_maps[iou_worst[5*j + int((i - 1)/4)][1]][0].reshape(7,7)
            plt.imshow(gaze_map, cmap="coolwarm")
            #plt.close()

            fig.add_subplot(rows, columns, i + 3)
            gaze_map = gaze_maps[iou_worst[5*j + int((i - 1)/4)][1]][0].reshape(7,7)
            gaze_binary = gaze_map > threshold
            plt.imshow(gaze_binary, cmap="coolwarm")
            #plt.close()

        #plt.title(f"Gaze Heatmap vs Ground Truth Segmentation Comparison, threshold of {threshold}")
        plt.savefig(f"./gaze_analysis_results/worst_iou_images/worst_{j}_threshold_{threshold}.png")
          
  
    ### determine top 20, bottom 20 iou values, obtain images for each of them, overlay segmentation region on them
    ### display gaze heatmap to the side, return all of these ## plot original image, image with segmentation, binary gaze heatmap, regular gaze heatmap

    

def gaze_time_segmentation(gaze_seqs, segmentation_masks, map_size):
    time = []
    in_seg = [] # 0 is false, 1 is true 


    for gaze_id, gaze_seq in gaze_seqs.items():
        img_path = gaze_id.split("/")[-1].split(".dcm")[0]
        rle_binary = segmentation_masks[img_path]

        if rle_binary != " -1":

            for time_step, gaze_info in enumerate(gaze_seq):
                ts = (time_step + 1)/len(gaze_seq)
                x,y, _ = gaze_info 
                x, y = np.clip([x, y], 0., 0.999)

                row = math.floor(y*map_size)
                col = math.floor(x*map_size)
                time.append(ts)
                in_seg.append(rle_binary[row][col])
               
                
    time = np.array(time)
    in_seg = np.array(in_seg)

    plt.scatter(time, in_seg)
    plt.xlabel("Time Step")
    plt.ylabel("Proportion of Being In Seg Mask")
    plt.savefig("./gaze_analysis_results/time_vs_seg.png")
    plt.close()

    return np.corrcoef(time, in_seg)[0,1]

def binarize_seg_mask(segmask, map_size):

    ret_seg = np.zeros((map_size,map_size))

    for i in range(segmask.shape[0]):
        for j in range(segmask.shape[1]):

            ## i = row, j = col
            val = segmask[i][j]

            small_row = int(i / math.ceil(segmask.shape[0]/map_size))
            small_col = int(j / math.ceil(segmask.shape[0]/map_size))

            if val == 1:
                ret_seg[small_row][small_col] = val


    return ret_seg

def get_segmentation_masks(map_size, gaze_sequences):
    # load seg dict
    seg_dict_pth = "/media/pneumothorax/rle_dict.pkl"
    with open(seg_dict_pth, "rb") as pkl_f:
        rle_dict = pickle.load(pkl_f)

    ret_rle = {}
    for gaze_id, val in tqdm(gaze_sequences.items(), total=len(gaze_sequences.keys())):
        img_path = gaze_id.split("/")[-1].split(".dcm")[0]
        rle = rle_dict[img_path]

        if rle != " -1":
            segmask_org = rle2mask(rle, 1024, 1024).T
            segmask = binarize_seg_mask(segmask_org, map_size)
            ret_rle[img_path] = segmask
        else:
            ret_rle[img_path] = rle 

    return ret_rle
    #print(rle_dict.keys())

def get_gaze_sequences():
    train_seqs, train_labels, train_gaze_ids = load_gaze_data("cxr_p", "train", 1, 0.2, False, 0)
    val_seqs, val_labels, val_gaze_ids = load_gaze_data("cxr_p", "val", 1, 0.2, False, 0)

    train_gaze_sequences = {gaze: seq for seq, gaze in zip(train_seqs, train_gaze_ids)}
    val_gaze_sequences = {gaze: seq for seq, gaze in zip(val_seqs, val_gaze_ids)}

    train_gaze_sequences.update(val_gaze_sequences)

    return train_gaze_sequences

def get_gaze_heatmaps(task):
    train_gaze_attribute_labels_dict = load_gaze_attribute_labels("cxr_p", "train", task, 0)
    val_gaze_attribute_labels_dict = load_gaze_attribute_labels("cxr_p", "val", task, 0)
    
    train_gaze_attribute_labels_dict.update(val_gaze_attribute_labels_dict)

    return train_gaze_attribute_labels_dict 


def main(threshold, threshold_max = .95, interval = 0.05, map_size = 7, image_analysis=False):

    task = "heatmap2" if map_size == 7 else "heatmap3"
    if image_analysis:
        task = "heatmap2"

    gaze_heatmaps = get_gaze_heatmaps(task) ## 7 x 7
    gaze_sequences = get_gaze_sequences() 
    segmentation_masks = get_segmentation_masks(map_size, gaze_sequences) ### map_size x map_size

    if image_analysis:
        heatmap_segmentation_visual_analysis(gaze_heatmaps, segmentation_masks, 0.1, gaze_sequences = gaze_sequences, map_size = 7)
        return 

   
    if map_size == 7:
        ### evaluate overlap between current 7x7 heatmap and segmentation mask

        metrics = {"threshold" : [], "iou_gaze_segs" : [], "iou_gaze_inv_segs": [], "overlap_gaze_inv_segs" : []}

        for t in np.arange(threshold, threshold_max + interval, interval):
            metrics["threshold"].append(t)
            print(f"Using a threshold of: {t}")
            iou_gaze_seg = gaze_heatmap_segmentation_mask(gaze_heatmaps, segmentation_masks, t, map_size, inverse_segmentation = False ,iou = False)
            print(f"Avergage IoU Between Gaze Heatmap and Ground Truth Segmentation Mask: {iou_gaze_seg}")

            ### evaluate overlap between current 7 x 7 heatmap and not segmentation mask regions
            iou_gaze_inv_seg = gaze_heatmap_segmentation_mask(gaze_heatmaps, segmentation_masks, t,  map_size, inverse_segmentation = True ,iou = True)
            overlap_gaze_inv_seg = gaze_heatmap_segmentation_mask(gaze_heatmaps, segmentation_masks, t,  map_size, inverse_segmentation = True ,iou = False)
            print(f"Average IoU Between Gaze Heatmap and Regions Not Contained within Ground Truth Segmentation Mask: {iou_gaze_inv_seg}")
            print(f"Average Overlap Between Gaze Heatmap and Regions Not Contained within Ground Truth Segmentation Mask: {overlap_gaze_inv_seg}")

            ### make iou vs time spent plots 
            gaze_heatmap_segmentation_iou_plots(gaze_sequences, segmentation_masks, gaze_heatmaps, t, map_size)

            metrics["iou_gaze_inv_segs"].append(iou_gaze_inv_seg)
            metrics["iou_gaze_segs"].append(iou_gaze_seg)
            metrics["overlap_gaze_inv_segs"].append(overlap_gaze_inv_seg)


        ### evaluate relationship/trend between gaze points being in segmentation region or being outside segmentation region and the time step in the seuqence (i.e plot)
        corr = gaze_time_segmentation(gaze_sequences, segmentation_masks, map_size)
        print(f"Correlation between gaze time step and being in ground truth segmentation mask: {corr}")

        ### evaluate relationship/trend between gaze points being in segmentation region or being outside segmentation region and the time spent on the patch 
        spent_corr = gaze_time_spent_segmentation(gaze_sequences, segmentation_masks, map_size)
        print(f"Correlation between gaze time spent and being in ground truth segmentation mask: {spent_corr}")

        metrics_df = pd.DataFrame.from_dict(metrics)
        print(metrics_df)

        metrics_json = metrics_df.to_json(orient="index")
        with open('./gaze_analysis_results/metrics.json', 'w') as f:
            json.dump(metrics_json, f)


    elif map_size == 224:
        ### evaluate overlap between current 224x224 heatmap and segmentation mask

        metrics = {"threshold" : [], "iou_gaze_segs" : [], "iou_gaze_inv_segs": [], "overlap_gaze_inv_segs" : []}

        for t in np.arange(threshold, threshold_max + interval, interval):
            metrics["threshold"].append(t)
            print(f"Using a threshold of: {t}")
            iou_gaze_seg = gaze_heatmap_segmentation_mask(gaze_heatmaps, segmentation_masks, t, map_size, inverse_segmentation = False ,iou = True)
            print(f"Avergage IoU Between Gaze Heatmap and Ground Truth Segmentation Mask: {iou_gaze_seg}")

            ### evaluate overlap between current 7 x 7 heatmap and not segmentation mask regions
            iou_gaze_inv_seg = gaze_heatmap_segmentation_mask(gaze_heatmaps, segmentation_masks, t,  map_size, inverse_segmentation = True ,iou = True)
            overlap_gaze_inv_seg = gaze_heatmap_segmentation_mask(gaze_heatmaps, segmentation_masks, t,  map_size, inverse_segmentation = True ,iou = False)
            print(f"Average IoU Between Gaze Heatmap and Regions Not Contained within Ground Truth Segmentation Mask: {iou_gaze_inv_seg}")
            print(f"Average Overlap Between Gaze Heatmap and Regions Not Contained within Ground Truth Segmentation Mask: {overlap_gaze_inv_seg}")

            ### make iou vs time spent plots 
            gaze_heatmap_segmentation_iou_plots(gaze_sequences, segmentation_masks, gaze_heatmaps, t, map_size)

            metrics["iou_gaze_inv_segs"].append(iou_gaze_inv_seg)
            metrics["iou_gaze_segs"].append(iou_gaze_seg)
            metrics["overlap_gaze_inv_segs"].append(overlap_gaze_inv_seg)


        ### evaluate relationship/trend between gaze points being in segmentation region or being outside segmentation region and the time step in the seuqence (i.e plot)
        corr = gaze_time_segmentation(gaze_sequences, segmentation_masks, map_size)
        print(f"Correlation between gaze time step and being in ground truth segmentation mask: {corr}")

        ### evaluate relationship/trend between gaze points being in segmentation region or being outside segmentation region and the time spent on the patch 
        spent_corr = gaze_time_spent_segmentation(gaze_sequences, segmentation_masks, map_size)
        print(f"Correlation between gaze time spent and being in ground truth segmentation mask: {spent_corr}")

        metrics_df = pd.DataFrame.from_dict(metrics)
        print(metrics_df)

        metrics_json = metrics_df.to_json(orient="index")
        with open('./gaze_analysis_results/metrics_224.json', 'w') as f:
            json.dump(metrics_json, f)
        

    else:
        raise ValueError("Map size not supported under current analysis")

if __name__ == "__main__":
    #main(threshold = 0, threshold_max=0.95, map_size = 224, image_analysis=True)
    main(threshold = 0, threshold_max=0.95, map_size = 7, image_analysis=False)

