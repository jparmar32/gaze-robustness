
from cgitb import small
import numpy as np
import torch
import pickle 
from tqdm import tqdm
import math

import matplotlib.pyplot as plt
from skimage.transform import resize
from utils import load_file_markers, get_data_transforms, load_gaze_attribute_labels, rle2mask, load_gaze_data

def compute_heatmap_iou(mask_1, mask_2):
    area_overlap = np.sum(np.logical_and(mask_1, mask_2))
    area_union = np.sum(mask_1) + np.sum(mask_2) - area_overlap
    return (area_overlap / area_union)


def compute_intersection_over_seg(gaze_mask, seg_mask):
    area_overlap = np.sum(np.logical_and(gaze_mask, seg_mask))
    return (area_overlap/np.sum(seg_mask))

def gaze_heatmap_segmentation_mask(gaze_maps, segmentation_masks, threshold, inverse_segmentation = False ,iou = True):
    
    metric = []

    for gaze_id, gaze_map in gaze_maps.items():
        gaze_map = gaze_map[0].reshape(7,7)
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
    plt.ylabel("In Seg Mask")
    plt.savefig("gaze_vs_seg.png")

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

def get_gaze_heatmaps():
    train_gaze_attribute_labels_dict = load_gaze_attribute_labels("cxr_p", "train", "heatmap2", 0)
    val_gaze_attribute_labels_dict = load_gaze_attribute_labels("cxr_p", "val", "heatmap2", 0)
    
    train_gaze_attribute_labels_dict.update(val_gaze_attribute_labels_dict)

    return train_gaze_attribute_labels_dict 


def main(threshold, map_size = 7):

    gaze_heatmaps = get_gaze_heatmaps() ## 7 x 7
    gaze_sequences = get_gaze_sequences() 
    segmentation_masks = get_segmentation_masks(map_size, gaze_sequences) ### map_size x map_size

    #print(gaze_heatmaps[list(gaze_heatmaps.keys())[0]][0].reshape(7,7).shape)
    #raise ValueError("test")

    ## TODO: implement iter over threshold 
   
    if map_size == 7:
        ### evaluate overlap between current 7x7 heatmap and segmentation mask
        iou_gaze_seg = gaze_heatmap_segmentation_mask(gaze_heatmaps, segmentation_masks, threshold, inverse_segmentation = False ,iou = True)
        print(f"Avergage IoU Between Gaze Heatmap and Ground Truth Segmentation Mask: {iou_gaze_seg}")

        ### evaluate overlap between current 7 x 7 heatmap and not segmentation mask regions
        iou_gaze_inv_seg = gaze_heatmap_segmentation_mask(gaze_heatmaps, segmentation_masks, threshold, inverse_segmentation = True ,iou = True)
        overlap_gaze_inv_seg = gaze_heatmap_segmentation_mask(gaze_heatmaps, segmentation_masks, threshold, inverse_segmentation = True ,iou = False)
        print(f"Average IoU Between Gaze Heatmap and Regions Not Contained within Ground Truth Segmentation Mask: {iou_gaze_inv_seg}")
        print(f"Average Overlap Between Gaze Heatmap and Regions Not Contained within Ground Truth Segmentation Mask: {overlap_gaze_inv_seg}")

        ### evaluate relationship/trend between gaze points being in segmentation region or being outside segmentation region and the time step in the seuqence (i.e plot)
        corr = gaze_time_segmentation(gaze_sequences, segmentation_masks, map_size)
        print(f"Correlation between gaze time step and being in ground truth segmentation mask: {corr}")

    elif map_size == 224:
        pass
        ## TODO: size up
    else:
        raise ValueError("Map size not supported under current analysis")

if __name__ == "__main__":
    main(threshold = 0, map_size = 7)

