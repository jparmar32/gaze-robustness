from operator import inv, le
import os
import pickle
import numpy as np

from torchvision import transforms
from functools import partial

from sklearn.model_selection import StratifiedShuffleSplit
from torch.optim.lr_scheduler import CosineAnnealingLR as CosineScheduler
import torch 
import pdb
import sklearn.metrics as skl
import cv2
import skimage.exposure as exposure

import matplotlib.pyplot as plt

def load_file_markers(
    source,
    split_type,
    ood_type,
    val_scale,
    seed,
    verbose=True,
    subclass=False
):
    """
    Returns: a list of file markers with image_path,label tuples
    """
    
    file_dir = os.path.join("./filemarkers", source)

    if split_type in ["train", "val"]:
        #TODO: for CXR-P, make filemarker default gold
        file_markers_dir = os.path.join(file_dir, "trainval_list_gold.pkl") #gold
        
        with open(file_markers_dir, "rb") as fp:
            file_markers = pickle.load(fp)

        labels = [fm[1] for fm in file_markers]
        sss = StratifiedShuffleSplit(
            n_splits=1, test_size=val_scale, random_state=seed
        )

        for train_ndx, val_ndx in sss.split(np.zeros(len(labels)), labels):
            file_markers_train = [file_markers[ndx] for ndx in train_ndx]
            file_markers_val = [file_markers[ndx] for ndx in val_ndx]

        
        if split_type == "train":
            file_markers = file_markers_train
        else:
            file_markers = file_markers_val

    elif split_type == "test":
        
        file_markers_dir = os.path.join(file_dir, "test_list.pkl")
        if subclass:
            file_markers_dir = os.path.join(file_dir, "test_list_tube.pkl")

        with open(file_markers_dir, "rb") as fp:
            file_markers = pickle.load(fp)

    if verbose:
        print(f"{len(file_markers)} files in {split_type} split...")

    
    if source == "cxr_a":
        # strip off the first part of img_pth
        file_markers = [(fm[0].split("/")[-1], fm[1]) for fm in file_markers]

    # shuffle file markers
    np.random.seed(seed)
    np.random.shuffle(file_markers)

    return file_markers




norm_stats = {
    "imagenet": ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    "cxr": ([0.48865], [0.24621]), #potential new values: [0.4889199], [0.2476612]
}


def get_data_transforms(dataset_name, normalization_type="none", gan = False):
    """Get data transforms based on dataset name"""
    num_channels = 1

    if normalization_type == "none":
        mean, std = [0] * num_channels, [1] * num_channels
    elif normalization_type == "imagenet":
        if num_channels != 3:
            raise ValueError("Cannot use imagenet statistics with ≠3 channels")
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    elif normalization_type == "train_images":
        if dataset_name in norm_stats:
            mean, std = norm_stats[dataset_name]
        else:
            mean, std = [0] * num_channels, [1] * num_channels
    else:
        raise ValueError(f"Unknown normalization type {normalization_type}")


    resize_val = 224 if gan else 224

    eval_transform = transforms.Compose(
        [
            transforms.Resize([resize_val, resize_val]),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    train_transform = eval_transform

    data_transforms = {
        "train": train_transform,
        "eval": eval_transform,
    }
    data_transforms["val"] = eval_transform
    data_transforms["test"] = eval_transform
    return data_transforms


class AverageMeter(object):
    """Computes and stores the average and current value
    Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val, self.avg, self.sum, self.count = [0] * 4

    def update(self, val, n=1):
        if isinstance(val, torch.Tensor):
            val = val.data.cpu().item()
        if isinstance(n, torch.Tensor):
            n = n.data.cpu().item()
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)

def accuracy(targets, probs, return_stats=False, as_percent=True):
    """Return the number of probs that match the associated ground truth label,
    along with the total number of probs/labels."""
    probs, targets = torch.as_tensor(probs), torch.as_tensor(targets)
    if len(probs.shape) == 1 or probs.shape[-1] == 1:
        predicted = (probs.data > 0.5).float()
    else:
        predicted = torch.argmax(probs.data, 1)
    total = targets.shape[0]
    correct = (predicted == targets).sum().item()
    acc = correct / total
    if as_percent:
        acc *= 100
    if return_stats:
        return acc, (correct, total)
    return acc

def compute_roc_auc(targets, probs):
    try:
        num_classes = len(set(np.array(targets)))
        if (
            num_classes < 2
            or len(probs.shape) < 1
            or len(probs.shape) > 2
            or (len(probs.shape) == 2 and probs.shape[1] != num_classes)
        ):
            raise ValueError
        elif num_classes == 2:
            if len(probs.shape) == 2:
                probs = probs[:, 1]
        else:
            if len(probs.shape) < 2:
                raise ValueError
        auroc = skl.roc_auc_score(targets, probs, multi_class="ovo")
    except ValueError:
        auroc = -1
    return auroc


def get_lrs(optimizer):
    lrs = []
    for param_group in optimizer.param_groups:
        if "lr" in param_group:
            lrs.append(param_group["lr"])
    return lrs


def build_scheduler(args, optimizer):
    scheduler_type = partial(CosineScheduler, T_max=args.epochs, eta_min=args.min_lr)
    scheduler = scheduler_type(optimizer)
    return scheduler

def make_heatmaps(gaze_seqs, num_patches=8, normalize_heatmaps=False):
    all_grids = np.zeros((len(gaze_seqs), 1, num_patches, num_patches), dtype=np.float32)
    for ndx, gaze_seq in enumerate(gaze_seqs):
        # loop through gaze seq and increment # of visits to each patch
        for (x,y,t) in gaze_seq:
            # make sure if x or y are > 1 then they are 1
            x, y = np.clip([x, y], 0., 0.999)
            patch_x, patch_y = int(x*num_patches), int(y*num_patches)
            all_grids[ndx, 0, patch_x, patch_y] += t
        if normalize_heatmaps:
            # Destroy total time information, as a diagnostic
            all_grids[ndx] /= np.sum(all_grids[ndx])
    return all_grids


def compute_avg_heatmap(gaze_seqs, grid_width = 8):
    
    heatmaps = []
    for gaze_seq in gaze_seqs:
        
        hm = [0] * grid_width * grid_width
        for x, y, t in gaze_seq:
            x = min(max(0, x), 0.9999)
            y = min(max(0, y), 0.9999)
            idx = int(x * grid_width) * grid_width + int(y * grid_width)
            hm[idx] += t
            
        heatmaps.append(hm)
        
    heatmaps = np.array(heatmaps)
        
    return np.mean(heatmaps,axis=0)


def load_gaze_data(source, split_type, train_scale, val_scale, gold, seed, return_img_pths=False, verbose=True):
    """
    Returns: a dictionary of (gaze_id: gaze_seq) for the split type and source
    """

    gaze_dict_pth = os.path.join('/home/jsparmar/gaze-robustness/gaze_data', source + '_gaze_data.pkl')

    with open(gaze_dict_pth, 'rb') as pkl_f:
        gaze_dict_all = pickle.load(pkl_f)

    # load file markers for split to know which gaze sequences to return
    
    file_markers = load_file_markers(source, split_type, False, val_scale, seed, False)

    gaze_seqs = []
    labels = []
    gaze_ids = []
    img_pths = []
    for img_pth, lab in file_markers:
        img_pths.append(img_pth)
        labels.append(lab)

        # extract gaze_id from img_pth

        # get gaze seq
        if img_pth in gaze_dict_all:
         
            gaze_ids.append(img_pth)

            if gaze_dict_all[img_pth] == []:
                gaze_seqs.append([[0.5,0.5,1]])
            else:
                gaze_seqs.append(gaze_dict_all[img_pth])
        else:
            gaze_seqs.append([[0.5,0.5,1]])

    gaze_seqs = np.array(gaze_seqs, dtype=object)
    labels = np.array(labels)
    gaze_ids = np.array(gaze_ids)
    if verbose: print(f'{len(gaze_seqs)} gaze sequences in {split_type} split...')

    if return_img_pths:
        return gaze_seqs, labels, img_pths
    return gaze_seqs, labels, gaze_ids


def load_gaze_attribute_labels(source, split_type, task, seed):
    """
    Creates helper task labels depending on gaze_mtl_task
    options are: loc1, loc2, time, diffusivity
    """

    # pull all gaze sequences
    seqs, labels, gaze_ids = load_gaze_data(source, split_type, 1, 0.2, False, seed)
    # create task_labels dict
    task_labels = {}
    for ndx,gaze_id in enumerate(gaze_ids):
        task_labels[gaze_id] = []
    
    
    if task == "heatmap1":
        grid_size = 4 #3
    elif task == "heatmap2":
        grid_size = 7 #2 
    else:
        gazemap_val = task.split("_")[1]
        grid_size = int(gazemap_val)

 
    heatmaps = make_heatmaps(seqs, grid_size).reshape(-1,grid_size*grid_size)
            #seg_masks = load_seg_masks(source)
    for ndx,gaze_id in enumerate(gaze_ids):
        task_labels[gaze_id].append(heatmaps[ndx,:].T/np.sum(heatmaps[ndx,:]))
    
    return task_labels
        

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

## this will be called in get_item within the dataloader (i.e works on one example)
def create_masked_image(x, segmentation_mask):
    """
    masked image is defined as: x_masked = x*seg + shuffle(x)*(1 - seg)
    to be used in get_item of dataloader 
    """

    inverse_segmentation_mask = 1 - segmentation_mask
    inverse_segmentation_mask = inverse_segmentation_mask.bool()
    inverse_segmentation_mask = inverse_segmentation_mask.unsqueeze(0)
    inverse_segmentation_mask = torch.cat([inverse_segmentation_mask, inverse_segmentation_mask, inverse_segmentation_mask])

    ### obtain the values contained at the inverse_segmentation_mask indices 
    assert inverse_segmentation_mask.shape == x.shape

    try:
        shuffled = x[inverse_segmentation_mask]

        ### shuffle these values
        shuffled = shuffled[torch.randperm(torch.numel(shuffled))]

        ### append the shuffled values to the original image times the segmentation mask
        x_masked = x*segmentation_mask
        x_masked[inverse_segmentation_mask] = shuffled
    except:
        import pdb; pdb.set_trace()

    return x_masked

## this will be called in get_item within the dataloader (i.e works on one example)
def create_masked_image_advanced_augmentations(x, segmentation_mask, augmentation="normal", masked_normalization_vals=None, seed=15):
    """
    masked image is defined as: x_masked = x*seg + shuffle(x)*(1 - seg)
    to be used in get_item of dataloader 
    """
   
    torch.manual_seed(seed)
    inverse_segmentation_mask = 1 - segmentation_mask
    inverse_segmentation_mask = inverse_segmentation_mask.bool()
    inverse_segmentation_mask = inverse_segmentation_mask.unsqueeze(0)
    inverse_segmentation_mask = torch.cat([inverse_segmentation_mask, inverse_segmentation_mask, inverse_segmentation_mask])

    ### obtain the values contained at the inverse_segmentation_mask indices 
    assert inverse_segmentation_mask.shape == x.shape
    assert torch.equal(inverse_segmentation_mask + segmentation_mask, torch.ones_like(x).long())
    
    if augmentation == 'normal':
        shuffled = x[inverse_segmentation_mask]

        ### shuffle these values
        shuffled = shuffled[torch.randperm(torch.numel(shuffled))]

        ### append the shuffled values to the original image times the segmentation mask
        x_masked = x*segmentation_mask
        x_masked[inverse_segmentation_mask] = shuffled

    ## For all of these how should I normalize
    elif augmentation == 'gaussian_noise':
        
        mean = 0
        var =  0.5

        ### different across slices
        #noise = torch.randn_like(x)*np.sqrt(var) + mean

        ### same across slices
        noise = torch.randn((x.shape[1], x.shape[2]))*np.sqrt(var) + mean
        noise = noise.unsqueeze(0)
        noise = torch.cat([noise, noise, noise])

        background = x + noise 
        x_masked = x*segmentation_mask + background*inverse_segmentation_mask

        normalize = transforms.Normalize([masked_normalization_vals['mean']]*x_masked.shape[0], [masked_normalization_vals['std']]*x_masked.shape[0])
        x_masked = normalize(x_masked)


    elif augmentation == 'gaussian_blur':


        gaussian_blur = transforms.GaussianBlur(kernel_size=15, sigma=15)

        ### different across slices
        #background = gaussian_blur(x)

        ### same across slices
        first_channel_x = x[0,:,:]
        first_channel_x = first_channel_x.unsqueeze(0)
        background = gaussian_blur(first_channel_x)
        background = torch.cat([background, background, background])

        x_masked = x*segmentation_mask + background*inverse_segmentation_mask

        normalize = transforms.Normalize([masked_normalization_vals['mean']]*x_masked.shape[0], [masked_normalization_vals['std']]*x_masked.shape[0])
        x_masked = normalize(x_masked)


    elif augmentation == 'color_jitter':

        jitter = transforms.ColorJitter(brightness=(0.1,1.1), contrast=(0.1,1.1), saturation=(0.1,1.1), hue=(-0.2,0.2))
        background = jitter(x)
        x_masked = x*segmentation_mask + background*inverse_segmentation_mask

        normalize = transforms.Normalize([masked_normalization_vals['mean']]*x_masked.shape[0], [masked_normalization_vals['std']]*x_masked.shape[0])
        x_masked = normalize(x_masked)

    ### To depricate
    elif augmentation == 'color_gamma':
        
        x_copy = x.detach().clone()
        background = transforms.functional.adjust_gamma(x_copy,0.7)
        x_masked = x*segmentation_mask + background*inverse_segmentation_mask

        normalize = transforms.Normalize([masked_normalization_vals['mean']]*x_masked.shape[0], [masked_normalization_vals['std']]*x_masked.shape[0])
        x_masked = normalize(x_masked)
        

    elif augmentation == 'sobel_horizontal':
        

        sobel_x = torch.transpose(x, 0,1)
        sobel_x = torch.transpose(sobel_x, 1,2)
        sobel_x = sobel_x.numpy()
        gray = cv2.cvtColor(sobel_x,cv2.COLOR_RGB2GRAY)

        # apply sobel derivatives
        sobelx = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=13)

        # optionally normalize to range 0 to 255 for proper display
        sobelx_img = torch.tensor(sobelx)
        sobelx_img = sobelx_img.unsqueeze(0)
        sobelx_img = torch.cat([sobelx_img, sobelx_img, sobelx_img])
        sobelx_img = sobelx_img.float()

        x_masked = x*segmentation_mask + sobelx_img*inverse_segmentation_mask
        normalize = transforms.Normalize([masked_normalization_vals['mean']]*x_masked.shape[0], [masked_normalization_vals['std']]*x_masked.shape[0])
        x_masked = normalize(x_masked)


    elif augmentation == 'sobel_magnitude':

        sobel_x = torch.transpose(x, 0,1)
        sobel_x = torch.transpose(sobel_x, 1,2)
        sobel_x = sobel_x.numpy()

        gray = cv2.cvtColor(sobel_x,cv2.COLOR_RGB2GRAY)

        # apply sobel derivatives
        sobelx = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=13)
        sobely = cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=13)

        # square 
        sobelx2 = cv2.multiply(sobelx,sobelx)
        sobely2 = cv2.multiply(sobely,sobely)

        # add together and take square root
        sobel_magnitude = cv2.sqrt(sobelx2 + sobely2)
    
        sobel_mag_img = torch.tensor(sobel_magnitude)
        sobel_mag_img = sobel_mag_img.unsqueeze(0)
        sobel_mag_img = torch.cat([sobel_mag_img, sobel_mag_img, sobel_mag_img])
        sobel_mag_img = sobel_mag_img.float()

        x_masked = x*segmentation_mask + sobel_mag_img*inverse_segmentation_mask
        normalize = transforms.Normalize([masked_normalization_vals['mean']]*x_masked.shape[0], [masked_normalization_vals['std']]*x_masked.shape[0])
        x_masked = normalize(x_masked)

    return x_masked

## implement in the batch case first, should 
def calculate_actdiff_loss(regular_activations, masked_activations, similarity_metric="l2"):
    """
    regular_activtations: list of activations produced by the original image in the model
    masked_activations: list of activations produced by the masked image in the mdodel
    """

    assert len(regular_activations) == len(masked_activations)

    if similarity_metric == "l2":
        metric = torch.nn.modules.distance.PairwiseDistance(p=2)
        all_dists = []
        for reg_act, masked_act in zip(regular_activations, masked_activations):
            all_dists.append(metric(reg_act.flatten().unsqueeze(0), masked_act.flatten().unsqueeze(0)))
            
    elif similarity_metric == "cosine":
        metric = torch.nn.CosineSimilarity(dim=0)
        all_dists = []
        for reg_act, masked_act in zip(regular_activations, masked_activations):
            all_dists.append(metric(reg_act.flatten(), masked_act.flatten()))
            

    #print(torch.hstack(all_dists))
    actdiff_loss = torch.sum(torch.hstack(all_dists))/len(all_dists)

    return(actdiff_loss)

## currently this is naive as these values are calculated from actdiff_lungmask at the 224x224 resolution and will
## not be correct for other segmentations/resolutions. Hence, we should consider a shift in how this is calculated
def get_masked_normalization(augmentation_type):
    
    if augmentation_type == "gaussian_noise":
        return {'mean': -0.002397208008915186, 'std': np.sqrt(1.4651013612747192)}

    elif augmentation_type == "gaussian_blur":
        return {'mean': 0.005687213037163019, 'std': np.sqrt(0.9313074946403503)}
 
    elif augmentation_type == "color_jitter":
        return {'mean': 0.2101794332265854, 'std': np.sqrt(0.10437733680009842)}

    elif augmentation_type == "sobel_horizontal":
        return {'mean': 2767.83740234375, 'std': np.sqrt(223935873024.0)}

    elif augmentation_type == "sobel_magnitude":
        return {'mean': 377082.25, 'std': np.sqrt(245810937856.0)}

    elif augmentation_type == "standard":
        return {'mean': 0.48865, 'std': 0.24621}

    else:
        return None 