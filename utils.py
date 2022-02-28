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
        file_markers_dir = os.path.join(file_dir, "trainval_list.pkl")
        
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
    "cxr": ([0.48865], [0.24621]),
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
        grid_size = 224

 
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

## implement in the batch case first, should 
def calculate_actdiff_loss(regular_activations, masked_activations):
    """
    regular_activtations: list of activations produced by the original image in the model
    masked_activations: list of activations produced by the masked image in the mdodel
    """

    assert len(regular_activations) == len(masked_activations)

    two_norm = torch.nn.modules.distance.PairwiseDistance(p=2)

    all_dists = []
    #L2 Distances between activations 
    for reg_act, masked_act in zip(regular_activations, masked_activations):
        all_dists.append(two_norm(reg_act.flatten().unsqueeze(0), masked_act.flatten().unsqueeze(0)))

    actdiff_loss = torch.sum(torch.hstack(all_dists))/len(all_dists)

    return(actdiff_loss)