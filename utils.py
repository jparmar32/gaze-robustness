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

def load_file_markers(
    source,
    split_type,
    ood_type,
    val_scale,
    seed,
    verbose=True
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


def get_data_transforms(dataset_name, normalization_type="none"):
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

    eval_transform = transforms.Compose(
        [
            transforms.Resize([224, 224]),
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