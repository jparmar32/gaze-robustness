import os
import pickle
import numpy as np

from torchvision import transforms

from sklearn.model_selection import StratifiedShuffleSplit

import pdb

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


