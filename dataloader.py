import os, sys
import numpy as np
import pickle
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from PIL import Image
import pydicom

from utils import load_file_markers, get_data_transforms

import pdb

class RoboGazeDataset(Dataset):

    """
    This dataset fetches both images and gaze sequences/heatmaps for two CXR dataset sources
    It also returns OOD datasets for evaluation 
    """

    def __init__(
        self,
        source,
        data_dir,
        split_type,
        transform,
        ood_type=None,
        val_scale=0.2,
        seed=0
    ):

        """
        source: <str> identifies which dataset to use (cxr_p or cxr_a)
        split_type: <str>  options are train, val, test
        transform: pytorch image transform for the specified split type
        ood_type: <str> which OOD dataset the test set should be (None, "age", "hospital")
        """

        self.source = source
        self.data_dir = data_dir
        self.split_type = split_type
        self.transform = transform
        self.file_markers = load_file_markers(
            source,
            split_type,
            ood_type,
            val_scale,
            seed,
        )

    def __len__(self):
        return len(self.file_markers)

    def __getitem__(self, idx):

        """
        Fetch index idx image and label from dataset.
        Args:
            idx: (int) index in [0, 1, ..., size_of_dataset-1]
        Returns:
            image: (Tensor) transformed image
            label: (int) corresponding label of image
        """

        img_id, label = self.file_markers[idx]

        if self.source == "cxr_a":
            img_pth = os.path.join(self.data_dir, f"cxr_ibm/dicom_images/{img_id}")
        elif self.source == "cxr_p":
            img_pth = os.path.join(self.data_dir, f"pneumothorax/dicom_images/{img_id}")
        else:
            raise ValueError(f"{self.source} not an implemented dataset.")

        ds = pydicom.dcmread(img_pth)
        img = ds.pixel_array
        img = Image.fromarray(np.uint8(img))

        img = self.transform(img)
        if img.shape[0] == 1:
            img = torch.cat([img, img, img])

        return img, label 



def fetch_dataloaders(
    source,
    data_dir,
    val_scale,
    seed,
    batch_size,
    num_workers
):

    
    transforms = get_data_transforms("cxr", normalization_type="train_images")
    dataloaders = []

    for split in ["train", "val", "test"]:

        dataset = RoboGazeDataset(
            source=source,
            data_dir=data_dir,
            split_type=split,
            transform=transforms[split],
            val_scale=val_scale,
            seed=seed,
        )

        dataloaders.append(
            DataLoader(
                dataset=dataset,
                shuffle=split == "train",
                batch_size=batch_size,
                num_workers=num_workers,
            )
        )

    return dataloaders


if __name__ == "__main__":
    
    dls = fetch_dataloaders("cxr_a","/media",0.2,0,32,4)

    # for (img,label) in dls[0]:
    #     pdb.set_trace()






