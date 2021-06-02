import os, sys
import numpy as np
import pickle
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from PIL import Image
import pydicom

from utils import load_file_markers, get_data_transforms, load_gaze_attribute_labels


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
        gaze_task = None,
        val_scale=0.2,
        seed=0,
        subclass=False
    ):

        """
        source: <str> identifies which dataset to use (cxr_p or cxr_a)
        split_type: <str>  options are train, val, test
        transform: pytorch image transform for the specified split type
        ood_type: <str> which OOD dataset the test set should be (None, "age", "hospital")
        """

        self.source = source
        self.gaze_task = gaze_task
        self.ood = False
        if self.source not in ['cxr_a', 'cxr_p']:
            self.ood = True

        self.data_dir = data_dir
        self.split_type = split_type
        self.transform = transform
        self.subclass = subclass
        self.file_markers = load_file_markers(
            source,
            split_type,
            self.ood,
            val_scale,
            seed,
            subclass=self.subclass
        )

        if self.split_type in ['train', 'val']:

            type_feature = None
            if self.gaze_task == "data_augment":
                type_feature = "heatmap1"
            else:
                type_feature = "heatmap2"
            gaze_attribute_labels_dict = load_gaze_attribute_labels(source, self.split_type, type_feature)
            self.gaze_features = gaze_attribute_labels_dict


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
        
        subclass_label = 0

        if self.source == "cxr_a":
            img_pth = os.path.join(self.data_dir, f"cxr_ibm/dicom_images/{img_id}")
        elif self.source == "cxr_p":
            img_pth = os.path.join(self.data_dir, f"pneumothorax/dicom_images/{img_id}")

            with open('/media/pneumothorax/cxr_tube_dict.pkl', 'rb') as f:          
                cxr_tube_dict = pickle.load(f)

            image_name = img_id.split("/")[-1].split(".dcm")[0]

            if self.subclass:
                subclass_label = 1 if cxr_tube_dict[image_name] is True else 0

        else:
            img_pth = img_id

        if self.ood:
            img = Image.open(img_pth)
        else:
            ds = pydicom.dcmread(img_pth)
            img = ds.pixel_array
            img = Image.fromarray(np.uint8(img))

   
        img = self.transform(img)
        
        if img.shape[0] == 1:
            img = torch.cat([img, img, img])

        if img.shape[0] >= 4:
            img = img[:3] 

        if self.split_type in ['train', 'val']:
            
            gaze_attribute = self.gaze_features[img_id][0]

            if self.gaze_task == "data_augment":

                gaze_map = gaze_attribute.reshape(4,4)
                gaze_binary_mask = torch.zeros(224, 224)
                divisor = int(224/4)

                for i in range(4):
                    for j in range(4):
                        if gaze_map[i,j] != 0:
                            gaze_binary_mask[(divisor)*(i):(divisor)*(i+1),(divisor)*(j):(divisor)*(j+1)] = torch.ones(divisor,divisor)

                gaze_binary_mask = gaze_binary_mask.unsqueeze(0)
                gaze_binary_mask= torch.cat([gaze_binary_mask, gaze_binary_mask, gaze_binary_mask])
                img = img*gaze_binary_mask
            
            return img, label, gaze_attribute

        else:
            return img, label, subclass_label



def fetch_dataloaders(
    source,
    data_dir,
    val_scale,
    seed,
    batch_size,
    num_workers,
    gaze_task = None, #either none, data augment, cam reg, cam reg convex
    ood_set = None,
    ood_shift = None,
    subclass = False
):

    
    transforms = get_data_transforms("cxr", normalization_type="train_images")
    dataloaders = {}

    for split in ["train", "val", "test"]:

        if ood_set is not None:
            if split == "test":
                source = f"{ood_set}/{source}/{ood_shift}"

        dataset = RoboGazeDataset(
            source=source,
            data_dir=data_dir,
            split_type=split,
            gaze_task = gaze_task,
            transform=transforms[split],
            val_scale=val_scale,
            seed=seed,
            subclass=subclass
        )

        dataloaders[split] = (
            DataLoader(
                dataset=dataset,
                shuffle=split == "train",
                batch_size=batch_size,
                num_workers=num_workers,
            )
        )

    return dataloaders


if __name__ == "__main__":
    
    #dls = fetch_dataloaders("cxr_p","/media",0.2,0,32,4, ood_set='chexpert', ood_shift='hospital')
    dls = fetch_dataloaders("cxr_p","/media",0.2,0,32,4, gaze_task="data_augment")

    dataiter = iter(dls['train'])

    for i in range(1):
        images, labels, subclass_label = dataiter.next()

        #print(subclass_label)
    # for (img,label) in dls[0]:
    #     pdb.set_trace()






