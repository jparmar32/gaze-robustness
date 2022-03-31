import numpy as np 
import pickle
import torch 
from torch.utils.data import Dataset
from skimage.filters import gaussian

### must return torch tensors

class SyntheticDataset(Dataset):
    def __init__(self, split='train', blur=0, train_length=500, val_length=128, test_length=128):

        self.split = split

        if split == 'train':
            self.length = train_length
        else:
            self.length = val_length

        self.blur = blur 

        ## load in labels
        with open(f"./actdiff/data/synth_hard/{split}_labels.pkl", "rb") as f:
            self.img_labels = pickle.load(f)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        img_path,label = self.img_labels[index]

        img_path = "./actdiff" + img_path[1:]
        seg_path = img_path.replace("img", "seg")

        image = np.load(img_path)
        seg = np.load(seg_path)

        # If there is a segmentation, blur it a bit.
        if (self.blur > 0) and (np.max(seg) != 0):
            seg = gaussian(seg, self.blur, output=seg)
            seg /= seg.max()

        seg = (seg > 0) * 1.0

        seg = torch.from_numpy(seg)
        image = torch.from_numpy(image)

        return image, label, seg