import os, sys
import numpy as np
import pickle
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from PIL import Image
import pydicom
import torchvision

from utils import load_file_markers, get_data_transforms, load_gaze_attribute_labels

import gan.generator as gan_generator
import acgan.generator as acgan_generator


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
        subclass=False,
        gan = False
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

        self.gan = gan
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
            gaze_attribute_labels_dict = load_gaze_attribute_labels(source, self.split_type, type_feature, seed)
            self.gaze_features = gaze_attribute_labels_dict

            self.average_heatmap = np.mean(list(self.gaze_features.values()), axis=0).squeeze()

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
        
        if not self.gan:
            if img.shape[0] == 1:
                img = torch.cat([img, img, img])

        if img.shape[0] >= 4:
            img = img[:3] 

        if self.gan:
            return img, label


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

            if self.gaze_task == "cam_reg_convex":
                gaze_attribute_dict = {"gaze_attribute": gaze_attribute, "average_hm":self.average_heatmap}

                return img, label, gaze_attribute_dict
            
            return img, label, gaze_attribute

        else:
            return img, label, subclass_label

class GanDataset(Dataset):
    def __init__(
        self,
        images,
        labels,
        gaze_attr):

        self.images = images
        self.labels = labels
        self.gaze_attr = gaze_attr

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx], self.gaze_attr[idx]


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
    subclass = False,
    gan_positive = None,
    gan_negative = None,
    gan_type = None
):

    
    transforms = get_data_transforms("cxr", normalization_type="train_images")
    dataloaders = {}

    for split in ["train", "val", "test"]:

        if ood_set is not None:
            if split == "test":
                source = f"{ood_set}/{source}/{ood_shift}"

        if split == 'train':
            if gan_positive is not None:
                original_dataset = RoboGazeDataset(
                source=source,
                data_dir=data_dir,
                split_type=split,
                gaze_task = gaze_task,
                transform=transforms[split],
                val_scale=val_scale,
                seed=seed,
                subclass=subclass)


                ## get positive and negative class amounts
                original_dataloader = DataLoader(
                    dataset=original_dataset,
                    shuffle=False,
                    batch_size=1,
                    num_workers=num_workers,
                )

                class_amounts = [0,0]
                for img, label, _ in original_dataloader:
                    class_amounts[label.item()] += 1


                if gan_type == "gan":
                    pos_generator = gan_generator.Generator_Advanced_224().cuda()
                    neg_generator = gan_generator.Generator_Advanced_224().cuda()
                    noise_size = 100

                    pos_generator.load_state_dict(torch.load(gan_positive + '/generator_best_ckpt.pt'))
                    neg_generator.load_state_dict(torch.load(gan_negative + '/generator_best_ckpt.pt'))


                    neg_noise = torch.randn(class_amounts[0], noise_size, 1, 1).cuda()
                    pos_noise = torch.randn(class_amounts[1], noise_size, 1, 1).cuda()

                    # Feed noise into the generator to create new images

                    neg_images = []
                    for i in range(class_amounts[0]):
                        neg_images.append(neg_generator(neg_noise[i].unsqueeze(dim=0)).detach())
                    neg_images = torch.cat(neg_images)

                    pos_images = []
                    for i in range(class_amounts[1]):
                        pos_images.append(pos_generator(pos_noise[i].unsqueeze(dim=0)).detach())
                    pos_images = torch.cat(pos_images)

                    #neg_images = neg_generator(neg_noise).detach()
                    #pos_images = pos_generator(pos_noise).detach()
                 
                    neg_labels = torch.zeros(neg_images.shape[0])
                    pos_labels = torch.ones(pos_images.shape[0])

                    neg_gaze_attr = torch.zeros(neg_images.shape[0])
                    pos_gaze_attr = torch.zeros(neg_images.shape[0])

                    positive_fake_data = GanDataset(images=pos_images, labels=pos_labels, gaze_attr=pos_gaze_attr)
                    negative_fake_data = GanDataset(images=neg_images, labels=neg_labels, gaze_attr=neg_gaze_attr)


                elif gan_type == "acgan":
                    pos_generator = acgan_generator.Generator_Advanced_224().cuda()
                    neg_generator = acgan_generator.Generator_Advanced_224().cuda()
                    noise_size = 110

                    pos_generator.load_state_dict(torch.load(gan_positive + '/generator_best_ckpt.pt'))
                    neg_generator.load_state_dict(torch.load(gan_negative + '/generator_best_ckpt.pt'))

                    neg_noise = torch.randn(class_amounts[0], noise_size, 1, 1).cuda()
                    pos_noise = torch.randn(class_amounts[1], noise_size, 1, 1).cuda()

                    # Feed noise into the generator to create new images
                    neg_images = []
                    for i in range(class_amounts[0]):
                        neg_images.append(neg_generator(neg_noise[i].unsqueeze(dim=0)).detach())
                    neg_images = torch.cat(neg_images)

                    pos_images = []
                    for i in range(class_amounts[1]):
                        pos_images.append(pos_generator(pos_noise[i].unsqueeze(dim=0)).detach())
                    pos_images = torch.cat(pos_images)
                    
                    #neg_images = neg_generator(neg_noise).detach()
                    #pos_images = pos_generator(pos_noise).detach()

                    neg_labels = torch.zeros(neg_images.shape[0])
                    pos_labels = torch.ones(pos_images.shape[0])

                    neg_gaze_attr = torch.zeros(neg_images.shape[0])
                    pos_gaze_attr = torch.zeros(neg_images.shape[0])

                    positive_fake_data = GanDataset(images=pos_images, labels=pos_labels, gaze_attr=pos_gaze_attr)
                    negative_fake_data = GanDataset(images=neg_images, labels=neg_labels, gaze_attr=neg_gaze_attr)

                elif gan_type == "wgan":
                    pass #TODO: varun to implement
                elif gan_type == "cgan":
                    pass #TODO: varun to implement

                dataset = torch.utils.data.ConcatDataset([original_dataset, positive_fake_data, negative_fake_data])

            else:
                dataset = RoboGazeDataset(
                source=source,
                data_dir=data_dir,
                split_type=split,
                gaze_task = gaze_task,
                transform=transforms[split],
                val_scale=val_scale,
                seed=seed,
                subclass=subclass)
        else:
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

def fetch_entire_dataloader(source,
    data_dir,
    val_scale,
    seed,
    batch_size,
    num_workers,
    gaze_task = None, #either none, data augment, cam reg, cam reg convex
    ood_set = None,
    ood_shift = None,
    subclass = False,
    gan = True,
    label_class = None):

    transforms = get_data_transforms("cxr", normalization_type="train_images", gan = gan)

    datasets = []
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
            subclass=subclass,
            gan=gan
        )

        datasets.append(dataset)


    concat_datasets = torch.utils.data.ConcatDataset(datasets)

    if label_class is not None:
        full_dataloader = DataLoader(
                    dataset=concat_datasets,
                    shuffle=False,
                    batch_size=1,
                    num_workers=num_workers,
                )
        class_indices = []
        for idx, (img, label) in enumerate(full_dataloader):
            if label.item() == label_class:
                class_indices.append(idx)
    
        class_dataset = torch.utils.data.Subset(concat_datasets, class_indices)

        return DataLoader(
                    dataset=class_dataset,
                    shuffle=True,
                    batch_size=batch_size,
                    num_workers=num_workers,
                )
    else:
        return DataLoader(
                    dataset=concat_datasets,
                    shuffle=True,
                    batch_size=batch_size,
                    num_workers=num_workers,
                )

if __name__ == "__main__":

    dls = fetch_dataloaders("cxr_p","/media",0.2,2,32,4, gaze_task=None, gan_positive = "/home/jsparmar/gaze-robustness/gan/positive_class", gan_negative = "/home/jsparmar/gaze-robustness/gan/negative_class", gan_type = 'gan')
    print(len(dls['train'].dataset))

    #dls = fetch_dataloaders("cxr_p","/media",0.2,0,32,4, ood_set='chexpert', ood_shift='hospital')
    #dls = fetch_dataloaders("cxr_p","/media",0.2,2,32,4, gaze_task="cam_reg_convex")
    #dl = fetch_entire_dataloader("cxr_p","/media",0.2,2,32,4, gaze_task=None, gan = True, label_class=1)
    #print(len(dl.dataset))
    #dataiter = iter(dl)
    #for i in range(1):
        #images, labels= dataiter.next()
        #print(images.shape)
        #grid_img = torchvision.utils.make_grid(images, nrow=8)
        #torchvision.utils.save_image(grid_img, 'downsampled_cxr.png')

    #dataiter = iter(dls['val'])

    #for i in range(1):
        #images, labels, gaze = dataiter.next()
        #print(len(dls['val'].dataset))
        #print(images[0, 0:10, 0:10])
        #print(images[1, 0:10, 0:10])
    

        #print(subclass_label)
    # for (img,label) in dls[0]:
    #     pdb.set_trace()'''


## dataloader for gan, downsample and 1 





