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
from skimage.transform import resize
import math
from skimage.filters import gaussian
import torch.nn as nn

from utils import load_file_markers, get_data_transforms, load_gaze_attribute_labels, rle2mask, create_masked_image, create_masked_image_advanced_augmentations, get_masked_normalization

import gan_training.gan.generator as gan_generator
import gan_training.acgan.generator as acgan_generator

import matplotlib.pyplot as plt


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
        gan = False,
        args = None
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

        self.IMG_SIZE = 224
        self.gan = gan
        self.data_dir = data_dir
        self.split_type = split_type
        self.transform = transform
        self.subclass = subclass
        self.args = args
        self.file_markers = load_file_markers(
            source,
            split_type,
            self.ood,
            val_scale,
            seed,
            subclass=self.subclass
        )

        self.seed = seed

        if self.split_type in ['train', 'val']:

            type_feature = None
            if self.gaze_task == "data_augment":
                type_feature = "heatmap1"
            elif self.gaze_task == "actdiff_gaze":
                type_feature = f"heatmap_{self.args.actdiff_gazemap_size}"
            else:
                type_feature = "heatmap2"

            gaze_attribute_labels_dict = load_gaze_attribute_labels(source, self.split_type, type_feature, seed)
            self.gaze_features = gaze_attribute_labels_dict

            self.average_heatmap = np.mean(list(self.gaze_features.values()), axis=0).squeeze()

            if self.args.machine == "meteor":
                seg_dict_path = "/media/pneumothorax/rle_dict.pkl"
            elif self.args.machine == "gemini":
                seg_dict_pth = "/media/4tb_hdd/CXR_observational/pneumothorax/rle_dict.pkl"
            else:
                raise ValueError("Machine type not known")

            with open(seg_dict_pth, "rb") as pkl_f:
                self.rle_dict = pickle.load(pkl_f)

        #naive method currently
        if self.gaze_task[:7] == "actdiff":
            #self.masked_normalization_values = get_masked_normalization(self.args.actdiff_augmentation_type)
            self.masked_normalization_values = get_masked_normalization('standard')
        

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


            if self.args.machine == "meteor":
                seg_dict_path = '/media/pneumothorax/cxr_tube_dict.pkl'
            elif self.args.machine == "gemini":
                tube_path = '/media/nvme_data/jupinder_cxr_robustness_results/cxr_tube_dict.pkl'
            else:
                raise ValueError("Machine type not known")

            with open(tube_path, 'rb') as f:          
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

            ### neet to return regular image, label, and masked image
            if self.gaze_task == "actdiff":

                rle = self.rle_dict[img_id.split("/")[-1].split(".dcm")[0]]
                y_true = rle != " -1"

                ### use the segmentation mask
                if y_true:
                    # extract segmask
                    segmask_org = rle2mask(rle, 1024, 1024).T

                    if self.args.actdiff_segmask_size != self.IMG_SIZE:

                        segmask_int = nn.functional.max_pool2d(torch.tensor(segmask_org).unsqueeze(0), int(1024/self.args.actdiff_segmask_size)).squeeze().numpy()
                        segmask_int = (segmask_int > 0) * 1
                        segmask = resize(segmask_int, (self.IMG_SIZE, self.IMG_SIZE))
                        segmask = (segmask > 0) * 1

                    else:
                        segmask = resize(segmask_org, (self.IMG_SIZE, self.IMG_SIZE))
                        segmask = (segmask > 0) * 1

                    segmask = torch.from_numpy(segmask)
                    segmask = torch.where(segmask > 0, torch.ones(segmask.shape), torch.zeros(segmask.shape)).long()
                    img_masked = create_masked_image(img, segmask)
                    return img, label, img_masked

                ## we don't have a segmentation mask for the image, as in actdiff set segmentation mask to all 1
                else:
                    segmask = torch.ones((self.IMG_SIZE,self.IMG_SIZE)).long()
                    img_masked = create_masked_image(img, segmask)
                    return img, label, img_masked

            ### need to return regular image, label, and masked image from the gaze heatmap
            if self.gaze_task == "actdiff_gaze":
                ### obtain 7 x 7 gaze heatmap
                gaze_map = gaze_attribute.reshape(self.args.actdiff_gazemap_size,self.args.actdiff_gazemap_size)
                gaze_map = (gaze_map > self.args.actdiff_gaze_threshold) * 1.0  

                ### resize up to 224 x 224
                gaze_map = resize(gaze_map, (self.IMG_SIZE,self.IMG_SIZE)) ### change to change_map_size
                gaze_map = torch.from_numpy(gaze_map)
                gaze_map = torch.where(gaze_map > 0, torch.ones(gaze_map.shape), torch.zeros(gaze_map.shape)).long()
                
                ### get masked image, 
                img_masked = create_masked_image(img, gaze_map)

                return img, label, img_masked

             ### neet to return regular image, label, and masked image
            if self.gaze_task == "actdiff_lungmask":


                if self.args.actdiff_segmentation_classes == 'positive':

                    ## we have lungmasks for these right now
                    if label == 1:
                        img_name = img_id.replace("/","_").split(".dcm")[0]
                        lung_mask = np.load(f"./lung_segmentations/annotations/{img_name}_lungmask.npy")
                        lung_mask = np.where(lung_mask > 0, np.ones(lung_mask.shape), np.zeros(lung_mask.shape))

                        if self.args.actdiff_lungmask_size != self.IMG_SIZE:
                            lung_mask_int = nn.functional.max_pool2d(torch.tensor(lung_mask).unsqueeze(0), int(224/self.args.actdiff_lungmask_size)).squeeze().numpy()
                            lung_mask_int = (lung_mask_int > 0) * 1
                            lung_mask = resize(lung_mask_int, (self.IMG_SIZE, self.IMG_SIZE))
                            lung_mask = (lung_mask > 0) * 1

                        lung_mask = torch.from_numpy(lung_mask)
                        lung_mask = torch.where(lung_mask > 0, torch.ones(lung_mask.shape), torch.zeros(lung_mask.shape)).long()

                        img_masked = create_masked_image_advanced_augmentations(img, lung_mask, augmentation=self.args.actdiff_augmentation_type, masked_normalization_vals=self.masked_normalization_values, seed=self.seed)
                        return img, label, img_masked

                    else:
                        segmask = torch.ones((self.IMG_SIZE,self.IMG_SIZE)).long()
                        img_masked = create_masked_image_advanced_augmentations(img, segmask, augmentation='normal', masked_normalization_vals=self.masked_normalization_values, seed=self.seed)
                        return img, label, img_masked

                elif self.args.actdiff_segmentation_classes == 'all':
                    
                    img_name = img_id.replace("/","_").split(".dcm")[0]
                    lung_mask = np.load(f"./lung_segmentations/annotations/{img_name}_lungmask.npy")
                    lung_mask = np.where(lung_mask > 0, np.ones(lung_mask.shape), np.zeros(lung_mask.shape))

                    if self.args.actdiff_lungmask_size != self.IMG_SIZE:
                        lung_mask_int = nn.functional.max_pool2d(torch.tensor(lung_mask).unsqueeze(0), int(224/self.args.actdiff_lungmask_size)).squeeze().numpy()
                        lung_mask_int = (lung_mask_int > 0) * 1
                        lung_mask = resize(lung_mask_int, (self.IMG_SIZE, self.IMG_SIZE))
                        lung_mask = (lung_mask > 0) * 1

                    lung_mask = torch.from_numpy(lung_mask)
                    lung_mask = torch.where(lung_mask > 0, torch.ones(lung_mask.shape), torch.zeros(lung_mask.shape)).long()

                    img_masked = create_masked_image_advanced_augmentations(img, lung_mask, augmentation=self.args.actdiff_augmentation_type, masked_normalization_vals=self.masked_normalization_values, seed=self.seed)
                    return img, label, img_masked


            if self.gaze_task == "segmentation_reg":
            
                rle = self.rle_dict[img_id.split("/")[-1].split(".dcm")[0]]
             
                y_true = rle != " -1"

                if y_true:

                    # extract segmask
                    segmask_org = rle2mask(rle, 1024, 1024).T
                    segmask = resize(segmask_org, (7,7))
                    segmask = torch.FloatTensor(segmask)
                    return img, label, segmask

                else:
                    segmask = -1*torch.ones((7,7))
                    return img, label, segmask
            
            if self.gaze_task is None:
                return img, label, 0

            return img, label, gaze_attribute

        else:

            if self.gaze_task == 'cam_error_analysis':
                return img, label, img_id
            
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
        img = self.images[idx]
        if img.shape[0] == 1:
                img = torch.cat([img, img, img])

        return img, int(self.labels[idx]), int(self.gaze_attr[idx])

class SyntheticDataset(Dataset):
    def __init__(self, split='train', blur=0, train_length=500, val_length=128, test_length=128):

        self.split = split

        if split == 'train':
            self.length = train_length
        elif split == "val":
            self.length = val_length
        else:
            self.length = test_length

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

        seg = torch.from_numpy(seg).long()
        image = torch.from_numpy(image)
        image = image.unsqueeze(0)
        image = torch.cat([image, image, image])
        image = image.float()
        img_masked = create_masked_image(image, seg)
        return image, int(label), img_masked


def fetch_dataloaders(
    source,
    data_dir,
    val_scale,
    seed,
    batch_size,
    num_workers,
    gaze_task = None, #either none, data augment, cam reg, cam reg convex actdiff
    ood_set = None,
    ood_shift = None,
    subclass = False,
    gan_positive = None,
    gan_negative = None,
    gan_type = None,
    args = None
):

    
    transforms = get_data_transforms("cxr", normalization_type="train_images")
    dataloaders = {}

    for split in ["train", "val", "test"]:

        if ood_set is not None:
            if split == "test":

                source = f"{args.machine}/{ood_set}/{source}/{ood_shift}"
                #source = f"{ood_set}/{source}/{ood_shift}"

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
                subclass=subclass,
                args=args)


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
                        neg_images.append(neg_generator(neg_noise[i].unsqueeze(dim=0)).detach().cpu())
                    neg_images = torch.cat(neg_images)

                    pos_images = []
                    for i in range(class_amounts[1]):
                        pos_images.append(pos_generator(pos_noise[i].unsqueeze(dim=0)).detach().cpu())
                    pos_images = torch.cat(pos_images)

                    #neg_images = neg_generator(neg_noise).detach()
                    #pos_images = pos_generator(pos_noise).detach()
                 

                    neg_labels = torch.zeros(neg_images.shape[0]).cpu().numpy()
                    pos_labels = torch.ones(pos_images.shape[0]).cpu().numpy()

                    neg_gaze_attr = torch.zeros(neg_images.shape[0]).cpu().numpy()
                    pos_gaze_attr = torch.zeros(neg_images.shape[0]).cpu().numpy()

                    positive_fake_data = GanDataset(images=pos_images, labels=pos_labels, gaze_attr=pos_gaze_attr)
                    negative_fake_data = GanDataset(images=neg_images, labels=neg_labels, gaze_attr=neg_gaze_attr)


                elif gan_type == "acgan":
                    noise_size = 110
                    pos_generator = acgan_generator.Generator_Advanced_224(1, noise_size).cuda()
                    neg_generator = acgan_generator.Generator_Advanced_224(1, noise_size).cuda()
                    

                    pos_generator.load_state_dict(torch.load(gan_positive + '/generator_best_ckpt.pt'))
                    neg_generator.load_state_dict(torch.load(gan_negative + '/generator_best_ckpt.pt'))

                    neg_noise = torch.randn(class_amounts[0], noise_size, 1, 1).cuda()
                    pos_noise = torch.randn(class_amounts[1], noise_size, 1, 1).cuda()

                    # Feed noise into the generator to create new images
                    neg_images = []
                    for i in range(class_amounts[0]):
                        neg_images.append(neg_generator(neg_noise[i].unsqueeze(dim=0)).detach().cpu())
                    neg_images = torch.cat(neg_images)

                    pos_images = []
                    for i in range(class_amounts[1]):
                        pos_images.append(pos_generator(pos_noise[i].unsqueeze(dim=0)).detach().cpu())
                    pos_images = torch.cat(pos_images)

                    #neg_images = neg_generator(neg_noise).detach()
                    #pos_images = pos_generator(pos_noise).detach()

                    neg_labels = torch.zeros(neg_images.shape[0]).cpu().numpy()
                    pos_labels = torch.ones(pos_images.shape[0]).cpu().numpy()

                    neg_gaze_attr = torch.zeros(neg_images.shape[0]).cpu().numpy()
                    pos_gaze_attr = torch.zeros(neg_images.shape[0]).cpu().numpy()

                    positive_fake_data = GanDataset(images=pos_images, labels=pos_labels, gaze_attr=pos_gaze_attr)
                    negative_fake_data = GanDataset(images=neg_images, labels=neg_labels, gaze_attr=neg_gaze_attr)

                dataset = torch.utils.data.ConcatDataset([original_dataset, positive_fake_data, negative_fake_data])

            else:
                if gaze_task == "actdiff" and source == "synth":
                    dataset = SyntheticDataset(split=split, blur=0.817838)
                else:
                    dataset = RoboGazeDataset(
                    source=source,
                    data_dir=data_dir,
                    split_type=split,
                    gaze_task = gaze_task,
                    transform=transforms[split],
                    val_scale=val_scale,
                    seed=seed,
                    subclass=subclass,
                    args=args)
        else:
            if gaze_task == "actdiff" and source == "synth":
                    dataset = SyntheticDataset(split=split, blur=0.817838)
            else:
                dataset = RoboGazeDataset(
                    source=source,
                    data_dir=data_dir,
                    split_type=split,
                    gaze_task = gaze_task,
                    transform=transforms[split],
                    val_scale=val_scale,
                    seed=seed,
                    subclass=subclass,
                    args=args
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

    dls = fetch_dataloaders("cxr_p","/media",0.2,2,1,4, gaze_task="actdiff_gaze")
    print(len(dls['train'].dataset))
    print(len(dls['val'].dataset))
    print(len(dls['test'].dataset))

    #dls = fetch_dataloaders("cxr_p","/media",0.2,0,32,4, ood_set='chexpert', ood_shift='hospital')
    #dls = fetch_dataloaders("cxr_p","/media",0.2,2,32,4, gaze_task="cam_reg_convex")
    #dl = fetch_entire_dataloader("cxr_p","/media",0.2,2,32,4, gaze_task=None, gan = True, label_class=1)
    #print(len(dl.dataset))
    #dataiter = iter(dl)

    dataiter = iter(dls['train'])
    for i in range(1):
        images, labels, seg = dataiter.next()
        #print(images.shape)
        #print(seg.shape)
        #print(images.shape)
        #grid_img = torchvision.utils.make_grid(images, nrow=8)
        #torchvision.utils.save_image(grid_img, 'downsampled_cxr.png')

    #dataiter = iter(dls['train'])

    #for i in range(1):
        #images, labels, gaze = dataiter.next()
        #print(len(dls['val'].dataset))
        #print(gaze.shape)
        #print(images[0, 0:10, 0:10])
        #print(images[1, 0:10, 0:10])
    

        #print(subclass_label)
    # for (img,label) in dls[0]:
    #     pdb.set_trace()'''


## dataloader for gan, downsample and 1 





