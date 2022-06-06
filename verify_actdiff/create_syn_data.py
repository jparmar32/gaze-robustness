import os
import json
import pickle 

import numpy as np 
import pandas as pd
from skimage.transform import resize

"""
goal: randomly generate a dataset of images with the following properties:
    training set:
        class 0: two cross symbols as the main predictor (add noise and random cropping of 
                 crosses to make less predictive). single solid rectangle in lower left 
                 corner as a distractor.
        class 1: one cross symbol as main predictor. single solid rectangle in lower 
                 right corner as distractor
    test set:
        Same as training but the relationship between the distractor is flipped   
         
"""


def add_cross(img, class_label, cross_dims=5, add_noise=False):
    """ Add the predictive cross(es) to the image arbirarily (may accidentally intersect the distractor, but
            that's ok). Class label defines how many times the cross is placed to ensure no overlap
    
        Args: 
            class_label: 0 or 1
            add_noise: True/False - if True, add blurring, randomly truncate cross limbs proportional to how
                        long they are
    """

    x_centres = np.random.choice(range(cross_dims, 28-cross_dims), size=[class_label + 1], replace=False)
    y_centres = np.random.choice(range(cross_dims, 28-cross_dims), size=[class_label + 1], replace=False)
    for i in range(class_label+1):

        centre_x = x_centres[i]
        centre_y = y_centres[i]
        img[centre_x-cross_dims:centre_x+cross_dims+1, centre_y] = 1
        img[centre_x, centre_y-cross_dims:centre_y+cross_dims+1] = 1

        
def add_distractor(img, class_label, tag_dims=[2,3], add_noise=False):
    """
    Add a distractor for prediction based on the class label - zero puts it on the left, one puts it on the
            right
        
        Args:
            tag_dims: list of the sizes for the rectangular distractor tag
            add_noise: True/False - if True, blurs the distractor and also shifts its location by some
                        random amount fixed within 2-5 pixels from the edges of the image
    """
    tag_buffer = 5
    size_x, size_y = img.shape
    if class_label == 0:
        # buffer of 5 from the border
        img[size_y-tag_buffer-tag_dims[0]:size_y-tag_buffer, tag_buffer:tag_buffer+tag_dims[1]] = 1
        tag_centre_x = (tag_buffer+tag_dims[1]) // 2
        tag_centre_y = 1
    else:
        img[size_y-tag_buffer-tag_dims[0]:size_y-tag_buffer, size_x-tag_buffer-tag_dims[1]:size_x-tag_buffer] = 1

        
def make_synthetic_dataset(length, mode, split='train', root="./data/synth_hard", img_size=28, seed=0):
    """ 
    Dataset Builder
    Parameters: 
        length: how many images should be generated
        mode: string of either train or test
    
    """
    os.makedirs(root, exist_ok=True)
    
    labels = []
    #np.random.seed(seed)
    
    for n in range(length):
        print("making ", n, " of ", length, " files")
        if n < length//2:
            label = 0
        else:
            label = 1 # even data split

        img_base = np.zeros([img_size,img_size])
        add_cross(img_base, label, cross_dims=np.random.randint(3,5))
        img_seg = np.zeros([img_size,img_size])
        img_seg[:,:] = img_base[:,:]
        
        if mode == 'distractor':
            add_distractor(img_base, label)

        ### switch the side of the confounder
        if split in ['val', 'test']:
            img_base = np.flip(img_base, axis=1)

        ### resize to larger resnet dimensions
        RESNET_DIM = 224
        img_base = resize(img_base, (RESNET_DIM, RESNET_DIM))
        img_base = np.where(img_base>0, np.ones((RESNET_DIM, RESNET_DIM)), np.zeros((RESNET_DIM, RESNET_DIM)))
        img_seg = resize(img_seg, (RESNET_DIM, RESNET_DIM))
        img_seg = np.where(img_seg>0, np.ones((RESNET_DIM, RESNET_DIM)), np.zeros((RESNET_DIM, RESNET_DIM)))
        
        # save image and segmentation map to file
        np.save("{}/{}_img_{}.npy".format(root, split, n), img_base)
        np.save("{}/{}_seg_{}.npy".format(root, split, n), img_seg)

        labels.append(("{}/{}_img_{}.npy".format(root, split, n),label))

    pickle.dump(labels,open(f'{root}/{split}_labels.pkl', 'wb'))

make_synthetic_dataset(length=500, mode='distractor', split='train')