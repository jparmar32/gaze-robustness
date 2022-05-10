import pickle
import os 
from glob import glob
import pandas as pd 
from sklearn.model_selection import train_test_split

small_test = './filemarkers/cxr_p/test_list.pkl'
small_trainval = './filemarkers/cxr_p/trainval_list_gold.pkl'

root_dir = '/media/4tb_hdd/CXR_observational/pneumothorax'
all_filepaths = sorted(glob(os.path.join(root_dir, "dicom_images/*/*/*.dcm")))

with open(os.path.join(root_dir, 'rle_dict.pkl'), "rb") as segs:
    image_segmentations = pickle.load(segs)

with open(small_test, "rb") as test_fp:
    small_test_filepaths = pickle.load(test_fp)

with open(small_trainval, "rb") as trainval_fp:
    small_trainval_filepaths = pickle.load(trainval_fp)

small_test_positive_count = 0
for val in small_test_filepaths:
    small_test_positive_count += val[1]

small_trainval_images = [img_id for img_id,label in small_trainval_filepaths]
small_test_class_ratio = small_test_positive_count/len(small_test_filepaths)

file_paths = []
labels = []

all_train_filepaths = []
all_test_filepaths = []

for filepath in all_filepaths:
    img_id= filepath.split('/media/4tb_hdd/CXR_observational/pneumothorax/dicom_images/')[1]
    img_name = img_id.split("/")[-1].split(".dcm")[0]

    if img_name in image_segmentations:
        if img_id not in small_trainval_images:
            rle = image_segmentations[img_name]
            if rle == " -1":
                file_paths.append(img_id)
                labels.append(0)
            else:
                file_paths.append(img_id)
                labels.append(1)

for img_id,label in small_trainval_filepaths:
    all_train_filepaths.append((img_id,label))

train_img_ids, test_img_ids, train_labels, test_labels = train_test_split(file_paths, labels, stratify=labels, test_size=int(0.2*len(image_segmentations)))



for img_id,label in zip(train_img_ids, train_labels):
    all_train_filepaths.append((img_id,label))

for img_id, label in zip(test_img_ids, test_labels):
    all_test_filepaths.append((img_id, label))

train_count = 0
for val in all_train_filepaths:
    train_count += val[1]

test_count = 0
for val in all_test_filepaths:
    test_count += val[1]

print(f"{len(all_train_filepaths)} in the trainval set and {len(all_test_filepaths)} in the test set")
print(f"{train_count/len(all_train_filepaths)} proportion of positives in the trainval set, and {test_count/len(all_test_filepaths)} proportions of positive in the test set")


with open('./filemarkers/cxr_p/new_trainval_all.pkl',"wb") as pkl_train:
        pickle.dump(all_train_filepaths,pkl_train)

with open('./filemarkers/cxr_p/new_test_all.pkl',"wb") as pkl_test:
        pickle.dump(all_test_filepaths, pkl_test)
'''

#new test filemarkers should be 20% of all filemarkers and have same class balance as previous test, no overlap with previous train



with open('./filemarkers/cxr_p/trainval_list_all.pkl', "rb") as fp:
        file_markers = pickle.load(fp)

print(len(file_markers))



with open('./filemarkers/cxr_p/test_list_all.pkl', "rb") as fp:
        file_markers_test = pickle.load(fp)

print(len(file_markers_test))

count = 0
for test in file_markers_test:
    if test in file_markers:
        count += 1

print(count)


root_dir = '/media/pneumothorax'
filepaths = sorted(glob(os.path.join(root_dir, "dicom_images/*/*/*.dcm")))
print(len(filepaths))'''