import pickle 
import csv
import numpy as np

'''with open('/home/jsparmar/gaze-robustness/filemarkers/cxr_p/test_list.pkl', 'rb') as f:
    data = pickle.load(f)

with open('/home/jsparmar/gaze-robustness/filemarkers/cxr_p/trainval_list.pkl', 'rb') as f:
    t_data = pickle.load(f)


with open('/media/pneumothorax/cxr_tube_dict.pkl', 'rb') as f:          
    cxr_tube_dict = pickle.load(f)

idx = 0 
test_list_tube = []
for img_id in data: 

    image_name = img_id[0].split("/")[-1].split(".dcm")[0]
    if image_name in cxr_tube_dict:
        test_list_tube.append(img_id)

with open('/home/jsparmar/gaze-robustness/filemarkers/cxr_p/test_list_tube.pkl', 'wb') as f:
    pickle.dump(test_list_tube, f)


with open('/home/jsparmar/gaze-robustness/gaze_data/cxr_p_gaze_data.pkl', 'rb') as f:
    data = pickle.load(f)

print(len(list(data.keys())))


segmask_dir = '/media/pneumothorax/train-rle.csv'
with open(segmask_dir) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')

    for ii,row in enumerate(csv_reader):
        print(row)


import torch

te = torch.ones(3,16,16)
gaze_binary_mask = torch.zeros(16, 16)

t = np.array([[0,1,2,0],[0,0,0,0], [0,0,0,0], [0,0,0,1]])

for i in range(4):
    for j in range(4):
        if t[i,j] != 0:
            gaze_binary_mask[(4)*(i):(4)*(i+1),(4)*(j):(4)*(j+1)] = torch.ones(4,4)

gaze_binary_mask = gaze_binary_mask.unsqueeze(0)
g = torch.cat([gaze_binary_mask, gaze_binary_mask, gaze_binary_mask])

print(te*g[1,:,:])

#print(gaze_binary_mask)'''

if cam_weight:
    assert (attn is None)
    n = len(targets)
    cum_losses = logits.new_zeros(n)
    cam_norms, eye_norms = [logits.new_zeros(n) for i in range(2)]

    for i in range(n):
        image = inputs_dict['images'][i,...] ##this i just getting thte specific image
        cam = get_CAM_from_img(image, model, targets[i].cpu()) ##getting CAM from image as functions bleow
        eye_hm = targets_dict['heatmap'][i,:,:].squeeze().float() #tthis is jut the heattmap of the image, make it 2d
        
        if eye_hm.shape != cam.shape:
            pool_dim = int(eye_hm.shape[0] / cam.shape[0])
            eye_hm = nn.functional.avg_pool2d(eye_hm.unsqueeze(0).unsqueeze(0), pool_dim).squeeze()

        eye_hm_norm = eye_hm / eye_hm.sum()
        cam_normalized = cam / cam.sum()


        if not (torch.isnan(cam_normalized).any() or torch.isnan(eye_hm_norm).any()):
            cum_losses[i] += cam_weight * torch.nn.functional.mse_loss(eye_hm_norm,cam_normalized,reduction='sum')
    a_loss = cum_losses.sum()  





