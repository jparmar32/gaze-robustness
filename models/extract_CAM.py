import cv2
import numpy as np
import torch
import torch.nn as nn

def get_final_conv_layer(model):
    if hasattr(model, 'layer4'):
        return model.layer4
    return model.layer3

def get_CAM_from_img(img,model,class_ndx,channels=1):
    features_blobs = []

    #defining hook for feature extractor
    def hook_feature(module, input, output):
        features_blobs.append(output)

    h = get_final_conv_layer(model).register_forward_hook(hook_feature)
    params = list(model.parameters())
    weight_softmax = params[-2]
    img = img.unsqueeze(0)
    logit = model(img)
    h.remove()
    CAMs = returnCAMTorch(features_blobs[0], weight_softmax, [int(class_ndx.item())]) # last arg is for which class 
    heatmap = CAMs[0]
    if channels == 3:
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # If you want image to show up in heatmap, uncomment below
    result = heatmap #* 0.7 + img * 0.5

    return result


def returnCAMTorch(feature_conv, weight_softmax, class_idx, size_upsample=None):
    bz, nc, h, w = feature_conv.shape   
    output_cam = []
    for idx in class_idx:
        cam = torch.mm(weight_softmax[class_idx], feature_conv.view((nc,h*w)))   
        cam = cam.view(h, w)
        cam = cam - torch.min(cam)
        cam_img = cam / torch.max(cam)
        if size_upsample:
            output_cam.append(nn.functional.interpolate(cam_img, size=size_upsample))
        else:
            output_cam.append(cam_img)

    return output_cam