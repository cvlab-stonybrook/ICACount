'''
Utils Functions
By Yifeng Huang(yifehuang@cs.stonybrook.edu)
Based on Viresh and Minh's code
Last Modified 2021.2.20
'''
import os
import cv2
import math
import random
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torchvision import transforms

MIN_HW = 384
MAX_HW = 1584
IM_NORM_MEAN = [0.485, 0.456, 0.406]
IM_NORM_STD = [0.229, 0.224, 0.225]

def Unet_extract_features(image_features, boxes, feat_map_keys, exemplar_scales=[0.9, 1.1]):
    '''

    :param image_features: Features extracted by ResNet50
    :param boxes: Exemplar
    :param feat_map_keys: which block in ResNet50
    :param exemplar_scales: Scaling Para
    :return: Correlation Result
    '''
    assert feat_map_keys in ['map2', 'map3', 'map4']

    # Get the Scaling
    Exemplar_num = boxes.shape[2]
    if feat_map_keys == 'map2':
        image_features = image_features['map2']
        Scaling = 4.0
    elif feat_map_keys == 'map3':
        image_features = image_features['map3']
        Scaling = 8.0
    elif feat_map_keys == 'map4':
        image_features = image_features['map4']
        Scaling = 16.0

    #Scale The box
    boxes = boxes[0][0]
    boxes_scaled = boxes / Scaling
    boxes_scaled[:, 0:2] = torch.floor(boxes_scaled[:, 0:2])
    boxes_scaled[:, 2:4] = torch.ceil(boxes_scaled[:, 2:4])
    boxes_scaled[:, 2:4] = boxes_scaled[:, 2:4] + 1  # make the end indices exclusive
    feat_h, feat_w = image_features.shape[-2], image_features.shape[-1]
    # make sure exemplars don't go out of bound
    boxes_scaled[:, 0:2] = torch.clamp_min(boxes_scaled[:, 0:2], 0)
    boxes_scaled[:, 2] = torch.clamp_max(boxes_scaled[:, 2], feat_h)
    boxes_scaled[:, 3] = torch.clamp_max(boxes_scaled[:, 3], feat_w)
    box_hs = boxes_scaled[:, 2] - boxes_scaled[:, 0]
    box_ws = boxes_scaled[:, 3] - boxes_scaled[:, 1]
    max_h = math.ceil(max(box_hs))
    max_w = math.ceil(max(box_ws))

    #Get Exemplar Features
    for j in range(0, Exemplar_num):
        y1, x1 = int(boxes_scaled[j, 0]), int(boxes_scaled[j, 1])
        y2, x2 = int(boxes_scaled[j, 2]), int(boxes_scaled[j, 3])
        # print(y1,y2,x1,x2,max_h,max_w)
        if j == 0:
            examples_features = image_features[:, :, y1:y2, x1:x2]
            if examples_features.shape[2] != max_h or examples_features.shape[3] != max_w:
                # examples_features = pad_to_size(examples_features, max_h, max_w)
                examples_features = F.interpolate(examples_features, size=(max_h, max_w), mode='bilinear')
        else:
            feat = image_features[:, :, y1:y2, x1:x2]
            if feat.shape[2] != max_h or feat.shape[3] != max_w:
                feat = F.interpolate(feat, size=(max_h, max_w), mode='bilinear')
                # feat = pad_to_size(feat, max_h, max_w)
            examples_features = torch.cat((examples_features, feat), dim=0)

    #Correlation
    h, w = examples_features.shape[2], examples_features.shape[3]
    features = F.conv2d(
        F.pad(image_features, ((int(w / 2)), int((w - 1) / 2), int(h / 2), int((h - 1) / 2))),
        examples_features
    )
    combined = features.permute([1, 0, 2, 3])

    #Scaling Correlation
    for scale in exemplar_scales:
        h1 = math.ceil(h * scale)
        w1 = math.ceil(w * scale)
        if h1 < 1:  # use original size if scaled size is too small
            h1 = h
        if w1 < 1:
            w1 = w
        examples_features_scaled = F.interpolate(examples_features, size=(h1, w1), mode='bilinear')
        features_scaled = F.conv2d(
            F.pad(image_features, ((int(w1 / 2)), int((w1 - 1) / 2), int(h1 / 2), int((h1 - 1) / 2))),
            examples_features_scaled)
        features_scaled = features_scaled.permute([1, 0, 2, 3])
        combined = torch.cat((combined, features_scaled), dim=1)
    return combined

def extract_features(feature_model, image, boxes,feat_map_keys=['map3','map4'], exemplar_scales=[0.9, 1.1]):
    N, M = image.shape[0], boxes.shape[2]
    """
    Getting features for the image N * C * H * W
    """
    Image_features = feature_model(image)
    """
    Getting features for the examples (N*M) * C * h * w
    """
    for ix in range(0,N):
        # boxes = boxes.squeeze(0)
        boxes = boxes[ix][0]
        cnter = 0
        Cnter1 = 0
        for keys in feat_map_keys:
            image_features = Image_features[keys][ix].unsqueeze(0)
            if keys == 'map1' or keys == 'map2':
                Scaling = 4.0
            elif keys == 'map3':
                Scaling = 8.0
            elif keys == 'map4':
                Scaling =  16.0
            else:
                Scaling = 32.0
            boxes_scaled = boxes / Scaling
            boxes_scaled[:, 1:3] = torch.floor(boxes_scaled[:, 1:3])
            boxes_scaled[:, 3:5] = torch.ceil(boxes_scaled[:, 3:5])
            boxes_scaled[:, 3:5] = boxes_scaled[:, 3:5] + 1 # make the end indices exclusive
            feat_h, feat_w = image_features.shape[-2], image_features.shape[-1]
            # make sure exemplars don't go out of bound
            boxes_scaled[:, 1:3] = torch.clamp_min(boxes_scaled[:, 1:3], 0)
            boxes_scaled[:, 3] = torch.clamp_max(boxes_scaled[:, 3], feat_h)
            boxes_scaled[:, 4] = torch.clamp_max(boxes_scaled[:, 4], feat_w)
            box_hs = boxes_scaled[:, 3] - boxes_scaled[:, 1]
            box_ws = boxes_scaled[:, 4] - boxes_scaled[:, 2]
            max_h = math.ceil(max(box_hs))
            max_w = math.ceil(max(box_ws))
            for j in range(0,M):
                y1, x1 = int(boxes_scaled[j,1]), int(boxes_scaled[j,2])
                y2, x2 = int(boxes_scaled[j,3]), int(boxes_scaled[j,4])
                #print(y1,y2,x1,x2,max_h,max_w)
                if j == 0:
                    examples_features = image_features[:,:,y1:y2, x1:x2]
                    if examples_features.shape[2] != max_h or examples_features.shape[3] != max_w:
                        #examples_features = pad_to_size(examples_features, max_h, max_w)
                        examples_features = F.interpolate(examples_features, size=(max_h,max_w),mode='bilinear')
                else:
                    feat = image_features[:,:,y1:y2, x1:x2]
                    if feat.shape[2] != max_h or feat.shape[3] != max_w:
                        feat = F.interpolate(feat, size=(max_h,max_w),mode='bilinear')
                        #feat = pad_to_size(feat, max_h, max_w)
                    examples_features = torch.cat((examples_features,feat),dim=0)
            """
            Convolving example features over image features
            """
            h, w = examples_features.shape[2], examples_features.shape[3]
            features =    F.conv2d(
                    F.pad(image_features, ((int(w/2)), int((w-1)/2), int(h/2), int((h-1)/2))),
                    examples_features
                )
            combined = features.permute([1,0,2,3])
            # computing features for scales 0.9 and 1.1
            for scale in exemplar_scales:
                    h1 = math.ceil(h * scale)
                    w1 = math.ceil(w * scale)
                    if h1 < 1: # use original size if scaled size is too small
                        h1 = h
                    if w1 < 1:
                        w1 = w
                    examples_features_scaled = F.interpolate(examples_features, size=(h1,w1),mode='bilinear')
                    features_scaled =    F.conv2d(F.pad(image_features, ((int(w1/2)), int((w1-1)/2), int(h1/2), int((h1-1)/2))),
                    examples_features_scaled)
                    features_scaled = features_scaled.permute([1,0,2,3])
                    combined = torch.cat((combined,features_scaled),dim=1)
            if cnter == 0:
                Combined = 1.0 * combined
            else:
                if Combined.shape[2] != combined.shape[2] or Combined.shape[3] != combined.shape[3]:
                    combined = F.interpolate(combined, size=(Combined.shape[2],Combined.shape[3]),mode='bilinear')
                Combined = torch.cat((Combined,combined),dim=1)
            cnter += 1
        if ix == 0:
            All_feat = 1.0 * Combined.unsqueeze(0)
        else:
            All_feat = torch.cat((All_feat,Combined.unsqueeze(0)),dim=0)
    return All_feat

class Unet_resizeImageWithGT(object):
    """
    If either the width or height of an image exceed a specified value, resize the image so that:
        1. The maximum of the new height and new width does not exceed a specified value
        2. The new height and new width are divisible by 8
        3. The aspect ratio is preserved
    No resizing is done if both height and width are smaller than the specified value
    By: Minh Hoai Nguyen (minhhoai@gmail.com)
    Modified by: Viresh
    """

    def __init__(self, MAX_HW=1504):
        self.max_hw = MAX_HW

    def __call__(self, sample):
        Normalize = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=IM_NORM_MEAN, std=IM_NORM_STD)])

        image, lines_boxes, density = sample['image'], sample['lines_boxes'], sample['gt_density']

        W, H = image.size
        if W > self.max_hw or H > self.max_hw:
            scale_factor = float(self.max_hw) / max(H, W)
            new_H = 8 * int(H * scale_factor / 8)
            new_W = 8 * int(W * scale_factor / 8)
            resized_image = transforms.Resize((new_H, new_W))(image)
            resized_density = cv2.resize(density, (new_W, new_H))
            orig_count = np.sum(density)
            new_count = np.sum(resized_density)

            if new_count > 0: resized_density = resized_density * (orig_count / new_count)

        else:
            scale_factor = 1
            resized_image = image
            resized_density = density
        boxes = list()
        for box in lines_boxes:
            box2 = [int(k * scale_factor) for k in box]
            y1, x1, y2, x2 = box2[0], box2[1], box2[2], box2[3]
            boxes.append([y1, x1, y2, x2])

        boxes = torch.Tensor(boxes).unsqueeze(0)
        resized_image = Normalize(resized_image)
        resized_density = torch.from_numpy(resized_density).unsqueeze(0).unsqueeze(0)
        sample = {'image': resized_image, 'boxes': boxes, 'gt_density': resized_density}
        return sample

class resizeImageWithGT(object):
    """
    If either the width or height of an image exceed a specified value, resize the image so that:
        1. The maximum of the new height and new width does not exceed a specified value
        2. The new height and new width are divisible by 8
        3. The aspect ratio is preserved
    No resizing is done if both height and width are smaller than the specified value
    By: Minh Hoai Nguyen (minhhoai@gmail.com)
    Modified by: Viresh
    """

    def __init__(self, MAX_HW=1504):
        self.max_hw = MAX_HW

    def __call__(self, sample):
        Normalize = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=IM_NORM_MEAN, std=IM_NORM_STD)])

        image, lines_boxes, density = sample['image'], sample['lines_boxes'], sample['gt_density']

        W, H = image.size
        if W > self.max_hw or H > self.max_hw:
            scale_factor = float(self.max_hw) / max(H, W)
            new_H = 8 * int(H * scale_factor / 8)
            new_W = 8 * int(W * scale_factor / 8)
            resized_image = transforms.Resize((new_H, new_W))(image)
            resized_density = cv2.resize(density, (new_W, new_H))
            orig_count = np.sum(density)
            new_count = np.sum(resized_density)

            if new_count > 0: resized_density = resized_density * (orig_count / new_count)

        else:
            scale_factor = 1
            resized_image = image
            resized_density = density
        boxes = list()
        for box in lines_boxes:
            box2 = [int(k * scale_factor) for k in box]
            y1, x1, y2, x2 = box2[0], box2[1], box2[2], box2[3]
            boxes.append([0, y1, x1, y2, x2])

        boxes = torch.Tensor(boxes).unsqueeze(0)
        resized_image = Normalize(resized_image)
        resized_density = torch.from_numpy(resized_density).unsqueeze(0).unsqueeze(0)
        sample = {'image': resized_image, 'boxes': boxes, 'gt_density': resized_density}
        return sample

TransformTrain = transforms.Compose([resizeImageWithGT(MAX_HW)])

def MincountLoss(output,boxes, deivce, use_gpu=True):
    ones = torch.ones(1)
    if use_gpu: ones = ones.to(deivce)
    Loss = 0.
    if boxes.shape[1] > 1:
        boxes = boxes.squeeze()
        for tempBoxes in boxes.squeeze():
            y1 = int(tempBoxes[1])
            y2 = int(tempBoxes[3])
            x1 = int(tempBoxes[2])
            x2 = int(tempBoxes[4])
            X = output[:,:,y1:y2,x1:x2].sum()
            if X.item() <= 1:
                Loss += F.mse_loss(X,ones)
    else:
        boxes = boxes.squeeze()
        y1 = int(boxes[1])
        y2 = int(boxes[3])
        x1 = int(boxes[2])
        x2 = int(boxes[4])
        X = output[:,:,y1:y2,x1:x2].sum()
        if X.item() <= 1:
            Loss += F.mse_loss(X,ones)
    return Loss

def PerturbationLoss(output,boxes, device, sigma=8, use_gpu=True):
    Loss = 0.
    if boxes.shape[1] > 1:
        boxes = boxes.squeeze()
        for tempBoxes in boxes.squeeze():
            y1 = int(tempBoxes[1])
            y2 = int(tempBoxes[3])
            x1 = int(tempBoxes[2])
            x2 = int(tempBoxes[4])
            out = output[:,:,y1:y2,x1:x2]
            GaussKernel = matlab_style_gauss2D(shape=(out.shape[2],out.shape[3]),sigma=sigma)
            GaussKernel = torch.from_numpy(GaussKernel).float()
            if use_gpu: GaussKernel = GaussKernel.to(device)
            Loss += F.mse_loss(out.squeeze(),GaussKernel)
    else:
        boxes = boxes.squeeze()
        y1 = int(boxes[1])
        y2 = int(boxes[3])
        x1 = int(boxes[2])
        x2 = int(boxes[4])
        out = output[:,:,y1:y2,x1:x2]
        Gauss = matlab_style_gauss2D(shape=(out.shape[2],out.shape[3]),sigma=sigma)
        GaussKernel = torch.from_numpy(Gauss).float()
        if use_gpu: GaussKernel = GaussKernel.to(device)
        Loss += F.mse_loss(out.squeeze(),GaussKernel)
    return Loss

def matlab_style_gauss2D(shape=(3,3),sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def gauss2D_density(shape,sigmas=None):
    """
        Generate 2D gaussian density
        shape: height, width
        sigmas: for height and width

    """
    m,n = [(ss-1.)/2. for ss in shape]

    if sigmas is None:
        sigmas = [m/2, n/2]

    y,x = np.ogrid[-m:m+1,-n:n+1]
    var_y = 2 * sigmas[0] * sigmas[0]
    var_x = 2 * sigmas[1] * sigmas[1]

    h = np.exp( - x*x/var_x - y*y/var_y)
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h