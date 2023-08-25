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

def interactive_loss(density, gt_density, mask, count_limit = 4):
    density = density * mask
    gt_density = gt_density * mask
    if gt_density.sum() >= count_limit:
        loss = max(0, count_limit - density.sum())
    else:
        count_gt_ceil = np.ceil(gt_density.sum().cpu().numpy())
        count_gt_floor = np.floor(gt_density.sum().cpu().numpy())
        loss = max(0, count_gt_floor - density.sum()) + max(0, density.sum() - count_gt_ceil)
    return loss

def sample_pixel(label):
  height, width = label.shape
  y = random.randint(0, height - 1)
  x = random.randint(0, width - 1)
  return y,x

def random_region_sample(label):
  max_label = np.max(label) + 1
  random_label = np.random.randint(0, int(max_label))
  return random_label

def zero_prior_region_sample(label, gt_density, device):
    max_label = np.max(label) + 1
    max_error = 0
    final_sample_label = None
    # For all region, calculate the loss
    for sample_label in range(max_label):
        # The Mask
        inter_mask = np.zeros((label.shape[0], label.shape[1]), dtype=np.uint8)
        inter_mask[label == sample_label] = 1
        inter_mask = torch.from_numpy(inter_mask).to(device)
        # Error
        gt_density_zero = gt_density * inter_mask
        if gt_density_zero.sum() == 0:
            final_sample_label = sample_label
            return final_sample_label
    random_label = np.random.randint(0, int(max_label))
    return random_label

def error_region_sample(label, density, gt_density, device):
  max_label = np.max(label) + 1
  max_error = 0
  final_sample_label = None
  #For all region, calculate the loss
  for sample_label in range(max_label):
    #The Mask
    inter_mask = np.zeros((label.shape[0], label.shape[1]), dtype=np.uint8)
    inter_mask[label == sample_label] = 1
    inter_mask = torch.from_numpy(inter_mask).to(device)
    #Error
    inter_error = interactive_loss(density, gt_density, inter_mask)
    #Max error
    if max_error < inter_error:
      max_error = inter_error
      final_sample_label = sample_label
  return final_sample_label

def baseline_random_region_sample(label, inter_time):
    max_label = np.max(label) + 1
    labels = list(range(max_label))
    random.shuffle(labels)
    label_list = labels[0:inter_time]
    return label_list

def baseline_error_region_sample(label, density, gt_density, inter_time):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = 'cpu'
    max_label = np.max(label) + 1
    error_array = np.zeros(max_label)
    final_sample_label = None
    # For all region, calculate the loss
    for sample_label in range(max_label):
        # The Mask
        inter_mask = np.zeros((label.shape[0], label.shape[1]), dtype=np.uint8)
        inter_mask[label == sample_label] = 1
        inter_mask = torch.from_numpy(inter_mask).to(device)
        # Error
        inter_error = interactive_loss(density, gt_density, inter_mask)
        # Max error
        error_array[sample_label] = inter_error
    label_list = []
    for i in range(inter_time):
        max_label = np.argmax(error_array)
        error_array[max_label] = -1
        label_list.append(max_label)
    return label_list

#Interactive loss for crowd counting upper bound and lower bound
def interactive_loss_crowd(density, gt_density, mask, count_limit = 50):
    density = density * mask
    gt_density = gt_density * mask
    if gt_density.sum() >= count_limit:
        loss = max(0, count_limit - density.sum())
    else:
        count_gt_ceil = np.ceil(gt_density.sum().cpu().numpy() / 10) * 10
        count_gt_floor = np.floor(gt_density.sum().cpu().numpy() / 10) * 10
        loss = max(0, count_gt_floor - density.sum()) + max(0, density.sum() - count_gt_ceil)
    return loss

def interactive_loss_double(density, gt_density, mask, count_limit = 8):
    density = density * mask
    gt_density = gt_density * mask
    if gt_density.sum() >= count_limit:
        loss = max(0, count_limit - density.sum())
    else:
        count_gt_ceil = np.ceil(gt_density.sum().cpu().numpy() / 2) * 2
        count_gt_floor = np.floor(gt_density.sum().cpu().numpy() / 2) * 2
        loss = max(0, count_gt_floor - density.sum()) + max(0, density.sum() - count_gt_ceil)
    return loss

def interactive_loss_list_crowd(density, gt_density, mask_list, count_limit = 50):
    gt_density_sum = 0
    density_sum = 0
    for mask in mask_list:
      mask = mask.cuda()
      density_sum += (density * mask).sum()
      gt_density_sum += (gt_density * mask).sum()
    if density_sum >= count_limit:
        loss = max(0, count_limit - density_sum)
    else:
        loss = max(0, gt_density_sum - density_sum) + max(0, density_sum - gt_density_sum)
    return loss

def error_region_sample_crowd(label, density, gt_density, device = torch.device('cuda')):
    max_label = np.max(label) + 1
    max_error = 0
    final_sample_label = None
    # For all region, calculate the loss
    for sample_label in range(max_label):
        # The Mask
        inter_mask = np.zeros((label.shape[0], label.shape[1]), dtype=np.uint8)
        inter_mask[label == sample_label] = 1
        inter_mask = torch.from_numpy(inter_mask).to(device)
        # Error
        inter_error = interactive_loss_crowd(density, gt_density, inter_mask)
        # Max error
        if max_error < inter_error:
            max_error = inter_error
            final_sample_label = sample_label
    return final_sample_label


def set_seed(seed = 3407):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False
    #torch.backends.cudnn.enabled = False

def interactive_loss_uncertain(density, gt_density, mask, count_limit = 4):
    density = density * mask
    gt_density = gt_density * mask
    if gt_density.sum() >= count_limit:
        loss = max(0, count_limit - density.sum())
        # under counting
        uncertain_state = -1
    elif gt_density.sum() == 0:
        loss = (density ** 2).sum()
        uncertain_state = 0
    else:
        count_gt_ceil = np.ceil(gt_density.sum().cpu().numpy())
        count_gt_floor = np.floor(gt_density.sum().cpu().numpy())
        loss = max(0, count_gt_floor - density.sum()) + max(0, density.sum() - count_gt_ceil)
        if density.sum() > count_gt_ceil:
          # overcounting
          uncertain_state = 1
        else:
          uncertain_state = -1
    return loss, uncertain_state

def get_uncertain_state(density, gt_density, mask, count_limit = 4):
    density = density * mask
    gt_density = gt_density * mask
    if gt_density.sum() >= count_limit:
        uncertain_state = -1
    else:
        count_gt_ceil = np.ceil(gt_density.sum().cpu().numpy())
        count_gt_floor = np.floor(gt_density.sum().cpu().numpy())
        if density.sum() > count_gt_ceil:
            # overcounting
            uncertain_state = 1
        elif density.sum() < count_gt_floor:
            uncertain_state = -1
        else:
            uncertain_state = 0
    return uncertain_state

def get_default_cfg():
    cfg = OmegaConf.load('./Config/default.yaml')
    return cfg

def merge_from_file(cfg, file):
    cfg2 = OmegaConf.load(file)
    cfg_new = OmegaConf.merge(cfg, cfg2)
    return cfg_new

def show_cfg(cfg):
    print(OmegaConf.to_yaml(cfg))

def save_cfg(cfg, save_path):
    OmegaConf.save(config = cfg, f = save_path)

def load_cfg(file_path: str):
    cfg = OmegaConf.load(file_path)
    return cfg