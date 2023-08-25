'''
FSC147 Dataset Loader and Preprocess
By Yifeng Huang(yifehuang@cs.stonybrook.edu)
Based on Viresh and Minh's code
Last Modified 2022.2.14
'''
from pycocotools.coco import COCO
import os
import cv2
import json
import torch
import numpy as np
from PIL import Image
from FamNet.utils import gauss2D_density
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

class FscBgDataset(Dataset):
    """Fewshot counting dataset with background FSC-147"""

    def __init__(self, root_dir, data_split, transform = None):
        assert data_split in ['train', 'val', 'test']
        anno_file = os.path.join(root_dir, 'json_annotationCombined_384_VarV2.json')
        data_split_file = os.path.join(root_dir, 'Train_Test_Val_FSC_147.json')

        with open(anno_file) as f:
            self.annotations = json.load(f)

        with open(data_split_file) as f:
            data_split_ids = json.load(f)

        self.im_dir = os.path.join(root_dir, 'images_384_VarV2')
        self.gt_dir = os.path.join(root_dir, 'gt_density_map_adaptive_384_VarV2')
        self.im_ids = data_split_ids[data_split]
        self.transform = transform

    def __len__(self):
        return len(self.im_ids)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        im_id = self.im_ids[idx]
        anno = self.annotations[im_id]

        bboxes = anno['box_examples_coordinates']
        dots = np.array(anno['points'])

        rects = list()
        for bbox in bboxes:
            x1 = bbox[0][0]
            y1 = bbox[0][1]
            x2 = bbox[2][0]
            y2 = bbox[2][1]
            rects.append([y1, x1, y2, x2])

        image = Image.open('{}/{}'.format(self.im_dir, im_id))
        image.load()

        W, H = image.size
        dots = np.maximum(dots, 0)
        dots[:, 0] = np.minimum(dots[:, 0], W - 1)
        dots[:, 1] = np.minimum(dots[:, 1], H - 1)

        # Get GT Density
        density_path = os.path.join(self.gt_dir, im_id.split(".jpg")[0] + ".npy")
        density = np.load(density_path).astype('float32')

        #Get Boxes
        boxes = np.array(rects)
        sample = {'im_id':im_id, 'image':image, 'dots':dots, 'boxes': boxes, 'gt_density':density}

        if self.transform:
            sample = self.transform(sample)

        return sample

class FSCD_LVIS_Dataset(Dataset):
    def __init__(self, data_path, split="train"):
        print("This data is fscd LVIS, with few exmplar boxes and points, split: {}".format(split), end="  ")
        pseudo_label_file = "instances_" + split + ".json"
        self.coco = COCO(os.path.join(data_path, "annotations", pseudo_label_file))
        self.image_ids = self.coco.getImgIds()

        self.img_root_path = os.path.join(data_path, "images", "all_images")
        self.density_root_path = os.path.join(data_path, "masks", "all_density")
        self.count_anno_file = os.path.join(data_path, "annotations", "count_" + split + ".json")
        self.count_anno = self.load_json(self.count_anno_file)
        print("with number of images: ", self.__len__())

    def load_json(self, json_file):
        with open(json_file, "r") as f:
            data = json.load(f)

        return data

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.coco.loadImgs([img_id])[0]
        img_file = img_info["file_name"]

        ex_bboxes = self.count_anno["annotations"][idx]["boxes"]
        rects = list()
        for bbox in ex_bboxes[:3]:
            x, y, w, h = bbox
            x1, y1 = x, y
            x2, y2 = x + w, y + h
            rects.append([y1, x1, y2, x2])

        img = Image.open(os.path.join(self.img_root_path, img_file))
        img = img.convert("RGB")
        boxes = np.array(rects)

        dots = self.count_anno["annotations"][idx]["points"]
        density_name = img_file.split('.')[0] + '.npy'
        density_path = os.path.join(self.density_root_path, density_name)
        density = np.load(density_path).astype('float32')
        img_name = img_file
        sample = {'im_id':img_name, 'image':img, 'dots':dots, 'boxes': boxes, 'gt_density':density}

        return sample