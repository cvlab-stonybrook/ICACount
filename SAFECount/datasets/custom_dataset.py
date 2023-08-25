from __future__ import division

import json
import os

import cv2
import numpy as np
import torch
import torch.distributed as dist
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler

from SAFECount.datasets.base_dataset import BaseDataset, BaseTransform
from SAFECount.datasets.transforms import RandomColorJitter
from pycocotools.coco import COCO

def build_custom_dataloader(cfg, training, distributed=False):

    normalize_fn = transforms.Normalize(mean=cfg["pixel_mean"], std=cfg["pixel_std"])

    if training:
        hflip = cfg.get("hflip", False)
        vflip = cfg.get("vflip", False)
        rotate = cfg.get("rotate", False)
        gamma = cfg.get("gamma", False)
        gray = cfg.get("gray", False)
        transform_fn = BaseTransform(
            cfg["input_size"], hflip, vflip, rotate, gamma, gray
        )
    else:
        transform_fn = BaseTransform(
            cfg["input_size"], False, False, False, False, False
        )

    if training and cfg.get("colorjitter", None):
        colorjitter_fn = RandomColorJitter.from_params(cfg["colorjitter"])
    else:
        colorjitter_fn = None

    dataset = CustomDataset(
        cfg["img_dir"],
        cfg["density_dir"],
        cfg["meta_file"],
        cfg["shot"],
        transform_fn=transform_fn,
        normalize_fn=normalize_fn,
        colorjitter_fn=colorjitter_fn,
    )
    sampler = RandomSampler(dataset)
    data_loader = DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        num_workers=cfg["workers"],
        pin_memory=True,
        sampler=sampler,
    )

    return data_loader

def build_FSCD_LVIS_dataloader(cfg, split, distributed=False):

    normalize_fn = transforms.Normalize(mean=cfg["pixel_mean"], std=cfg["pixel_std"])
    fscdlvis_root_dir = cfg.get("fscdlvis_root_dir", False)
    if split == 'train':
        hflip = cfg.get("hflip", False)
        vflip = cfg.get("vflip", False)
        rotate = cfg.get("rotate", False)
        gamma = cfg.get("gamma", False)
        gray = cfg.get("gray", False)
        transform_fn = BaseTransform(
            cfg["input_size"], hflip, vflip, rotate, gamma, gray
        )
    else:
        transform_fn = BaseTransform(
            cfg["input_size"], False, False, False, False, False
        )

    if split == 'train' and cfg.get("colorjitter", None):
        colorjitter_fn = RandomColorJitter.from_params(cfg["colorjitter"])
    else:
        colorjitter_fn = None

    dataset = FSCD_LVISDataset(
        data_path = fscdlvis_root_dir,
        split = split,
        transform_fn=transform_fn,
        normalize_fn=normalize_fn,
        colorjitter_fn=colorjitter_fn,
    )
    sampler = RandomSampler(dataset)
    data_loader = DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        num_workers=cfg["workers"],
        pin_memory=True,
        sampler=sampler,
    )

    return data_loader

class CustomDataset(BaseDataset):
    def __init__(
        self,
        img_dir,
        density_dir,
        meta_file,
        shot,
        transform_fn,
        normalize_fn,
        colorjitter_fn=None,
    ):
        self.img_dir = img_dir
        self.density_dir = density_dir
        self.meta_file = meta_file
        self.shot = shot
        self.transform_fn = transform_fn
        self.normalize_fn = normalize_fn
        self.colorjitter_fn = colorjitter_fn

        # construct metas
        if isinstance(meta_file, str):
            meta_file = [meta_file]
        self.metas = []
        for _meta_file in meta_file:
            with open(_meta_file, "r+") as f_r:
                for line in f_r:
                    meta = json.loads(line)
                    self.metas.append(meta)

    def __len__(self):
        return len(self.metas)

    def __getitem__(self, index):
        meta = self.metas[index]
        # read img
        img_name = meta["filename"]
        img_path = os.path.join(self.img_dir, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
        # read density
        density_name = meta["density"]
        density_path = os.path.join(self.density_dir, density_name)
        density = np.load(density_path)
        # get boxes, h, w
        boxes = meta["boxes"]
        if self.shot:
            boxes = boxes[: self.shot]
        # transform
        if self.transform_fn:
            image, density, boxes, _ = self.transform_fn(
                image, density, boxes, [], (height, width)
            )
        if self.colorjitter_fn:
            image = self.colorjitter_fn(image)
        image = transforms.ToTensor()(image)
        density = transforms.ToTensor()(density)
        boxes = torch.tensor(boxes, dtype=torch.float64)
        if self.normalize_fn:
            image = self.normalize_fn(image)
        return {
            "filename": img_name,
            "height": height,
            "width": width,
            "image": image,
            "density": density,
            "boxes": boxes,
        }

class FSCD_LVISDataset(BaseDataset):
    def __init__(
        self,
        data_path,
        split,
        transform_fn,
        normalize_fn,
        colorjitter_fn=None,
    ):
        print('Loading FSCD_LVIS Dataset......')
        pseudo_label_file = "instances_" + split + ".json"
        self.coco = COCO(os.path.join(data_path, "annotations", pseudo_label_file))
        self.image_ids = self.coco.getImgIds()
        self.transform_fn = transform_fn
        self.normalize_fn = normalize_fn
        self.colorjitter_fn = colorjitter_fn
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

    def __getitem__(self, index):
        img_id = self.image_ids[index]
        img_info = self.coco.loadImgs([img_id])[0]
        img_file = img_info["file_name"]

        ex_bboxes = self.count_anno["annotations"][index]["boxes"]
        rects = list()
        for bbox in ex_bboxes[:3]:
            x, y, w, h = bbox
            x1, y1 = x, y
            x2, y2 = x + w, y + h
            rects.append([y1, x1, y2, x2])
        boxes = np.array(rects)

        img_path = os.path.join(self.img_root_path, img_file)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
        # read density
        density_name = img_file.split('.')[0] + '.npy'
        density_path = os.path.join(self.density_root_path, density_name)
        density = np.load(density_path)

        # transform
        if self.transform_fn:
            image, density, boxes, _ = self.transform_fn(
                image, density, boxes, [], (height, width)
            )
        if self.colorjitter_fn:
            image = self.colorjitter_fn(image)
        image = transforms.ToTensor()(image)
        density = transforms.ToTensor()(density)
        boxes = torch.tensor(boxes, dtype=torch.float64)
        if self.normalize_fn:
            image = self.normalize_fn(image)
        return {
            "filename": img_file,
            "height": height,
            "width": width,
            "image": image,
            "density": density,
            "boxes": boxes,
        }