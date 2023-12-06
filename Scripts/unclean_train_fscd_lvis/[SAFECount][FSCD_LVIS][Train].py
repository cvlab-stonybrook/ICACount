import os
import copy
import random
import torch
import numpy as np
import datetime
from Dataset import FSCD_LVIS_Dataset
from torch.utils.data import DataLoader
from utils import extract_features, TransformTrain, set_seed, MincountLoss, PerturbationLoss
from Model import Resnet50FPN, weights_normal_init, CountRegressor
import torch.nn.functional as F
from omegaconf import OmegaConf
from config import save_cfg
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ExponentialLR, StepLR

result = {}
seed = 3407
os.environ['PYTHONHASHSEED'] = str(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

from utils import (
        interactive_loss,
        random_region_sample,
        save_cfg,
        interactive_loss_uncertain,
    )
from SAFECount.safecount import MySafecount
from SAFECount.utils.misc_helper import (
    create_logger,
    get_current_time,
    load_state,
    save_checkpoint,
    set_random_seed,
    to_device,
)
from SAFECount.datasets.data_builder import build_dataloader
from easydict import EasyDict
import yaml

def get_optimizer(parameters, config):
    if config.type == "Adam":
        return torch.optim.Adam(parameters, **config.kwargs)
    elif config.type == "SGD":
        return torch.optim.SGD(parameters, **config.kwargs)
    elif config.type == "AdamW":
        return torch.optim.AdamW(parameters, **config.kwargs)
    else:
        raise NotImplementedError

def get_scheduler(optimizer, config):
    if config["type"] == "StepLR":
        return StepLR(optimizer, **config["kwargs"])
    elif config["type"] == "ExponentialLR":
        return ExponentialLR(optimizer, **config["kwargs"])
    else:
        raise NotImplementedError

cfg_dir = '/home/yifeng/CVPR2023/Code/SAFECount/FSCD_LVIS.yaml'
with open(cfg_dir) as f:
    config = EasyDict(yaml.load(f, Loader=yaml.FullLoader))

train_loader, val_loader, test_loader = build_dataloader(config.dataset, distributed=False)
model = MySafecount(config)
if torch.cuda.is_available():
    device = "cuda:2"
else:
    device = 'cpu'
model.to(device)
model.train()
lr_scale_backbone = config.trainer["lr_scale_backbone"]
if lr_scale_backbone == 0:
    model.module.backbone.eval()
    for p in model.module.backbone.parameters():
        p.requires_grad = False
    # parameters not include backbone
    parameters = [
        p for n, p in model.module.named_parameters() if "backbone" not in n
    ]
else:
    assert lr_scale_backbone > 0 and lr_scale_backbone <= 1
    parameters = [
        {
            "params": [
                p
                for n, p in model.module.named_parameters()
                if "backbone" not in n and p.requires_grad
            ],
            "lr": config.trainer.optimizer.kwargs.lr,
        },
        {
            "params": [
                p
                for n, p in model.module.named_parameters()
                if "backbone" in n and p.requires_grad
            ],
            "lr": lr_scale_backbone * config.trainer.optimizer.kwargs.lr,
        },
    ]

optimizer = get_optimizer(parameters, config.trainer.optimizer)
lr_scheduler = get_scheduler(optimizer, config.trainer.lr_scheduler)
last_epoch = 0
criterion = torch.nn.MSELoss().to(device)

def train_one_epoch(train_loader, model, optimizer, criterion, lr_scheduler):
    model.train()
    if lr_scale_backbone == 0:
        model.module.backbone.eval()
        for p in model.module.backbone.parameters():
            p.requires_grad = False

    train_loss = 0.
    train_mae = 0.
    train_rmse = 0.
    for i, sample in enumerate(train_loader):
        iter = i + 1
        current_lr = lr_scheduler.get_lr()[0]
        sample = to_device(sample, device=torch.device(device))
        # forward
        density_pred = model(sample)  # 1 x 1 x h x w
        density = sample["density"]
        loss = 250 * criterion(density.squeeze(), density_pred.squeeze())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        pred_cnt = torch.sum(density_pred).item()
        gt_cnt = torch.sum(density).item()
        cnt_err = abs(pred_cnt - gt_cnt)
        train_mae += cnt_err
        train_rmse += cnt_err ** 2

    train_loss = torch.Tensor([train_loss]).cuda()
    iter = torch.Tensor([iter]).cuda()
    train_mae = (train_mae / iter.item())
    train_rmse = (train_rmse / iter.item()) ** 0.5
    train_loss = train_loss.item() / iter.item()
    return train_mae, train_rmse, train_loss

def eval(val_loader, model, criterion):
    model.eval()

    val_loss = 0.
    val_mae = 0.
    val_rmse = 0.
    for i, sample in enumerate(val_loader):
        iter = i + 1
        sample = to_device(sample, device=torch.device(device))
        # forward
        outputs = model(sample)  # 1 x 1 x h x w
        density_pred = model(sample)  # 1 x 1 x h x w
        density = sample["density"]
        loss = 250 * criterion(density.squeeze(), density_pred.squeeze())
        val_loss += loss.item()
        pred_cnt = torch.sum(density_pred).item()
        gt_cnt = torch.sum(density).item()
        cnt_err = abs(pred_cnt - gt_cnt)
        val_mae += cnt_err
        val_rmse += cnt_err ** 2
    val_loss = torch.Tensor([val_loss]).cuda()
    iter = torch.Tensor([iter]).cuda()
    val_loss = val_loss.item() / iter.item()
    val_mae = (val_mae / iter.item())
    val_rmse = (val_rmse / iter.item()) ** 0.5
    return val_mae, val_rmse, val_loss

Best_MAE = 1e6
result = {}
exp_name = '[SAFECount][FSCD_LVIS][Train]'
log_root = '/home/yifeng/FeatureRefine/Log/'
current_time = datetime.datetime.now()
experiment_date = current_time.strftime("%m-%d-%Y %H-%M")
exp_name = '[' + experiment_date + ']' + exp_name
log_save_dir = os.path.join(log_root, exp_name)
TBwriter = SummaryWriter(log_save_dir)
for ep in range(last_epoch, config.trainer.epochs):
    result[ep] = {}
    print('###################Train#####################')
    train_mae, train_rmse, train_loss = train_one_epoch(train_loader, model, optimizer, criterion, lr_scheduler)
    print('Train MAE: ', train_mae)
    print('Train RMSE: ', train_rmse)
    result[ep]['Train MAE'] = train_mae
    result[ep]['Train RMSE'] = train_rmse
    TBwriter.add_scalar('train/MAE', train_mae, ep + 1)
    TBwriter.add_scalar('train/RMSE', train_rmse, ep + 1)
    TBwriter.add_scalar('train/Count loss', train_loss, ep + 1)
    print('Evaluation:')

    val_mae, val_rmse, val_loss = eval(val_loader, model, criterion)
    print('Val MAE: ', val_mae)
    print('Val RMSE: ', val_rmse)
    result[ep]['Val MAE'] = val_mae
    result[ep]['Val RMSE'] = val_rmse
    TBwriter.add_scalar('val/MAE', val_mae, ep + 1)
    TBwriter.add_scalar('val/RMSE', val_rmse, ep + 1)
    if val_mae < Best_MAE:
        if not os.path.exists(log_save_dir):
            os.makedirs(log_save_dir)
        model_save_dir = os.path.join(log_save_dir, 'FSCD_LVIS_SAFECount.pth')
        torch.save(model.state_dict(), model_save_dir)
        Best_MAE = val_mae
    print('Best MAE:', Best_MAE)
    result_conf = OmegaConf.create(result)
    Result_save_path = os.path.join(log_save_dir, 'Result.yaml')
    save_cfg(result_conf, Result_save_path)