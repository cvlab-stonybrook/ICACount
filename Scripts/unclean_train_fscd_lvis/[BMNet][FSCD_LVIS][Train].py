import os
import math
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

result = {}
seed = 3407
os.environ['PYTHONHASHSEED'] = str(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

if torch.cuda.is_available():
    device = "cuda:3"
else:
    device = 'cpu'

from BMnet.FSC147_dataset import Build_FSCD_LVIS, batch_collate_fn
from utils import extract_features, TransformTrain, MincountLoss, PerturbationLoss, interactive_loss, random_region_sample, save_cfg
from BMnet.config import cfg
from BMnet.models import build_model, build_model_train
from BMnet.loss import get_loss

Save_dir = '/home/yifeng/FeatureRefine/Log/'
cfg.merge_from_file('/home/yifeng/FeatureRefine/Code/BMnet/config/bmnet+_fsc147.yaml')
model = build_model_train(cfg)
train_dataset, val_dataset, test_dataset = Build_FSCD_LVIS(cfg)
data_loader_train= DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=batch_collate_fn)
data_loader_val = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=batch_collate_fn)
criterion = get_loss(cfg, device)
criterion.to(device)
model.to(device)

n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('number of params:', n_parameters)

param_dicts = [
    {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
    {
        "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
        "lr": cfg.TRAIN.lr_backbone,
    },
]

if cfg.TRAIN.optimizer == "AdamW":
    optimizer = torch.optim.AdamW(param_dicts, lr=cfg.TRAIN.lr,
                                  weight_decay=cfg.TRAIN.weight_decay)
elif cfg.TRAIN.optimizer == "Adam":
    optimizer = torch.optim.Adam(param_dicts, lr=cfg.TRAIN.lr)
elif cfg.TRAIN.optimizer == "SGD":
    optimizer = torch.optim.SGD(param_dicts, lr=cfg.TRAIN.lr,
                                weight_decay=cfg.TRAIN.weight_decay)
else:
    raise NotImplementedError

lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, cfg.TRAIN.lr_drop)


def train_one_epoch(model, criterion, data_loader, optimizer, device, epoch, max_norm = 0):
    model.train()
    criterion.train()
    loss_sum = 0
    loss_counting = 0
    loss_contrast = 0
    train_mae = 0
    train_rmse = 0
    for idx, sample in enumerate(data_loader):
        img, patches, targets, _ = sample
        img = img.to(device)
        patches['patches'] = patches['patches'].to(device)
        patches['scale_embedding'] = patches['scale_embedding'].to(device)
        density_map = targets['density_map'].to(device)
        pt_map = targets['pt_map'].to(device)
        outputs = model(img, patches, is_train=True)
        dest = outputs['density_map']
        counting_loss, contrast_loss = criterion(outputs, density_map, pt_map)
        loss = counting_loss if isinstance(contrast_loss, int) else counting_loss + contrast_loss
        gt_cnt = targets['gtcount'].item()
        pred_cnt = dest.sum().item()
        cnt_err = abs(pred_cnt - gt_cnt)
        train_mae += cnt_err
        train_rmse += cnt_err ** 2
        loss_value = loss.item()
        if not math.isfinite(loss_value):
            continue

        loss_sum += loss_value
        loss_contrast += contrast_loss if isinstance(contrast_loss, int) else contrast_loss.item()
        loss_counting += counting_loss.item()

        optimizer.zero_grad()
        loss.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

    train_mae = (train_mae / len(data_loader))
    train_rmse = (train_rmse / len(data_loader)) ** 0.5
    return loss_sum / len(data_loader), train_mae, train_rmse


@torch.no_grad()
def evaluate(model, data_loader, device):
    mae = 0
    mse = 0
    model.eval()
    for idx, sample in enumerate(data_loader):
        img, patches, targets, _ = sample
        img = img.to(device)
        patches['patches'] = patches['patches'].to(device)
        patches['scale_embedding'] = patches['scale_embedding'].to(device)
        gtcount = targets['gtcount']
        with torch.no_grad():
            outputs = model(img, patches, is_train=False)
        error = torch.abs(outputs.sum() - gtcount.item()).item()
        mae += error
        mse += error ** 2

    mae = mae / len(data_loader)
    mse = mse / len(data_loader)
    mse = mse ** 0.5

    return mae, mse

Best_MAE = 1e6
result = {}
exp_name = '[BMNet][FSCD_LVIS][Train]'
log_root = '/home/yifeng/FeatureRefine/Log/'
current_time = datetime.datetime.now()
experiment_date = current_time.strftime("%m-%d-%Y %H-%M")
exp_name = '[' + experiment_date + ']' + exp_name
log_save_dir = os.path.join(log_root, exp_name)
TBwriter = SummaryWriter(log_save_dir)
for ep in range(cfg.TRAIN.start_epoch, cfg.TRAIN.epochs):
    result[ep] = {}
    train_loss, train_mae, train_rmse = train_one_epoch(
        model, criterion, data_loader_train, optimizer, device, ep,
        cfg.TRAIN.clip_max_norm)
    print('Train MAE: ', train_mae)
    print('Train RMSE: ', train_rmse)
    result[ep]['Train MAE'] = train_mae
    result[ep]['Train RMSE'] = train_rmse
    TBwriter.add_scalar('train/MAE', train_mae, ep + 1)
    TBwriter.add_scalar('train/RMSE', train_rmse, ep + 1)
    TBwriter.add_scalar('train/Count loss', train_loss, ep + 1)
    print('Evaluation:')
    val_mae, val_rmse = evaluate(model, data_loader_val, device)
    print('Val MAE: ', val_mae)
    print('Val RMSE: ', val_rmse)
    result[ep]['Val MAE'] = val_mae
    result[ep]['Val RMSE'] = val_rmse
    TBwriter.add_scalar('val/MAE', val_mae, ep + 1)
    TBwriter.add_scalar('val/RMSE', val_rmse, ep + 1)
    if val_mae < Best_MAE:
        if not os.path.exists(log_save_dir):
            os.makedirs(log_save_dir)
        model_save_dir = os.path.join(log_save_dir, 'FSCD_LVIS_BMNet.pth')
        torch.save(model.state_dict(), model_save_dir)
        Best_MAE = val_mae
    print('Best MAE:', Best_MAE)
    result_conf = OmegaConf.create(result)
    Result_save_path = os.path.join(log_save_dir, 'Result.yaml')
    save_cfg(result_conf, Result_save_path)