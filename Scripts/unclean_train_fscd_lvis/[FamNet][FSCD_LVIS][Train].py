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

result = {}
seed = 30
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

Root_dir = '/home/yifeng/data/FSCD_LVIS/'
train_dataset = FSCD_LVIS_Dataset(Root_dir, 'train')
val_dataset = FSCD_LVIS_Dataset(Root_dir, 'val')
test_dataset = FSCD_LVIS_Dataset(Root_dir, 'test')
resnet50_conv = Resnet50FPN().to(device)
resnet50_conv.to(device)
regressor = CountRegressor(6, pool='mean')
regressor.to(device)
regressor.train()

optimizer = torch.optim.Adam(regressor.parameters(), lr=1e-6)
weights_normal_init(regressor, dev=0.001)
counting_criterion = torch.nn.MSELoss(reduction = 'sum').to(device)

exp_name = '[FamNet][FSCD_LVIS][Train]'
log_root = '/home/yifeng/FeatureRefine/Log/'
current_time = datetime.datetime.now()
experiment_date = current_time.strftime("%m-%d-%Y %H-%M")
exp_name = '[' + experiment_date + ']' + exp_name
log_save_dir = os.path.join(log_root, exp_name)
TBwriter = SummaryWriter(log_save_dir)
resnet50_conv.eval()
regressor.train()
MAPS = ['map3', 'map4']
Scales = [0.8, 1.2]
def train():
    batch_index = 0
    train_mae = 0
    train_rmse = 0
    all_count_loss = 0
    batch_loss = 0
    idx_list = list(range(len(train_dataset)))
    random.shuffle(idx_list)
    for idx in idx_list:
        batch_index += 1
        sample = train_dataset[idx]
        im_id, image, boxes, dots, density = sample['im_id'], sample['image'], sample['boxes'], sample['dots'], sample['gt_density']
        sample = {'image': image, 'lines_boxes': boxes, 'gt_density': density}
        sample = TransformTrain(sample)
        image, boxes, gt_density = sample['image'], sample['boxes'], sample['gt_density']

        image = image.to(device)
        boxes = boxes.to(device)
        image.required_grad = False
        with torch.no_grad():
            features = extract_features(resnet50_conv, image.unsqueeze(0), boxes.unsqueeze(0), MAPS, Scales)
        features.required_grad = False
        pred_density = regressor(features)
        if pred_density.shape[2] != gt_density.shape[2] or pred_density.shape[3] != gt_density.shape[3]:
            orig_count = gt_density.sum().detach().item()
            gt_density = F.interpolate(gt_density, size=(pred_density.shape[2], pred_density.shape[3]), mode='bilinear')
            new_count = gt_density.sum().detach().item()
            if new_count > 0: gt_density = gt_density * (orig_count / new_count)

        gt_density = gt_density.to(device)
        counting_loss = counting_criterion(pred_density, gt_density)
        all_count_loss += counting_loss.item()
        pred_cnt = torch.sum(pred_density).item()
        gt_cnt = torch.sum(gt_density).item()
        cnt_err = abs(pred_cnt - gt_cnt)
        train_mae += cnt_err
        train_rmse += cnt_err ** 2
        optimizer.zero_grad()
        counting_loss.backward()
        optimizer.step()
    all_count_loss = all_count_loss / len(idx_list)
    train_mae = (train_mae / len(idx_list))
    train_rmse = (train_rmse / len(idx_list))**0.5
    return all_count_loss, train_mae, train_rmse

def val():
    idx_list = list(range(len(val_dataset)))
    random.shuffle(idx_list)
    batch_index = 0.
    val_mae = 0.
    val_rmse = 0.
    all_count_loss = 0.
    for idx in idx_list:
        batch_index += 1
        sample = val_dataset[idx]
        im_id, image, boxes, dots, density = sample['im_id'], sample['image'], sample['boxes'], sample['dots'], sample[
            'gt_density']
        sample = {'image': image, 'lines_boxes': boxes, 'gt_density': density}
        sample = TransformTrain(sample)
        image, boxes, gt_density = sample['image'].to(device), sample['boxes'].to(device), sample['gt_density'].to(device)

        with torch.no_grad():
            features = extract_features(resnet50_conv, image.unsqueeze(0), boxes.unsqueeze(0), MAPS, Scales)
            features.required_grad = False
            pred_density = regressor(features)
        if pred_density.shape[2] != gt_density.shape[2] or pred_density.shape[3] != gt_density.shape[3]:
            orig_count = gt_density.sum().detach().item()
            gt_density = F.interpolate(gt_density, size=(pred_density.shape[2], pred_density.shape[3]), mode='bilinear')
            new_count = gt_density.sum().detach().item()
            if new_count > 0: gt_density = gt_density * (orig_count / new_count)

        counting_loss = counting_criterion(pred_density.squeeze(), gt_density.squeeze())
        all_count_loss += counting_loss
        pred_cnt = torch.sum(pred_density).item()
        gt_cnt = torch.sum(gt_density).item()
        cnt_err = abs(pred_cnt - gt_cnt)
        val_mae += cnt_err
        val_rmse += cnt_err ** 2
    all_count_loss = all_count_loss / len(idx_list)
    val_mae = (val_mae / len(idx_list))
    val_rmse = (val_rmse / len(idx_list)) ** 0.5
    return all_count_loss, val_mae, val_rmse


Best_MAE = 1e6
Epoch = 150
for ep in range(Epoch):
    # Warm up
    print(ep)
    if (ep + 1) >= 5 and (ep + 1) % 5 == 0:
        for para_group in optimizer.param_groups:
            para_group['lr'] = para_group['lr'] * 0.5
    result[ep] = {}
    print('###################Train#####################')
    all_count_loss, train_mae, train_rmse = train()
    print('Train loss: ', all_count_loss)
    print('Train MAE: ', train_mae)
    print('Train RMSE: ', train_rmse)
    result[ep]['Train MAE'] = train_mae
    result[ep]['Train RMSE'] = train_rmse
    TBwriter.add_scalar('train/MAE', train_mae, ep + 1)
    TBwriter.add_scalar('train/RMSE', train_rmse, ep + 1)
    TBwriter.add_scalar('train/Count loss', all_count_loss, ep + 1)
    print('Evaluation:')
    all_count_loss, val_mae, val_rmse = val()
    print('Val MAE: ', val_mae)
    print('Val RMSE: ', val_rmse)
    result[ep]['Val MAE'] = val_mae
    result[ep]['Val RMSE'] = val_rmse
    TBwriter.add_scalar('val/MAE', val_mae, ep + 1)
    TBwriter.add_scalar('val/RMSE', val_rmse, ep + 1)
    if val_mae < Best_MAE:
        if not os.path.exists(log_save_dir):
            os.makedirs(log_save_dir)
        model_save_dir = os.path.join(log_save_dir, 'FSCD_LVIS_regressor.pth')
        torch.save(regressor.state_dict(), model_save_dir)
        Best_MAE = val_mae
    print('Best MAE:', Best_MAE)
    result_conf = OmegaConf.create(result)
    Result_save_path = os.path.join(log_save_dir, 'Result.yaml')
    save_cfg(result_conf, Result_save_path)