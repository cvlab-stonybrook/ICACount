import os
import sys
import copy
import tqdm
import torch
import random
import datetime
import argparse
import warnings
import numpy as np
from omegaconf import OmegaConf
from IPSE.utils import load_cfg, show_cfg

warnings.filterwarnings("ignore", "The parameter 'pretrained' is deprecated.*", category=UserWarning)
warnings.filterwarnings("ignore", "Arguments other than a weight enum or.*are deprecated.*", category=UserWarning)
warnings.filterwarnings("ignore", "Using a target size.*that is different to the input size.*", category=UserWarning)

parser = argparse.ArgumentParser(description="ICACount Experiment")
parser.add_argument('--split', type=str, default='test', required=False)
parser.add_argument('--gpu_id', type=int, default=0, required=False)
parser.add_argument('--dataset', type=str, default='fscdlvis', required=False)
args = parser.parse_args()
assert args.split in ['test', 'val'], 'Split not supported'
assert args.dataset in ['fsc147', 'fscdlvis'], 'Dataset not supported'

default_cfg = load_cfg('Configs/Default.yaml')
if args.dataset == 'fsc147':
    cfg = OmegaConf.merge(default_cfg, load_cfg('Configs/FamNet_fsc.yaml'))
else:
    cfg = OmegaConf.merge(default_cfg, load_cfg('Configs/FamNet_fscd.yaml'))
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
current_time = datetime.datetime.now()
experiment_date = current_time.strftime("%m-%d-%Y %H-%M")
exp_name = '[{}][{}][{}][{}]'.format(experiment_date, args.dataset, 'FamNet', args.split)
print(exp_name)

Inter_time = cfg.INTER_TIME
SEED_LIST = cfg.SEED_LIST
ADLR = cfg.ADLR
ADGS = cfg.ADGS
INLR = cfg.INLR
INGS = cfg.INGS
Root_dir = cfg.DATA_ROOT_DIR
Save_dir = cfg.LOG_DIR
Cp_dir = cfg.CP_DIR
if args.dataset == 'fsc147':
    Root_dir = os.path.join(Root_dir, 'FSC_147')
    Cp_dir = os.path.join(Cp_dir, 'FSC_147')
else:
    Root_dir = os.path.join(Root_dir, 'FSCD_LVIS')
    Cp_dir = os.path.join(Cp_dir, 'FSCD_LVIS')
assert os.path.exists(Root_dir), 'Root dir not exists'
if not os.path.exists(Save_dir):
    os.makedirs(Save_dir)

Final_mae = []
Final_rmse = []
for i in range(Inter_time + 1):
    Final_mae.append([])
    Final_rmse.append([])

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = 'cpu'

for seed in SEED_LIST:
    print('Current seed: ', seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_default_dtype(torch.float32)
    sys.path.append("..")
    import torch.optim as optim
    import torch.nn.functional as F
    from FamNet.Dataset import FscBgDataset, FSCD_LVIS_Dataset
    from FamNet.utils import extract_features, TransformTrain, MincountLoss, PerturbationLoss
    from FamNet.Model import Resnet50FPN, F_CountRegressor_CS
    from IPSE.IP_seg import VIS
    from IPSE.utils import get_uncertain_state, random_region_sample, save_cfg, interactive_loss_uncertain

    # Test Time Adaptation weight in learning to count everything
    weight_mincount = 1e-9
    weight_perturbation = 1e-4
    if args.dataset == 'fsc147':
        Test_Adaptation = True
    else:
        Test_Adaptation = False
    Model_dir = os.path.join(Cp_dir, 'FamNet.pth')
    if args.dataset == 'fsc147':
        val_dataset = FscBgDataset(Root_dir, 'val', False)
        test_dataset = FscBgDataset(Root_dir, 'test', False)
    else:
        val_dataset = FSCD_LVIS_Dataset(Root_dir, 'val')
        test_dataset = FSCD_LVIS_Dataset(Root_dir, 'test')

    if args.split == 'test':
        eval_dataset = test_dataset
    else:
        eval_dataset = val_dataset

    resnet50_conv = Resnet50FPN()
    resnet50_conv.to(device)
    regressor = F_CountRegressor_CS(6, pool='mean')
    regressor.load_state_dict(torch.load(Model_dir, map_location='cpu'), strict=False)
    regressor.to(device)
    resnet50_conv.eval()
    regressor.eval()

    def adapted_inference(Debug = False):
        inter_result = []
        for i in range(Inter_time + 1):
            inter_result.append([])
        MAPS = ['map3', 'map4']
        Scales = [0.9, 1.1]
        idx_list = list(range(len(eval_dataset)))
        cnt = 0
        total_adapt_time = []
        pbar = tqdm.tqdm(total=len(idx_list))
        for idx in idx_list:
            if Debug:
                print('Debug Mode!!')
            if Debug:
                if idx == 3:
                    break
            test_sample = eval_dataset[idx]
            im_id, image, boxes, dots, density = test_sample['im_id'], test_sample['image'], test_sample['boxes'], \
                                                 test_sample['dots'], test_sample[
                                                     'gt_density']
            sample = {'image': image, 'lines_boxes': boxes, 'gt_density': density}
            sample = TransformTrain(sample)
            image, boxes, gt_density = sample['image'], sample['boxes'], sample['gt_density']
            image = image.to(device)
            boxes = boxes.to(device)
            gt_density = gt_density.to(device)
            with torch.no_grad():
                features = extract_features(resnet50_conv, image.unsqueeze(0), boxes.unsqueeze(0), MAPS, Scales)
            adapted_regressor = copy.deepcopy(regressor)
            adapted_regressor.to(device)

            # Perform Test Time Adaptation
            if Test_Adaptation:
                adapted_regressor.train()
                optimizer = optim.Adam(adapted_regressor.parameters(), lr=ADLR)
                features.required_grad = True
                for step in range(0, ADGS):
                    optimizer.zero_grad()
                    output, _ = adapted_regressor(features)
                    lCount = weight_mincount * MincountLoss(output, boxes, device)
                    lPerturbation = weight_perturbation * PerturbationLoss(output, boxes, device, sigma=8)
                    loss = lCount + lPerturbation
                    if torch.is_tensor(loss):
                        loss.backward()
                        optimizer.step()

            features.required_grad = False
            output, simifeat = adapted_regressor(features)
            output = output.squeeze()
            pred_cnt = output.sum().item()
            if args.dataset == 'fsc147':
                gt_cnt = dots.shape[0]
            else:
                gt_cnt = len(dots)
            cnt = cnt + 1
            err = gt_cnt - pred_cnt
            inter_result[0].append(err)
            inter_error_result = []
            inter_adapt_time = []
            inter_error_result.append(np.abs(err))
            Result_dict[im_id] = {}
            Result_dict[im_id]['Error'] = []
            Result_dict[im_id]['Time'] = []

            inter_mask_list = []
            adapted_regressor.reset_refinement_module(features.shape[-2], features.shape[-1])
            adapted_regressor.to(device)
            for int_time in range(Inter_time):
                # Inference
                features.required_grad = False
                output = adapted_regressor.inter_inference(simifeat)
                # VIS
                density = output.squeeze().detach().cpu().numpy()
                visual = VIS(density)
                visual.solve()
                label = visual.Llabel

                # Sample Region Randomly
                sample_label = random_region_sample(label)
                inter_mask = np.zeros((label.shape[0], label.shape[1]), dtype=np.uint8)
                inter_mask[label == sample_label] = 1
                inter_mask = torch.from_numpy(inter_mask).to(device)
                inter_mask_list.append(inter_mask)
                if int_time < 1:
                    gt_density = F.interpolate(gt_density, size=(inter_mask.shape[0], inter_mask.shape[1]),
                                               mode='bilinear').squeeze()
                over_counting_num = 0
                under_counting_num = 0
                for inmask in inter_mask_list:
                    uncertain_state = get_uncertain_state(output, gt_density, inmask)
                    if uncertain_state == 1:
                        over_counting_num += 1
                    elif uncertain_state == -1:
                        under_counting_num += 1
                # First term
                scale_1 = min(1, np.exp(((int_time + 1) - 3) / 2))

                # Second term
                if over_counting_num == 0 or under_counting_num == 0:
                    scale_2 = 1
                else:
                    over_p = over_counting_num / (over_counting_num + under_counting_num)
                    under_p = under_counting_num / (over_counting_num + under_counting_num)
                    uncertain = (over_p * np.log(over_p)) + (under_p * np.log(under_p))
                    scale_2 = 1 + uncertain
                scale = (scale_1 + scale_2) / 2
                scale_INLR = INLR * scale
                scale_INGS = np.rint(INGS / scale).astype(np.int32)

                optimizer_inter = optim.Adam(
                    [adapted_regressor.ch_scale, adapted_regressor.ch_bias, adapted_regressor.sp_scale,
                     adapted_regressor.sp_bias], lr=scale_INLR)
                features.required_grad = True
                adaptation_start_time = datetime.datetime.now()
                for step in range(0, scale_INGS):
                    optimizer_inter.zero_grad()
                    output = adapted_regressor.inter_inference(simifeat)
                    output = output.squeeze()

                    # Local Adaptation loss
                    local_region_loss = 0.
                    for inmask in inter_mask_list:
                        inter_loss, uncertain_state = interactive_loss_uncertain(output, gt_density, inmask)
                        local_region_loss += inter_loss

                    # Global Adaptation Loss
                    all_inter_mask = np.zeros((label.shape[0], label.shape[1]), dtype=np.uint8)
                    for int_mask in inter_mask_list:
                        all_inter_mask += int_mask.cpu().numpy()
                    all_inter_mask = torch.from_numpy(all_inter_mask).to(device)
                    new_count_limit = 4 * len(inter_mask_list)
                    global_region_loss, _ = interactive_loss_uncertain(output, gt_density, all_inter_mask,
                                                                       new_count_limit)

                    inertial_loss = ((adapted_regressor.ch_scale - 1) ** 2).sum() + (
                            adapted_regressor.ch_bias ** 2).sum() + (
                                            (adapted_regressor.sp_scale - 1) ** 2).sum() + (
                                            adapted_regressor.sp_bias ** 2).sum()
                    inter_loss = 0.5 * local_region_loss + 0.5 * global_region_loss + 1e-3 * inertial_loss
                    if torch.is_tensor(inter_loss):
                        inter_loss.backward()
                        optimizer_inter.step()
                adaptation_end_time = datetime.datetime.now()
                adapt_time = (adaptation_end_time - adaptation_start_time).total_seconds()
                Result_dict[im_id]['Time'].append(adapt_time)
                inter_adapt_time.append(adapt_time)
                features.required_grad = False
                output = adapted_regressor.inter_inference(simifeat)
                pred_cnt = output.sum().item()
                if args.dataset == 'fsc147':
                    gt_cnt = dots.shape[0]
                else:
                    gt_cnt = len(dots)
                cnt = cnt + 1
                err = gt_cnt - pred_cnt
                inter_result[int_time + 1].append(err)
                Result_dict[im_id]['Error'].append(np.abs(err).item())
                inter_error_result.append(np.abs(err))
            assert len(inter_mask_list) == Inter_time
            total_time = sum(inter_adapt_time)
            total_adapt_time.append(total_time)
            #print(idx, '/', len(idx_list), total_time,  inter_error_result)
            pbar.set_postfix(idx=str(im_id), inter_error=",".join(map(str, inter_error_result)))
            pbar.update(1)
        pbar.close()
        return inter_result, total_adapt_time

    Result_dict = {}
    Result_dict['FinalMAE'] = []
    Result_dict['FinalRMSE'] = []
    Test_Adaptation = True
    show_detail = False

    Inter_result, total_adapt_time = adapted_inference()
    for inter_time in range(Inter_time + 1):
        image_errs = Inter_result[inter_time]
        image_errs = np.array(image_errs)
        mse = np.sqrt(np.mean(np.square(image_errs)))
        mae = np.mean(np.abs(image_errs))
        Result_dict['FinalMAE'].append(mae.item())
        Result_dict['FinalRMSE'].append(mse.item())
        Final_mae[inter_time].append(mae)
        Final_rmse[inter_time].append(mse)
        print('mae {}, mse {}\n'.format(mae, mse))
    avg_adapt_time = sum(total_adapt_time) / len(total_adapt_time)
    print('Average adaptation time: ', avg_adapt_time)
    Result_dict['AvgAdaptTime'] = avg_adapt_time
    res_save_dir = os.path.join(Save_dir, exp_name)
    result_conf = OmegaConf.create(Result_dict)
    if not os.path.exists(res_save_dir):
        os.makedirs(res_save_dir)
    Result_save_path = os.path.join(res_save_dir, str(seed) + '_Result.yaml')
    save_cfg(result_conf, Result_save_path)

Final_Result_dict = {}
Final_Result_dict['FinalMAE'] = []
Final_Result_dict['FinalRMSE'] = []
for inter_time in range(Inter_time + 1):
    avg_mae = sum(Final_mae[inter_time]) / len(Final_mae[inter_time])
    avg_rmse = sum(Final_rmse[inter_time]) / len(Final_rmse[inter_time])
    Final_Result_dict['FinalMAE'].append(avg_mae.item())
    Final_Result_dict['FinalRMSE'].append(avg_rmse.item())
    print('Interaction: {}, mae {}, mse {}\n'.format(inter_time, avg_mae, avg_rmse))
    res_save_dir = os.path.join(Save_dir, exp_name)
    result_conf = OmegaConf.create(Final_Result_dict)
    if not os.path.exists(res_save_dir):
        os.makedirs(res_save_dir)
    Result_save_path = os.path.join(res_save_dir, 'Overall_Result.yaml')
    save_cfg(result_conf, Result_save_path)