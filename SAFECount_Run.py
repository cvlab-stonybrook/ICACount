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

cfg = load_cfg('./Configs/SAFECount.yaml')
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
current_time = datetime.datetime.now()
experiment_date = current_time.strftime("%m-%d-%Y %H-%M")
exp_name = '[{}][{}][{}][{}]'.format(experiment_date, args.dataset, 'SAFECount', args.split)
print(exp_name)

Inter_time = cfg.INTER_TIME
SEED_LIST = cfg.SEED_LIST
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
    print('Current seed:', seed)
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
    from easydict import EasyDict
    import yaml
    import datetime
    from IPSE.IP_seg import VIS
    import torch.optim as optim
    import torch.nn.functional as F
    from IPSE.utils import (
        interactive_loss,
        random_region_sample,
        save_cfg,
        interactive_loss_uncertain,
        get_uncertain_state,
    )
    from SAFECount.safecount import MySafecount, MySafecount_CS
    from SAFECount.utils.misc_helper import (
        create_logger,
        get_current_time,
        load_state,
        save_checkpoint,
        set_random_seed,
        to_device,
    )
    from SAFECount.datasets.data_builder import build_dataloader
    if args.dataset == 'fsc147':
        cfg_dir = './SAFECount/FSC147.yaml'
    else:
        cfg_dir = './SAFECount/FSCD_LVIS.yaml'
    with open(cfg_dir) as f:
        config = EasyDict(yaml.load(f, Loader=yaml.FullLoader))
    if args.dataset == 'fsc147':
        config['dataset']['img_dir'] = os.path.join(Root_dir, 'images_384_VarV2')
        config['dataset']['density_dir'] = os.path.join(Root_dir, 'gt_density_map_adaptive_384_VarV2')
    if args.dataset == 'fscdlvis':
        config['dataset']['fscdlvis_root_dir'] = Root_dir
    model = MySafecount_CS(config)
    if args.dataset == 'fsc147':
        checkpoint = torch.load(os.path.join(Cp_dir, 'SAFECount.pth.tar'), map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'], strict=False)
    else:
        checkpoint = torch.load(os.path.join(Cp_dir, 'SAFECount.pth'), map_location='cpu')
        model.load_state_dict(checkpoint, strict=False)

    train_loader, val_loader, test_loader = build_dataloader(config.dataset, distributed=False)
    model.eval()
    model.to(device)

    if args.split == 'test':
        eval_loader = test_loader
    else:
        eval_loader = val_loader

    def adapted_inference(Debug=False):
        inter_result = [[], [], [], [], [], []]
        cnt = 0
        total_adapt_time = []
        pbar = tqdm.tqdm(total=len(eval_loader))
        for idx, sample in enumerate(eval_loader):
            if Debug:
                print('Debug Mode!!')
            if Debug:
                if idx == 3:
                    break
            model.module.reset_refine_module()
            model.to(device)
            sample = to_device(sample, device=torch.device(device))
            output, refine_feat = model(sample)
            output = output.squeeze()
            pred_cnt = output.sum().item()
            gt_density = sample["density"]
            gt_cnt = torch.sum(gt_density).item()
            cnt = cnt + 1
            err = gt_cnt - pred_cnt
            inter_result[0].append(err)
            inter_error_result = []
            inter_adapt_time = []
            inter_error_result.append(np.abs(err))
            im_id = sample['filename'][0]
            Result_dict[im_id] = {}
            Result_dict[im_id]['Error'] = []
            Result_dict[im_id]['Time'] = []
            inter_mask_list = []
            for int_time in range(Inter_time):
                output = model.module.inter_inference(refine_feat)
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
                optimizer_inter = optim.Adam([model.module.ch_scale,
                                              model.module.ch_bias,
                                              model.module.sp_scale,
                                              model.module.sp_bias], lr=scale_INLR)
                adaptation_start_time = datetime.datetime.now()
                for step in range(0, scale_INGS):
                    optimizer_inter.zero_grad()
                    output = model.module.inter_inference(refine_feat)
                    output = output.squeeze()

                    # Local Adaptation loss
                    local_region_loss = 0.
                    over_counting_num = 0
                    under_counting_num = 0
                    for inmask in inter_mask_list:
                        inter_loss, uncertain_state = interactive_loss_uncertain(output, gt_density, inmask)
                        local_region_loss += inter_loss
                        if uncertain_state == 1:
                            over_counting_num += 1
                        elif uncertain_state == -1:
                            under_counting_num += 1

                    # Global Adaptation Loss
                    all_inter_mask = np.zeros((label.shape[0], label.shape[1]), dtype=np.uint8)
                    large_region_num = 0
                    for int_mask in inter_mask_list:
                        if (gt_density * int_mask).sum() <= 4:
                            all_inter_mask += int_mask.cpu().numpy()
                            large_region_num += 1
                    all_inter_mask = torch.from_numpy(all_inter_mask).to(device)
                    new_count_limit = 4 * len(inter_mask_list)
                    global_region_loss, _ = interactive_loss_uncertain(output, gt_density, all_inter_mask,
                                                                       new_count_limit)

                    inertial_loss = ((adapted_regressor.ch_scale - 1) ** 2).sum() + (
                            adapted_regressor.ch_bias ** 2).sum() + (
                                            (adapted_regressor.sp_scale - 1) ** 2).sum() + (
                                            adapted_regressor.sp_bias ** 2).sum()
                    region_num = len(inter_mask_list)
                    inter_loss = region_num / (
                                large_region_num + region_num) * local_region_loss + large_region_num / (
                                             large_region_num + region_num) * global_region_loss + 1e-3 * inertial_loss
                    if torch.is_tensor(inter_loss):
                        inter_loss.backward()
                        optimizer_inter.step()
                adaptation_end_time = datetime.datetime.now()
                adapt_time = (adaptation_end_time - adaptation_start_time).total_seconds()
                Result_dict[im_id]['Time'].append(adapt_time)
                inter_adapt_time.append(adapt_time)
                output = model.module.inter_inference(refine_feat)
                pred_cnt = output.sum().item()
                density = sample["density"]
                gt_cnt = torch.sum(density).item()
                cnt = cnt + 1
                err = gt_cnt - pred_cnt
                inter_result[int_time + 1].append(err)
                Result_dict[im_id]['Error'].append(np.abs(err).item())
                inter_error_result.append(np.abs(err))

            assert len(inter_mask_list) == Inter_time
            total_time = sum(inter_adapt_time)
            total_adapt_time.append(total_time)
            #print(idx, '/', len(eval_loader), total_time, inter_error_result)
            pbar.set_postfix(idx=str(im_id), inter_error=",".join(map(str, inter_error_result)), refresh=True)
            pbar.update(1)
        return inter_result, total_adapt_time


    Result_dict = {}
    Result_dict['FinalMAE'] = []
    Result_dict['FinalRMSE'] = []

    Inter_result, total_adapt_time = adapted_inference()
    for inter_time in range(Inter_time + 1):
        image_errs = Inter_result[inter_time]
        image_errs = np.array(image_errs)
        mse = np.sqrt(np.mean(np.square(image_errs)))
        mae = np.mean(np.abs(image_errs))
        Final_mae[inter_time].append(mae)
        Final_rmse[inter_time].append(mse)
        Result_dict['FinalMAE'].append(mae.item())
        Result_dict['FinalRMSE'].append(mse.item())
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