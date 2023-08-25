'''
Model(ResNet50 Feature Extractor and Regressor)
By Yifeng Huang(yifehuang@cs.stonybrook.edu)
Based on Viresh and Minh's code
Last Modified 2022.2.14
'''
import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import copy
import numpy as np

#Normal Model
class Resnet50FPN(nn.Module):
    def __init__(self):
        super(Resnet50FPN, self).__init__()
        self.resnet = torchvision.models.resnet50(pretrained=True)
        children = list(self.resnet.children())
        self.conv1 = nn.Sequential(*children[:4])
        self.conv2 = children[4]
        self.conv3 = children[5]
        self.conv4 = children[6]
    def forward(self, im_data):
        feat = OrderedDict()
        feat_map = self.conv1(im_data)
        feat_map = self.conv2(feat_map)
        feat_map3 = self.conv3(feat_map)
        feat_map4 = self.conv4(feat_map3)
        feat['map3'] = feat_map3
        feat['map4'] = feat_map4
        return feat

class CountRegressor(nn.Module):
    def __init__(self, input_channels,pool='mean'):
        super(CountRegressor, self).__init__()
        self.pool = pool
        self.regressor = nn.Sequential(
            nn.Conv2d(input_channels, 196, 7, padding=3),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(196, 128, 5, padding=2),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(64, 32, 1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 1),
            nn.ReLU(),
        )

    def forward(self, im):
        output = self.regressor(im.squeeze(0))
        if self.pool == 'mean':
            output = torch.mean(output, dim=(0),keepdim=True)
            return output
        elif self.pool == 'max':
            output, _ = torch.max(output, 0,keepdim=True)
            return output

class BgCountRegressor(nn.Module):
    def __init__(self, input_channels,pool='mean'):
        super(BgCountRegressor, self).__init__()
        self.pool = pool
        self.Shared_regressor = nn.Sequential(
            nn.Conv2d(input_channels, 196, 7, padding=3),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(196, 128, 5, padding=2),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
        )

        self.count = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(64, 32, 1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 1),
            nn.ReLU(),
        )

        self.bg = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(64, 32, 1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, im):
        output = self.Shared_regressor(im.squeeze(0))
        count_output = self.count(output)
        bg_output = self.bg(output)
        if self.pool == 'mean':
            count_output = torch.mean(count_output, dim=(0),keepdim=True)
            bg_output = torch.mean(bg_output, dim=(0),keepdim=True)
            return count_output, bg_output
        elif self.pool == 'max':
            count_output, _ = torch.max(count_output, 0,keepdim=True)
            bg_output, _ = torch.max(bg_output, 0, keepdim=True)
            return count_output, bg_output

class FixedCountRegressor(nn.Module):
  def __init__(self, input_channels, regressor):
    super(FixedCountRegressor, self).__init__()
    self.regressor = copy.deepcopy(regressor.regressor[0:9])
  def forward(self, im):
    output = self.regressor(im.squeeze(0))
    return output

class AdaptedCountRegressor(nn.Module):
  def __init__(self, input_channels, regressor):
    super(AdaptedCountRegressor, self).__init__()
    self.regressor = copy.deepcopy(regressor.regressor[9:13])
  def forward(self, im):
    output = self.regressor(im)
    output = torch.mean(output, dim=(0),keepdim=True)
    return output

def weights_normal_init(model, dev=0.01):
    if isinstance(model, list):
        for m in model:
            weights_normal_init(m, dev)
    else:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, dev)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, dev)


def weights_xavier_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('relu'))
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

class F_CountRegressor(nn.Module):
    def __init__(self, input_channels,pool='mean'):
        super(F_CountRegressor, self).__init__()
        self.pool = pool
        self.regressor = nn.Sequential(
            nn.Conv2d(input_channels, 196, 7, padding=3),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(196, 128, 5, padding=2),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(64, 32, 1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 1),
            nn.ReLU(),
        )
        self.ch_scale = torch.nn.Parameter(torch.Tensor(np.ones(196)), requires_grad=True)
        self.ch_bias = torch.nn.Parameter(torch.Tensor(np.zeros(196)), requires_grad=True)

    def forward(self, im):
      with torch.no_grad():
        simi = im.squeeze(0)
        for i in range(3):
          simi = self.regressor[i](simi)
      simi.required_grad = True
      refine_simi = simi * self.ch_scale.view(1, -1, 1, 1) + self.ch_bias.view(1, -1, 1, 1)
      for i in range(3, len(self.regressor)):
        refine_simi = self.regressor[i](refine_simi)
      if self.pool == 'mean':
        output = torch.mean(refine_simi, dim=(0),keepdim=True)
        return output
      elif self.pool == 'max':
        output, _ = torch.max(refine_simi, 0,keepdim=True)
        return output

class F_CountRegressor_SC(nn.Module):
    def __init__(self, input_channels, pool='mean'):
        super(F_CountRegressor_SC, self).__init__()
        self.pool = pool
        self.regressor = nn.Sequential(
            nn.Conv2d(input_channels, 196, 7, padding=3),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(196, 128, 5, padding=2),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(64, 32, 1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 1),
            nn.ReLU(),
        )

    def reset_refinement_module(self, height, width):
        self.ch_scale = torch.nn.Parameter(torch.Tensor(np.ones(196)), requires_grad=True)
        self.ch_bias = torch.nn.Parameter(torch.Tensor(np.zeros(196)), requires_grad=True)
        self.sp_scale = torch.nn.Parameter(torch.Tensor(np.ones((2 * height, 2 * width))),
                                           requires_grad=True)
        self.sp_bias = torch.nn.Parameter(torch.Tensor(np.zeros((2 * height, 2 * width))),
                                          requires_grad=True)

    def inter_inference(self, simifeat):
        simifeat.required_grad = True
        refine_simi = (simifeat * self.sp_scale.unsqueeze(0).unsqueeze(0) + self.sp_bias.unsqueeze(0).unsqueeze(0)) \
                      * self.ch_scale.view(1, -1, 1, 1) + self.ch_bias.view(1, -1, 1, 1)
        for i in range(3, len(self.regressor)):
            refine_simi = self.regressor[i](refine_simi)
        if self.pool == 'mean':
            output = torch.mean(refine_simi, dim=(0),keepdim=True)
            return output
        elif self.pool == 'max':
            output, _ = torch.max(refine_simi, 0,keepdim=True)
            return output

    def forward(self, im):
        simi = im.squeeze(0)
        for i in range(3):
            simi = self.regressor[i](simi)
        simi_feat = copy.deepcopy(simi.detach())
        for i in range(3, len(self.regressor)):
            simi = self.regressor[i](simi)
        if self.pool == 'mean':
            output = torch.mean(simi, dim=(0),keepdim=True)
            return output, simi_feat
        elif self.pool == 'max':
            output, _ = torch.max(simi, 0,keepdim=True)
            return output, simi_feat

class F_CountRegressor_S(nn.Module):
    def __init__(self, input_channels, pool='mean'):
        super(F_CountRegressor_S, self).__init__()
        self.pool = pool
        self.regressor = nn.Sequential(
            nn.Conv2d(input_channels, 196, 7, padding=3),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(196, 128, 5, padding=2),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(64, 32, 1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 1),
            nn.ReLU(),
        )

    def reset_refinement_module(self, height, width):
        self.sp_scale = torch.nn.Parameter(torch.Tensor(np.ones((2 * height, 2 * width))),
                                           requires_grad=True)
        self.sp_bias = torch.nn.Parameter(torch.Tensor(np.zeros((2 * height, 2 * width))),
                                          requires_grad=True)

    def inter_inference(self, simifeat):
        simifeat.required_grad = True
        refine_simi = simifeat * self.sp_scale.unsqueeze(0).unsqueeze(0) + self.sp_bias.unsqueeze(0).unsqueeze(0)
        for i in range(3, len(self.regressor)):
            refine_simi = self.regressor[i](refine_simi)
        if self.pool == 'mean':
            output = torch.mean(refine_simi, dim=(0),keepdim=True)
            return output
        elif self.pool == 'max':
            output, _ = torch.max(refine_simi, 0,keepdim=True)
            return output

    def forward(self, im):
        simi = im.squeeze(0)
        for i in range(3):
            simi = self.regressor[i](simi)
        simi_feat = copy.deepcopy(simi.detach())
        for i in range(3, len(self.regressor)):
            simi = self.regressor[i](simi)
        if self.pool == 'mean':
            output = torch.mean(simi, dim=(0),keepdim=True)
            return output, simi_feat
        elif self.pool == 'max':
            output, _ = torch.max(simi, 0,keepdim=True)
            return output, simi_feat


class F_CountRegressor_C(nn.Module):
    def __init__(self, input_channels, pool='mean'):
        super(F_CountRegressor_C, self).__init__()
        self.pool = pool
        self.regressor = nn.Sequential(
            nn.Conv2d(input_channels, 196, 7, padding=3),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(196, 128, 5, padding=2),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(64, 32, 1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 1),
            nn.ReLU(),
        )

    def reset_refinement_module(self, height = None, width = None):
        self.ch_scale = torch.nn.Parameter(torch.Tensor(np.ones(196)), requires_grad=True)
        self.ch_bias = torch.nn.Parameter(torch.Tensor(np.zeros(196)), requires_grad=True)

    def inter_inference(self, simifeat):
        simifeat.required_grad = True
        refine_simi = (simifeat * self.ch_scale.view(1, -1, 1, 1) + self.ch_bias.view(1, -1, 1,1))
        for i in range(3, len(self.regressor)):
            refine_simi = self.regressor[i](refine_simi)
        if self.pool == 'mean':
            output = torch.mean(refine_simi, dim=(0),keepdim=True)
            return output
        elif self.pool == 'max':
            output, _ = torch.max(refine_simi, 0,keepdim=True)
            return output

    def forward(self, im):
        simi = im.squeeze(0)
        for i in range(3):
            simi = self.regressor[i](simi)
        simi_feat = copy.deepcopy(simi.detach())
        for i in range(3, len(self.regressor)):
            simi = self.regressor[i](simi)
        if self.pool == 'mean':
            output = torch.mean(simi, dim=(0),keepdim=True)
            return output, simi_feat
        elif self.pool == 'max':
            output, _ = torch.max(simi, 0,keepdim=True)
            return output, simi_feat


class F_CountRegressor_CS(nn.Module):
    def __init__(self, input_channels, pool='mean'):
        super(F_CountRegressor_CS, self).__init__()
        self.pool = pool
        self.regressor = nn.Sequential(
            nn.Conv2d(input_channels, 196, 7, padding=3),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(196, 128, 5, padding=2),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(64, 32, 1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 1),
            nn.ReLU(),
        )

    def reset_refinement_module(self, height, width):
        self.ch_scale = torch.nn.Parameter(torch.Tensor(np.ones(196)), requires_grad=True)
        self.ch_bias = torch.nn.Parameter(torch.Tensor(np.zeros(196)), requires_grad=True)
        self.sp_scale = torch.nn.Parameter(torch.Tensor(np.ones((2 * height, 2 * width))),
                                           requires_grad=True)
        self.sp_bias = torch.nn.Parameter(torch.Tensor(np.zeros((2 * height, 2 * width))),
                                          requires_grad=True)

    def inter_inference(self, simifeat):
        simifeat.required_grad = True
        refine_simi = (simifeat * self.ch_scale.view(1, -1, 1, 1) + self.ch_bias.view(1, -1, 1,1)) \
                      * self.sp_scale.unsqueeze(0).unsqueeze(0) + self.sp_bias.unsqueeze(0).unsqueeze(0)
        for i in range(3, len(self.regressor)):
            refine_simi = self.regressor[i](refine_simi)
        if self.pool == 'mean':
            output = torch.mean(refine_simi, dim=(0),keepdim=True)
            return output
        elif self.pool == 'max':
            output, _ = torch.max(refine_simi, 0,keepdim=True)
            return output

    def forward(self, im):
        simi = im.squeeze(0)
        for i in range(3):
            simi = self.regressor[i](simi)
        simi_feat = copy.deepcopy(simi.detach())
        for i in range(3, len(self.regressor)):
            simi = self.regressor[i](simi)
        if self.pool == 'mean':
            output = torch.mean(simi, dim=(0),keepdim=True)
            return output, simi_feat
        elif self.pool == 'max':
            output, _ = torch.max(simi, 0,keepdim=True)
            return output, simi_feat

class F_CountRegressor_NLCS(nn.Module):
    def __init__(self, input_channels, pool='mean'):
        super(F_CountRegressor_NLCS, self).__init__()
        self.pool = pool
        self.regressor = nn.Sequential(
            nn.Conv2d(input_channels, 196, 7, padding=3),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(196, 128, 5, padding=2),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(64, 32, 1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 1),
            nn.ReLU(),
        )

    def reset_refinement_module(self, height, width, act = None):
        self.ch_scale = torch.nn.Parameter(torch.Tensor(np.zeros(196)), requires_grad=True)
        self.ch_bias = torch.nn.Parameter(torch.Tensor(np.zeros(196)), requires_grad=True)
        self.sp_scale = torch.nn.Parameter(torch.Tensor(np.zeros((2 * height, 2 * width))),
                                           requires_grad=True)
        self.sp_bias = torch.nn.Parameter(torch.Tensor(np.zeros((2 * height, 2 * width))),
                                          requires_grad=True)
        if act == 'ReLU':
            self.refine_act = nn.ReLU()
        else:
            self.refine_act = nn.LeakyReLU(0.3)

    def inter_inference(self, simifeat):
        simifeat.required_grad = True
        refine_simi = simifeat + self.refine_act((simifeat * self.ch_scale.view(1, -1, 1, 1) + self.ch_bias.view(1, -1, 1,1)) \
                      * self.sp_scale.unsqueeze(0).unsqueeze(0) + self.sp_bias.unsqueeze(0).unsqueeze(0))
        for i in range(3, len(self.regressor)):
            refine_simi = self.regressor[i](refine_simi)
        if self.pool == 'mean':
            output = torch.mean(refine_simi, dim=(0),keepdim=True)
            return output
        elif self.pool == 'max':
            output, _ = torch.max(refine_simi, 0,keepdim=True)
            return output

    def forward(self, im):
        simi = im.squeeze(0)
        for i in range(3):
            simi = self.regressor[i](simi)
        simi_feat = copy.deepcopy(simi.detach())
        for i in range(3, len(self.regressor)):
            simi = self.regressor[i](simi)
        if self.pool == 'mean':
            output = torch.mean(simi, dim=(0),keepdim=True)
            return output, simi_feat
        elif self.pool == 'max':
            output, _ = torch.max(simi, 0,keepdim=True)
            return output, simi_feat

class F_CountRegressor_NLS(nn.Module):
    def __init__(self, input_channels, pool='mean'):
        super(F_CountRegressor_NLS, self).__init__()
        self.pool = pool
        self.regressor = nn.Sequential(
            nn.Conv2d(input_channels, 196, 7, padding=3),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(196, 128, 5, padding=2),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(64, 32, 1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 1),
            nn.ReLU(),
        )

    def reset_refinement_module(self, height, width):
        self.sp_scale = torch.nn.Parameter(torch.Tensor(np.zeros((2 * height, 2 * width))),
                                           requires_grad=True)
        self.sp_bias = torch.nn.Parameter(torch.Tensor(np.zeros((2 * height, 2 * width))),
                                          requires_grad=True)
        self.refine_act = nn.LeakyReLU(0.3)

    def inter_inference(self, simifeat):
        simifeat.required_grad = True
        refine_simi = simifeat + self.refine_act(simifeat * self.sp_scale.unsqueeze(0).unsqueeze(0) + self.sp_bias.unsqueeze(0).unsqueeze(0))
        for i in range(3, len(self.regressor)):
            refine_simi = self.regressor[i](refine_simi)
        if self.pool == 'mean':
            output = torch.mean(refine_simi, dim=(0),keepdim=True)
            return output
        elif self.pool == 'max':
            output, _ = torch.max(refine_simi, 0,keepdim=True)
            return output

    def forward(self, im):
        simi = im.squeeze(0)
        for i in range(3):
            simi = self.regressor[i](simi)
        simi_feat = copy.deepcopy(simi.detach())
        for i in range(3, len(self.regressor)):
            simi = self.regressor[i](simi)
        if self.pool == 'mean':
            output = torch.mean(simi, dim=(0),keepdim=True)
            return output, simi_feat
        elif self.pool == 'max':
            output, _ = torch.max(simi, 0,keepdim=True)
            return output, simi_feat

class F_CountRegressor_NLC(nn.Module):
    def __init__(self, input_channels, pool='mean'):
        super(F_CountRegressor_NLC, self).__init__()
        self.pool = pool
        self.regressor = nn.Sequential(
            nn.Conv2d(input_channels, 196, 7, padding=3),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(196, 128, 5, padding=2),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(64, 32, 1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 1),
            nn.ReLU(),
        )

    def reset_refinement_module(self, height, width):
        self.ch_scale = torch.nn.Parameter(torch.Tensor(np.zeros(196)), requires_grad=True)
        self.ch_bias = torch.nn.Parameter(torch.Tensor(np.zeros(196)), requires_grad=True)
        self.refine_act = nn.LeakyReLU(0.3)

    def inter_inference(self, simifeat):
        simifeat.required_grad = True
        refine_simi = simifeat + self.refine_act((simifeat * self.ch_scale.view(1, -1, 1, 1) + self.ch_bias.view(1, -1, 1,1)))
        for i in range(3, len(self.regressor)):
            refine_simi = self.regressor[i](refine_simi)
        if self.pool == 'mean':
            output = torch.mean(refine_simi, dim=(0),keepdim=True)
            return output
        elif self.pool == 'max':
            output, _ = torch.max(refine_simi, 0,keepdim=True)
            return output

    def forward(self, im):
        simi = im.squeeze(0)
        for i in range(3):
            simi = self.regressor[i](simi)
        simi_feat = copy.deepcopy(simi.detach())
        for i in range(3, len(self.regressor)):
            simi = self.regressor[i](simi)
        if self.pool == 'mean':
            output = torch.mean(simi, dim=(0),keepdim=True)
            return output, simi_feat
        elif self.pool == 'max':
            output, _ = torch.max(simi, 0,keepdim=True)
            return output, simi_feat

class F_CountRegressor_CS_raw(nn.Module):
    def __init__(self, input_channels, pool='mean'):
        super(F_CountRegressor_CS_raw, self).__init__()
        self.pool = pool
        self.regressor = nn.Sequential(
            nn.Conv2d(input_channels, 196, 7, padding=3),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(196, 128, 5, padding=2),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(64, 32, 1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 1),
            nn.ReLU(),
        )

    def reset_refinement_module(self, height, width):
        self.ch_scale = torch.nn.Parameter(torch.Tensor(np.ones(6)), requires_grad=True)
        self.ch_bias = torch.nn.Parameter(torch.Tensor(np.zeros(6)), requires_grad=True)
        self.sp_scale = torch.nn.Parameter(torch.Tensor(np.ones((height, width))),
                                           requires_grad=True)
        self.sp_bias = torch.nn.Parameter(torch.Tensor(np.zeros((height, width))),
                                          requires_grad=True)

    def inter_inference(self, simifeat):
        simifeat.required_grad = True
        refine_simi = (simifeat * self.ch_scale.view(1, -1, 1, 1) + self.ch_bias.view(1, -1, 1,1)) \
                      * self.sp_scale.unsqueeze(0).unsqueeze(0) + self.sp_bias.unsqueeze(0).unsqueeze(0)
        for i in range(0, len(self.regressor)):
            refine_simi = self.regressor[i](refine_simi)
        if self.pool == 'mean':
            output = torch.mean(refine_simi, dim=(0),keepdim=True)
            return output
        elif self.pool == 'max':
            output, _ = torch.max(refine_simi, 0,keepdim=True)
            return output

    def forward(self, im):
        simi = im.squeeze(0)
        for i in range(0):
            simi = self.regressor[i](simi)
        simi_feat = copy.deepcopy(simi.detach())
        for i in range(0, len(self.regressor)):
            simi = self.regressor[i](simi)
        if self.pool == 'mean':
            output = torch.mean(simi, dim=(0),keepdim=True)
            return output, simi_feat
        elif self.pool == 'max':
            output, _ = torch.max(simi, 0,keepdim=True)
            return output, simi_feat

class F_CountRegressor_CS_exem(nn.Module):
    def __init__(self, input_channels , exem_num, height, width, pool='mean'):
        super(F_CountRegressor_CS_exem, self).__init__()
        self.pool = pool
        self.regressor = nn.Sequential(
            nn.Conv2d(input_channels, 196, 7, padding=3),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(196, 128, 5, padding=2),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(64, 32, 1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 1),
            nn.ReLU(),
        )
        self.exem_num = exem_num
        self.ch_scale = torch.nn.Parameter(torch.Tensor(np.ones((exem_num, 196))), requires_grad=True)
        self.ch_bias = torch.nn.Parameter(torch.Tensor(np.zeros((exem_num, 196))), requires_grad=True)
        self.sp_scale = torch.nn.Parameter(torch.Tensor(np.ones((exem_num, 2 * height, 2 * width))),
                                           requires_grad=True)
        self.sp_bias = torch.nn.Parameter(torch.Tensor(np.zeros((exem_num, 2 * height, 2 * width))),
                                          requires_grad=True)

    def forward(self, im):
      with torch.no_grad():
        simi = im.squeeze(0)
        for i in range(3):
          simi = self.regressor[i](simi)

      simi.required_grad = True
      refine_simi = (simi * self.ch_scale.view(self.exem_num, 196, 1, 1) + self.ch_bias.view(self.exem_num, 196, 1, 1)) * self.sp_scale.unsqueeze(1) + self.sp_bias.unsqueeze(1)
      for i in range(3, len(self.regressor)):
        refine_simi = self.regressor[i](refine_simi)
      if self.pool == 'mean':
        output = torch.mean(refine_simi, dim=(0),keepdim=True)
        return output
      elif self.pool == 'max':
        output, _ = torch.max(refine_simi, 0,keepdim=True)
        return output

class F_CountRegressor_CS_128(nn.Module):
    def __init__(self, input_channels, pool='mean'):
        super(F_CountRegressor_CS_128, self).__init__()
        self.pool = pool
        self.regressor = nn.Sequential(
            nn.Conv2d(input_channels, 196, 7, padding=3),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(196, 128, 5, padding=2),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(64, 32, 1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 1),
            nn.ReLU(),
        )

    def reset_refinement_module(self, height, width):
        self.ch_scale = torch.nn.Parameter(torch.Tensor(np.ones(128)), requires_grad=True)
        self.ch_bias = torch.nn.Parameter(torch.Tensor(np.zeros(128)), requires_grad=True)
        self.sp_scale = torch.nn.Parameter(torch.Tensor(np.ones((4 * height, 4 * width))),
                                           requires_grad=True)
        self.sp_bias = torch.nn.Parameter(torch.Tensor(np.zeros((4 * height, 4 * width))),
                                          requires_grad=True)

    def inter_inference(self, simifeat):
        simifeat.required_grad = True
        refine_simi = (simifeat * self.ch_scale.view(1, -1, 1, 1) + self.ch_bias.view(1, -1, 1,1)) \
                      * self.sp_scale.unsqueeze(0).unsqueeze(0) + self.sp_bias.unsqueeze(0).unsqueeze(0)
        for i in range(6, len(self.regressor)):
            refine_simi = self.regressor[i](refine_simi)
        if self.pool == 'mean':
            output = torch.mean(refine_simi, dim=(0),keepdim=True)
            return output
        elif self.pool == 'max':
            output, _ = torch.max(refine_simi, 0,keepdim=True)
            return output

    def forward(self, im):
        simi = im.squeeze(0)
        for i in range(6):
            simi = self.regressor[i](simi)
        simi_feat = copy.deepcopy(simi.detach())
        for i in range(6, len(self.regressor)):
            simi = self.regressor[i](simi)
        if self.pool == 'mean':
            output = torch.mean(simi, dim=(0),keepdim=True)
            return output, simi_feat
        elif self.pool == 'max':
            output, _ = torch.max(simi, 0,keepdim=True)
            return output, simi_feat

class F_CountRegressor_CS_64(nn.Module):
    def __init__(self, input_channels, pool='mean'):
        super(F_CountRegressor_CS_64, self).__init__()
        self.pool = pool
        self.regressor = nn.Sequential(
            nn.Conv2d(input_channels, 196, 7, padding=3),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(196, 128, 5, padding=2),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(64, 32, 1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 1),
            nn.ReLU(),
        )

    def reset_refinement_module(self, height, width):
        self.ch_scale = torch.nn.Parameter(torch.Tensor(np.ones(64)), requires_grad=True)
        self.ch_bias = torch.nn.Parameter(torch.Tensor(np.zeros(64)), requires_grad=True)
        self.sp_scale = torch.nn.Parameter(torch.Tensor(np.ones((8 * height, 8 * width))),
                                           requires_grad=True)
        self.sp_bias = torch.nn.Parameter(torch.Tensor(np.zeros((8 * height, 8 * width))),
                                          requires_grad=True)

    def inter_inference(self, simifeat):
        simifeat.required_grad = True
        refine_simi = (simifeat * self.ch_scale.view(1, -1, 1, 1) + self.ch_bias.view(1, -1, 1,1)) \
                      * self.sp_scale.unsqueeze(0).unsqueeze(0) + self.sp_bias.unsqueeze(0).unsqueeze(0)
        for i in range(9, len(self.regressor)):
            refine_simi = self.regressor[i](refine_simi)
        if self.pool == 'mean':
            output = torch.mean(refine_simi, dim=(0),keepdim=True)
            return output
        elif self.pool == 'max':
            output, _ = torch.max(refine_simi, 0,keepdim=True)
            return output

    def forward(self, im):
        simi = im.squeeze(0)
        for i in range(9):
            simi = self.regressor[i](simi)
        simi_feat = copy.deepcopy(simi.detach())
        for i in range(9, len(self.regressor)):
            simi = self.regressor[i](simi)
        if self.pool == 'mean':
            output = torch.mean(simi, dim=(0),keepdim=True)
            return output, simi_feat
        elif self.pool == 'max':
            output, _ = torch.max(simi, 0,keepdim=True)
            return output, simi_feat


class F_CountRegressor_SC_128(nn.Module):
    def __init__(self, input_channels , height, width, pool='mean'):
        super(F_CountRegressor_SC_128, self).__init__()
        self.pool = pool
        self.regressor = nn.Sequential(
            nn.Conv2d(input_channels, 196, 7, padding=3),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(196, 128, 5, padding=2),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(64, 32, 1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 1),
            nn.ReLU(),
        )

        self.ch_scale = torch.nn.Parameter(torch.Tensor(np.ones(128)), requires_grad=True)
        self.ch_bias = torch.nn.Parameter(torch.Tensor(np.zeros(128)), requires_grad=True)
        self.sp_scale = torch.nn.Parameter(torch.Tensor(np.ones((4 * height, 4 * width))),
                                           requires_grad=True)
        self.sp_bias = torch.nn.Parameter(torch.Tensor(np.zeros((4 * height, 4 * width))),
                                          requires_grad=True)
        self.refine_act = nn.LeakyReLU(0.3)

    def forward(self, im):
      with torch.no_grad():
        simi = im.squeeze(0)
        for i in range(6):
          simi = self.regressor[i](simi)

      simi.required_grad = True
      refine_simi = (simi * self.sp_scale.unsqueeze(0).unsqueeze(0) + self.sp_bias.unsqueeze(0).unsqueeze(0)) * self.ch_scale.view(1, -1, 1, 1) + self.ch_bias.view(1, -1, 1, 1)
      for i in range(6, len(self.regressor)):
        refine_simi = self.regressor[i](refine_simi)
      if self.pool == 'mean':
        output = torch.mean(refine_simi, dim=(0),keepdim=True)
        return output
      elif self.pool == 'max':
        output, _ = torch.max(refine_simi, 0,keepdim=True)
        return output

class F_CountRegressor_SC_64(nn.Module):
    def __init__(self, input_channels , height, width, pool='mean'):
        super(F_CountRegressor_SC_64, self).__init__()
        self.pool = pool
        self.regressor = nn.Sequential(
            nn.Conv2d(input_channels, 196, 7, padding=3),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(196, 128, 5, padding=2),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(64, 32, 1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 1),
            nn.ReLU(),
        )

        self.ch_scale = torch.nn.Parameter(torch.Tensor(np.ones(64)), requires_grad=True)
        self.ch_bias = torch.nn.Parameter(torch.Tensor(np.zeros(64)), requires_grad=True)
        self.sp_scale = torch.nn.Parameter(torch.Tensor(np.ones((8 * height, 8 * width))),
                                           requires_grad=True)
        self.sp_bias = torch.nn.Parameter(torch.Tensor(np.zeros((8 * height, 8 * width))),
                                          requires_grad=True)
        self.refine_act = nn.LeakyReLU(0.3)

    def forward(self, im):
      with torch.no_grad():
        simi = im.squeeze(0)
        for i in range(9):
          simi = self.regressor[i](simi)

      simi.required_grad = True
      refine_simi = (simi * self.sp_scale.unsqueeze(0).unsqueeze(0) + self.sp_bias.unsqueeze(0).unsqueeze(0)) * self.ch_scale.view(1, -1, 1, 1) + self.ch_bias.view(1, -1, 1, 1)
      for i in range(9, len(self.regressor)):
        refine_simi = self.regressor[i](refine_simi)
      if self.pool == 'mean':
        output = torch.mean(refine_simi, dim=(0),keepdim=True)
        return output
      elif self.pool == 'max':
        output, _ = torch.max(refine_simi, 0,keepdim=True)
        return output