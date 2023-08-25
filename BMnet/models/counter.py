"""
Counter modules.
"""
import copy
import numpy as np
import torch
from torch import nn

def get_counter(cfg):
    counter_name = cfg.MODEL.counter
    if counter_name == 'density_x16':
        return DensityX16_CS(counter_dim=cfg.MODEL.counter_dim)
    else:
        raise NotImplementedError

def get_counter_train(cfg):
    counter_name = cfg.MODEL.counter
    if counter_name == 'density_x16':
        return DensityX16(counter_dim=cfg.MODEL.counter_dim)
    else:
        raise NotImplementedError
        

class DensityX16(nn.Module):
    def __init__(self, counter_dim):
        super().__init__()
        self.regressor =  nn.Sequential(
                                    nn.Conv2d(counter_dim, 196, 7, padding=3),
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
                                    nn.UpsamplingBilinear2d(scale_factor=2),
                                    nn.Conv2d(32, 1, 1),
                                    nn.ReLU()
                                )
        self._weight_init_()
        
    def forward(self, features):
        features = self.regressor(features)
        return features
        
    def _weight_init_(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                # nn.init.kaiming_uniform_(
                #         m.weight, 
                #         mode='fan_in', 
                #         nonlinearity='relu'
                #         )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class DensityX16_CS(nn.Module):
    def __init__(self, counter_dim):
        super().__init__()
        self.regressor = nn.Sequential(
            nn.Conv2d(counter_dim, 196, 7, padding=3),
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
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(32, 1, 1),
            nn.ReLU()
        )

    def reset_refine_module(self, height, width):
        self.ch_scale = torch.nn.Parameter(torch.Tensor(np.ones(196)), requires_grad=True)
        self.ch_bias = torch.nn.Parameter(torch.Tensor(np.zeros(196)), requires_grad=True)
        self.sp_scale = torch.nn.Parameter(torch.Tensor(np.ones((height, width))),
                                           requires_grad=True)
        self.sp_bias = torch.nn.Parameter(torch.Tensor(np.zeros((height, width))),
                                          requires_grad=True)

    def inter_inference(self, refine_feat):
        refine_feat.required_grad = True
        output = (refine_feat * self.ch_scale.view(1, -1, 1, 1) + self.ch_bias.view(1, -1, 1, 1)) * self.sp_scale.unsqueeze(
            0).unsqueeze(0) + self.sp_bias.unsqueeze(0).unsqueeze(0)
        for i in range(3, len(self.regressor)):
            output = self.regressor[i](output)
        return output

    def forward(self, features):
        with torch.no_grad():
            for i in range(0, 3):
                features = self.regressor[i](features)
            refine_feat = copy.deepcopy(features)
            for i in range(3, len(self.regressor)):
                features = self.regressor[i](features)
            return features, refine_feat

