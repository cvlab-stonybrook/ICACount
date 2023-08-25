import torch

from BMnet.models.backbone import build_backbone
from BMnet.models.counter import get_counter, get_counter_train
from BMnet.models.epf_extractor import build_epf_extractor
from BMnet.models.refiner import build_refiner
from BMnet.models.matcher import build_matcher
from BMnet.models.class_agnostic_counting_model import CACModel

def build_model(cfg):
    backbone = build_backbone(cfg)
    epf_extractor = build_epf_extractor(cfg)
    refiner = build_refiner(cfg)
    matcher = build_matcher(cfg)
    counter = get_counter(cfg)
    model = CACModel(backbone, epf_extractor, refiner, matcher, counter, cfg.MODEL.hidden_dim)
    
    return model


def build_model_train(cfg):
    backbone = build_backbone(cfg)
    epf_extractor = build_epf_extractor(cfg)
    refiner = build_refiner(cfg)
    matcher = build_matcher(cfg)
    counter = get_counter_train(cfg)
    model = CACModel(backbone, epf_extractor, refiner, matcher, counter, cfg.MODEL.hidden_dim)

    return model
    
    
    