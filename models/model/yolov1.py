import torch
import torch.nn as nn
import torch.nn.functional as F
from models.builder import MODEL_REGISTRY, build_backbone, build_neck, build_head

from models.backbones.darknet53 import Darknet53


@MODEL_REGISTRY.register('yolov1')
class YoloV1(nn.Module):
    def __init__(self, cfg):
        super(YoloV1, self).__init__()

        self.backbone = build_backbone(cfg.MODEL.BACKBONE)
        if cfg.MODEL.NECK.NAME != '':
            self.neck = build_neck(cfg.MODEL.NECK)
        self.head = build_head(cfg.MODEL.HEAD)

