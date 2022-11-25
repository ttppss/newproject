from tools.registry import Registry
from yacs.config import CfgNode


BACKBONE_REGISTRY = Registry()
NECK_REGISTRY = Registry()
HEAD_REGISTRY = Registry()
MODEL_REGISTRY = Registry()
LOSS_REGISTRY = Registry()


def build_backbone(backbone_cfg: CfgNode):
    backbone = BACKBONE_REGISTRY[backbone_cfg.BACKBONE.NAME]()
    return backbone


def build_neck(neck_cfg: CfgNode):
    neck = NECK_REGISTRY[neck_cfg.NACK.NAME](neck_cfg)
    return neck


def build_head(head_cfg: CfgNode):
    head = HEAD_REGISTRY[head_cfg.NAME](head_cfg)
    return head


def build_model(model_cfg: CfgNode):
    model = MODEL_REGISTRY[model_cfg.MODEL.NAME](model_cfg)
    return model


def build_loss(loss_cfg: CfgNode):
    model = LOSS_REGISTRY[loss_cfg.LOSS.NAME]()
    return model