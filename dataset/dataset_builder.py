from tools.registry import Registry
from yacs.config import CfgNode

DATASET_REGISTRY = Registry()


def build_dataset(cfg: CfgNode):
    dataset = DATASET_REGISTRY[cfg.DATASET.DATASET](cfg)
    return dataset