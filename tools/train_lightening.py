import argparse
import copy
import os
import os.path as osp
import time
import warnings

import torch

import sys
sys.path.append('/home/zinan/pycharmproject/newproject')
# sys.path.append('/data2/zinan_xiong/newproject')

# import configs.config
from configs import get_cfg_defaults, update_cfg
from tools.utils import create_logger, get_criterion, get_optimizer, get_scheduler
from visdom import Visdom
from tools.utils import count_flop, collate_fn, visualize_data, visualize_data_with_bbox

from models.builder import build_backbone, build_model
from dataset.dataset_builder import build_dataset

import pytorch_lightning as pl
from models.lightning_model import LitAutoEncoder

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    cfg = get_cfg_defaults()
    args = parse_args()
    update_cfg(cfg, args)

    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = 0 if args.gpus is None else range(args.gpus)

    # create work_dir and logger
    logger, final_output_dir = create_logger(cfg, args.config)

    logger.info('\nConfiguration: \n{}\n'.format(cfg))
    logger.info('\n' + '*' * 60)

    model = build_model(cfg)

    # decide if we use data parallel of not according to number of GPU.
    # if cfg.gpu_ids > 0:
    #     model = torch.nn.DataParallel(model, device_ids=cfg.GPUS)
    # else:
    #     model = model

    logger.info(f"\nmodel structure: \n{model}")
    logger.info('\n' + '*' * 60)

    # calculate the number of parameters and FLOPs using dummy data.
    dummy_data = torch.randn(1, 3, cfg.DATASET.IMAGE_SIZE[0], cfg.DATASET.IMAGE_SIZE[1])
    count_flop(model, dummy_data)

    # construct the dataset
    train_dataset, val_dataset = build_dataset(cfg)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=cfg.TRAIN.SHUFFLE,
        num_workers=cfg.TRAIN.NUM_WORKERS,
        collate_fn=collate_fn
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.TRAIN.NUM_WORKERS,
        collate_fn=collate_fn
    )

    visualize_data_with_bbox(train_loader, 'training data', 'training data visualization')
    visualize_data_with_bbox(val_loader, 'validation data', 'validation data visualization')

    automodel = LitAutoEncoder(model)

    # todo: wield: if not using accelerator='cpu', it will cause an error: Expected all tensors to be on the same device.
    trainer = pl.Trainer(limit_train_batches=256, max_epochs=3, log_every_n_steps=10, accelerator='cpu', gpus='0')
    trainer.fit(model=automodel, train_dataloaders=train_loader, val_dataloaders=val_loader)



if __name__ == '__main__':
    main()



