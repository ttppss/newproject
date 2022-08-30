import argparse
import copy
import os
import os.path as osp
import time
import warnings

import torch

from configs import cfg, update_cfg
from tools.utils import create_logger
from visdom import Visdom
from tools.utils import count_flop

from models.builder import build_backbone

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

    model = build_backbone(cfg.MODEL)

    # decide if we use data parallel of not according to number of GPU.
    if cfg.gpu_ids > 0:
        model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).to(cfg.DEVICE)
    else:
        model = model.to(cfg.DEVICE)
    logger.info(f"\nmodel structure: \n{model}")
    logger.info('\n' + '*' * 60)

    # calculate the number of parameters and FLOPs using dummy data.
    dummy_data = torch.randn(1, 3, cfg.TRAIN.INPUT_SIZE[0], cfg.TRAIN.INPUT_SIZE[1]).to(cfg.DEVICE)
    count_flop(model, dummy_data)

if __name__ == '__main__':
    main()



