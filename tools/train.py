import argparse
import copy
import os
import os.path as osp
import time
import warnings
import numpy as np

import torch

import sys
sys.path.append('/home/zinan/pycharmproject/newproject')

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# import configs.config
from configs import get_cfg_defaults, update_cfg
from tools.utils import create_logger, get_criterion, get_optimizer, get_scheduler
from visdom import Visdom
from tools.utils import count_flop, collate_fn, visualize_data, visualize_data_with_bbox, train_detection, val_detection

from models.builder import build_backbone, build_model, build_loss
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    if cfg.GPUS:
        model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).to(device)
    else:
        model = model.to(device)
    logger.info(f"\nmodel structure: \n{model}")
    logger.info('\n' + '*' * 60)

    # calculate the number of parameters and FLOPs using dummy data.
    dummy_data = torch.randn(1, 3, cfg.DATASET.IMAGE_SIZE[0], cfg.DATASET.IMAGE_SIZE[1]).to(device)
    count_flop(model, dummy_data)

    # construct the dataset
    train_dataset, val_dataset = build_dataset(cfg)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=cfg.TRAIN.SHUFFLE,
        num_workers=cfg.TRAIN.NUM_WORKERS,
        collate_fn=collate_fn,
        drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.TRAIN.NUM_WORKERS,
        collate_fn=collate_fn,
        drop_last=True
    )

    visualize_data_with_bbox(train_loader, 'training data', 'training data visualization')
    visualize_data_with_bbox(val_loader, 'validation data', 'validation data visualization')

    # automodel = LitAutoEncoder(model)
    #
    # trainer = pl.Trainer(limit_train_batches=16, max_epochs=3, log_every_n_steps=1, accelerator='gpu', devices=1)
    # trainer.fit(model=automodel, train_dataloaders=train_loader, val_dataloaders=val_loader)

    best_perf = 0.0
    best_model = False
    last_epoch = -1
    best_val_loss = np.inf
    optimizer = get_optimizer(cfg, model)
    # criterion = get_criterion(cfg)
    criterion = build_loss(cfg.TRAIN)
    checkpoint_file = os.path.join(final_output_dir, 'checkpoint.pth')

    if cfg.AUTO_RESUME and os.path.exists(checkpoint_file):
        logger.info('******** Loading checkpoint file {} ********'.format(checkpoint_file))
        checkpoint = torch.load(checkpoint_file)
        best_perf = checkpoint['perf']
        last_epoch = checkpoint['epoch']
        model.module.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info('******** checkpoint loaded, continue training from epoch {} ********'.format(last_epoch))

    lr_scheduler = get_scheduler(cfg, optimizer)

    for epoch in range(cfg.TRAIN.MAX_EPOCH):
        train_loss = train_detection(cfg, epoch, model, optimizer, criterion, lr_scheduler, train_loader,
                                      train_dataset, device)
        logger.info('Validating Model...')
        val_loss = val_detection(cfg, epoch, model, optimizer, criterion,
                                 val_loader, val_dataset, final_output_dir, best_val_loss, device)

        if best_val_loss > val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(final_output_dir, 'model_best.pth'))

        if epoch % cfg.TRAIN.SAVE_EVERY_N_EPOCH == 0:
            torch.save(model.state_dict(), os.path.join(final_output_dir, 'epoch_' + str(epoch) + '.pth'))

        # if val_accuracy > best_perf:
        #     best_perf = val_accuracy
        #     best_model = True
        # else:
        #     best_model = False

        logger.info('saving checkpoint to {}'.format(final_output_dir))
        # save_checkpoint(
        #     {'epoch': epoch + 1,
        #      'model': cfg.MODEL.NAME,
        #      'state_dict': model.module.state_dict(),
        #      'best_state_dict': model.module.state_dict(),
        #      'perf': val_accuracy,
        #      'optimizer': optimizer.state_dict(),
        #      }, best_model, final_output_dir
        # )

    final_model_state_file = os.path.join(final_output_dir, 'final_state.pth')
    logger.info('**************** best validation accuracy: {} ****************'.format(best_val_loss))
    logger.info('******** saving final model state to {} ********'.format(final_model_state_file))
    torch.save(model.module.state_dict(), final_model_state_file)



if __name__ == '__main__':
    main()



