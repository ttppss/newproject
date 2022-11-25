from pathlib import Path
import logging
import time
import os
import inspect
import torch
import torchvision.transforms.functional as F
import torch.nn as nn
from torchvision.utils import draw_bounding_boxes
import numpy as np
import torch.optim as optim
from tqdm import tqdm

from fvcore.nn import FlopCountAnalysis, flop_count_table
import torchvision
from visdom import Visdom

from tools.meter import AverageMeter

logger = logging.getLogger(__name__)
viz = Visdom()

CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
               'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
               'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
               'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
               'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
               'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
               'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
               'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
               'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
               'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
               'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
               'scissors', 'teddy bear', 'hair drier', 'toothbrush')

COCO_LABEL_MAP = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8,
                      9: 9, 10: 10, 11: 11, 13: 12, 14: 13, 15: 14, 16: 15, 17: 16,
                      18: 17, 19: 18, 20: 19, 21: 20, 22: 21, 23: 22, 24: 23, 25: 24,
                      27: 25, 28: 26, 31: 27, 32: 28, 33: 29, 34: 30, 35: 31, 36: 32,
                      37: 33, 38: 34, 39: 35, 40: 36, 41: 37, 42: 38, 43: 39, 44: 40,
                      46: 41, 47: 42, 48: 43, 49: 44, 50: 45, 51: 46, 52: 47, 53: 48,
                      54: 49, 55: 50, 56: 51, 57: 52, 58: 53, 59: 54, 60: 55, 61: 56,
                      62: 57, 63: 58, 64: 59, 65: 60, 67: 61, 70: 62, 72: 63, 73: 64,
                      74: 65, 75: 66, 76: 67, 77: 68, 78: 69, 79: 70, 80: 71, 81: 72,
                      82: 73, 84: 74, 85: 75, 86: 76, 87: 77, 88: 78, 89: 79, 90: 80}



def create_logger(cfg, cfg_name):
    assert len(cfg.OUTPUT_PATH) > 0, 'Error: The output directory need to be specified.'
    output_dir = Path(cfg.OUTPUT_PATH)
    if not output_dir.exists():
        print('=> creating {}'.format(output_dir))
        output_dir.mkdir()

    dataset = cfg.DATASET.TYPE
    model = cfg.MODEL.NAME
    cfg_name = os.path.basename(cfg_name).split('.')[0]

    time_str = time.strftime('%Y-%m-%d-%H-%M')

    final_output_dir = output_dir / dataset / model / cfg_name / time_str

    print('=> creating {}'.format(final_output_dir))
    final_output_dir.mkdir(parents=True, exist_ok=True)

    log_file = '{}_{}.log'.format(cfg_name, time_str)
    final_log_file = final_output_dir / log_file
    logging.basicConfig(filename=str(final_log_file))
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    return logger, str(final_output_dir)


def build_from_cfg(cfg, registry, default_args=None):
    """Build a module from config dict.
    Args:
        cfg (dict): Config dict. It should at least contain the key "type".
        registry (:obj:`Registry`): The registry to search the type from.
        default_args (dict, optional): Default initialization arguments.
    Returns:
        obj: The constructed object.
    """
    assert isinstance(cfg, dict) and 'type' in cfg
    assert isinstance(default_args, dict) or default_args is None  # 两个是断言，相当于判断，否的话抛出异常。

    args = cfg.copy()  # args相当于temp中间变量，是个字典。
    obj_type = args.pop('type')  # 字典的pop作用：移除序列中key为‘type’的元素，并且返回该元素的值
    if isinstance(obj_type, str):
        obj_type = registry.get(obj_type)  # 获取obj_type的value。
        # 如果obj_type已经注册到注册表registry中，即在属性_module_dict中，则obj_type 不为None
        if obj_type is None:
            raise KeyError('{} is not in the {} registry'.format(
                obj_type, registry.name))
    elif not inspect.isclass(obj_type):
        raise TypeError('type must be a str or valid type, but got {}'.format(
            type(obj_type)))
    if default_args is not None:
        for name, value in default_args.items():  # items()返回字典的键值对用于遍历
            args.setdefault(name, value)
            # 将default_args的键值对加入到args中，将模型和训练配置进行整合，然后送入类中返回

    return obj_type(**args)


def count_flop(model, inp):
    flops = FlopCountAnalysis(model, inp)
    logger.info('\n********** # of parameters and FLOPs **********\n')
    logger.info(flop_count_table(flops))


def collate_fn(batch):
    """
    Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader).
    This describes how to combine these tensors of different sizes. We use lists.
    Note: this need not be defined in this Class, can be standalone.
    :param batch: an iterable of N sets from __getitem__()
    :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
    """

    images = list()
    bboxes = list()
    labels = list()

    for b in batch:
        images.append(b[0])
        bboxes.append(b[1])
        labels.append(b[2])

    images = torch.stack(images, dim=0)

    return images, bboxes, labels


def xywh2xyxy(bboxes):
    bboxes[:, 2] = bboxes[:, 0] + bboxes[:, 2]
    bboxes[:, 3] = bboxes[:, 1] + bboxes[:, 3]
    return bboxes


def xyxy2xywh(bboxes):
    """
    change the coordinate from [xmin, ymin, xmax, ymax] to [xmin, ymin, width, height]
    :param bboxes: tensor of shape [N, 4], N is the number of bounding boxes.
    :return: bboxes: tensor of shape [N, 4] in transformed value.
    """
    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
    bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
    return bboxes

def visualize_data(data_loader, window_name, title):
    # TODO: may need a more elegant way to put the images together
    # when dealing with object detection, the returning data format is like [[img, label], [img, label]]
    inputs = next(iter(data_loader))

    # add something. If visualize bbox, then...
    inp = inputs[0]
    out = torchvision.utils.make_grid(inp, nrow=5)
    inp = torch.transpose(out, 0, 2)
    mean = torch.FloatTensor([0.485, 0.456, 0.406])
    std = torch.FloatTensor([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = torch.transpose(inp, 0, 2)
    viz.images(inp, win=window_name, opts={'title': title})


def visualize_data_with_bbox(data_loader, window_name, title):
    inputs = next(iter(data_loader))
    for i in range(inputs[0].shape[0]):
        labels_num = inputs[2][i]
        # draw_bounding_boxes only accepts string, so convert it from int to class name.
        labels = [CLASSES[j] for j in labels_num]
        inp = torch.transpose(inputs[0][i], 0, 2)
        mean = torch.FloatTensor([0.485, 0.456, 0.406])
        std = torch.FloatTensor([0.229, 0.224, 0.225])
        inp = (std * 255) * inp + mean * 255
        inp = torch.transpose(inp, 0, 2)
        inp = draw_bounding_boxes(inp.byte(), inputs[1][i], labels, width=2, font_size=80)
        inputs[0][i] = inp
    out = torchvision.utils.make_grid(inputs[0], nrow=5)
    viz.images(out, win=window_name, opts={'title': title})


class SquarePad:
    def __call__(self, image):
        w, h = image.size
        max_wh = np.max([w, h])
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = (hp, vp, hp, vp)
        return F.pad(image, padding, 0, 'constant')


def get_criterion(cfg):
    if cfg.TRAIN.LOSS.NAME == 'crossentropy':
        loss = nn.CrossEntropyLoss()
    elif cfg.TRAIN.LOSS.NAME == 'smoothl1':
        loss = nn.L1Loss()
    else:
        raise ValueError("loss %s is not defined." % cfg.TRAIN.LOSS.NAME)

    return loss


def get_optimizer(cfg, model):
    if cfg.TRAIN.OPTIMIZER == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=cfg.TRAIN.LR,
            # momentum=cfg.TRAIN.MOMENTUM,
            weight_decay=cfg.TRAIN.WEIGHT_DECAY,
        )
    elif cfg.TRAIN.OPTIMIZER == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=cfg.TRAIN.LR,
        )
    return optimizer


def get_scheduler(cfg, optimizer):
    if cfg.TRAIN.SCHEDULER == 'exp':
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg.TRAIN.LR_DECAY)
    elif cfg.TRAIN.SCHEDULER == 'step':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.TRAIN.STEP_SIZE,
                                                       gamma=cfg.TRAIN.LR_DECAY)
    elif cfg.TRAIN.SCHEDULER == 'multistep':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.TRAIN.LR_STEPS,
                                                            gamma=cfg.TRAIN.LR_DECAY)
    elif cfg.TRAIN.SCHEDULER == 'cos':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.TRAIN.COSINE_T_MAX)
    return lr_scheduler


def train_detection(cfg, epoch, model, optimizer, criterion, lr_scheduler, train_loader, train_dataset, device):
    losses = AverageMeter()
    h, w = cfg.DATASET.IMAGE_SIZE
    bs = cfg.TRAIN.BATCH_SIZE
    S, B, C = cfg.MODEL.YOLOV1.GRID_SIZE, cfg.MODEL.YOLOV1.NUM_PREDICTIONS_PER_LOCATION, cfg.MODEL.NUM_CLASSES
    targets = torch.zeros(bs, S, S, 5 * B + C)

    model.train()
    print('Current Learning Rate: {:.5f}'.format(lr_scheduler.get_last_lr()[0]))

    # print(next(iter(train_loader)))

    for ii, (data, bbox, labels) in tqdm(enumerate(train_loader)):
        inputs = data.to(device)
        bboxes = [box for box in bbox]
        labels = [label for label in labels]
        optimizer.zero_grad()

        for idx, bbox in enumerate(bboxes):
            try:
                # bbox = xyxy2xywh(bbox)
                bbox /= torch.Tensor([w, h, w, h])
                target = encode(cfg, bbox, labels[idx])
                targets[idx, :] = target
            except:
                print('value error: ', bbox)
                pass

        predicted = model(inputs)
        targets = targets.to(device)
        loss = criterion(predicted, targets)
        loss.backward()
        optimizer.step()
        # total_loss += totalloss
        # _, pred = torch.max(score[1], 1)

        # correct += torch.sum(pred == targets)
        # epoch_accuracy = correct / len(train_dataset)
        losses.update(loss.item(), inputs.size(0))

        if ii != 0 and (ii % cfg.TRAIN.PRINT_FREQ == 0 or ii + 1 == len(train_loader)):
            # print('reached breakpoint')
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Training Loss {loss.val:.5f} ({loss.avg:.5f})\t'.format(
                epoch, ii, len(train_loader), loss=losses)
            print(msg)
            logger.info(msg)

    lr_scheduler.step()
    return losses.avg


def val_detection(cfg, epoch, model, optimizer, criterion, val_loader, val_dataset, final_output_dir, best_val_loss, device):
    model.eval()
    correct = 0
    total_loss = 0
    val_losses = AverageMeter()
    h, w = cfg.DATASET.IMAGE_SIZE
    bs = cfg.TRAIN.BATCH_SIZE
    S, B, C = cfg.MODEL.YOLOV1.GRID_SIZE, cfg.MODEL.YOLOV1.NUM_PREDICTIONS_PER_LOCATION, cfg.MODEL.NUM_CLASSES
    targets = torch.zeros(bs, S, S, 5 * B + C)

    # print(next(iter(train_loader)))

    for ii, (data, bbox, labels) in tqdm(enumerate(val_loader)):
        inputs = data.to(device)
        bboxes = [box for box in bbox]
        labels = [label for label in labels]
        optimizer.zero_grad()

        for idx, bbox in enumerate(bboxes):
            try:
                # bbox = xyxy2xywh(bbox)
                bbox /= torch.Tensor([w, h, w, h])
                target = encode(cfg, bbox, labels[idx])
                targets[idx, :] = target
            except:
                print('value error: ', bbox)
                pass

        with torch.no_grad():
            predicted = model(inputs)
        targets = targets.to(device)
        loss = criterion(predicted, targets)

        # total_loss += totalloss
        # _, pred = torch.max(score[1], 1)

        # correct += torch.sum(pred == targets)
        # epoch_accuracy = correct / len(train_dataset)
        val_losses.update(loss.item(), inputs.size(0))

        if ii + 1 == len(val_loader):
            # print('reached breakpoint')
            msg = 'Validataion Loss {loss.val:.5f} ({loss.avg:.5f})\t'.format(
                loss=val_losses)
            print(msg)
            logger.info(msg)

    return val_losses.avg

def encode(cfg, boxes, labels):
    """ Encode box coordinates and class labels as one target tensor.
    Args:
        boxes: (tensor) [[x1, y1, x2, y2]_obj1, ...], normalized from 0.0 to 1.0 w.r.t. image width/height.
        labels: (tensor) [c_obj1, c_obj2, ...]
    Returns:
        An encoded tensor sized [S, S, 5 x B + C], 5=(x, y, w, h, conf)
    """

    S, B, C = cfg.MODEL.YOLOV1.GRID_SIZE, cfg.MODEL.YOLOV1.NUM_PREDICTIONS_PER_LOCATION, cfg.MODEL.NUM_CLASSES
    N = 5 * B + C

    target = torch.zeros(S, S, N)
    cell_size = 1.0 / float(S)
    boxes_wh = boxes[:, 2:] - boxes[:, :2]  # width and height for each box, [n, 2]
    boxes_xy = (boxes[:, 2:] + boxes[:, :2]) / 2.0  # center x & y for each box, [n, 2]
    for b in range(boxes.size(0)):
        xy, wh, label = boxes_xy[b], boxes_wh[b], int(labels[b])

        ij = (xy / cell_size).ceil() - 1.0
        i, j = int(ij[0]), int(ij[1])  # y & x index which represents its location on the grid.
        x0y0 = ij * cell_size  # x & y of the cell left-top corner.
        xy_normalized = (xy - x0y0) / cell_size  # x & y of the box on the cell, normalized from 0.0 to 1.0.

        # TBM, remove redundant dimensions from target tensor.
        # To remove these, loss implementation also has to be modified.
        for k in range(B):
            s = 5 * k
            target[j, i, s:s + 2] = xy_normalized
            target[j, i, s + 2:s + 4] = wh
            target[j, i, s + 4] = 1.0
        target[j, i, 5 * B + label] = 1.0

    return target