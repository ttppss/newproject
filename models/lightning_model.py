import pytorch_lightning as pl
from tools.utils import get_criterion, get_optimizer, get_scheduler
from configs import cfg
from tools.utils import xyxy2xywh
from models.loss import yolov1_Loss
import torch


class LitAutoEncoder(pl.LightningModule):
    def __init__(self, model, grid_size=7, num_bboxes=2, num_classes=80):
        super().__init__()
        self.model = model
        self.S = grid_size
        self.B = num_bboxes
        self.C = num_classes

    def forward(self, x):
        output = self.model(x)
        return output

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        # bboxes is a list of length 'batch', each element of the list is a tensor of shape (n, 4)
        # labels is a list of length 'batch', each element of the list is an array of size (n,)
        image, bboxes, labels = batch
        _, _, h, w = image.shape
        targets = torch.zeros(image.shape[0], self.S, self.S, 5 * self.B + self.C)
        # targets = torch.zeros()

        # todo: need to change the format of bbox here,
        #  change it from absolute coordinate to related coordinate.
        if cfg.DATASET.TYPE == 'coco':
            # if yolo, then change the coordinate to x,y,w,h.
            if 'yolo' in cfg.MODEL.NAME:
                for idx, bbox in enumerate(bboxes):
                    try:
                        # bbox = xyxy2xywh(bbox)
                        bbox /= torch.Tensor([w, h, w, h])
                        target = self.encode(bbox, labels[idx])
                        targets[idx, :] = target
                    except:
                        pass


        # out is of shape [batch, 7, 7, 30] for this project
        out = self.model(image)
        loss = yolov1_Loss()(out, targets)
        # Logging to TensorBoard by default
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        image, bboxes, labels = batch
        _, _, h, w = image.shape
        targets = torch.zeros(image.shape[0], self.S, self.S, 5 * self.B + self.C)
        # targets = torch.zeros()

        # todo: need to change the format of bbox here,
        #  change it from absolute coordinate to related coordinate.
        if cfg.DATASET.TYPE == 'coco':
            # if yolo, then change the coordinate to x,y,w,h.
            if 'yolo' in cfg.MODEL.NAME:
                for idx, bbox in enumerate(bboxes):
                    try:
                        # bbox = xyxy2xywh(bbox)
                        bbox /= torch.Tensor([w, h, w, h])
                        target = self.encode(bbox, labels[idx])
                        targets[idx, :] = target
                    except:
                        pass

        # out is of shape [batch, 7, 7, 30] for this project
        out = self.model(image)
        loss = yolov1_Loss()(out, targets)
        # Logging to TensorBoard by default
        self.log("validataion_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = get_optimizer(cfg, self.model)
        return optimizer

    def encode(self, boxes, labels):
        """ Encode box coordinates and class labels as one target tensor.
        Args:
            boxes: (tensor) [[x1, y1, x2, y2]_obj1, ...], normalized from 0.0 to 1.0 w.r.t. image width/height.
            labels: (tensor) [c_obj1, c_obj2, ...]
        Returns:
            An encoded tensor sized [S, S, 5 x B + C], 5=(x, y, w, h, conf)
        """

        S, B, C = self.S, self.B, self.C
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
