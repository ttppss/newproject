import pytorch_lightning as pl
from tools.utils import get_criterion, get_optimizer, get_scheduler
from configs import cfg
from tools.utils import xyxy2xywh


class LitAutoEncoder(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        image, bboxes, labels = batch

        # todo: need to change the format of bbox here,
        #  change it from absolute coordinate to related coordinate.
        if cfg.DATASET.TYPE == 'coco':
            # if yolo, then change the coordinate to x,y,w,h.
            if 'yolo' in cfg.MODEL.NAME:
                for bbox in bboxes:
                    bbox = xyxy2xywh(bbox)

        out = self.model(image)
        loss = get_criterion(cfg)(out, image)
        # Logging to TensorBoard by default
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = get_optimizer(cfg, self.model)
        return optimizer
