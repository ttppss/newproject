import torch
import torch.nn as nn
import torch.nn.functional as F
from models.builder import MODEL_REGISTRY, build_backbone, build_neck, build_head

from models.backbones.darknet53 import darknet53


@MODEL_REGISTRY.register('yolov1')
class YoloV1(nn.Module):
    def __init__(self, cfg):
        super(YoloV1, self).__init__()

        self.feature_size = 7
        self.num_bboxes = 2
        self.num_classes = cfg.MODEL.NUM_CLASSES

        self.backbone = darknet53()
        self.conv_layers = self._make_conv_layers()
        self.fc_layers = self._make_fc_layers()

    def _make_conv_layers(self):
        net = nn.Sequential(
            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(1024, 1024, 3, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),

            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(1024, 1024, 3, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True)
        )
        return net

    def _make_fc_layers(self):
        S, B, C = self.feature_size, self.num_bboxes, self.num_classes

        net = nn.Sequential(
            Flatten(),
            nn.Linear(7 * 7 * 1024, 4096),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.5, inplace=False),  # is it okay to use Dropout with BatchNorm?
            nn.Linear(4096, S * S * (5 * B + C)),
            nn.Sigmoid()
        )

        return net

    def forward(self, x):
        S, B, C = self.feature_size, self.num_bboxes, self.num_classes

        x = self.backbone(x)
        x = self.conv_layers(x)
        x = self.fc_layers(x)

        x = x.view(-1, S, S, 5 * B + C)
        return x


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)