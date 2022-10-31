import os
import os.path as osp
import sys
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import cv2
from PIL import Image
import numpy as np
from pycocotools.coco import COCO
import torchvision.datasets as dataset
from torch.utils.data import DataLoader
from .dataset_builder import DATASET_REGISTRY
from yacs.config import CfgNode
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tools.utils import SquarePad


@DATASET_REGISTRY.register('coco')
def build_coco_dataset(cfg: CfgNode):
    if cfg.DATASET.ALBUMENTATION:
        data_transforms = {
            'train': A.Compose([
                # https://github.com/albumentations-team/albumentations/issues/718
                A.LongestMaxSize(max_size=max(cfg.DATASET.IMAGE_SIZE[0], cfg.DATASET.IMAGE_SIZE[1]), interpolation=1),
                A.PadIfNeeded(min_height=min(cfg.DATASET.IMAGE_SIZE[0], cfg.DATASET.IMAGE_SIZE[1]),
                              min_width=min(cfg.DATASET.IMAGE_SIZE[0], cfg.DATASET.IMAGE_SIZE[1]),
                              border_mode=0, value=(0, 0, 0)),
                A.Resize(cfg.DATASET.IMAGE_SIZE[0], cfg.DATASET.IMAGE_SIZE[1]),
                A.HorizontalFlip(),
                A.VerticalFlip(),
                A.RandomRotate90(),
                A.RandomBrightnessContrast(),
                A.Normalize(),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'])),
            'test': A.Compose([
                A.LongestMaxSize(max_size=max(cfg.DATASET.IMAGE_SIZE[0], cfg.DATASET.IMAGE_SIZE[1]), interpolation=1),
                A.PadIfNeeded(min_height=min(cfg.DATASET.IMAGE_SIZE[0], cfg.DATASET.IMAGE_SIZE[1]),
                              min_width=min(cfg.DATASET.IMAGE_SIZE[0], cfg.DATASET.IMAGE_SIZE[1]),
                              border_mode=0, value=(0, 0, 0)),
                A.Resize(cfg.DATASET.IMAGE_SIZE[0], cfg.DATASET.IMAGE_SIZE[1]),
                A.Normalize(),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'])),
        }
    else:
        # TODO: the transformation below doesn't work, since the bounding box will not adjust accordingly.
        #  need to change when other things are finished.
        data_transforms = {
            'train': transforms.Compose(
                [
                    SquarePad(),
                    transforms.Resize((cfg.DATASET.IMAGE_SIZE[0], cfg.DATASET.IMAGE_SIZE[1])),
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.RandomAdjustSharpness(0.5),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]
            ),
            'test': transforms.Compose(
                [
                    SquarePad(),
                    transforms.Resize((cfg.DATASET.IMAGE_SIZE[0], cfg.DATASET.IMAGE_SIZE[1])),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]
            )
        }
    train_dataset = COCODetection(cfg.DATASET.TRAIN_DATA_ROOT, cfg.DATASET.TRAIN_ANNO, cfg,
                                  transform=data_transforms['train'])
    val_dataset = COCODetection(cfg.DATASET.TEST_DATA_ROOT, cfg.DATASET.TEST_ANNO, cfg,
                                transform=data_transforms['test'])
    return train_dataset, val_dataset


class COCODetection(data.Dataset):
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

    def __init__(self, image_path, info_file, cfg: CfgNode, transform=None,
                 target_transform=None, has_gt=True):
        self.root = image_path
        self.coco = COCO(info_file)
        self.ids = list(self.coco.imgToAnns.keys())  # 标签数目 小于样本数目，说明有的图像没有标签

        if len(self.ids) == 0 or not has_gt:  # 如果没有标签或者不需要GT，则直接使用image
            self.ids = list(sorted(self.coco.imgs.keys()))
        self.transform = transform
        self.target_transform = target_transform

        self.data_infos = self.load_annotations(info_file)

        self.has_gt = has_gt
        self.cfg = cfg

    def __len__(self):
        return len(self.ids)

    def load_annotations(self, ann_file):
        """Load annotation from COCO style annotation file.

        Args:
            ann_file (str): Path of annotation file.

        Returns:
            list[dict]: Annotation info from COCO api.
        """

        self.coco = COCO(ann_file)
        # The order of returned `cat_ids` will not
        # change with the order of the CLASSES
        self.cat_ids = self.coco.get_cat_ids(cat_names=self.CLASSES)

        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.img_ids = self.coco.get_img_ids()
        data_infos = []
        total_ann_ids = []
        for i in self.img_ids:
            info = self.coco.load_imgs([i])[0]
            info['filename'] = info['file_name']
            data_infos.append(info)
            ann_ids = self.coco.get_ann_ids(img_ids=[i])
            total_ann_ids.extend(ann_ids)
        assert len(set(total_ann_ids)) == len(
            total_ann_ids), f"Annotation ids in '{ann_file}' are not unique!"
        return data_infos

    def get_ann_info(self, idx):
        """Get COCO annotation by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """

        img_id = self.data_infos[idx]['id']
        ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
        ann_info = self.coco.load_anns(ann_ids)
        return self._parse_ann_info(self.data_infos[idx], ann_info)

    def get_cat_ids(self, idx):
        """Get COCO category ids by index.

        Args:
            idx (int): Index of data.

        Returns:
            list[int]: All categories in the image of specified index.
        """

        img_id = self.data_infos[idx]['id']
        ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
        ann_info = self.coco.load_anns(ann_ids)
        return [ann['category_id'] for ann in ann_info]

    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,\
                labels, masks, seg_map. "masks" are raw annotations and not \
                decoded into binary masks.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []
        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
                gt_masks_ann.append(ann.get('segmentation', None))

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        seg_map = img_info['filename'].replace('jpg', 'png')

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            seg_map=seg_map)

        return ann

    def xyxy2xywh(self, bbox):
        """Convert ``xyxy`` style bounding boxes to ``xywh`` style for COCO
        evaluation.

        Args:
            bbox (numpy.ndarray): The bounding boxes, shape (4, ), in
                ``xyxy`` order.

        Returns:
            list[float]: The converted bounding boxes, in ``xywh`` order.
        """

        _bbox = bbox.tolist()
        return [
            _bbox[0],
            _bbox[1],
            _bbox[2] - _bbox[0],
            _bbox[3] - _bbox[1],
        ]

    def _load_image(self, id: int):
        path = self.coco.loadImgs(id)[0]["file_name"]
        image = Image.open(os.path.join(self.root, path)).convert("RGB")
        return np.array(image)

    def _load_target(self, id: int):
        return self.coco.loadAnns(self.coco.getAnnIds(id))

    def __getitem__(self, index):
        # id = self.ids[index]
        # image = self._load_image(id)
        # anno = self.get_ann_info(index)
        # bboxes = anno["bboxes"]
        # class_labels = anno['labels']

        img_id = self.data_infos[index]['id']
        image = self._load_image(img_id)
        annos = self.get_ann_info(index)
        bboxes = annos['bboxes']
        class_labels = annos['labels']

        # note: The code below also works, but the code above is more neat.
        #  Also, the code above can generate CORRECT class label, however, the code below need to be transformed
        #  using the COCO_LABEL_MAP above, otherwise there will be a mismatch.
        # img_id = self.ids[index]
        # ann_ids = self.coco.getAnnIds(imgIds=img_id)
        # ann_info = self.coco.load_anns(ann_ids)
        # image = self._load_image(img_id)
        # bboxes = [anno["bbox"] for anno in ann_info]
        # class_labels = [anno["category_id"] for anno in ann_info]

        if self.transform is not None:
            if self.cfg.DATASET.ALBUMENTATION:
                transformed = self.transform(image=image, bboxes=bboxes, class_labels=class_labels)
                transformed_image = transformed['image']
                transformed_image = transformed_image
                transformed_bboxes = torch.as_tensor(transformed['bboxes'])
            else:
                image = Image.fromarray(image)
                transformed_image = self.transform(image)
                transformed_bboxes = bboxes

        # Note: after using get_anno_info, the output bounding box is in the format of [xmin, ymin, xmax, ymax].
        #  so I changed the parameter of BboxParams to 'pascal_voc' instead of 'coco'.
        return transformed_image, transformed_bboxes, class_labels


# if __name__=='__main__':
#     dataset = COCODetection(val_image, val_info, transform=data_transforms)
#     loader = DataLoader(dataset)
#
#     for info in loader:
#         print(info)
