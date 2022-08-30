from yacs.config import CfgNode as CN

_C = CN()

_C.ENV = 'default'
_C.DEVICE = 'cuda'
_C.GPUS = [0, ]
_C.AUTO_RESUME = True
_C.TASK = ''

_C.DATASET = CN()
_C.DATASET.IMAGE_SIZE = [224, 224]
_C.DATASET.DATASET = 'dogcat'
_C.DATASET.DATA_ROOT = ''
_C.DATASET.TRAIN_DATA_ROOT = 'data/train/'
_C.DATASET.TEST_DATA_ROOT = 'data/test1/'

_C.MODEL = CN()
_C.MODEL.NAME = 'baseline'
_C.MODEL.NUM_CLASSES = 20
_C.MODEL.DEVICE = 'cuda'

_C.MODEL.BACKBONE = CN()
_C.MODEL.BACKBONE.NAME = 'resnet50'
_C.MODEL.BACKBONE.PRETRAINED_WEIGHT = False
_C.MODEL.BACKBONE.PRETRAINED_PATH = 'checkpoints/model_99.pth'

_C.MODEL.NECK = CN()
_C.MODEL.NECK.NAME = ''
_C.MODEL.NECK.IN_CHANNELS = []
_C.MODEL.NECK.OUT_CHANNELS = 256

_C.MODEL.HEAD = CN()
_C.MODEL.HEAD.NAME = ''
_C.MODEL.HEAD.ACTIVATION = 'leaky_relu'
_C.MODEL.HEAD.OUTPUT_DIM = 1000
_C.MODEL.HEAD.INPUT_DIM = 1280  # MobileNet_V2
_C.MODEL.HEAD.HIDDEN_DIMS = [512, 256]
_C.MODEL.HEAD.BN = True
_C.MODEL.HEAD.DROPOUT = -1.0
_C.MODEL.HEAD.NUM_CLASSES = 21
_C.MODEL.HEAD.NUM_BOXES = []

_C.MODEL.ANCHOR_GENERATOR = CN()
_C.MODEL.ANCHOR_GENERATOR.FMAP_DIMS = []
_C.MODEL.ANCHOR_GENERATOR.OBJ_SCALES = []
_C.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = []


# SSD configurations
_C.MODEL.SSD = CN()
_C.MODEL.SSD.MIN_SCORE = 0.0
_C.MODEL.SSD.MAX_OVERLAP = 0.0
_C.MODEL.SSD.TOP_K = 200

# YOLOv1 configurations
_C.MODEL.YOLOV1 = CN()
_C.MODEL.YOLOV1.NUM_PREDICTIONS_PER_LOCATION = 2
_C.MODEL.YOLOV1.MIN_SCORE = 0.05
_C.MODEL.YOLOV1.IOU_THRESHOLD = 0.5


_C.TRAIN = CN()
_C.TRAIN.INPUT_SIZE = [448, 448]
_C.TRAIN.BATCH_SIZE = 128
_C.TRAIN.USE_GPU = True
_C.TRAIN.NUM_WORKERS = 4
_C.TRAIN.SHUFFLE = True
_C.TRAIN.PRINT_FREQ = 5
_C.TRAIN.MAX_EPOCH = 100
_C.TRAIN.OPTIMIZER = 'adam'
_C.TRAIN.SCHEDULER = 'exp'
_C.TRAIN.STEP_SIZE = 10
_C.TRAIN.LR_STEPS = [40, 70, 90]
_C.TRAIN.LR = 0.01
_C.TRAIN.LR_DECAY = 0.95
_C.TRAIN.WEIGHT_DECAY = 1e-4
_C.TRAIN.COSINE_T_MAX = 20

_C.TRAIN.LOSS = CN()
_C.TRAIN.LOSS.NAME = ''

_C.OUTPUT_PATH = '../output'


def get_cfg_defaults():
    return _C.clone()


# TODO: print warning and add configs automatically if not in config file
# but I think in most cases, it should be prohibited, so maybe just throw an error.
def update_cfg(cfg, args):
    cfg.defrost()
    cfg.merge_from_file(args.config)
    if args.work_dir:
        cfg.OUTPUT_PATH = args.work_dir
    # if args.data_path:
    #     cfg.DATASET.DATA_ROOT = args.data_path
    # cfg.freeze()
