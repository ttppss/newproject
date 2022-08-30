from pathlib import Path
import logging
import time
import os
import inspect

from fvcore.nn import FlopCountAnalysis, flop_count_table

logger = logging.getLogger(__name__)


def create_logger(cfg, cfg_name):
    assert len(cfg.OUTPUT_PATH) > 0, 'Error: The output directory need to be specified.'
    output_dir = Path(cfg.OUTPUT_PATH)
    if not output_dir.exists():
        print('=> creating {}'.format(output_dir))
        output_dir.mkdir()

    dataset = cfg.DATASET.DATASET
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