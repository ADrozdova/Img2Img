import mmcv
import numpy as np
import torch
from omegaconf import read_write

import src.model as module_arch
from src.segmentation.datasets import (COCOObjectDataset, PascalContextDataset,
                                       PascalVOCDataset)
from src.segmentation.evaluation import (GROUP_PALETTE, build_seg_inference)


def init_gen(config, device, local_rank):
    logger = config.get_logger("train")
    gen_B = config.init_obj(config["generator"], module_arch)
    gen_A = config.init_obj(config["generator"], module_arch)

    if local_rank == 0:
        logger.info(gen_A)

    gen_B = gen_B.to(device)
    gen_A = gen_A.to(device)

    gen_A = torch.nn.parallel.DistributedDataParallel(
        gen_A, device_ids=[local_rank], output_device=local_rank
    )
    gen_B = torch.nn.parallel.DistributedDataParallel(
        gen_B, device_ids=[local_rank], output_device=local_rank
    )

    return gen_A, gen_B


def init_disc(config, device, local_rank):
    logger = config.get_logger("train")
    disc_A = config.init_obj(config["discriminator"], module_arch)
    disc_B = config.init_obj(config["discriminator"], module_arch)

    if local_rank == 0:
        logger.info(disc_A)

    disc_A = disc_A.to(device)
    disc_B = disc_B.to(device)

    disc_A = torch.nn.parallel.DistributedDataParallel(
        disc_A, device_ids=[local_rank], output_device=local_rank
    )
    disc_B = torch.nn.parallel.DistributedDataParallel(
        disc_B, device_ids=[local_rank], output_device=local_rank
    )

    return disc_A, disc_B


def get_seg_model(cfg, model, text_transform, dataset, additional_classes):
    if dataset == 'voc' or dataset == 'Pascal VOC':
        dataset_class = PascalVOCDataset
        seg_cfg = 'src/segmentation/configs/_base_/datasets/pascal_voc12.py'
    elif dataset == 'coco' or dataset == 'COCO':
        dataset_class = COCOObjectDataset
        seg_cfg = 'src/segmentation/configs/_base_/datasets/coco.py'
    elif dataset == 'context' or dataset == 'Pascal Context':
        dataset_class = PascalContextDataset
        seg_cfg = 'src/segmentation/configs/_base_/datasets/pascal_context.py'
    else:
        raise ValueError('Unknown dataset: {}'.format(dataset))
    with read_write(cfg):
        cfg.evaluate.seg.cfg = seg_cfg

    dataset_cfg = mmcv.Config()
    dataset_cfg.CLASSES = list(dataset_class.CLASSES)
    dataset_cfg.PALETTE = dataset_class.PALETTE.copy()

    if len(additional_classes) > 0:
        additional_classes = list(
            set(additional_classes) - set(dataset_cfg.CLASSES))
        dataset_cfg.CLASSES.extend(additional_classes)
        dataset_cfg.PALETTE.extend(GROUP_PALETTE[np.random.choice(
            list(range(len(GROUP_PALETTE))), len(additional_classes))])
    seg_model = build_seg_inference(model, dataset_cfg, text_transform,
                                    cfg.evaluate.seg)
    return seg_model
