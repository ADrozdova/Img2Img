import os
import os.path as osp
import warnings
from collections import namedtuple

import mmcv
import numpy as np
import torch
from mmcv.cnn.utils import revert_sync_batchnorm
from mmcv.image import tensor2imgs
from mmcv.parallel import collate, scatter
from omegaconf import read_write

from src.datasets import build_text_transform
from src.model import build_model
from src.segmentation.datasets import (COCOObjectDataset, PascalContextDataset,
                                       PascalVOCDataset)
from src.segmentation.evaluation import (GROUP_PALETTE, build_seg_demo_pipeline,
                                         build_seg_inference)
from src.utils import get_config, load_checkpoint, prepare_device

warnings.filterwarnings("ignore", category=UserWarning)

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


class LoadImage:
    """A simple pipeline to load image."""

    def __call__(self, results):
        """Call function to load images into results.
        Args:
            results (dict): A result dict contains the file name
                of the image to be read.
        Returns:
            dict: ``results`` will be returned containing loaded image.
        """

        if isinstance(results['img'], str):
            results['filename'] = results['img']
            results['ori_filename'] = results['img']
        else:
            results['filename'] = None
            results['ori_filename'] = None
        img = mmcv.imread(results['img'])
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        return results


def inference(cfg, model, test_pipeline, text_transform, vis_modes, dataset, additional_classes, input_img, output_dir):
    if dataset == 'voc' or dataset == 'Pascal VOC':
        dataset_class = PascalVOCDataset
        seg_cfg = 'segmentation/configs/_base_/datasets/pascal_voc12.py'
    elif dataset == 'coco' or dataset == 'COCO':
        dataset_class = COCOObjectDataset
        seg_cfg = 'segmentation/configs/_base_/datasets/coco_object164k.py'
    elif dataset == 'context' or dataset == 'Pascal Context':
        dataset_class = PascalContextDataset
        seg_cfg = 'segmentation/configs/_base_/datasets/pascal_context.py'
    else:
        raise ValueError('Unknown dataset: {}'.format(dataset))
    with read_write(cfg):
        cfg.evaluate.seg.cfg = seg_cfg

    dataset_cfg = mmcv.Config()
    dataset_cfg.CLASSES = list(dataset_class.CLASSES)
    dataset_cfg.PALETTE = dataset_class.PALETTE.copy()

    if len(additional_classes) > 0:
        additional_classes = additional_classes.split(',')
        additional_classes = list(
            set(additional_classes) - set(dataset_cfg.CLASSES))
        dataset_cfg.CLASSES.extend(additional_classes)
        dataset_cfg.PALETTE.extend(GROUP_PALETTE[np.random.choice(
            list(range(len(GROUP_PALETTE))), len(additional_classes))])
    seg_model = build_seg_inference(model, dataset_cfg, text_transform,
                                    cfg.evaluate.seg)

    device = next(seg_model.parameters()).device
    # prepare data
    data = dict(img=input_img)
    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)
    if next(seg_model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    else:
        data['img_metas'] = [i.data[0] for i in data['img_metas']]
    with torch.no_grad():
        result = seg_model(return_loss=False, rescale=True, **data)

    img_tensor = data['img'][0]
    img_metas = data['img_metas'][0]
    imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
    assert len(imgs) == len(img_metas)

    out_file_dict = dict()
    for img, img_meta in zip(imgs, img_metas):
        h, w, _ = img_meta['img_shape']
        img_show = img[:h, :w, :]

        ori_h, ori_w = img_meta['ori_shape'][:-1]
        img_show = mmcv.imresize(img_show, (ori_w, ori_h))

        for vis_mode in vis_modes:
            out_file = osp.join(output_dir, 'vis_imgs', vis_mode,
                                f'{vis_mode}.jpg')
            seg_model.show_result(img_show, img_tensor.to(device), result,
                                  out_file, vis_mode)
            out_file_dict[vis_mode] = out_file

    return out_file_dict


def main(local_rank):
    checkpoint_url = 'https://github.com/xvjiarui/GroupViT/releases/download/v1.0.0/group_vit_gcc_yfcc_30e-74d335e6.pth'
    cfg_path = 'config/group_vit_gcc_yfcc_30e.yml'
    output_dir = 'demo/output'
    vis_modes = ['input_pred_label', 'final_group']

    PSEUDO_ARGS = namedtuple('PSEUDO_ARGS',
                             ['cfg', 'opts', 'resume', 'vis', 'local_rank'])

    args = PSEUDO_ARGS(
        cfg=cfg_path, opts=[], resume=checkpoint_url, vis=vis_modes, local_rank=0)

    cfg = get_config(args)

    with read_write(cfg):
        cfg.evaluate.eval_only = True

    device = prepare_device(local_rank)
    torch.distributed.init_process_group(backend="nccl")

    model = build_model(cfg.model)
    model = revert_sync_batchnorm(model)
    model.to(device)
    model.eval()

    load_checkpoint(cfg, model, None, None)

    text_transform = build_text_transform(False, cfg.data.text_aug, with_dc=False)
    test_pipeline = build_seg_demo_pipeline()

    voc_results = inference(cfg, model, test_pipeline, text_transform, vis_modes, 'voc', [], 'examples/voc.jpg', output_dir)


if __name__ == "__main__":
    main(int(os.environ["LOCAL_RANK"]))
