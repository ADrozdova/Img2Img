import os
import warnings
from collections import namedtuple

import mmcv
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from mmcv.cnn.utils import revert_sync_batchnorm
from mmcv.image import tensor2imgs
from mmcv.parallel import collate, scatter
from omegaconf import read_write
from torch import optim
from torchvision import transforms

from src.datasets import build_text_transform
from src.loss import GramMSELoss, GramMatrix
from src.model import build_model
from src.model.styletransfer_vgg import VGG
from src.segmentation.datasets import (COCOObjectDataset, PascalContextDataset,
                                       PascalVOCDataset)
from src.segmentation.evaluation import (GROUP_PALETTE, build_seg_demo_pipeline,
                                         build_seg_inference)
from src.utils import get_config, load_checkpoint
from src.utils import prepare_device

warnings.filterwarnings("ignore", category=UserWarning)

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def get_model(cfg, model, text_transform, dataset, additional_classes):
    if dataset == 'voc' or dataset == 'Pascal VOC':
        dataset_class = PascalVOCDataset
        seg_cfg = 'src/segmentation/configs/_base_/datasets/pascal_voc12.py'
    elif dataset == 'coco' or dataset == 'COCO':
        dataset_class = COCOObjectDataset
        seg_cfg = 'src/segmentation/configs/_base_/datasets/coco_object164k.py'
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


def get_seg(seg_model, test_pipeline, input_img):
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

    return result[0]


def run_styleransfer(vgg, style_image, content_image, device, img_size):
    # pre and post processing for images
    prep = transforms.Compose([transforms.Resize(img_size),
                               transforms.ToTensor(),
                               transforms.Lambda(lambda x: x[torch.LongTensor([2, 1, 0])]),  # turn to BGR
                               transforms.Normalize(mean=[0.40760392, 0.45795686, 0.48501961],  # subtract imagenet mean
                                                    std=[1, 1, 1]),
                               transforms.Lambda(lambda x: x.mul_(255)),
                               ])
    postpa = transforms.Compose([transforms.Lambda(lambda x: x.mul_(1. / 255)),
                                 transforms.Normalize(mean=[-0.40760392, -0.45795686, -0.48501961],  # add imagenet mean
                                                      std=[1, 1, 1]),
                                 transforms.Lambda(lambda x: x[torch.LongTensor([2, 1, 0])]),  # turn to RGB
                                 ])
    postpb = transforms.Compose([transforms.ToPILImage()])

    def postp(tensor):  # to clip results in the range [0,1]
        t = torch.clamp(postpa(tensor), min=0.0, max=1.0)
        return postpb(t)

    imgs = [Image.open(style_image), Image.open(content_image)]
    imgs_torch = [prep(img).unsqueeze(0).to(device) for img in imgs]

    # imgs_torch = [img.unsqueeze(0).to(device) for img in imgs_torch]
    style_image, content_image = imgs_torch

    opt_img = content_image.data.clone()
    opt_img.requires_grad = True

    # define layers, loss functions, weights and compute optimization targets
    style_layers = ['r11', 'r21', 'r31', 'r41', 'r51']
    content_layers = ['r42']
    loss_layers = style_layers + content_layers
    loss_fns = [GramMSELoss()] * len(style_layers) + [nn.MSELoss()] * len(content_layers)

    loss_fns = [loss_fn.to(device) for loss_fn in loss_fns]

    # these are good weights settings:
    style_weights = [1e3 / n ** 2 for n in [64, 128, 256, 512, 512]]
    content_weights = [1e0]
    weights = style_weights + content_weights

    # compute optimization targets
    style_targets = [GramMatrix()(A).detach() for A in vgg(style_image, style_layers)]
    content_targets = [A.detach() for A in vgg(content_image, content_layers)]
    targets = style_targets + content_targets

    # run style transfer
    max_iter = 500
    show_iter = 50
    optimizer = optim.LBFGS([opt_img])
    n_iter = [0]

    while n_iter[0] <= max_iter:
        def closure():
            optimizer.zero_grad()
            out = vgg(opt_img, loss_layers)
            layer_losses = [weights[a] * loss_fns[a](A, targets[a]) for a, A in enumerate(out)]
            loss = sum(layer_losses)
            loss.backward()
            n_iter[0] += 1
            # print loss
            if n_iter[0] % show_iter == (show_iter - 1):
                print('Iteration: %d, loss: %f' % (n_iter[0] + 1, loss.item()))
            return loss

        optimizer.step(closure)

    # display result
    out_img = postp(opt_img.data[0].squeeze())
    return np.asarray(out_img)


def inference(cfg, model, test_pipeline, text_transform, dataset, input_img, output_file, part_to_style, vgg_path, device):
    seg_model = get_model(cfg, model, text_transform, dataset, list(part_to_style.keys()))
    seg = get_seg(seg_model, test_pipeline, input_img)

    # get network
    vgg = VGG()
    vgg.load_state_dict(torch.load(vgg_path))
    for param in vgg.parameters():
        param.requires_grad = False

    vgg = vgg.to(device)
    result = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)

    for part, style_image in part_to_style.items():

        label = seg_model.CLASSES.index(part)
        stylized = run_styleransfer(vgg, style_image, input_img, device, (seg.shape[0], seg.shape[1]))

        result[seg == label, :] = stylized[seg == label, :]

    img_result = Image.fromarray(np.uint8(result))
    img_result.save(output_file)


def main(local_rank):
    checkpoint_url = 'https://github.com/xvjiarui/GroupViT/releases/download/v1.0.0/group_vit_gcc_yfcc_30e-74d335e6.pth'
    cfg_path = 'src/configs/group_vit_gcc_yfcc_30e.yml'
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

    part_to_style = {"face": "vangogh_starry_night.jpg", "background": "patterned_leaves.jpg"}

    inference(cfg, model, test_pipeline, text_transform, 'voc', './test_dataset/12.jpg', "group_styletransfer_output.jpg", part_to_style, "vgg_conv.pth",
              device)


if __name__ == "__main__":
    main(int(os.environ["LOCAL_RANK"]))
