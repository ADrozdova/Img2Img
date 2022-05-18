import os
import argparse
from collections import namedtuple
import collections

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from mmcv.cnn.utils import revert_sync_batchnorm
from omegaconf import read_write
from torch import optim
from torchvision import transforms

from src.datasets import build_text_transform
from src.loss import GramMSELoss, GramMatrix
from src.model import build_model
from src.model.styletransfer_vgg import VGG
from src.segmentation.evaluation import build_seg_demo_pipeline
from src.utils import get_config, load_checkpoint, prepare_device, get_seg
from src.utils.init_models import get_seg_model, get_model_from_cfg
from src.utils.parse_config import ConfigParser


def run_styleransfer(vgg, style_image, content_image, device, img_size, max_iter=500):
    # get network

    # pre and post processing for images
    prep = transforms.Compose([transforms.Resize(img_size),
                               transforms.ColorJitter(brightness=.005, hue=.003),
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


def inference(cfg, model, test_pipeline, text_transform, device):
    img_names = config["data"]["image_path"]
    if isinstance(img_names, str):
        img_names = [img_names]
    texts_all = config["styletransfer"]["texts"]

    if isinstance(texts_all, str):
        texts_all = [texts_all]

    all_words = []

    img_cnt = int(len(img_names) / 6)

    idx_start = img_cnt * config.local_rank
    idx_end = img_cnt * (config.local_rank + 1) if config.local_rank < 5 else len(img_names)

    img_names = img_names[idx_start:idx_end]
    texts_all = texts_all[idx_start:idx_end]

    print(img_names, len(img_names), len(texts_all))

    for text in texts_all:
        if isinstance(text, collections.OrderedDict):
            all_words.extend(list(text.keys()))
        else:
            all_words.extend(text.split())

    seg_model = get_seg_model(
        cfg, model, text_transform, config["groupvit"]["dataset"], all_words
    )
    vgg = VGG()
    vgg.load_state_dict(torch.load(config["styletransfer"]["vgg_path"]))
    for param in vgg.parameters():
        param.requires_grad = False

    vgg = vgg.to(device)

    for img_name, part_to_style in zip(img_names, texts_all):
        input_img = os.path.join(config["data"]["input"], img_name)
        seg = get_seg(seg_model, test_pipeline, input_img)

        for trial in range(config["data"]["n_trials"]):
            # background (to include unlabeled pixels)
            background = "background" if 'background' in part_to_style else list(part_to_style.keys())[0]
            result = run_styleransfer(vgg, part_to_style[background], input_img, device, (seg.shape[0], seg.shape[1]))

            for part, style_image in part_to_style.items():
                if part == background:
                    continue

                label = seg_model.CLASSES.index(part)

                if len(result[seg == label, :]) == 0:  # class not found
                    continue

                stylized = run_styleransfer(vgg, style_image, input_img, device, (seg.shape[0], seg.shape[1]))

                result[seg == label, :] = stylized[seg == label, :]

            out_file = img_name.split(".")[0] + "_" + str(trial) + ".jpg"
            if not os.path.exists(os.path.join(config["data"]["output"], img_name.split(".")[0])):
                os.mkdir(os.path.join(config["data"]["output"], img_name.split(".")[0]))

            Image.fromarray(np.uint8(result)).save(os.path.join(config["data"]["output"], img_name.split(".")[0], out_file))


def main(config):
    device = prepare_device(config.local_rank)
    torch.distributed.init_process_group(backend="nccl")

    model, cfg = get_model_from_cfg(config, device)

    inference(cfg, model, build_seg_demo_pipeline(), build_text_transform(False, cfg.data.text_aug, with_dc=False), device)


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )

    config = ConfigParser.from_args(
        args, [], local_rank=int(os.environ["LOCAL_RANK"])
    )
    main(config)
