import os
import warnings
from argparse import Namespace
from collections import namedtuple

import clip
import numpy as np
import torch
from PIL import Image
from mmcv.cnn.utils import revert_sync_batchnorm
from omegaconf import read_write
from torch import optim
from torchvision import models
from torchvision import transforms
from torchvision.transforms.functional import adjust_contrast

from src.datasets import build_text_transform
from src.datasets.imagenet_template import full_imagenet_templates
from src.model import UNet
from src.model import build_model
from src.segmentation.evaluation import (build_seg_demo_pipeline)
from src.utils import get_config, load_checkpoint
from src.utils import prepare_device, get_seg, compose_text_with_templates, load_image2, get_features, img_normalize, \
    get_image_prior_losses, clip_normalize
from src.utils.init_models import get_seg_model

warnings.filterwarnings("ignore", category=UserWarning)

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def run_styleransfer(vgg, source, image_dir, text, img_size, device, training_iterations=100):
    img_height, img_width = img_size

    training_args = {
        "lambda_tv": 2e-3,
        "lambda_patch": 9000,
        "lambda_dir": 500,
        "content_weight": 150,
        "crop_size": 128,
        "num_crops": 64,
        "img_height": img_height,
        "img_width": img_width,
        "max_step": training_iterations,
        "lr": 5e-4,
        "thresh": 0.7,
        "content_path": image_dir,
        "text": text
    }
    args = Namespace(**training_args)

    content_image = load_image2(image_dir, img_height=args.img_height, img_width=args.img_width)
    content_image = content_image.to(device)

    content_features = get_features(img_normalize(content_image, device), vgg)

    style_net = UNet()
    style_net.to(device)

    optimizer = optim.Adam(style_net.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

    total_loss_epoch = []
    output_image = content_image

    cropper = transforms.Compose([
        transforms.RandomCrop(args.crop_size)
    ])
    augment = transforms.Compose([
        transforms.RandomPerspective(fill=0, p=1, distortion_scale=0.5),
        transforms.Resize(224)
    ])

    clip_model, preprocess = clip.load('ViT-B/32', device, jit=False)

    with torch.no_grad():
        template_text = compose_text_with_templates(text, full_imagenet_templates)
        tokens = clip.tokenize(template_text).to(device)
        text_features = clip_model.encode_text(tokens).detach()
        text_features = text_features.mean(axis=0, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        template_source = compose_text_with_templates(source, full_imagenet_templates)
        tokens_source = clip.tokenize(template_source).to(device)
        text_source = clip_model.encode_text(tokens_source).detach()
        text_source = text_source.mean(axis=0, keepdim=True)
        text_source /= text_source.norm(dim=-1, keepdim=True)
        source_features = clip_model.encode_image(clip_normalize(content_image, device))
        source_features /= (source_features.clone().norm(dim=-1, keepdim=True))

    num_crops = args.num_crops
    for epoch in range(0, args.max_step + 1):

        scheduler.step()
        target = style_net(content_image, use_sigmoid=True).to(device)
        target.requires_grad_(True)

        target_features = get_features(img_normalize(target, device), vgg)

        content_loss = 0

        content_loss += torch.mean((target_features['conv4_2'] - content_features['conv4_2']) ** 2)
        content_loss += torch.mean((target_features['conv5_2'] - content_features['conv5_2']) ** 2)

        loss_patch = 0
        img_proc = []
        for n in range(num_crops):
            target_crop = cropper(target)
            target_crop = augment(target_crop)
            img_proc.append(target_crop)

        img_proc = torch.cat(img_proc, dim=0)
        img_aug = img_proc

        image_features = clip_model.encode_image(clip_normalize(img_aug, device))
        image_features /= (image_features.clone().norm(dim=-1, keepdim=True))

        img_direction = (image_features - source_features)
        img_direction /= img_direction.clone().norm(dim=-1, keepdim=True)

        text_direction = (text_features - text_source).repeat(image_features.size(0), 1)
        text_direction /= text_direction.norm(dim=-1, keepdim=True)
        loss_temp = (1 - torch.cosine_similarity(img_direction, text_direction, dim=1))
        loss_temp[loss_temp < args.thresh] = 0
        loss_patch += loss_temp.mean()

        glob_features = clip_model.encode_image(clip_normalize(target, device))
        glob_features /= (glob_features.clone().norm(dim=-1, keepdim=True))

        glob_direction = (glob_features - source_features)
        glob_direction /= glob_direction.clone().norm(dim=-1, keepdim=True)

        loss_glob = (1 - torch.cosine_similarity(glob_direction, text_direction, dim=1)).mean()

        reg_tv = args.lambda_tv * get_image_prior_losses(target)

        total_loss = args.lambda_patch * loss_patch + args.content_weight * content_loss + reg_tv + args.lambda_dir * loss_glob
        total_loss_epoch.append(total_loss)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            print("After %d iters:" % epoch)
            print('Total loss: ', total_loss.item())
            print('Content loss: ', content_loss.item())
            print('patch loss: ', loss_patch.item())
            print('dir loss: ', loss_glob.item())
            print('TV loss: ', reg_tv.item())
            output_image = target.clone()

    output_image = torch.clamp(output_image, 0, 1).squeeze()
    output_image = adjust_contrast(output_image, 1.5)
    output_image = transforms.ToPILImage()(output_image.cpu())
    return output_image


def inference(cfg, model, test_pipeline, text_transform, dataset, input_img, output_file, part_to_style, device):
    seg_model = get_seg_model(cfg, model, text_transform, dataset, list(part_to_style.keys()))
    seg = get_seg(seg_model, test_pipeline, input_img)

    # get network
    vgg = models.vgg19(pretrained=True).features
    vgg.to(device)

    # background (to include unlabeled pixels)

    result = run_styleransfer(vgg, "a Photo", input_img, part_to_style['background'], (seg.shape[0], seg.shape[1]), device)

    for part, style in part_to_style.items():
        if part == "background":
            continue

        label = seg_model.CLASSES.index(part)

        if len(result[seg == label, :]) == 0:  # class not found
            continue

        stylized = run_styleransfer(vgg, "a Photo", input_img, style, (seg.shape[0], seg.shape[1]),
                                  device)

        result[seg == label, :] = stylized[seg == label, :]

    img_result = Image.fromarray(np.uint8(result))
    img_result.save(output_file)


def main(local_rank):
    checkpoint_url = 'https://github.com/xvjiarui/GroupViT/releases/download/v1.0.0/group_vit_gcc_yfcc_30e-74d335e6.pth'
    cfg_path = 'src/configs/group_vit_gcc_yfcc_30e.yml'
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

    part_to_style = {"background": "vangogh starry night", "face": "black pencil sketch"}

    inference(cfg, model, test_pipeline, text_transform, 'context', './test_dataset/25.jpg', "group_clipstyler_output.jpg", part_to_style, device)


if __name__ == "__main__":
    main(int(os.environ["LOCAL_RANK"]))
