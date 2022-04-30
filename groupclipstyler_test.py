import os
import warnings
import argparse
from argparse import Namespace
from collections import namedtuple
import collections

import clip
import numpy as np
import torch
from PIL import Image
from mmcv.cnn.utils import revert_sync_batchnorm
from omegaconf import read_write
from torch import optim
from torchvision import models, transforms
from torchvision.transforms.functional import adjust_contrast
from src.loss import GramMSELoss, GramMatrix
from src.datasets import build_text_transform
from src.datasets.imagenet_template import full_imagenet_templates
from src.model import UNet, build_model
from src.segmentation.evaluation import (build_seg_demo_pipeline)
from src.utils import get_config, load_checkpoint
from src.utils import prepare_device, get_seg, compose_text_with_templates, load_image2, get_features, img_normalize, \
    get_image_prior_losses, clip_normalize
from src.utils.init_models import get_seg_model
from src.utils.parse_config import ConfigParser


warnings.filterwarnings("ignore", category=UserWarning)

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def run_styleransfer(vgg, content, image_path, text, training_args, gram_args, seg, seg_model, img_size, device):
    img_height, img_width = img_size
    source = " ".join(content)

    content_image = load_image2(image_path, img_height=img_height, img_width=img_width)
    content_image = content_image.to(device)

    content_features = get_features(img_normalize(content_image, device), vgg)

    style_net = UNet()
    style_net.to(device)

    optimizer = optim.Adam(style_net.parameters(), lr=training_args["lr"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

    total_loss_epoch = []
    output_image = content_image

    cropper = transforms.Compose([
        transforms.RandomCrop(training_args["crop_size"])
    ])
    augment = transforms.Compose([
        transforms.RandomPerspective(fill=0, p=1, distortion_scale=0.5),
        transforms.Resize(224)
    ])

    patches_seg = seg.unfold(0, gram_args["patch_size"], gram_args["patch_step"]).unfold(1, gram_args["patch_size"], gram_args["patch_step"])
    patches_coords = []
    patches_classes = []
    for i in range(patches_seg.shape[0]):
        for j in range(patches_seg.shape[1]):
            seg_class = torch.unique(patches_seg[i, j])
            if len(seg_class) == 1:
                patches_coords.append((i, j))
                patches_classes.append(seg_class[0].item())

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

    num_crops = training_args["num_crops"]
    for epoch in range(0, training_args["max_step"] + 1):
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
        loss_temp[loss_temp < training_args["thresh"]] = 0
        loss_patch += loss_temp.mean()

        glob_features = clip_model.encode_image(clip_normalize(target, device))
        glob_features /= (glob_features.clone().norm(dim=-1, keepdim=True))

        glob_direction = (glob_features - source_features)
        glob_direction /= glob_direction.clone().norm(dim=-1, keepdim=True)

        loss_glob = (1 - torch.cosine_similarity(glob_direction, text_direction, dim=1)).mean()

        reg_tv = training_args["lambda_tv"] * get_image_prior_losses(target)

        loss_gram = 0

        if gram_args["lambda_gram"] != 0:
            target_patches = target.unfold(2, gram_args["patch_size"], gram_args["patch_step"]).unfold(3, gram_args["patch_size"], gram_args["patch_step"])
            indices = np.random.choice(range(len(patches_coords)), int(len(patches_coords) * gram_args["patch_rate"]), replace=False)

            for i in range(len(indices)):
                for j in range(i):
                    a, b = indices[i], indices[j]
                    loss_gram_patch = GramMSELoss()(target_patches[:, :, patches_coords[a][0], patches_coords[a][1], :],
                                                    GramMatrix()(target_patches[:, :, patches_coords[b][0], patches_coords[b][1], :]))
                    if patches_classes[a] == patches_classes[b]:
                        loss_gram += loss_gram_patch
                    else:
                        loss_gram -= loss_gram_patch

        total_loss = training_args["lambda_patch"] * loss_patch + training_args["content_weight"] * content_loss +\
                     reg_tv + training_args["lambda_dir"] * loss_glob + gram_args["lambda_gram"] * loss_gram
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
            if isinstance(loss_gram, int):
                print('Gram MSE loss', loss_gram)
            else:
                print('Gram MSE loss', loss_gram.item())
            output_image = target.clone()

    output_image = torch.clamp(output_image, 0, 1).squeeze()
    output_image = adjust_contrast(output_image, 1.5)
    output_image = transforms.ToPILImage()(output_image.cpu())
    output_image = output_image.resize(img_size)
    return np.asarray(output_image)


def inference(cfg, model, test_pipeline, text_transform, config, device):
    img_names = config["data"]["image_path"]
    if isinstance(img_names, str):
        img_names = [img_names]
    texts_all = config["clipstyler"]["args"]["texts"]

    if isinstance(texts_all, str):
        texts_all = [texts_all]
    all_words = []
    for text in texts_all:
        all_words.extend(text.split())

    seg_model = get_seg_model(cfg, model, text_transform, config["groupvit"]["dataset"], all_words)

    # get network
    vgg = models.vgg19(pretrained=True).features
    vgg.to(device)
    for parameter in vgg.parameters():
        parameter.requires_grad_(False)

    for img_name, text_style in zip(img_names, texts_all):
        input_img = os.path.join(config["data"]["input"], img_name)
        seg = get_seg(seg_model, test_pipeline, input_img)
        content = []

        for word in text_style.split():
            if len(seg[seg == seg_model.CLASSES.index(word)]) > 0:
                content.append(word)

        gram_args_all = config["clipstyler"]["gram_args"]
        for patch_size, patch_step in gram_args_all["patch_size_step"]:
            for lambda_gram in gram_args_all["lambda_gram"]:
                for patch_rate in gram_args_all["patch_rate"]:
                    gram_args = {"patch_size": patch_size, "patch_step": patch_step, "lambda_gram": lambda_gram, "patch_rate": patch_rate}

                    print()
                    print("Image:", img_name, "parameters:", gram_args)
                    print()

                    stylized = run_styleransfer(vgg, content, input_img, text_style, config["clipstyler"]["args"], gram_args, torch.from_numpy(seg), seg_model, (seg.shape[0], seg.shape[1]), device)
                    out_path = os.path.join(config["data"]["output"], img_name.split(".")[0] + "_psz_" + str(patch_size) +
                                            "_pst_" + str(patch_step) + "_lg_" + str(lambda_gram) + "_pr_" +
                                            str(patch_rate) + ".jpg")
                    Image.fromarray(np.uint8(stylized)).save(out_path)


def main(config):
    PSEUDO_ARGS = namedtuple('PSEUDO_ARGS',
                             ['cfg', 'opts', 'resume', 'vis', 'local_rank'])

    args = PSEUDO_ARGS(
        cfg=config["groupvit"]["cfg_path"], opts=[], resume=config["groupvit"]["checkpoint_url"], vis=config["groupvit"]["vis_modes"], local_rank=config.local_rank)

    cfg = get_config(args)

    with read_write(cfg):
        cfg.evaluate.eval_only = True

    device = prepare_device(config.local_rank)
    torch.distributed.init_process_group(backend="nccl")

    model = build_model(cfg.model)
    model = revert_sync_batchnorm(model)
    model.to(device)
    model.eval()

    load_checkpoint(cfg, model, None, None)

    text_transform = build_text_transform(False, cfg.data.text_aug, with_dc=False)
    test_pipeline = build_seg_demo_pipeline()

    # style_text = "person to sketch with black pencil background to watercolour"

    inference(cfg, model, test_pipeline, text_transform, config, device)


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )

    CustomArgs = collections.namedtuple("CustomArgs", "flags type target")
    options = [
        CustomArgs(["--img", "--image"], type=str, target="data;image_path"),
        CustomArgs(["--txt", "--text"], type=str, target="clipstyler;args;text"),
        CustomArgs(["--out", "--output"], type=str, target="data;output"),
        CustomArgs(["--vit_dataset", "--groupvit_dataset"], type=str, target="groupvit;dataset"),
    ]
    config = ConfigParser.from_args(args, options, local_rank=int(os.environ["LOCAL_RANK"]))
    main(config)
