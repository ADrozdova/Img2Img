import argparse
import collections
import os
import warnings

import clip
import numpy as np
import torch
from PIL import Image
from torch import optim
from torchvision import models, transforms
from torchvision.transforms.functional import adjust_contrast
import torch.nn as nn
import itertools

from src.datasets import build_text_transform
from src.datasets.imagenet_template import full_imagenet_templates
from src.loss import GramMSELoss, GramMatrix
from src.model import UNet
from src.segmentation.evaluation import build_seg_demo_pipeline
from src.utils import (
    prepare_device,
    get_seg,
    compose_text_with_templates,
    load_image2,
    get_features,
    img_normalize,
    get_image_prior_losses,
    clip_normalize,
    get_patches_idx,
    get_text_feats,
    get_text_source,
    get_source_features,
    dependency_parse,
    img_dir
)
from src.utils.init_models import get_seg_model, get_model_from_cfg
from src.utils.parse_config import ConfigParser

warnings.filterwarnings("ignore", category=UserWarning)

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def run_styleransfer(
    vgg,
    content,
    image_path,
    text,
    text_parsed,
    classes,
    training_args,
    patch_args,
    seg,
    img_size,
    device,
):
    img_height, img_width = img_size
    source = " ".join(content) if training_args["get_content"] else "a Picture"

    content_image = load_image2(image_path, img_height=img_height, img_width=img_width)
    content_image = content_image.to(device)

    content_features = get_features(img_normalize(content_image, device), vgg)

    style_net = UNet()
    style_net.to(device)

    optimizer = optim.Adam(style_net.parameters(), lr=training_args["lr"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

    total_loss_epoch = []
    output_image = content_image

    cropper = transforms.Compose([transforms.RandomCrop(training_args["crop_size"])])
    augment = transforms.Compose(
        [
            transforms.RandomPerspective(fill=0, p=1, distortion_scale=0.5),
            transforms.Resize(224),
        ]
    )

    clip_model, preprocess = clip.load("ViT-B/32", device, jit=False)

    if patch_args["lambda_gram"] != 0 or patch_args["lambda_clip_patch"] != 0:
        patches_coords, patches_classes = get_patches_idx(seg, patch_args["patch_size"], patch_args["patch_step"])

        content_img_patches = content_image.unfold(
            2, patch_args["patch_size"], patch_args["patch_step"]
        ).unfold(3, patch_args["patch_size"], patch_args["patch_step"])

        patches_content = [content_img_patches[:, :, coords[0], coords[1], :] for coords in patches_coords]
        patches_features = get_source_features(clip_model, device, torch.cat(patches_content, dim=0))

    text_features = get_text_feats(clip, clip_model, text, device)
    text_source = get_text_source(clip, clip_model, source, device)
    source_features = get_source_features(clip_model, device, content_image)

    parts_text_dirs = dict()
    for part, text_part in text_parsed.items():
        text_features_part = get_text_feats(clip, clip_model, text, device)

        text_source_part = get_text_source(clip, clip_model, part, device)

        parts_text_dirs[classes.index(part)] = text_features_part - text_source_part


    num_crops = training_args["num_crops"]
    for epoch in range(0, training_args["max_step"] + 1):
        scheduler.step()
        target = style_net(content_image, use_sigmoid=True).to(device)
        target.requires_grad_(True)

        target_features = get_features(img_normalize(target, device), vgg)

        content_loss = torch.mean((target_features["conv4_2"] - content_features["conv4_2"]) ** 2)
        content_loss += torch.mean((target_features["conv5_2"] - content_features["conv5_2"]) ** 2)

        img_proc = []
        for _ in range(num_crops):
            img_proc.append(augment(cropper(target)))

        img_aug = torch.cat(img_proc, dim=0)

        img_direction = img_dir(img_aug, source_features, clip_model, device)
        text_direction = (text_features - text_source).repeat(img_aug.size(0), 1)
        text_direction /= text_direction.norm(dim=-1, keepdim=True)
        loss_temp = 1 - torch.cosine_similarity(img_direction, text_direction, dim=1)
        loss_temp[loss_temp < training_args["thresh"]] = 0

        loss_patch = loss_temp.mean()

        glob_features = clip_model.encode_image(clip_normalize(target, device))
        glob_features /= glob_features.clone().norm(dim=-1, keepdim=True)

        glob_direction = glob_features - source_features
        glob_direction /= glob_direction.clone().norm(dim=-1, keepdim=True)

        loss_glob = (1 - torch.cosine_similarity(glob_direction, text_direction, dim=1)).mean()

        reg_tv = training_args["lambda_tv"] * get_image_prior_losses(target)

        loss_gram = 0
        loss_clip_patches = 0

        if patch_args["lambda_gram"] != 0  or patch_args["lambda_clip_patch"] != 0:
            target_patches = target.unfold(
                2, patch_args["patch_size"], patch_args["patch_step"]
            ).unfold(3, patch_args["patch_size"], patch_args["patch_step"])
            indices = np.random.choice(
                range(len(patches_coords)),
                int(len(patches_coords) * patch_args["patch_rate"]),
                replace=False,
            )

            patches_selected = []
            for i in range(len(indices)):
                patches_selected.append(target_patches[:, :, patches_coords[indices[i]][0], patches_coords[indices[i]][1], :])

            patches_selected = torch.cat(patches_selected, dim=0)
            patches_gram = GramMatrix()(patches_selected)

            text_dirs_patches = []
            source_features_patches = []

            for i in range(len(indices)):
                a = indices[i]
                text_dirs_patches.append(parts_text_dirs[patches_classes[a]])
                source_features_patches.append(patches_features[a:a+1])

                if patch_args["lambda_gram"] != 0:
                    for j in range(i):
                        loss_gram_patch = nn.MSELoss()(patches_gram[i], patches_gram[j])
                        if patches_classes[a] == patches_classes[indices[j]]:
                            loss_gram += loss_gram_patch
                        else:
                            loss_gram -= loss_gram_patch
            if patch_args["lambda_clip_patch"] != 0:
                image_features_patches = clip_model.encode_image(clip_normalize(augment(patches_selected), device))
                image_features_patches /= image_features_patches.clone().norm(dim=-1, keepdim=True)

                source_features_patches = torch.cat(source_features_patches, dim=0)

                img_direction_patches = image_features_patches - source_features_patches
                img_direction_patches /= img_direction_patches.clone().norm(dim=-1, keepdim=True)

                text_dirs_patches = torch.cat(text_dirs_patches, dim=0)
                text_dirs_patches /= text_dirs_patches.norm(dim=-1, keepdim=True)

                loss_clip_patches = 1 - torch.cosine_similarity(img_direction_patches, text_dirs_patches, dim=1)
                # loss_clip_patches[loss_clip_patches < training_args["thresh"]] = 0
                loss_clip_patches = loss_clip_patches.mean()

        total_loss = (
            training_args["lambda_patch"] * loss_patch
            + training_args["content_weight"] * content_loss
            + reg_tv
            + training_args["lambda_dir"] * loss_glob
            + patch_args["lambda_gram"] * loss_gram
            + patch_args["lambda_clip_patch"] * loss_clip_patches
        )
        total_loss_epoch.append(total_loss)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            print("After %d iters:" % epoch)
            print("Total loss: ", total_loss.item())
            print("Content loss: ", content_loss.item())
            print("patch loss: ", loss_patch.item())
            print("dir loss: ", loss_glob.item())
            print("TV loss: ", reg_tv.item())
            if isinstance(loss_gram, int):
                print("Gram MSE loss", loss_gram)
            else:
                print("Gram MSE loss", loss_gram.item())
            if isinstance(loss_clip_patches, int):
                print("Clip patch loss", loss_clip_patches)
            else:
                print("Clip patch loss", loss_clip_patches.item())
            output_image = target.clone()

    output_image = adjust_contrast(torch.clamp(output_image, 0, 1).squeeze(), 1.5)
    output_image = transforms.ToPILImage()(output_image.cpu())
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
        if isinstance(text, collections.OrderedDict):
            all_words.extend(list(text.keys()))
        else:
            all_words.extend(text.split())

    seg_model = get_seg_model(
        cfg, model, text_transform, config["groupvit"]["dataset"], all_words
    )

    # get network
    vgg = models.vgg19(pretrained=True).features
    vgg.to(device)
    for parameter in vgg.parameters():
        parameter.requires_grad_(False)

    for img_name, text_style in zip(img_names, texts_all):
        input_img = os.path.join(config["data"]["input"], img_name)
        seg = get_seg(seg_model, test_pipeline, input_img)

        content = []
        if config["clipstyler"]["args"]["get_content"]:
            for word in text_style.split():
                if len(seg[seg == seg_model.CLASSES.index(word)]) > 0:
                    content.append(word)

        johnny = dependency_parse(content, text_style) if isinstance(text_style, str) else text_style
        print(johnny)

        patch_args_all = config["clipstyler"]["patch_args"]

        to_iter = [range(config["data"]["n_trials"]), patch_args_all["patch_size_step"], patch_args_all["lambda_gram"],
                   patch_args_all["patch_rate"], patch_args_all["lambda_clip_patch"]]

        for item in itertools.product(*to_iter):
            trial, (patch_size, patch_step), lambda_gram, patch_rate, lambda_clip_patch = item
            patch_args = {
                "patch_size": patch_size,
                "patch_step": patch_step,
                "lambda_gram": lambda_gram,
                "patch_rate": patch_rate,
                "lambda_clip_patch": lambda_clip_patch
            }

            print("\nImage:", img_name, "parameters:", patch_args, "\n")

            if config["data"]["use_masks"]:
                # just so there are no black spots
                background = "background" if 'background' in text_style else list(text_style.keys())[0]
                result = run_styleransfer(vgg, content, input_img, text_style[background], johnny, seg_model.CLASSES,
                                          config["clipstyler"]["args"], patch_args, torch.from_numpy(seg),
                                          (seg.shape[0], seg.shape[1]), device)

                for part, style in text_style.items():
                    label = seg_model.CLASSES.index(part)

                    if len(result[seg == label, :]) == 0 or part == "background":  # class not found
                        continue

                    stylized = run_styleransfer(vgg, content, input_img, style, johnny, seg_model.CLASSES,
                                                config["clipstyler"]["args"], patch_args, torch.from_numpy(seg),
                                                (seg.shape[0], seg.shape[1]), device)

                    result[seg == label, :] = stylized[seg == label, :]

                out_path = os.path.join(config["data"]["output"], img_name.split(".")[0])
                if not os.path.exists(out_path):
                    os.mkdir(out_path)
                Image.fromarray(np.uint8(result)).save(os.path.join(out_path, str(trial) + ".jpg"))
            else:
                stylized = run_styleransfer(vgg, content, input_img, text_style, johnny, seg_model.CLASSES,
                                            config["clipstyler"]["args"], patch_args, torch.from_numpy(seg),
                                            (seg.shape[0], seg.shape[1]), device)

                out_path = os.path.join(config["data"]["output"], img_name.split(".")[0] + "_psz_" + str(patch_size) +
                                        "_pst_" + str(patch_step) + "_lg_" + str(lambda_gram) + "_pr_" + str(patch_rate) +
                                        "_lcp_" + str(lambda_clip_patch))
                if not os.path.exists(out_path):
                    os.mkdir(out_path)
                Image.fromarray(np.uint8(stylized)).save(os.path.join(out_path, str(trial) + ".jpg"))


def main(config):
    device = prepare_device(config.local_rank)
    torch.distributed.init_process_group(backend="nccl")

    model, cfg = get_model_from_cfg(config, device)

    inference(cfg, model, build_seg_demo_pipeline(), build_text_transform(False, cfg.data.text_aug, with_dc=False),
              config, device)


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
        CustomArgs(
            ["--vit_dataset", "--groupvit_dataset"], type=str, target="groupvit;dataset"
        ),
    ]
    config = ConfigParser.from_args(
        args, options, local_rank=int(os.environ["LOCAL_RANK"])
    )
    main(config)
