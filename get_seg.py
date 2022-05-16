import argparse
import collections
import os
import pickle

import numpy as np
import torch
from PIL import Image

from src.datasets import build_text_transform
from src.segmentation.evaluation import build_seg_demo_pipeline
from src.utils import prepare_device, get_seg
from src.utils.init_models import get_seg_model, get_model_from_cfg
from src.utils.parse_config import ConfigParser


def run_seg(cfg, model, test_pipeline, text_transform, config, device):
    img_names = config["data"]["image_path"]
    if isinstance(img_names, str):
        img_names = [img_names]
    texts_all = config["clipstyler"]["args"]["texts"]

    if isinstance(texts_all, str):
        texts_all = [texts_all]

    all_words = []

    img_names = [img_names[config.local_rank]]
    texts_all = [texts_all[config.local_rank]]

    for text in texts_all:
        if isinstance(text, collections.OrderedDict):
            all_words.extend(list(text.keys()))
        else:
            all_words.extend(text.split())

    seg_model = get_seg_model(
        cfg, model, text_transform, config["groupvit"]["dataset"], all_words
    )

    for img_name, text_style in zip(img_names, texts_all):
        input_img = os.path.join(config["data"]["input"], img_name)
        seg = get_seg(seg_model, test_pipeline, input_img)

        parts_idx = dict()
        if config["clipstyler"]["args"]["get_content"]:
            for word in text_style.split():
                if len(seg[seg == seg_model.CLASSES.index(word)]) > 0:
                    parts_idx[word] = seg_model.CLASSES.index(word)

        np.save(os.path.join(config["data"]["output"], img_name.split(".")[0] + "_seg" + ".npy"), seg)
        pickle_file = os.path.join(config["data"]["output"], img_name.split(".")[0] + "_parts_idx" + ".pickle")
        with open(pickle_file, 'wb') as handle:
            pickle.dump(parts_idx, handle, protocol=pickle.HIGHEST_PROTOCOL)


def main(config):
    device = prepare_device(config.local_rank)
    torch.distributed.init_process_group(backend="nccl")

    model, cfg = get_model_from_cfg(config, device)

    run_seg(cfg, model, build_seg_demo_pipeline(), build_text_transform(False, cfg.data.text_aug, with_dc=False),
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