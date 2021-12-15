import argparse
import collections
import warnings

import numpy as np
import itertools
from PIL import Image

import torch
import os
from torchvision import transforms
import src.loss as module_loss
import src.model as module_arch
from src.datasets.utils import get_dataloaders
from src.trainer import Trainer
from src.utils import prepare_device, img_to_jpeg
from src.utils.parse_config import ConfigParser

warnings.filterwarnings("ignore", category=UserWarning)

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def main(config):
    gen_B = config.init_obj(config["generator"], module_arch)
    gen_A = config.init_obj(config["generator"], module_arch)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config["n_gpu"])

    gen_B = gen_B.to(device)
    gen_A = gen_A.to(device)

    if len(device_ids) > 1:
        gen_B = torch.nn.DataParallel(gen_B, device_ids=device_ids)
        gen_A = torch.nn.DataParallel(gen_A, device_ids=device_ids)

    params = config['test']

    checkpoint = torch.load(params['checkpoint_file'])
    state_dict = checkpoint["state_dict_gen_a"]
    gen_A.load_state_dict(state_dict)
    state_dict = checkpoint["state_dict_gen_b"]
    gen_B.load_state_dict(state_dict)

    run_model(gen_A, params["img_folder_A"], params["save_dir_A"])
    run_model(gen_B, params["img_folder_B"], params["save_dir_B"])


def run_model(model, img_folder, save_dir):
    files = os.listdir(img_folder)

    for file in files:
        image = transforms.ToTensor()(Image.open(os.path.join(img_folder, file)))

        result = model(image).squeeze(0)
        img_to_jpeg(result, os.path.join(save_dir, file))


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )

    config = ConfigParser.from_args(args)
    main(config)
