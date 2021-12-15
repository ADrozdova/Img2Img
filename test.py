import argparse
import os
import warnings

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

import src.model as module_arch
from src.datasets.dataset import TestDataset
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

    run_model(gen_A, params["img_folder_A"], params["save_dir_A"], device)
    run_model(gen_B, params["img_folder_B"], params["save_dir_B"], device)


def run_model(model, img_folder, save_dir, device):
    dataset = TestDataset(img_folder)

    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=1)

    for batch_idx, batch in tqdm(
            enumerate(dataloader),
            desc="test",
            total=len(dataloader),
    ):
        file, image = batch
        image = image.to(device)
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