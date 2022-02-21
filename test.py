import argparse
import os
import warnings

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from pathlib import Path

import src.model as module_arch
from src.datasets.dataset import TestDataset
from src.utils import prepare_device, img_to_jpeg
from src.utils.parse_config import ConfigParser
from src.utils.init_models import init_gen

warnings.filterwarnings("ignore", category=UserWarning)

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def main(config):
    device = prepare_device(config.local_rank)
    torch.distributed.init_process_group(backend="nccl")

    gen_A, gen_B = init_gen(config, device, config.local_rank)

    params = config["test"]

    checkpoint = torch.load(params["checkpoint_file"])

    gen_A = torch.nn.parallel.DistributedDataParallel(
        gen_A, device_ids=[config.local_rank], output_device=config.local_rank
    )
    gen_B = torch.nn.parallel.DistributedDataParallel(
        gen_B, device_ids=[config.local_rank], output_device=config.local_rank
    )

    state_dict = checkpoint["state_dict_gen_A"]
    gen_A.load_state_dict(state_dict)
    state_dict = checkpoint["state_dict_gen_B"]
    gen_B.load_state_dict(state_dict)

    resize = None
    if "resize" in params:
        resize = params["resize"]

    if "img_folder_A" in params:
        run_model(
            gen_B,
            params["img_folder_A"],
            params["save_dir_B"],
            params["save_dir_A_true"],
            device,
            resize,
            params["batch_size"],
        )
    if "img_folder_B" in params:
        run_model(
            gen_A,
            params["img_folder_B"],
            params["save_dir_A"],
            params["save_dir_B_true"],
            device,
            resize,
            params["batch_size"],
        )


def run_model(
    model, img_folder, save_dir, save_dir_true, device, resize=None, batch_size=1
):
    if resize is not None:
        transform = transforms.Compose(
            [
                transforms.Resize((resize, resize)),
            ]
        )
    else:
        transform = None
    dataset = TestDataset(img_folder, transform=transform)

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    Path(save_dir_true).mkdir(parents=True, exist_ok=True)

    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size)

    for batch_idx, batch in tqdm(
        enumerate(dataloader),
        desc="test",
        total=len(dataloader),
    ):
        files, images = batch
        images = images.to(device)
        result = model(images)
        result = torch.clamp(result, min=0.0, max=1.0)
        
        if batch_idx == 0:
            print(result[0])
            print("TORCH MIN", torch.min(result[0]))
            print("TORCH MAX", torch.max(result[0]))

        for i in range(len(result)):
            img_to_jpeg(result[i], os.path.join(save_dir, files[i]))
            img_to_jpeg(images[i], os.path.join(save_dir_true, files[i]))


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-r",
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )

    config = ConfigParser.from_args(args, local_rank=int(os.environ["LOCAL_RANK"]))
    main(config)
