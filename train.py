import argparse
import collections

import numpy as np
import torch

import src.loss as module_loss
from src.datasets.utils import get_dataloaders
from src.trainer import Trainer
from src.utils import prepare_device
from src.utils.parse_config import ConfigParser

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def main(config):
    # setup data_loader instances
    torch.cuda.set_device(config.local_rank)
    torch.distributed.init_process_group(backend="nccl")

    dataloaders = get_dataloaders(config)

    # prepare for (multi-device) GPU training
    device = prepare_device(config["n_gpu"])

    loss_module = config.init_obj(config["loss"], module_loss).to(device)

    trainer = Trainer(
        loss_module,
        config=config,
        device=device,
        local_rank=config.local_rank,
        data_loader_A=dataloaders["train_loader_A"],
        data_loader_B=dataloaders["train_loader_B"],
        valid_data_loader_A=dataloaders["val_loader_A"],
        valid_data_loader_B=dataloaders["val_loader_B"],
        adversarial=loss_module.adversarial,
    )

    trainer.train()


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Train img2img options")
    args.add_argument(
        "--local_rank",
        type=int,
        help="rank for DDP",
    )
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

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple("CustomArgs", "flags type target")
    options = [
        CustomArgs(["--lr", "--learning_rate"], type=float, target="optimizer;args;lr"),
        CustomArgs(
            ["--bs", "--batch_size"], type=int, target="data_loader;args;batch_size"
        ),
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
