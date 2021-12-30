import argparse
import collections
import warnings

import numpy as np
import itertools
import torch

import src.loss as module_loss
import src.model as module_arch
from src.datasets.utils import get_dataloaders
from src.trainer import Trainer
from src.utils import prepare_device
from src.utils.parse_config import ConfigParser

warnings.filterwarnings("ignore", category=UserWarning)

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def main(config):
    logger = config.get_logger("train")

    # setup data_loader instances
    dataloaders = get_dataloaders(config)

    gen_B = config.init_obj(config["generator"], module_arch)
    gen_A = config.init_obj(config["generator"], module_arch)

    logger.info(gen_A)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config["n_gpu"])

    loss_module = config.init_obj(config["loss"], module_loss).to(device)

    if loss_module.adversarial:
        disc_A = config.init_obj(config["discriminator"], module_arch)
        disc_B = config.init_obj(config["discriminator"], module_arch)
        logger.info(disc_A)
    else:
        disc_A = None
        disc_B = None

    gen_B = gen_B.to(device)
    gen_A = gen_A.to(device)

    if loss_module.adversarial:
        disc_A = disc_A.to(device)
        disc_B = disc_B.to(device)

    if len(device_ids) > 1:
        gen_B = torch.nn.DataParallel(gen_B, device_ids=device_ids)
        gen_A = torch.nn.DataParallel(gen_A, device_ids=device_ids)
        if loss_module.adversarial:
            disc_A = torch.nn.DataParallel(disc_A, device_ids=device_ids)
            disc_B = torch.nn.DataParallel(disc_B, device_ids=device_ids)

    trainable_params = filter(lambda p: p.requires_grad, itertools.chain(gen_A.parameters(), gen_B.parameters()))
    optimizer_G = config.init_obj(config["optimizer"], torch.optim, trainable_params)

    if loss_module.adversarial:
        trainable_params = filter(lambda p: p.requires_grad, disc_A.parameters())
        optimizer_DA = config.init_obj(config["optimizer"], torch.optim, trainable_params)

        trainable_params = filter(lambda p: p.requires_grad, disc_B.parameters())
        optimizer_DB = config.init_obj(config["optimizer"], torch.optim, trainable_params)
    else:
        optimizer_DA, optimizer_DB = None, None

    # lr_scheduler = config.init_obj(config["lr_scheduler"], torch.optim.lr_scheduler, optimizer)

    trainer = Trainer(
        gen_B,
        gen_A,
        disc_A,
        disc_B,
        loss_module,
        optimizer_G,
        optimizer_DA,
        optimizer_DB,
        config=config,
        device=device,
        data_loader_A=dataloaders["train_loader_A"],
        data_loader_B=dataloaders["train_loader_B"],
        valid_data_loader_A=dataloaders["val_loader_A"],
        valid_data_loader_B=dataloaders["val_loader_B"]
    )

    trainer.train()


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
