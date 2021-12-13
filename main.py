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
    # setup data_loader instances
    dataloaders = get_dataloaders(config)

    # build model architecture, then print to console
    gen_B= config.init_obj(config["Generator"], module_arch)
    gen_A = config.init_obj(config["Generator"], module_arch)

    disc_A = config.init_obj(config["Discriminator"], module_arch)
    disc_B = config.init_obj(config["Discriminator"], module_arch)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config["n_gpu"])

    gen_B = gen_B.to(device)
    gen_A = gen_A.to(device)
    disc_A = disc_A.to(device)
    disc_B = disc_B.to(device)

    if len(device_ids) > 1:
        gen_B = torch.nn.DataParallel(gen_B, device_ids=device_ids)
        gen_A = torch.nn.DataParallel(gen_A, device_ids=device_ids)
        disc_A = torch.nn.DataParallel(disc_A, device_ids=device_ids)
        disc_B = torch.nn.DataParallel(disc_B, device_ids=device_ids)

    loss_module = config.init_obj(config["loss"], module_loss).to(device)

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, itertools.chain(gen_B.parameters(), gen_A.parameters(),
                                                                         disc_A.parameters(), disc_B.parameters()))
    optimizer = config.init_obj(config["optimizer"], torch.optim, trainable_params)
    # lr_scheduler = config.init_obj(config["lr_scheduler"], torch.optim.lr_scheduler, optimizer)


    trainer = Trainer(
        gen_B,
        gen_A,
        disc_A,
        disc_B,
        loss_module,
        optimizer,
        config=config,
        device=device,
        data_loader=dataloaders["train"],
        valid_data_loader=dataloaders["val"],
        len_epoch=config["trainer"].get("len_epoch", None)
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

    config = ConfigParser.from_args(args)
    main(config)