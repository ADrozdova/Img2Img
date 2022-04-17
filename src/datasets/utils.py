import torch

from .dataset import ImageFolderDataset
from torchvision import transforms
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from .eval_sampler import DistributedEvalSampler


def get_dataloaders(config):
    params = config["dataset"]

    transform = None
    if "resize" in params:
        transform = transforms.Compose(
            [
                transforms.Resize((params["resize"], params["resize"])),
            ]
        )

    train_set_A = ImageFolderDataset(
        params["img_folder"],
        params["parts_train"][0],
        url=params["url"],
        transform=transform,
    )
    train_set_B = ImageFolderDataset(
        params["img_folder"],
        params["parts_train"][1],
        url=params["url"],
        transform=transform,
    )

    if "parts_val" not in params:
        train_size = int(len(train_set_A) * params["valid_rate"])

        train_set_A, val_set_A = torch.utils.data.random_split(
            train_set_A,
            [train_size, len(train_set_A) - train_size],
            generator=torch.Generator().manual_seed(42),
        )

        train_size = int(len(train_set_B) * params["valid_rate"])

        train_set_B, val_set_B = torch.utils.data.random_split(
            train_set_B,
            [train_size, len(train_set_B) - train_size],
            generator=torch.Generator().manual_seed(42),
        )

    else:
        val_set_A = ImageFolderDataset(
            params["img_folder"],
            params["parts_val"][0],
            url=params["url"],
            transform=transform,
        )
        val_set_B = ImageFolderDataset(
            params["img_folder"],
            params["parts_val"][1],
            url=params["url"],
            transform=transform,
        )

    dataloaders = dict()
    sampler = DistributedSampler(train_set_A)
    dataloaders["train_loader_A"] = DataLoader(
        dataset=train_set_A,
        shuffle=(sampler is None),
        batch_size=params["batch_size_train"],
        pin_memory=True,
        num_workers=params["num_workers_train"],
        sampler=sampler
    )
    sampler = DistributedSampler(train_set_B)
    dataloaders["train_loader_B"] = DataLoader(
        dataset=train_set_B,
        shuffle=(sampler is None),
        batch_size=params["batch_size_train"],
        pin_memory=True,
        num_workers=params["num_workers_train"],
        sampler=sampler
    )

    sampler = DistributedEvalSampler(val_set_A) if params["ddp_val"] else None
    dataloaders["val_loader_A"] = DataLoader(
        dataset=val_set_A,
        shuffle=(sampler is None),
        batch_size=params["batch_size_val"],
        pin_memory=True,
        num_workers=params["num_workers_val"],
        sampler=sampler
    )

    sampler = DistributedEvalSampler(val_set_B) if params["ddp_val"] else None
    dataloaders["val_loader_B"] = DataLoader(
        dataset=val_set_B,
        shuffle=(sampler is None),
        batch_size=params["batch_size_val"],
        pin_memory=True,
        num_workers=params["num_workers_val"],
        sampler=sampler
    )

    return dataloaders
