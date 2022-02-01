import torch

from .dataset import ImageFolderDataset
from torchvision import transforms


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

    dataloaders["train_loader_A"] = torch.utils.data.DataLoader(
        dataset=train_set_A,
        batch_size=params["batch_size_train"],
        shuffle=True,
        pin_memory=True,
        num_workers=8,
    )

    dataloaders["train_loader_B"] = torch.utils.data.DataLoader(
        dataset=train_set_B,
        batch_size=params["batch_size_train"],
        shuffle=True,
        pin_memory=True,
        num_workers=8,
    )

    dataloaders["val_loader_A"] = torch.utils.data.DataLoader(
        dataset=val_set_A,
        batch_size=params["batch_size_val"],
        shuffle=True,
        pin_memory=True,
        num_workers=8,
    )

    dataloaders["val_loader_B"] = torch.utils.data.DataLoader(
        dataset=val_set_B,
        batch_size=params["batch_size_val"],
        shuffle=True,
        pin_memory=True,
        num_workers=8,
    )

    return dataloaders
