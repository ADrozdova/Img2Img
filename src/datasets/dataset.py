import os

import torch
import torchvision
from PIL import Image
from src.utils.util import load_dataset


class ImageFolderDataset(torch.utils.data.Dataset):
    def __init__(self, img_folder, part, url=None, transform=None):
        self.img_folder = img_folder
        if not os.path.exists(self.img_folder):
            load_dataset(url, self.img_folder + ".zip")
        self.transform = transform
        self.part = part
        self.all_imgs = os.listdir(os.path.join(img_folder, self.part))

    def __len__(self):
        return len(self.all_imgs)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_folder, self.part, self.all_imgs[idx])
        image = torchvision.transforms.ToTensor()(Image.open(img_path))
        if self.transform:
            image = self.transform(image)
        return image


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, img_folder, url=None, transform=None):
        self.img_folder = img_folder
        if not os.path.exists(self.img_folder):
            load_dataset(url, self.img_folder + ".zip")
        self.transform = transform
        self.all_imgs = os.listdir(img_folder)

    def __len__(self):
        return len(self.all_imgs)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_folder, self.part, self.all_imgs[idx])
        image = torchvision.transforms.ToTensor()(Image.open(img_path))
        if self.transform:
            image = self.transform(image)
        return self.all_imgs[idx], image
