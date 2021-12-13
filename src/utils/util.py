import os
import subprocess
import zipfile

import torch


def prepare_device(n_gpu_use):
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print(
            "Warning: There's no GPU available on this machine,"
            "training will be performed on CPU."
        )
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(
            f"Warning: The number of GPU's configured to use is {n_gpu_use}, but only {n_gpu} are "
            "available on this machine."
        )
        n_gpu_use = n_gpu
    device = torch.device("cuda:0" if n_gpu_use > 0 else "cpu")
    list_ids = list(range(n_gpu_use))
    return device, list_ids

def load_dataset(url, zipname):
    if url is None:
        raise RuntimeError("Can't load dataset with None url")

    subprocess.run(
        ["wget", url, "-O", zipname],
    )

    with zipfile.ZipFile(zipname, 'r') as zip_ref:
        zip_ref.extractall(".")

    os.remove(zipname)