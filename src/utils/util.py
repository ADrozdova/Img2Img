import json
import os
import subprocess
import zipfile
from collections import OrderedDict
from itertools import repeat
from pathlib import Path

import pandas as pd
import torch
from torchvision.io import write_jpeg

ROOT_PATH = Path(__file__).absolute().resolve().parent.parent.parent


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def read_json(fname):
    fname = Path(fname)
    with fname.open("rt") as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open("wt") as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def inf_loop(data_loader):
    """wrapper function for endless data loader."""
    for loader in repeat(data_loader):
        yield from loader


def prepare_device(local_rank):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    device = torch.device('cuda', local_rank) if torch.cuda.is_available() else torch.device("cpu")
    return device


def load_dataset(url, zipname):
    if url is None:
        raise RuntimeError("Can't load dataset with None url")

    subprocess.run(
        ["wget", url, "-O", zipname],
    )

    with zipfile.ZipFile(zipname, 'r') as zip_ref:
        zip_ref.extractall(".")

    os.remove(zipname)


def img_to_jpeg(image, filename):
    write_jpeg((image.cpu() * 255).to(torch.uint8), filename=filename)


class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=["total", "counts", "average"])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        # if self.writer is not None:
        #     self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)

    def keys(self):
        return self._data.total.keys()