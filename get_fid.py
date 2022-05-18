from tqdm import tqdm
from functools import partialmethod

tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)


from pytorch_fid.fid_score import calculate_fid_given_paths
import torch
import os
import numpy as np
from PIL import Image
import pickle
import shutil
import argparse
from src.utils.parse_config import ConfigParser

import warnings
warnings.filterwarnings("ignore")


def mask_images(dir, seg_npy, idx, out_dir):
    for image_name in os.listdir(dir):
        if ".jpg" not in image_name and ".png" not in image_name:
            continue
        img = np.asarray(Image.open(os.path.join(dir, image_name)))
        img[seg_npy != idx] = 0
        Image.fromarray(np.uint8(img)).save(os.path.join(out_dir, image_name))


def run_fid(base, paths, seg_file, parts_idx_path, styles, num_workers=None, device=None, dims=2048,
            batch_size=50):
    if device is None:
        device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
    else:
        device = torch.device(device)

    if num_workers is None:
        num_avail_cpus = len(os.sched_getaffinity(0))
        num_workers = min(num_avail_cpus, 8)

    if len(os.listdir(paths)) == 0:
        return

    seg_npy = np.load(seg_file)
    with open(parts_idx_path, 'rb') as handle:
        parts_idx = pickle.load(handle)

    added_dirs = []

    parts = sorted(list(parts_idx.keys()))
    fids = []

    for i in range(min(2, len(parts))):
        part = parts[i]
        style = styles[1 - i]

        out_dir_base = os.path.join(base, style, str(part))
        # print("out_dir_base", out_dir_base)
        added_dirs.append(out_dir_base)
        if not os.path.exists(out_dir_base):
            os.mkdir(out_dir_base)

        mask_images(os.path.join(base, style), seg_npy, parts_idx[part], out_dir_base)

        out_dir_paths = os.path.join(paths, str(part))
        # print("out_dir_paths", out_dir_paths)
        added_dirs.append(out_dir_paths)
        if not os.path.exists(out_dir_paths):
            os.mkdir(out_dir_paths)

        mask_images(paths, seg_npy, parts_idx[part], out_dir_paths)

        fids.append(calculate_fid_given_paths([out_dir_base, out_dir_paths],
                                              batch_size,
                                              device,
                                              dims,
                                              num_workers))
        # print("\ndirs:", part, fid_value, "\n")
    # print(added_dirs)
    for added in added_dirs:
        shutil.rmtree(added)
    return np.mean(fids)


def main(config):
    params = config["params"]
    styles_dict = {0: ["black_pencil_2", "vangogh_starry_night"],
                       1: ["the_wave", "wheatfield"],
                       2: ["black_pencil_2", "monet_sunrise"],
                       3: ["the_wave", "the_kiss"],
                       4: ["the_kiss", "venus"],
                       5: ["monet_sunrise", "vangogh_starry_night"],
                       6: ["venus", "wheatfield"]
                       }

    for out_dir in params["out_dir"]:
        fids = []
        for image in params["images"]:
            pseudohash = (int(image.split("_")[0]) + int(image.split("_")[1])) % len(styles_dict)
            fids.append(run_fid(os.path.join(params["base"]),
                    os.path.join(out_dir, image),
                    os.path.join(params["seg_dir"], image + "_seg.npy"),
                    os.path.join(params["seg_dir"], image + "_parts_idx.pickle"),
                    styles_dict[pseudohash]))
        print("mean fid for", out_dir, np.mean(fids))


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )

    config = ConfigParser.from_args(args, [], 0)
    main(config)

