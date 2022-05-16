from pytorch_fid.fid_score import calculate_fid_given_paths
import torch
import os
import numpy as np
from PIL import Image
import pickle
import shutil
import argparse
from src.utils.parse_config import ConfigParser


def mask_images(dir, seg_npy, idx, out_dir):
    for image_name in os.listdir(dir):
        if ".jpg" not in image_name and ".png" not in image_name:
            continue
        img = np.asarray(Image.open(os.path.join(dir, image_name)))
        img[seg_npy != idx] = 0
        Image.fromarray(np.uint8(img)).save(os.path.join(out_dir, image_name))


def run_fid(base, paths, seg_file, parts_idx_path, num_workers=None, device=None, dims=2048, batch_size=50):
    dirs = os.listdir(paths)

    if device is None:
        device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
    else:
        device = torch.device(device)

    if num_workers is None:
        num_avail_cpus = len(os.sched_getaffinity(0))
        num_workers = min(num_avail_cpus, 8)

    if len(dirs) == 0:
        return

    if not os.path.isdir(os.path.join(paths, dirs[0])):
        dirs = ["."]

    seg_npy = np.load(seg_file)
    with open(parts_idx_path, 'rb') as handle:
        parts_idx = pickle.load(handle)

    added_dirs = []

    for part, idx in parts_idx.items():
        out_dir = os.path.join(base, str(part))
        added_dirs.append(out_dir)
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        mask_images(base, seg_npy, idx, out_dir)

    for dir in dirs:

        for part, idx in parts_idx.items():
            out_dir = os.path.join(paths, dir, str(part))
            added_dirs.append(out_dir)
            if not os.path.exists(out_dir):
                os.mkdir(out_dir)
            mask_images(os.path.join(paths, dir), seg_npy, idx, out_dir)

            fid_value = calculate_fid_given_paths([os.path.join(base, str(part)), out_dir],
                                                  batch_size,
                                                  device,
                                                  dims,
                                                  num_workers)
            print("\ndirs:", out_dir, os.path.join(base, str(part)), 'FID: ', fid_value, "\n")

    for added in added_dirs:
        shutil.rmtree(added)


def main(config):
    params = config["params"]
    if "data" in params:
        for test in params["data"]:
            run_fid(*test)
    else:
        for image in params["images"]:
            run_fid(os.path.join(params["base"], image),
                    os.path.join(params["out_dir"], image),
                    os.path.join(params["seg_dir"], image + "_seg.npy"),
                    os.path.join(params["seg_dir"], image + "_parts_idx.pickle"))


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
