import json
import os
import subprocess
import zipfile
from collections import OrderedDict
from itertools import repeat
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from mmcv.image import tensor2imgs
from mmcv.parallel import collate, scatter
from torchvision import transforms
from torchvision.io import write_jpeg
import spacy

from src.datasets.imagenet_template import full_imagenet_templates

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


def img_denormalize(image, device):
    mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).to(device)
    mean = mean.view(1, -1, 1, 1)
    std = std.view(1, -1, 1, 1)

    image = image * std + mean
    return image


def img_normalize(image, device):
    mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).to(device)
    mean = mean.view(1, -1, 1, 1)
    std = std.view(1, -1, 1, 1)

    image = (image - mean) / std
    return image


def clip_normalize(image, device):
    image = F.interpolate(image, size=224, mode='bicubic')
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).to(device)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).to(device)
    mean = mean.view(1, -1, 1, 1)
    std = std.view(1, -1, 1, 1)

    image = (image - mean) / std
    return image


def get_image_prior_losses(inputs_jit):
    diff1 = inputs_jit[:, :, :, :-1] - inputs_jit[:, :, :, 1:]
    diff2 = inputs_jit[:, :, :-1, :] - inputs_jit[:, :, 1:, :]
    diff3 = inputs_jit[:, :, 1:, :-1] - inputs_jit[:, :, :-1, 1:]
    diff4 = inputs_jit[:, :, :-1, :-1] - inputs_jit[:, :, 1:, 1:]

    loss_var_l2 = torch.norm(diff1) + torch.norm(diff2) + torch.norm(diff3) + torch.norm(diff4)

    return loss_var_l2


def load_image2(img_path, img_height=None, img_width=None):
    image = Image.open(img_path)
    if img_width is not None:
        image = image.resize((img_width, img_height))  # change image size to (3, img_size, img_size)

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    image = transform(image)[:3, :, :].unsqueeze(0)

    return image


def get_features(image, model, layers=None):
    if layers is None:
        layers = {'0': 'conv1_1',
                  '5': 'conv2_1',
                  '10': 'conv3_1',
                  '19': 'conv4_1',
                  '21': 'conv4_2',
                  '28': 'conv5_1',
                  '31': 'conv5_2'
                  }
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x

    return features


def im_convert2(tensor):
    """ Display a tensor as an image. """
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze(0)  # change size to (channel, height, width)

    image = image.transpose(1, 2, 0)
    # change into unnormalized image
    image = image.clip(0, 1)  # in the previous steps, we change PIL image(0, 255) into tensor(0.0, 1.0), so convert it

    return image


def get_seg(seg_model, test_pipeline, input_img):
    device = next(seg_model.parameters()).device
    # prepare data
    data = dict(img=input_img)
    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)
    if next(seg_model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    else:
        data['img_metas'] = [i.data[0] for i in data['img_metas']]
    with torch.no_grad():
        result = seg_model(return_loss=False, rescale=True, **data)

    img_tensor = data['img'][0]
    img_metas = data['img_metas'][0]
    imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
    assert len(imgs) == len(img_metas)

    return result[0]


def compose_text_with_templates(text: str, templates=full_imagenet_templates) -> list:
    return [template.format(text) for template in templates]


def get_patches_idx(seg, patch_size, patch_step):
    patches_seg = seg.unfold(
        0, patch_size, patch_step
    ).unfold(1, patch_size, patch_step)
    patches_coords = []
    patches_classes = []
    for i in range(patches_seg.shape[0]):
        for j in range(patches_seg.shape[1]):
            seg_class = torch.unique(patches_seg[i, j])
            if len(seg_class) == 1:
                patches_coords.append((i, j))
                patches_classes.append(seg_class[0].item())

    return patches_coords, patches_classes


def get_text_feats(clip, clip_model, text, device):
    with torch.no_grad():
        template_text = compose_text_with_templates(text, full_imagenet_templates)
        tokens = clip.tokenize(template_text).to(device)
        text_features = clip_model.encode_text(tokens).detach()
        text_features = text_features.mean(axis=0, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    return text_features


def get_text_source(clip, clip_model, source, device):
    with torch.no_grad():
        template_source = compose_text_with_templates(source, full_imagenet_templates)
        tokens_source = clip.tokenize(template_source).to(device)
        text_source = clip_model.encode_text(tokens_source).detach()
        text_source = text_source.mean(axis=0, keepdim=True)
        text_source /= text_source.norm(dim=-1, keepdim=True)
    return text_source


def get_source_features(clip_model, device, content_image):
    with torch.no_grad():
        source_features = clip_model.encode_image(clip_normalize(content_image, device))
        source_features /= source_features.clone().norm(dim=-1, keepdim=True)
    return source_features


def dependency_parse(words, sentence):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(sentence)
    token_texts = [tok.text for tok in doc]

    result = dict()

    for word in words:
        idx = token_texts.index(word)
        nodes = [doc[idx]]
        word_children_idx = []
        while len(nodes) > 0:
            node = nodes.pop()
            for child in node.children:
                if child.text not in words:
                    nodes.append(child)
            if node.text not in words:
                word_children_idx.append(node.i)

        result[word] = [token_texts[idx] for idx in sorted(word_children_idx)]
        result[word] = " ".join(result[word])
    return result


def img_dir(img_aug, source_features, clip_model, device):
    image_features = clip_model.encode_image(clip_normalize(img_aug, device))
    image_features /= image_features.clone().norm(dim=-1, keepdim=True)

    img_direction = image_features - source_features
    img_direction /= img_direction.clone().norm(dim=-1, keepdim=True)
    return img_direction
