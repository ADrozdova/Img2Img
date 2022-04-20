import os

import torch
import torch.nn as nn
from PIL import Image
from torch import optim
from torchvision import transforms

from src.loss import GramMSELoss, GramMatrix
from src.model.styletransfer_vgg import VGG
from src.utils import prepare_device, img_to_jpeg

# pre and post processing for images
img_size = 512
prep = transforms.Compose([transforms.Scale(img_size),
                           transforms.ToTensor(),
                           transforms.Lambda(lambda x: x[torch.LongTensor([2, 1, 0])]),  # turn to BGR
                           transforms.Normalize(mean=[0.40760392, 0.45795686, 0.48501961],  # subtract imagenet mean
                                                std=[1, 1, 1]),
                           transforms.Lambda(lambda x: x.mul_(255)),
                           ])
postpa = transforms.Compose([transforms.Lambda(lambda x: x.mul_(1. / 255)),
                             transforms.Normalize(mean=[-0.40760392, -0.45795686, -0.48501961],  # add imagenet mean
                                                  std=[1, 1, 1]),
                             transforms.Lambda(lambda x: x[torch.LongTensor([2, 1, 0])]),  # turn to RGB
                             ])
postpb = transforms.Compose([transforms.ToPILImage()])


def postp(tensor):  # to clip results in the range [0,1]
    t = postpa(tensor)
    t[t > 1] = 1
    t[t < 0] = 0
    img = postpb(t)
    return img


model_path = "vgg_conv.pth"

# get network
vgg = VGG()
vgg.load_state_dict(torch.load(model_path))
for param in vgg.parameters():
    param.requires_grad = False

local_rank = int(os.environ["LOCAL_RANK"])

device = prepare_device(local_rank)
torch.distributed.init_process_group(backend="nccl")

vgg = torch.nn.parallel.DistributedDataParallel(
    vgg, device_ids=[local_rank], output_device=local_rank
)

image_dir = "/images"

# load images, ordered as [style_image, content_image]
img_dirs = [image_dir, image_dir]
img_names = ['vangogh_starry_night.jpg', 'Tuebingen_Neckarfront.jpg']
imgs = [Image.open(img_dirs[i] + name) for i, name in enumerate(img_names)]
imgs_torch = [prep(img) for img in imgs]

imgs_torch = [img.unsqueeze(0).to(device) for img in imgs_torch]
style_image, content_image = imgs_torch

opt_img = content_image.data.clone()

# define layers, loss functions, weights and compute optimization targets
style_layers = ['r11', 'r21', 'r31', 'r41', 'r51']
content_layers = ['r42']
loss_layers = style_layers + content_layers
loss_fns = [GramMSELoss()] * len(style_layers) + [nn.MSELoss()] * len(content_layers)

loss_fns = [loss_fn.to(device) for loss_fn in loss_fns]

# these are good weights settings:
style_weights = [1e3 / n ** 2 for n in [64, 128, 256, 512, 512]]
content_weights = [1e0]
weights = style_weights + content_weights

# compute optimization targets
style_targets = [GramMatrix()(A).detach() for A in vgg(style_image, style_layers)]
content_targets = [A.detach() for A in vgg(content_image, content_layers)]
targets = style_targets + content_targets

# run style transfer
max_iter = 500
show_iter = 50
optimizer = optim.LBFGS([opt_img])
n_iter = [0]

while n_iter[0] <= max_iter:
    def closure():
        optimizer.zero_grad()
        out = vgg(opt_img, loss_layers)
        layer_losses = [weights[a] * loss_fns[a](A, targets[a]) for a, A in enumerate(out)]
        loss = sum(layer_losses)
        loss.backward()
        n_iter[0] += 1
        # print loss
        if n_iter[0] % show_iter == (show_iter - 1):
            print('Iteration: %d, loss: %f' % (n_iter[0] + 1, loss.item()))
        return loss


    optimizer.step(closure)

# display result
out_img = postp(opt_img.data[0].cpu().squeeze())

img_to_jpeg(out_img, "styletransfer_output.jpg")
