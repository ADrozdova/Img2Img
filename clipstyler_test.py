import os
from argparse import Namespace

import clip
import torch.nn
import torch.optim as optim
from torchvision import transforms, models
from torchvision.transforms.functional import adjust_contrast

from src.datasets.imagenet_template import full_imagenet_templates
from src.model import UNet
from src.utils.util import load_image2, get_features, img_normalize, clip_normalize, \
    get_image_prior_losses, prepare_device

device = prepare_device(int(os.environ["LOCAL_RANK"]))
VGG = models.vgg19(pretrained=True).features
VGG.to(device)

for parameter in VGG.parameters():
    parameter.requires_grad_(False)

source = "a Photo"

text = "Sketch with black pencil"  # @param {"type": "string"}

image_dir = "/images/face.jpg"  # @param {type: "string"}

training_iterations = 100  # @param {type: "integer"}

training_args = {
    "lambda_tv": 2e-3,
    "lambda_patch": 9000,
    "lambda_dir": 500,
    "lambda_c": 150,
    "crop_size": 128,
    "num_crops": 64,
    "img_height": 512,
    "img_width": 512,
    "max_step": training_iterations,
    "lr": 5e-4,
    "thresh": 0.7,
    "content_path": image_dir,
    "text": text
}

args = Namespace(**training_args)


def compose_text_with_templates(text: str, templates=full_imagenet_templates) -> list:
    return [template.format(text) for template in templates]


content_path = args.content_path
content_image = load_image2(content_path, img_height=args.img_height, img_width=args.img_width)

content_image = content_image.to(device)

content_features = get_features(img_normalize(content_image, device), VGG)

target = content_image.clone().requires_grad_(True).to(device)

style_net = UNet()
style_net.to(device)

style_weights = {'conv1_1': 0.1,
                 'conv2_1': 0.2,
                 'conv3_1': 0.4,
                 'conv4_1': 0.8,
                 'conv5_1': 1.6}

content_weight = args.lambda_c

show_every = 100
optimizer = optim.Adam(style_net.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
steps = args.max_step

content_loss_epoch = []
style_loss_epoch = []
total_loss_epoch = []

output_image = content_image

m_cont = torch.mean(content_image, dim=(2, 3), keepdim=False).squeeze(0)
m_cont = [m_cont[0].item(), m_cont[1].item(), m_cont[2].item()]

cropper = transforms.Compose([
    transforms.RandomCrop(args.crop_size)
])
augment = transforms.Compose([
    transforms.RandomPerspective(fill=0, p=1, distortion_scale=0.5),
    transforms.Resize(224)
])

clip_model, preprocess = clip.load('ViT-B/32', device, jit=False)

prompt = args.text

with torch.no_grad():
    template_text = compose_text_with_templates(prompt, full_imagenet_templates)
    tokens = clip.tokenize(template_text).to(device)
    text_features = clip_model.encode_text(tokens).detach()
    text_features = text_features.mean(axis=0, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    template_source = compose_text_with_templates(source, full_imagenet_templates)
    tokens_source = clip.tokenize(template_source).to(device)
    text_source = clip_model.encode_text(tokens_source).detach()
    text_source = text_source.mean(axis=0, keepdim=True)
    text_source /= text_source.norm(dim=-1, keepdim=True)
    source_features = clip_model.encode_image(clip_normalize(content_image, device))
    source_features /= (source_features.clone().norm(dim=-1, keepdim=True))

num_crops = args.num_crops
for epoch in range(0, steps + 1):

    scheduler.step()
    target = style_net(content_image, use_sigmoid=True).to(device)
    target.requires_grad_(True)

    target_features = get_features(img_normalize(target, device), VGG)

    content_loss = 0

    content_loss += torch.mean((target_features['conv4_2'] - content_features['conv4_2']) ** 2)
    content_loss += torch.mean((target_features['conv5_2'] - content_features['conv5_2']) ** 2)

    loss_patch = 0
    img_proc = []
    for n in range(num_crops):
        target_crop = cropper(target)
        target_crop = augment(target_crop)
        img_proc.append(target_crop)

    img_proc = torch.cat(img_proc, dim=0)
    img_aug = img_proc

    image_features = clip_model.encode_image(clip_normalize(img_aug, device))
    image_features /= (image_features.clone().norm(dim=-1, keepdim=True))

    img_direction = (image_features - source_features)
    img_direction /= img_direction.clone().norm(dim=-1, keepdim=True)

    text_direction = (text_features - text_source).repeat(image_features.size(0), 1)
    text_direction /= text_direction.norm(dim=-1, keepdim=True)
    loss_temp = (1 - torch.cosine_similarity(img_direction, text_direction, dim=1))
    loss_temp[loss_temp < args.thresh] = 0
    loss_patch += loss_temp.mean()

    glob_features = clip_model.encode_image(clip_normalize(target, device))
    glob_features /= (glob_features.clone().norm(dim=-1, keepdim=True))

    glob_direction = (glob_features - source_features)
    glob_direction /= glob_direction.clone().norm(dim=-1, keepdim=True)

    loss_glob = (1 - torch.cosine_similarity(glob_direction, text_direction, dim=1)).mean()

    reg_tv = args.lambda_tv * get_image_prior_losses(target)

    total_loss = args.lambda_patch * loss_patch + content_weight * content_loss + reg_tv + args.lambda_dir * loss_glob
    total_loss_epoch.append(total_loss)

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        print("After %d iters:" % epoch)
        print('Total loss: ', total_loss.item())
        print('Content loss: ', content_loss.item())
        print('patch loss: ', loss_patch.item())
        print('dir loss: ', loss_glob.item())
        print('TV loss: ', reg_tv.item())
        output_image = target.clone()

output_image = torch.clamp(output_image, 0, 1)
output_image = adjust_contrast(output_image, 1.5)
output_image = transforms.ToPILImage()(output_image.cpu())
output_image.save("clipstyler_output.jpg")
