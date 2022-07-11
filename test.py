from icon_image import gen_icon
from IPython import display
from base64 import b64encode
from omegaconf import OmegaConf
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from taming.models import cond_transformer, vqgan
import taming.modules
import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF
from torch.optim.lr_scheduler import StepLR
from tqdm.notebook import tqdm
from shutil import move
import os

from CLIP import clip
import kornia.augmentation as K
import numpy as np
import imageio
from PIL import ImageFile, Image

ImageFile.LOAD_TRUNCATED_IMAGES = True

# @title Start AI Image Generation!


metadata = PngInfo()
for k, v in gen_config.items():
    try:
        metadata.add_text("AI_ " + k, str(v))
    except UnicodeEncodeError:
        pass

if init_image_icon or target_image_icon:
    for k, v in icon_config.items():
        try:
            metadata.add_text("AI_Icon_ " + k, str(v))
        except UnicodeEncodeError:
            pass

model_names = {
    "vqgan_imagenet_f16_16384": "ImageNet 16384",
    "vqgan_imagenet_f16_1024": "ImageNet 1024",
    "vqgan_openimages_f16_8192": "OpenImages 8912",
    "wikiart_1024": "WikiArt 1024",
    "wikiart_16384": "WikiArt 16384",
    "coco": "COCO-Stuff",
    "faceshq": "FacesHQ",
    "sflckr": "S-FLCKR",
}
name_model = model_names[model_name]

if seed == -1:
    seed = None
if init_image == "None":
    init_image = None
if target_images == "None" or not target_images:
    model_target_images = []
else:
    model_target_images = target_images.split("|")
    model_target_images = [image.strip() for image in model_target_images]

model_texts = [phrase.strip() for phrase in texts.split("|")]
if model_texts == [""]:
    model_texts = []


args = argparse.Namespace(
    prompts=model_texts,
    image_prompts=model_target_images,
    noise_prompt_seeds=[],
    noise_prompt_weights=[],
    size=[width, height],
    init_image=init_image,
    init_weight=0.0,
    clip_model="ViT-B/32",
    vqgan_config=f"{model_name}.yaml",
    vqgan_checkpoint=f"{model_name}.ckpt",
    step_size=learning_rate,
    cutn=32,
    cut_pow=1.0,
    display_freq=images_interval,
    seed=seed,
)
from urllib.request import urlopen

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
if model_texts:
    print("Using texts:", model_texts)
if model_target_images:
    print("Using image prompts:", model_target_images)
if args.seed is None:
    seed = torch.seed()
else:
    seed = args.seed
torch.manual_seed(seed)
print("Using seed:", seed)

model = load_vqgan_model(args.vqgan_config, args.vqgan_checkpoint).to(device)
perceptor = (
    clip.load(args.clip_model, jit=False)[0].eval().requires_grad_(False).to(device)
)
# clock=deepcopy(perceptor.visual.positional_embedding.data)
# perceptor.visual.positional_embedding.data = clock/clock.max()
# perceptor.visual.positional_embedding.data=clamp_with_grad(clock,0,1)

cut_size = perceptor.visual.input_resolution

f = 2 ** (model.decoder.num_resolutions - 1)
make_cutouts = MakeCutouts(cut_size, args.cutn, cut_pow=args.cut_pow)

toksX, toksY = args.size[0] // f, args.size[1] // f
sideX, sideY = toksX * f, toksY * f

if args.vqgan_checkpoint == "vqgan_openimages_f16_8192.ckpt":
    e_dim = 256
    n_toks = model.quantize.n_embed
    z_min = model.quantize.embed.weight.min(dim=0).values[None, :, None, None]
    z_max = model.quantize.embed.weight.max(dim=0).values[None, :, None, None]
else:
    e_dim = model.quantize.e_dim
    n_toks = model.quantize.n_e
    z_min = model.quantize.embedding.weight.min(dim=0).values[None, :, None, None]
    z_max = model.quantize.embedding.weight.max(dim=0).values[None, :, None, None]
# z_min = model.quantize.embedding.weight.min(dim=0).values[None, :, None, None]
# z_max = model.quantize.embedding.weight.max(dim=0).values[None, :, None, None]

# normalize_imagenet = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                            std=[0.229, 0.224, 0.225])

if args.init_image:
    if "http" in args.init_image:
        img = Image.open(urlopen(args.init_image))
    else:
        img = Image.open(args.init_image)
    pil_image = img.convert("RGB")
    if pil_image.size != (width, height):
        print(f"Resizing source image to {width}x{height}")
        pil_image = pil_image.resize((sideX, sideY), Image.LANCZOS)
    pil_tensor = TF.to_tensor(pil_image)
    z, *_ = model.encode(pil_tensor.to(device).unsqueeze(0) * 2 - 1)
else:
    one_hot = F.one_hot(
        torch.randint(n_toks, [toksY * toksX], device=device), n_toks
    ).float()
    # z = one_hot @ model.quantize.embedding.weight
    if args.vqgan_checkpoint == "vqgan_openimages_f16_8192.ckpt":
        z = one_hot @ model.quantize.embed.weight
    else:
        z = one_hot @ model.quantize.embedding.weight
    z = z.view([-1, toksY, toksX, e_dim]).permute(0, 3, 1, 2)
    z = torch.rand_like(z) * 2
z_orig = z.clone()
z.requires_grad_(True)
opt = optim.Adam([z], lr=args.step_size)
scheduler = StepLR(opt, step_size=5, gamma=0.95)

normalize = transforms.Normalize(
    mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]
)


pMs = []

for prompt in args.prompts:
    txt, weight, stop = parse_prompt(prompt)
    embed = perceptor.encode_text(clip.tokenize(txt).to(device)).float()
    pMs.append(Prompt(embed, weight, stop).to(device))

for prompt in args.image_prompts:
    path, weight, stop = parse_prompt(prompt)
    img = Image.open(path)
    pil_image = img.convert("RGB")
    img = resize_image(pil_image, (sideX, sideY))
    batch = make_cutouts(TF.to_tensor(img).unsqueeze(0).to(device))
    embed = perceptor.encode_image(normalize(batch)).float()
    pMs.append(Prompt(embed, weight, stop).to(device))

for seed, weight in zip(args.noise_prompt_seeds, args.noise_prompt_weights):
    gen = torch.Generator().manual_seed(seed)
    embed = torch.empty([1, perceptor.visual.output_dim]).normal_(generator=gen)
    pMs.append(Prompt(embed, weight).to(device))


def synth(z):
    if args.vqgan_checkpoint == "vqgan_openimages_f16_8192.ckpt":
        z_q = vector_quantize(z.movedim(1, 3), model.quantize.embed.weight).movedim(
            3, 1
        )
    else:
        z_q = vector_quantize(z.movedim(1, 3), model.quantize.embedding.weight).movedim(
            3, 1
        )
    return clamp_with_grad(model.decode(z_q).add(1).div(2), 0, 1)


@torch.no_grad()
def checkin(i, losses):
    losses_str = ", ".join(f"{loss.item():g}" for loss in losses)
    tqdm.write(f"i: {i}, loss: {sum(losses).item():g}, losses: {losses_str}")
    out = synth(z)
    TF.to_pil_image(out[0].cpu()).save("progress.png", pnginfo=metadata)
    display.display(display.Image("progress.png"))


def ascend_txt():
    # global i
    out = synth(z)
    iii = perceptor.encode_image(normalize(make_cutouts(out))).float()

    result = []

    if args.init_weight:
        # result.append(F.mse_loss(z, z_orig) * args.init_weight / 2)
        result.append(
            F.mse_loss(z, torch.zeros_like(z_orig))
            * ((1 / torch.tensor(i * 2 + 1)) * args.init_weight)
            / 2
        )
    for prompt in pMs:
        result.append(prompt(iii))
    img = np.array(
        out.mul(255).clamp(0, 255)[0].cpu().detach().numpy().astype(np.uint8)
    )[:, :, :]
    img = np.transpose(img, (1, 2, 0))
    img = Image.fromarray(img)
    # imageio.imwrite(f'./steps/{i:03d}.png', np.array(img))

    img.save(f"./steps/{i:03d}.png", pnginfo=metadata)
    return result


def train(i):
    opt.zero_grad()
    lossAll = ascend_txt()
    if i % args.display_freq == 0:
        checkin(i, lossAll)

    loss = sum(lossAll)
    loss.backward()
    opt.step()
    scheduler.step()
    with torch.no_grad():
        z.copy_(z.maximum(z_min).minimum(z_max))


try:
    for i in tqdm(range(max_steps)):
        train(i)
    checkin(max_steps, ascend_txt())
except KeyboardInterrupt:
    pass
