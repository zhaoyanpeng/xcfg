import os, sys, re
import copy, time, pickle, json
import numpy as np
import argparse

from PIL import Image
from tqdm import tqdm
from collections import Counter, defaultdict
from torchvision.transforms.functional import InterpolationMode

import torch
import torch.nn as nn
import torch.utils.data as data

import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from clip import load 
from cvap.util import seed_all_rng

seed_all_rng(1213) # Yann's random luck

# Data path options
parser = argparse.ArgumentParser()
parser.add_argument('--data_root', default='', type=str, help='')
parser.add_argument('--npz_token', default='', type=str, help='')
parser.add_argument('--flickr_root', default='', type=str, help='')
parser.add_argument('--mscoco_root', default='', type=str, help='')
parser.add_argument('--flickr_out_root', default='', type=str, help='')
parser.add_argument('--mscoco_out_root', default='', type=str, help='')
parser.add_argument('--clip_model_root', default='', type=str, help='')
parser.add_argument('--clip_model_name', default='', type=str, help='')
parser.add_argument('--batch_size', default=8, type=int, help='')
parser.add_argument('--peep_rate', default=1, type=int, help='')
parser.add_argument('--num_proc', default=1, type=int, help='')
cfg = parser.parse_args()
echo=print

def build_clip_encoder(cfg):
    model, _ = load(
        cfg.clip_model_name, cfg.clip_model_root, device="cpu", jit=False
    )
    model = model.train(False)
    return model 

def build_resnet101_encoder(cfg):
    resnet101 = models.resnet101(pretrained=True)
    model = torch.nn.Sequential(*(list(resnet101.children())[:-1]))
    model = model.train(False)
    return model

def build_resnet152_encoder(cfg):
    resnet101 = models.resnet152(pretrained=True)
    model = torch.nn.Sequential(*(list(resnet101.children())[:-1]))
    model = model.train(False)
    return model

def resnet_transform(resize_size=256, crop_size=224):
    """ (1) https://github.com/pytorch/vision/blob/d2bfd639e46e1c5dc3c177f889dc7750c8d137c7/references/classification/train.py#L111 
        (2) https://github.com/pytorch/vision/blob/65676b4ba1a9fd4417293cb16f690d06a4b2fb4b/references/classification/presets.py#L44
        (3) or https://github.com/pytorch/examples/blob/42e5b996718797e45c46a25c55b031e6768f8440/imagenet/main.py#L105
        (4) or https://pytorch.org/hub/pytorch_vision_resnet 
    """
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    return transforms.Compose([
        transforms.Resize(resize_size, interpolation=InterpolationMode.BILINEAR),
        transforms.CenterCrop(crop_size),
        lambda image: image.convert("RGB"),
        transforms.ToTensor(),
        #transforms.PILToTensor(),                   # equal to ...
        #transforms.ConvertImageDtype(torch.float),  # ... ToTensor() 
        transforms.Normalize(mean=mean, std=std),
    ])
    """
    PILToTensor() results in warnings: torchvision/transforms/functional.py:169: UserWarning: The given NumPy array is not writeable, and PyTorch does not support non-writeable tensors. This means you can write to the underlying (supposedly non-writeable) NumPy array using the tensor. You may want to copy the array to protect its data or make it writeable before converting it to a tensor. This type of warning will be suppressed for the rest of this program. (Triggered internally at  /pytorch/torch/csrc/utils/tensor_numpy.cpp:143.
    """

def clip_transform(n_px=224):
    """ https://github.com/openai/CLIP/blob/573315e83f07b53a61ff5098757e8fc885f1703e/clip/clip.py#L76
    """
    mean = (0.48145466, 0.4578275, 0.40821073)
    std = (0.26862954, 0.26130258, 0.27577711)
    return transforms.Compose([
        transforms.Resize(n_px, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(n_px),
        lambda image: image.convert("RGB"),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

def create_mscoco_data_list(cfg):
    """ all mscoco images. https://cocodataset.org/#download
    """
    image_list = list()
    def per_split(root):
        for root, dir, files in os.walk(root):
            if len(dir) > 0:
                continue
            for fname in files:
                if fname.endswith(".jpg"):
                    image_list.append((fname[:-4], f"{root}/{fname}"))
    root = f"{cfg.mscoco_root}"
    for root, dir, files in os.walk(root):
        for sub_dir in dir:
            if sub_dir not in ["train2014", "test2014", "val2014"]:
                continue
            per_split(f"{root}/{sub_dir}")
    return image_list 

def create_flickr_data_list(cfg):
    """ all flickr images. http://hockenmaier.cs.illinois.edu/DenotationGraph/data/
    """
    image_list = list()
    root = f"{cfg.flickr_root}/flickr30k-images"
    for root, dir, files in os.walk(root):
        if len(dir) > 0:
            continue
        for fname in files:
            if fname.endswith(".jpg"):
                image_list.append((fname[:-4], f"{root}/{fname}"))
    return image_list 

class ImageDatasetSrc(torch.utils.data.Dataset):
    """ `__getitem__' loads raw file from disk.
    """
    def __init__(self, cfg, data_list, transform=None):
        self.dataset = list()
        for iline, line in enumerate(data_list):
            self.dataset.append(line)
            if iline < 8:
                pass #print(caption, self.dataset[-1])
        self.transform_resnet = resnet_transform()
        self.transform_clip = clip_transform()
        self.length = len(self.dataset)
        self.cfg = cfg

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        record = self.dataset[index]
        image = Image.open(record[1])
        image_resnet = self.transform_resnet(image)
        image_clip = self.transform_clip(image)
        item = {"name": record[0], "image_resnet": image_resnet, "image_clip": image_clip}
        return item

class ImageCollator:
    def __init__(self, device=torch.device("cpu")):
        # RuntimeError: cannot pin 'torch.cuda.FloatTensor' only dense CPU tensors can be pinned
        # when pin_memory is true, the collator has to return CPU tensors
        self.device = device

    def __call__(self, records):
        union = { 
            k: [record.get(k) for record in records] for k in set().union(*records) 
        } 
        return (
            torch.stack(union["image_clip"], dim=0),
            torch.stack(union["image_resnet"], dim=0),
            union["name"],
        )

def build_image_loader(cfg, data_list, transform):
    dataset = ImageDatasetSrc(cfg, data_list, transform)
    sampler = torch.utils.data.SequentialSampler(dataset)
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=cfg.batch_size,
        collate_fn=ImageCollator(),
        num_workers=cfg.num_proc,
        pin_memory=True,
        sampler=sampler,
        drop_last=False,
    )
    return dataloader

def save_image_npz(names, clip_npz_root, resnet_npz_root, z_clip=None, z_resnet=None):
    for i, name in enumerate(names):
        if z_clip is not None:
            np.savez_compressed(
                f"{clip_npz_root}/{name}", v=z_clip[i] 
            )
        if z_resnet is not None:
            np.savez_compressed(
                f"{resnet_npz_root}/{name}", v=z_resnet[i] 
            )

def encode_images(cfg, clip, resnet, dataloader, clip_npz_root, resnet_npz_root):
    nsample = 0
    start_time = time.time()
    for ibatch, (image_clip, image_resnet, names) in enumerate(dataloader):
        image_clip = image_clip.cuda(0, non_blocking=True)
        image_resnet = image_resnet.cuda(0, non_blocking=True)
        #echo(image_clip.shape, image_resnet.shape)

        if clip is not None:
            z_clip = clip.encode_image(image_clip)
            z_clip = z_clip.cpu().numpy()
        else:
            z_clip = None

        z_resnet = resnet(image_resnet).squeeze() 
        z_resnet = z_resnet.cpu().numpy()
        #echo(z_clip.shape, z_resnet.shape)
        #echo(names[:10])

        save_image_npz(names, clip_npz_root, resnet_npz_root, z_clip=z_clip, z_resnet=z_resnet)
        nsample += image_clip.shape[0]
        if (ibatch + 1) % cfg.peep_rate == 0:
            echo(f"--step {ibatch + 1:08d} {nsample / (time.time() - start_time):.2f} samples/s")
        #break

def main_encode_flickr_images(cfg):
    clip = None #build_clip_encoder(cfg).cuda()
    resnet = build_resnet152_encoder(cfg).cuda()
    #resnet = build_resnet101_encoder(cfg).cuda()

    flickr_images = create_flickr_data_list(cfg)
    flickr_loader = build_image_loader(cfg, flickr_images, resnet_transform())
    echo(f"Total {len(flickr_loader)} / {len(flickr_loader.dataset)} flickr batches.")
    
    clip_npz_root = f"{cfg.flickr_root}/clip-b32"
    resnet_npz_root = f"{cfg.flickr_root}/resn-152"

    for output_dir in [clip_npz_root, resnet_npz_root]: 
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    encode_images(
        cfg, clip, resnet, flickr_loader, clip_npz_root, resnet_npz_root
    )

def main_encode_mscoco_images(cfg):
    clip = None #build_clip_encoder(cfg).cuda()
    resnet = build_resnet152_encoder(cfg).cuda()
    #resnet = build_resnet101_encoder(cfg).cuda()

    mscoco_images = create_mscoco_data_list(cfg)
    mscoco_loader = build_image_loader(cfg, mscoco_images, resnet_transform())
    echo(f"Total {len(mscoco_loader)} / {len(mscoco_loader.dataset)} mscoco batches.")

    clip_npz_root = f"{cfg.mscoco_root}/clip-b32"
    resnet_npz_root = f"{cfg.mscoco_root}/resn-152"
    #resnet_npz_root = f"{cfg.mscoco_root}/resn-101"

    for output_dir in [clip_npz_root, resnet_npz_root]: 
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    encode_images(
        cfg, clip, resnet, mscoco_loader, clip_npz_root, resnet_npz_root
    )

def main_collect_mscoco_npz(cfg):
    def per_split(id_file, ipath, ofile):
        vectors = list()
        with open(id_file, "r") as fr:
            for line in fr:
                id = line.strip().split(".")[0]
                npz_file = f"{ipath}/{id}.npz"
                vector = np.load(npz_file)["v"]
                vectors.append(vector)
        vectors = np.stack(vectors, axis=0)
        np.save(ofile, vectors)
        echo(f"saved {vectors.shape} in {ofile}")
    
    def main_per_split(split):
        npz_root = f"{cfg.mscoco_root}/{cfg.npz_token}"
        ofile = f"{cfg.mscoco_out_root}/{split}_{cfg.npz_token}.npy"
        id_file = f"{cfg.mscoco_out_root}/{split}.id"
        echo(f"src: {npz_root}\ntgt: {ofile}\nids: {id_file}")
        per_split(id_file, npz_root, ofile)

    main_per_split("train")
    main_per_split("test")
    main_per_split("val")

def main_collect_flickr_npz(cfg):
    def per_split(id_file, ipath, ofile):
        vectors = list()
        with open(id_file, "r") as fr:
            for line in fr:
                id = line.strip().split(".")[0]
                npz_file = f"{ipath}/{id}.npz"
                vector = np.load(npz_file)["v"]
                vectors.append(vector)
        vectors = np.stack(vectors, axis=0)
        np.save(ofile, vectors)
        echo(f"saved {vectors.shape} in {ofile}")
    
    def main_per_split(split):
        npz_root = f"{cfg.flickr_root}/{cfg.npz_token}"
        ofile = f"{cfg.flickr_out_root}/{split}_{cfg.npz_token}.npy"
        id_file = f"{cfg.flickr_out_root}/{split}.id"
        echo(f"src: {npz_root}\ntgt: {ofile}\nids: {id_file}")
        per_split(id_file, npz_root, ofile)

    main_per_split("train")
    main_per_split("test")
    main_per_split("val")

if __name__ == '__main__':
    echo(cfg)

    #main_collect_mscoco_npz(cfg)
    #main_collect_flickr_npz(cfg)

    with torch.no_grad(): 
        #main_encode_flickr_images(cfg)
        #main_encode_mscoco_images(cfg)
        pass
    pass
