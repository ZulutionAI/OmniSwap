import json
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image, ImageOps
from .utils import get_new_size


def c_crop(image):
    width, height = image.size
    new_size = min(width, height)
    left = (width - new_size) / 2
    top = (height - new_size) / 2
    right = (width + new_size) / 2
    bottom = (height + new_size) / 2
    return image.crop((left, top, right, bottom))


def c_pad(image):
    width, height = image.size
    new_size = max(width, height)
    left = int((new_size - width) / 2)
    top = int((new_size - height) / 2)
    right = new_size - (left + width)
    bottom = new_size - (top + height)
    border = (left, top, right, bottom)
    return ImageOps.expand(image, border, fill=(255, 255, 255))


class StyleLoRAImageDataset(Dataset):
    def __init__(self, img_dir_list, img_size=512, style_tags=None):
        self.images = []
        for img_dir in img_dir_list:
            self.images += [os.path.join(img_dir, i) for i in os.listdir(img_dir) if '.jpg' in i or '.png' in i or '.webp' in i or '.jpeg' in i]
            # for folder in os.listdir(img_dir):
            #     image_folder_path = os.path.join(img_dir, folder)
            #     self.images += [os.path.join(image_folder_path, i) for i in os.listdir(image_folder_path) if '.jpg' in i or '.png' in i or '.webp' in i or '.jpeg' in i]
        print(f"Got {len(self.images)} imgs now")
        self.images.sort()
        self.img_size = img_size
        self.style_tags = style_tags

    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, idx):
        try:
            img = Image.open(self.images[idx])
            img = img.convert("RGB")
            # if max(img.size) / min(img.size) < 1.5:
            #     img = c_crop(img)
            # else:
            #     img = c_pad(img)
            # img = img.resize((self.img_size, self.img_size))
            new_w, new_h = get_new_size(img, self.img_size)
            new_w = new_w //32 * 32
            new_h = new_h //32 * 32
            img = img.resize((new_w, new_h))

            img = torch.from_numpy((np.array(img) / 127.5) - 1)
            img = img.permute(2, 0, 1)
            # json_path = '.'.join(self.images[idx].split('.')[:-1]) + '.json'
            # prompt = json.load(open(json_path))['caption']
            img_path = self.images[idx]

            prompt_path = '.'.join(img_path.split('.')[:-1]) + '.txt'
            # with open(prompt_path, 'r') as f:
            #     prompt = ''.join(f.readlines())
            prompt = 'The character is crying.'
            
            # prompt_down = "Teardrops streaming down on face;"
            # prompt_around = "Lower eyelids filled with tears;"
            # if prompt_path.split('/')[-2] == 'crying_down':
            #     prompt = prompt_down + prompt
            # elif prompt_path.split('/')[-2] == 'crying_around_eye':
            #     prompt = prompt_around + prompt
            # else:
            #     prompt = prompt_around[:-1] + ',' + prompt_down + prompt

            if self.style_tags is not None:
                prompt = self.style_tags + prompt
            # print(prompt)
            return img, prompt
        except Exception as e:
            print(e)
            return self.__getitem__(random.randint(0, len(self.images) - 1))


def loader(train_batch_size, num_workers, **args):
    dataset = StyleLoRAImageDataset(**args)
    return DataLoader(dataset, batch_size=train_batch_size, num_workers=num_workers, shuffle=True)
