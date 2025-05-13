import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import json
import random
import cv2

def c_crop(image):
    width, height = image.size
    new_size = min(width, height)
    left = (width - new_size) / 2
    top = (height - new_size) / 2
    right = (width + new_size) / 2
    bottom = (height + new_size) / 2
    return image.crop((left, top, right, bottom))

class CustomImageDatasetv2(Dataset):
    def __init__(self, img_dir, img_size=512):
        self.images = [os.path.join(img_dir, i) for i in os.listdir(img_dir) if '.jpg' in i or '.png' in i]
        self.images.sort()
        self.img_size = img_size

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.images[idx])
            img = c_crop(img)
            img = img.resize((self.img_size, self.img_size))
            img = torch.from_numpy((np.array(img) / 127.5) - 1)
            img = img.permute(2, 0, 1)
            json_path = self.images[idx].split('.')[0] + '.json'
            prompt = json.load(open(json_path))['caption']
            return img, prompt
        except Exception as e:
            print(e)
            return self.__getitem__(random.randint(0, len(self.images) - 1))

def c_crop_1344x768(image):
    width, height = image.size
    # 比例是1.75
    if height / width > 1.75: # crop height
        top = (height - int(width * 1.75)) // 2
        bottom = top
        right = 0
        left = 0
    else:
        right = (width - int(height / 1.75)) // 2
        left = right
        bottom = 0
        top = 0

    # print("left, top, right, bottom:", left, top, right, bottom)
    return image.crop((left, top, width - right, height - bottom))


class CustomImageDatasetv0(Dataset):
    def __init__(self, img_dir, img_size=512):
        # self.images = [os.path.join(img_dir, i) for i in os.listdir(img_dir) if '.jpg' in i or '.png' in i]
        # path = "/mnt/wangxuekuan/code/layout/dataset_split/skeleton_trainv1.txt"

        # f = open(path, "r")
        # all_path = f.readlines()
        
        plist = [
            "/mnt/wangxuekuan/code/layout/dataset_split/skeleton_trainv8_vcg_0.txt_new.txt",
            "/mnt/wangxuekuan/code/layout/dataset_split/skeleton_trainv8_vcg_1.txt_new.txt",
            "/mnt/wangxuekuan/code/layout/dataset_split/skeleton_trainv8_vcg_2.txt_new.txt",
            "/mnt/wangxuekuan/code/layout/dataset_split/skeleton_trainv8_vcg_3.txt_new.txt",
            # "/mnt/wangxuekuan/code/layout/dataset_split/skeleton_trainv8_reelshort.txt_new.txt",
            # "/mnt/wangxuekuan/code/layout/dataset_split/skeleton_trainv8_pinterest.txt_new.txt",
            "/mnt/wangxuekuan/code/layout/dataset_split/skeleton_trainv8_flux_1.txt_new.txt",
            "/mnt/wangxuekuan/code/layout/dataset_split/skeleton_trainv8_flux_0.txt_new.txt"
        ]

        all_path = []
        for p in plist:
            f = open(p, "r")
            all_path += f.readlines()
        
        #  
        # self.json_path_list = list(map(lambda x: x[3:].strip(), all_path))
        self.json_path_list = all_path
        # random.shuffle(self.json_path_list)

        # self.json_path_list[:10000].sort()

        self.img_size = (768, 1344)

    def __len__(self):
        return len(self.json_path_list)

    def __getitem__(self, idx):
        # try:
            # 找到图片路径
            # 
        try:
            json_path = self.json_path_list[idx].split(", ")[1].strip()
            image_path = self.json_path_list[idx].split(", ")[2].strip()
            
            img = Image.open(image_path)
            # image = np.asarray(image, dtype=np.float32) # 转化为array
            # image = image[:, :, :3] 

            img_size = img.size # width, height
            # pad image
            img = c_crop_1344x768(img)
            img = img.resize(self.img_size)
            # hint = canny_processor(img)
            prompt = json.load(open(json_path))
            # json_path = self.images[idx].split('.')[0] + '.json'
             
            caption_list = []
            if "caption_detail" in prompt.keys():
                caption_list.append("caption_detail")
            elif "caption_simple" in prompt.keys():
                caption_list.append("caption_simple")
            elif "caption_action" in prompt.keys():
                caption_list.append("caption_action")
            elif "caption_character_desc" in prompt.keys():
                caption_list.append("caption_character_desc")
            
            caption_type = random.choice(caption_list)
            prompt_caption = prompt[caption_type].lower()

            if caption_type in ["caption_action", "caption_character_desc"]:
                if random.choice([0, 1]) == 0 and "caption_env" in prompt.keys():
                    prompt_caption += "," + ", ".join(prompt["caption_env"])
                    
            # single caption
            prompt_single_caption = ""

            kpt_list = []
            if random.choice([0, 1]) == 0:
                for idx, item in enumerate(prompt["segments"]):
                    item_caption = ""
                    if "caption" not in list(item.keys()):
                        if "attribute" in list(item.keys()) and "appearance" in list(item.keys()) and "action" in list(item.keys()):
                            item_caption = ",".join(item["attribute"]) + ",".join(item["appearance"]) + ",".join(item["action"]) 
                    else:
                        if type(item["caption"]) == str:
                            item_caption = item["caption"]
                        else:
                            item_caption = ",".join(item["caption"])
                    
                    if item_caption != "":
                        prompt_single_caption += f" &,subject{idx}," + item_caption
                    

            num_character = len(prompt["segments"])      
            prompt = f"{num_character}characters," + prompt_caption + prompt_single_caption
   
            img = torch.from_numpy((np.array(img) / 127.5) - 1)
            img = img.permute(2, 0, 1)[:3, ...]

            if random.choice([0,1,1,1,1]) == 0:
                prompt = ""
      
            return img, prompt

        except Exception as e:
            print(e)
            return self.__getitem__(random.randint(0, len(self.json_path_list) - 1))


class CustomImageDataset(Dataset):
    def __init__(self, img_dir, img_size=512):
        # self.images = [os.path.join(img_dir, i) for i in os.listdir(img_dir) if '.jpg' in i or '.png' in i]
        # path = "/mnt/wangxuekuan/code/layout/dataset_split/skeleton_trainv1.txt"

        # f = open(path, "r")
        # all_path = f.readlines()
        
        plist = [
            "/mnt/wangxuekuan/code/layout/dataset_split/skeleton_trainv8_vcg_0.txt_new.txt",
            "/mnt/wangxuekuan/code/layout/dataset_split/skeleton_trainv8_vcg_1.txt_new.txt",
            "/mnt/wangxuekuan/code/layout/dataset_split/skeleton_trainv8_vcg_2.txt_new.txt",
            "/mnt/wangxuekuan/code/layout/dataset_split/skeleton_trainv8_vcg_3.txt_new.txt",
            # "/mnt/wangxuekuan/code/layout/dataset_split/skeleton_trainv8_reelshort.txt_new.txt",
            # "/mnt/wangxuekuan/code/layout/dataset_split/skeleton_trainv8_pinterest.txt_new.txt",
            "/mnt/wangxuekuan/code/layout/dataset_split/skeleton_trainv8_flux_1.txt_new.txt",
            "/mnt/wangxuekuan/code/layout/dataset_split/skeleton_trainv8_flux_0.txt_new.txt"
        ]

        plist = [
            "/mnt2/wangxuekuan/code/x-flux.bk/data/controlnet/train_skeleton_v0.txt"
        ]
        all_path = []
        for p in plist:
            f = open(p, "r")
            all_path += f.readlines()
        
        #  
        # self.json_path_list = list(map(lambda x: x[3:].strip(), all_path))
        self.json_path_list = all_path
        # random.shuffle(self.json_path_list)

        # self.json_path_list[:10000].sort()

        self.img_size = (768, 1344)

    def __len__(self):
        return len(self.json_path_list)

    def __getitem__(self, idx):
        # try:
            # 找到图片路径
            # 
        # try:
            json_path = self.json_path_list[idx].split(", ")[0].strip()
            image_path = self.json_path_list[idx].split(", ")[1].strip()
            skeleton_path = self.json_path_list[idx].split(", ")[2].strip()
            
            data = cv2.imread(skeleton_path)
            kernel = np.ones((7, 7),np.uint8)
            # 进行膨胀操作
            dilated_image = cv2.dilate(data, kernel, iterations=2)
            dilated_image = cv2.cvtColor(dilated_image, cv2.COLOR_BGR2GRAY)
            dilated_image[dilated_image > 20] = 255
            dilated_image[dilated_image < 20] = 0
            weights = (dilated_image / 255 + 1) 
            # dilated_image = cv2.dilate(dilated_image, kernel, iterations=1)
            # Image.fromnumpy
            weights = Image.fromarray(weights)
            weights = c_crop_1344x768(weights)

            img = Image.open(image_path)
            # image = np.asarray(image, dtype=np.float32) # 转化为array
            # image = image[:, :, :3] 

            img_size = img.size # width, height
            # pad image
            img = c_crop_1344x768(img)
            img = img.resize(self.img_size)
            # hint = canny_processor(img)
            prompt = json.load(open(json_path))
            # json_path = self.images[idx].split('.')[0] + '.json'
             
            caption_list = []
            if "caption_detail" in prompt.keys():
                caption_list.append("caption_detail")
            elif "caption_simple" in prompt.keys():
                caption_list.append("caption_simple")
            elif "caption_action" in prompt.keys():
                caption_list.append("caption_action")
            elif "caption_character_desc" in prompt.keys():
                caption_list.append("caption_character_desc")
            
            caption_type = random.choice(caption_list)
            prompt_caption = prompt[caption_type].lower()

            if caption_type in ["caption_action", "caption_character_desc"]:
                if random.choice([0, 1]) == 0 and "caption_env" in prompt.keys():
                    prompt_caption += "," + ", ".join(prompt["caption_env"])
                    
            # single caption
            prompt_single_caption = ""

            kpt_list = []
            if random.choice([0, 1]) == 0:
                for idx, item in enumerate(prompt["segments"]):
                    item_caption = ""
                    if "caption" not in list(item.keys()):
                        if "attribute" in list(item.keys()) and "appearance" in list(item.keys()) and "action" in list(item.keys()):
                            item_caption = ",".join(item["attribute"]) + ",".join(item["appearance"]) + ",".join(item["action"]) 
                    else:
                        if type(item["caption"]) == str:
                            item_caption = item["caption"]
                        else:
                            item_caption = ",".join(item["caption"])
                    
                    if item_caption != "":
                        prompt_single_caption += f" &,subject{idx}," + item_caption
                    

            num_character = len(prompt["segments"])      
            prompt = f"{num_character}characters," + prompt_caption + prompt_single_caption
   
            img = torch.from_numpy((np.array(img) / 127.5) - 1)
            img = img.permute(2, 0, 1)[:3, ...]

            if random.choice([0,1,1,1,1]) == 0:
                prompt = ""

            weights = weights.resize((weights.size[0]//8, weights.size[1]//8))
            weights = np.array(weights)
            weights = torch.from_numpy(weights)
            return img, prompt, weights

        # except Exception as e:
        #     print(e)
        #     return self.__getitem__(random.randint(0, len(self.json_path_list) - 1))

def loader(train_batch_size, num_workers, **args):
    dataset = CustomImageDataset(**args)
    return DataLoader(dataset, batch_size=train_batch_size, num_workers=num_workers, shuffle=True)