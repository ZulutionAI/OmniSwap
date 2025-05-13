import json
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image, ImageOps
# import transforms
from torchvision import transforms

from transformers import CLIPImageProcessor
from insightface.app import FaceAnalysis

import cv2

def canny_processor(image, low_threshold=100, high_threshold=200):
    image = np.array(image)
    image = cv2.Canny(image, low_threshold, high_threshold)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    canny_image = Image.fromarray(image)
    return canny_image

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

def expand_bbox(img, bbox, expansion_factor=1.2):
    """
    Expand the bounding box by a given factor while ensuring it stays within the image boundaries.
    """
    x1, y1, x2, y2 = map(int, bbox)

    # Calculate width and height of the original bbox
    width = x2 - x1
    height = y2 - y1

    # Increase width and height by the expansion factor
    new_width = int(width * (expansion_factor))
    new_height = int(height * (expansion_factor))

    # Calculate new x1, y1, x2, y2 ensuring the bbox stays within image boundaries
    new_x1 = max(0, x1 - (new_width - width) // 2)
    new_y1 = max(0, y1 - (new_height - height) // 2)
    new_x2 = min(img.width, x2 + (new_width - width) // 2)
    new_y2 = min(img.height, y2 + (new_height - height) // 2)

    # Ensure new_x2 and new_y2 are not smaller than new_x1 and new_y1
    new_x2 = max(new_x1 + 1, new_x2)
    new_y2 = max(new_y1 + 1, new_y2)

    return [new_x1, new_y1, new_x2, new_y2]

def c_crop_1344x768(image):
    width, height = image.size
    # 比例是1.75
    if height / width > 1.75: # crop height
        top = 0 #(height - int(width * 1.75)) // 2
        bottom = (height - int(width * 1.75))
        right = 0
        left = 0
    else:
        right = (width - int(height / 1.75)) // 2
        left = right
        bottom = 0
        top = 0
    
    # print("left, top, right, bottom:", left, top, right, bottom)
    return image.crop((left, top, width - right, height - bottom))

def c_crop_1344x768(image):
    width, height = image.size
    # 比例是1.75
    if height / width > 1.75: # crop height
        top = 0 #(height - int(width * 1.75)) // 2
        bottom = (height - int(width * 1.75))
        right = 0
        left = 0
    else:
        right = (width - int(height / 1.75)) // 2
        left = right
        bottom = 0
        top = 0
    
    # print("left, top, right, bottom:", left, top, right, bottom)
    return image.crop((left, top, width - right, height - bottom))

class IpAdaptorImageDataset(Dataset):
    def __init__(self, args):
        super().__init__()

        data_num = args.get("data_num", -1)
        box_type = args.get("box_type", "face")
        data_aug = args.get("aug", False)
        self.face_pad = args.get("face_pad", False)
        center_crop = args.get("center_crop", False)

        self.face_embedding = args.get("face_embedding", False)

        size=(448, 768)
        if center_crop is True:
            size = (512, 512)

        t_drop_rate=0.1
        i_drop_rate=0.1
        ti_drop_rate=0.1
        use_face=True

        self.center_crop = center_crop

        self.box_type = box_type

        self.size = size
        self.i_drop_rate = i_drop_rate
        self.t_drop_rate = t_drop_rate
        self.ti_drop_rate = ti_drop_rate

        self.use_face = use_face
        json_file = "/mnt2/wangxuekuan/code/x-flux.bk/IP-Adapter-Flux/data/MSDBv2_single_with_face.json"
        txt_file = "/mnt/huangyuqiu/scripts/face_embedding/results/vcg_temp/single_people_filter_face_area.txt"

        self.data = open(txt_file).readlines() 
        
        if data_num != -1:
            #self.data = json.load(open(json_file))[:data_num]  # list of dict: [{"image_file": "1.png", "text": "A dog"}]
            self.data = open(txt_file).readlines()[:data_num] * 100
        else:
            # self.data = json.load(open(json_file))
            self.data = open(txt_file).readlines()


        # self.transform = transforms.Compose([
        #     transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BILINEAR),
        #     transforms.CenterCrop(self.size),
        #     transforms.ToTensor(),
        #     transforms.Normalize([0.5], [0.5]),
        # ])
        if self.use_face and data_aug:
            self.face_transform = transforms.Compose([
                transforms.RandomResizedCrop(
                    224, scale=(0.8, 1.2), ratio=(0.8, 1.2), interpolation=transforms.InterpolationMode.BICUBIC
                    ),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)], p=0.2,
                ),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.2),
                transforms.RandomSolarize(threshold=128, p=0.2),
            ])
        else:
            self.face_transform = transforms.Compose([
                transforms.RandomResizedCrop(
                    224, scale=(1.0, 1.0), ratio=(1.0, 1.0), interpolation=transforms.InterpolationMode.BICUBIC
                    ),
                # transforms.RandomHorizontalFlip(),
                # transforms.RandomApply(
                #     [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)], p=0.2,
                # ),
                # transforms.RandomGrayscale(p=0.2),
                # transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.2),
                # transforms.RandomSolarize(threshold=128, p=0.2),
            ])


        self.clip_image_processor = CLIPImageProcessor()
        self.error_idx = []

        # if self.face_embedding == True:
        #     self.face_model = FaceAnalysis(providers=['CPUExecutionProvider'])
        #     self.face_model.prepare(ctx_id=-1, det_size=(640, 640))

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if idx in self.error_idx:
            return self.__getitem__(random.randint(0, len(self.data) - 1))
        # try:
        if True:
            # item = self.data[idx] 
            #
            image_file = self.data[idx].strip()
            json_path = image_file.replace("/mnt2/huangyuqiu/share/vcg_images_122W_1/img/", "/mnt2/wangxuekuan/code/x-flux.bk/IP-Adapter-Flux/data/MSDBv2_single/json/").replace(".jpg", ".json")
            # json_path = "/mnt/shuiyunhao/task/IP-Adapter-Flux/" + item["json_file"]
            # image_file = "/mnt/shuiyunhao/task/IP-Adapter-Flux/" +item["image_file"]
            # print(json_path, image_file)
            # print(
            
            # read json
            with open(json_path, 'r') as f:
                info_dict = json.load(f)
            prob = random.random()
            # if prob < 0.5:
            prompt = info_dict["caption_detail"]
            # else:
            #     prompt = info_dict["caption_simple"]

            # read image         
            raw_image = Image.open(image_file)

            if self.center_crop is True:
                bbox = np.array(info_dict['segments'][0]["bbox"])
                bbox[bbox < 0] = 0

                # 计算中心裁剪的位置,同时确保bbox在裁剪后的图像中
                img_w, img_h = raw_image.size
                crop_size = min(img_w, img_h)
                left = max(0, min((img_w - crop_size) // 2, bbox[0]))
                top = max(0, min((img_h - crop_size) // 2, bbox[1]))
                right = min(img_w, max(left + crop_size, bbox[2]))
                bottom = top + (right - left)
                
                # 裁剪并应用其他变换
                if top == bbox[1]:
                    top = random.randint(0, int(bbox[1]))
                    bottom = top + crop_size

            else:
                img_w, img_h = raw_image.size
                left = 0
                top = 0
                right = img_w - 1
                bottom = img_h - 1

            # 读取bbox信息
            if self.box_type == "face" and "face_bbox" in list(info_dict['segments'][0].keys()):
                # print("face")
                bbox = np.array(info_dict['segments'][0]["face_bbox"])
                if self.face_pad:
                    w = bbox[2] - bbox[0]
                    h = bbox[3] - bbox[1]
                    if w > h:
                        bbox[1] -= (w - h) // 2
                        bbox[3] += (w - h) // 2
                    else:
                        bbox[0] -= (h - w) // 2
                        bbox[2] += (h - w) // 2

            elif self.box_type == "all":
                bbox = np.array([left, top, right, bottom])
            else:
                # print("body")
                bbox = np.array(info_dict['segments'][0]["bbox"])
            
            bbox[bbox < 0] = 0

            # 计算中心裁剪的位置,同时确保bbox在裁剪后的图像中
             

            cropped_image = raw_image.crop((left, top, right, bottom))

            if self.center_crop is False:
                cropped_image = c_crop_1344x768(cropped_image)
            
            # cropped_image.save("cropped_image.jpg")

            # image = self.transform(cropped_image.convert("RGB"))
        
            image = cropped_image.convert("RGB").resize(self.size)

            hint = canny_processor(image)
            hint = torch.from_numpy((np.array(hint) / 127.5) - 1)
            hint = hint.permute(2, 0, 1)
            
            # self.face_embedding:
            #     faces_embedding = self.face_model.get(np.array(image))

            # try:
            #     bbox = np.array(info_dict['segments'][0]["face_bbox"])
            #     bbox[bbox<0] = 0
            #     #face_image = raw_image.crop(bbox)
                
            # except:
            bbox = np.array(info_dict['segments'][0]["bbox"])
            bbox[bbox<0] = 0
            
            #print("Can't detect face, use whole body img!")

            expansion_factor = random.choice([1.0, 1.0, random.uniform(1.0, 1.8)])
            bbox = expand_bbox(raw_image, bbox, expansion_factor)

            # np.zero(())
            size = cropped_image.size
            print(size)
            ip_atten_mask = np.zeros((size[1], size[0]))
            ip_atten_mask[bbox[3]:bbox[1], bbox[2]:bbox[0]] = 1
            ip_atten_mask = cv2.resize(ip_atten_mask, self.size)

            face_image = raw_image.crop(bbox)
            # face_image.save("face_image.jpg")
            if face_image.size[0] > 64 and face_image.size[1] > 64:
                face_image = self.face_transform(face_image.convert("RGB"))
                clip_image = self.clip_image_processor(images=face_image, return_tensors="pt").pixel_values[0]
                drop_image_embed = 0
                rand_num = random.random()
                if rand_num < self.i_drop_rate:
                    drop_image_embed = 1
                    clip_image = torch.zeros_like(clip_image)
                elif rand_num < (self.i_drop_rate + self.t_drop_rate):
                    prompt = ""
                elif rand_num < (self.i_drop_rate + self.t_drop_rate + self.ti_drop_rate):
                    prompt = ""
                    drop_image_embed = 1
                    clip_image = torch.zeros_like(clip_image)
                
                # prompt = ""
                image = np.array(image).transpose(2, 0, 1) / 127.5 - 1
                # print(image.shape)
                # print(image.max(), image.min())
                # print(faces_embedding)
                # self.face_embedding:
                #     if len(faces_embedding) == 0:
                #         faces_embedding = faces_embedding[0].embedding
                #     else:
                #         faces_embedding = np.zeros(512)
                    
                #     faces_embedding = torch.from_numpy(faces_embedding)
                # else:
                faces_embedding = torch.from_numpy(np.zeros(512))
                ip_atten_mask = torch.from_numpy(ip_atten_mask)
                # size
                
                return image, prompt, clip_image, faces_embedding, hint, ip_atten_mask #, drop_image_embed
            else:
                print("small - data")
                self.error_idx.append(idx)
                return self.__getitem__(random.randint(0, len(self.data) - 1))

        # except Exception as e:
        #     print(e)
        #     self.error_idx.append(idx)
        #     return self.__getitem__(random.randint(0, len(self.data) - 1))


def loader(train_batch_size, num_workers, **args):
    # print(args)
    dataset = IpAdaptorImageDataset(args)
    return DataLoader(dataset, batch_size=train_batch_size, num_workers=num_workers, shuffle=True)
