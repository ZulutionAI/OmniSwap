from datetime import datetime
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
import os
from PIL import Image
import cv2
import numpy as np
import json
import glob
import concurrent.futures
from tqdm import tqdm
import scipy
import copy
import face_recognition
import math
import torch.nn.functional as F
from insightface.utils import face_align

import time

from .utils import get_new_size, c_cropv2, draw_handpose, draw_bodypose, convert_open_to_mmpose, canny_processor, get_skeleton_keypoints_from_facebook308, combine_canny_skeleton
from .utils import fill_mask, get_sapiens_sam_path, check_bbox_overlap, calculate_mask_external_rectangle, pad_bbox, canny_processor, c_crop, c_pad, expand_bbox, c_crop_1344x768, get_sapiens_path

def gen_control_image(raw_image, json_path, image_file, skeleton_sapiens_path, mask_sapiens_path):
    prob = random.random()
    if "danbooru_anime" in json_path or "anime_pictures" in json_path:
        hint_skeleton = canny_processor(raw_image)
        hint_skeleton = np.array(hint_skeleton)
    else:
        if prob > 0.7:
            if "/skeleton_sapiens/" in skeleton_sapiens_path or "/skeleton_sapiens_308/" in skeleton_sapiens_path:
                hint_skeleton = combine_canny_skeleton(img_path=image_file, \
                                            skeleton_path=skeleton_sapiens_path, 
                                            seg_path=mask_sapiens_path, 
                                            is_facebook=True, add_canny=False).astype(np.uint8)
            else:
                hint_skeleton = combine_canny_skeleton(img_path=image_file, \
                                            skeleton_path=skeleton_sapiens_path, 
                                            seg_path=mask_sapiens_path, 
                                            is_facebook=False, add_canny=False).astype(np.uint8)
        elif prob > 0.1:
            hint_skeleton = canny_processor(raw_image)
            hint_skeleton = np.array(hint_skeleton)
        else:
            hint_skeleton = np.zeros_like(np.array(raw_image))

    return hint_skeleton

def mask_aug(input_mask, aug_type="dilate"):
    input_mask = input_mask.astype(np.uint8)
    kernel_size = random.randint(1, 7)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    if aug_type == "dilate": 
        iterations_n = random.choice([3, 4, 5])
        input_mask = cv2.dilate(input_mask, kernel, iterations=iterations_n)
    elif aug_type == "dilate_small":
        iterations_n = random.choice([1, 2])
        input_mask = cv2.dilate(input_mask, kernel, iterations=iterations_n)
    elif aug_type == "erode": 
        input_mask = cv2.erode(input_mask, kernel, iterations=iterations_n)

    input_mask[input_mask>0] = 1
    input_mask[input_mask<=0] = 0
    
    return input_mask.astype(bool)


def crop_norm_image(hint_mask_image, target_size, value_norm_flag = False):
    hint_mask_image = Image.fromarray(hint_mask_image)
    hint_mask_image = c_cropv2(hint_mask_image)
    hint_mask_image = hint_mask_image.convert("RGB").resize(target_size)
    if value_norm_flag:
        hint_mask_image = torch.from_numpy((np.array(hint_mask_image) / 127.5) - 1)
    else:
        hint_mask_image = torch.from_numpy(np.array(hint_mask_image))
    hint_mask_image = hint_mask_image.permute(2, 0, 1)
    return hint_mask_image

def get_per_mask_region(mask_sapiens_path, mask_sam_path, seg_idx, left_arm_flag=False, right_arm_flag=False, left_leg_flag=False, right_leg_flag=False, torso_flag=False):
    if mask_sapiens_path is not None and mask_sam_path is not None:
        if mask_sapiens_path.endswith("npy"):
            mask_info = np.load(mask_sapiens_path)
        else:
            mask_info = scipy.sparse.load_npz(mask_sapiens_path).toarray()
        if mask_sam_path.endswith("npy"):
            mask_info_sam = np.load(mask_sam_path)
        else:
            mask_info_sam = scipy.sparse.load_npz(mask_sam_path).toarray()
        person_mask = np.logical_and(mask_info, mask_info_sam==seg_idx)
        mask_info[~person_mask] = 0
        face_mask = mask_info == 2
        for xxxx in [23, 24, 25, 26, 27]:
            face_mask = np.logical_or(face_mask, mask_info == xxxx) 
        # face_mask = mask_info == 2
        face_mask = face_mask.astype(bool)
        hair_mask = mask_info == 3
        # 23: Lower_Lip
        # 24: Upper_Lip
        # 25: Lower_Teeth
        # 26: Upper_Teeth
        # 27: Tongue
        cloth_mask = mask_info == 8
        for xxxx in [9, 17, 18, 12, 22]:
            cloth_mask = np.logical_or(cloth_mask, mask_info == xxxx)
        cloth_mask = cloth_mask.astype(bool)
        # 8: Left_Shoe
        # 9: Left_Sock
        # 17: Right_Shoe
        # 18: Right_Sock
        # 12: Lower_Clothing
        # 22: Upper_Clothing
        body_mask = np.zeros_like(mask_info)
        if left_arm_flag:
            for xxxx in [5, 6, 10]:
                body_mask = np.logical_or(body_mask, mask_info == xxxx)
        if right_arm_flag:
            for xxxx in [14, 15, 19]:
                body_mask = np.logical_or(body_mask, mask_info == xxxx)
        if left_leg_flag:
            for xxxx in [4, 7, 11]:
                body_mask = np.logical_or(body_mask, mask_info == xxxx)
        if right_leg_flag:
            for xxxx in [13, 16, 20]:
                body_mask = np.logical_or(body_mask, mask_info == xxxx)
        if torso_flag:
            body_mask = np.logical_or(body_mask, mask_info == 21)
        body_mask = np.logical_or(body_mask, cloth_mask)
        body_mask = body_mask.astype(bool)

    else:
        mask_info = None
        face_mask = None
        hair_mask = None
        cloth_mask = None
        body_mask = None

    return face_mask, hair_mask, cloth_mask, body_mask, mask_info

def fill_image_noise(face_image, face_mask, hair_mask, atten_box, hair_type):
    # 创建一个与face_image大小相同的噪声图像
    h, w = face_image.size
    noise = np.random.randint(0, 256, size=(w, h, 3), dtype=np.uint8)
    
    # 将face_image转换为numpy数组
    face_image_np = np.array(face_image)
    
    face_mask_resized = face_mask[atten_box[1]:atten_box[3], atten_box[0]:atten_box[2]]
     
    # 合并face和hair的掩码
    # print(face_mask_resized, hair_mask_resized)
    if hair_type and random.random() > 0.8:
        if hair_mask is not None:
            hair_mask_resized = hair_mask[atten_box[1]:atten_box[3], atten_box[0]:atten_box[2]]
            combined_mask = np.logical_or(face_mask_resized, hair_mask_resized)
        else:
            combined_mask = face_mask_resized.astype(bool)
    else:
        combined_mask = face_mask_resized.astype(bool)
    # 使用掩码将face和hair以外的区域替换为噪声
    face_image_np[~combined_mask] = noise[~combined_mask]
    # print(face_image_np)
    # 将numpy数组转回Image对象
    face_image = Image.fromarray(face_image_np)

    return face_image

def update_face_box_by_mask(face_bbox, face_mask, hair_mask, face_type, hair_type, cloth_type):
    # ref image
    all_mask = None
    mask_box = None
    if face_type == "box":
        if hair_type:
            hair_box = calculate_mask_external_rectangle(hair_mask)
            if hair_box is not None:
                mask_box = [min(face_bbox[0], hair_box[0]), min(face_bbox[1], hair_box[1]), 
                            max(face_bbox[2], hair_box[2]), max(face_bbox[3], hair_box[3])]

    else: # mask
        if hair_type:
            all_mask = np.logical_or(face_mask, hair_mask)
            all_mask = all_mask.astype(bool)
        else:
            all_mask = face_mask.astype(bool)
        
        mask_box = calculate_mask_external_rectangle(all_mask)
    
    drag_flag = True
    if mask_box is not None:
        if cloth_type is False:
            if (mask_box[2] - mask_box[0]) * (mask_box[3] - mask_box[1]) < (face_bbox[2] - face_bbox[0]) * (face_bbox[3] - face_bbox[1]) * 2.5:
                face_bbox = mask_box
                drag_flag = False
        else:
            face_bbox = mask_box
            drag_flag = False
        
    return face_bbox, all_mask, drag_flag

class IpAdaptorImageDataset(Dataset):
    def __init__(self, args):
        super().__init__()

        data_num = args.get("data_num", -1)
        box_type = args.get("box_type", "face")
        data_aug = args.get("aug", False)
        self.face_pad = args.get("face_pad", False)
        center_crop = args.get("center_crop", False)
        self.genal_caption_flag = args.get("genal_caption_flag", False)

        self.face_embedding = args.get("face_embedding", False)
        self.atten_mask_type = args.get("atten_mask_type", "face")
        self.ip_number = args.get("ip_number", 1)
        self.ani_picture = args.get("ani_picture", True)
        self.real_picture = args.get("real_picture", True)
        self.have_cloth = args.get("have_cloth", False)
        self.do_only_cloth = args.get("do_only_cloth", False)
        self.max_side_len = args.get("max_side_len", 1024)
        self.face_align = args.get("face_align", False) 
        self.clip_size = args.get("clip_image_size", 224)

        self.face_min_area = args.get("face_min_area", 0.04)

        size=(448, 768)

        t_drop_rate=0.1
        i_drop_rate=0.1
        ti_drop_rate=0.1
   
        self.center_crop = center_crop

        self.box_type = box_type

        self.target_size = (args.get("img_size_w", 448), args.get("img_size_h", 768))
        self.i_drop_rate = i_drop_rate
        self.t_drop_rate = t_drop_rate
        self.ti_drop_rate = ti_drop_rate
        self.use_face = True

        plist = []
        if self.real_picture:
            plist += [
                "/mnt2/wangxuekuan/code/x-flux.bk/data/ip_adapter/train_ip_vcg_80w.txt",
                "/mnt2/wangxuekuan/code/x-flux.bk/data/ip_adapter/train_ip_flux_1.txt",
                "/mnt2/wangxuekuan/code/x-flux.bk/data/ip_adapter/train_ip_vcg_122w.txt",
                "/mnt2/wangxuekuan/code/x-flux.bk/data/ip_adapter/train_ip_vcg_chat_history_0_300_two_people.txt",
                "/mnt2/wangxuekuan/code/x-flux.bk/data/ip_adapter/train_ip_vcg_reelshort_0_200.txt",
            ]

        if self.ani_picture: #and not self.do_only_cloth:
            plist += [
                "/mnt/wangxuekuan/code/layout/dataset_split/danbooru_anime.txt",
                "/mnt/wangxuekuan/code/layout/dataset_split/flux_anime_1character.txt",
                "/mnt/wangxuekuan/code/layout/dataset_split/flux_anime_2characters.txt",
                "/mnt2/wangxuekuan/code/x-flux.bk/data/ip_adapter/anime_pictures_46w.txt",
            ]

        plist += [
            "/mnt2/wangxuekuan/code/x-flux.bk/data/ip_adapter/tuchong.txt",
        ]
        
        # plist += [
        #     '/mnt2/zhenghaoyu/share/IGPair/IGPair_seperate/IGPair_cloth_pair.txt',
        #     '/mnt2/zhenghaoyu/share/IGPair/IGPair_seperate/IGPair_cloth_pair.txt',
        #     '/mnt2/zhenghaoyu/share/IGPair/IGPair_seperate/IGPair_cloth_pair.txt',
        #     '/mnt2/zhenghaoyu/share/IGPair/IGPair_seperate/IGPair_cloth_pair.txt',
        #     '/mnt2/zhenghaoyu/share/IGPair/IGPair_seperate/IGPair_cloth_pair.txt',
        # ]

        self.data_tmp = []
        for p in plist:
            self.data_tmp += open(p).readlines() 
        self.data = self.data_tmp

        if data_num != -1:
            self.data = self.data[:data_num] * 100

        if self.use_face and data_aug:
            self.face_transform = transforms.Compose([
                transforms.RandomRotation(degrees=30),
                transforms.RandomResizedCrop(
                    224, scale=(0.8, 1.2), ratio=(0.8, 1.2), interpolation=transforms.InterpolationMode.BICUBIC
                    ),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)], p=0.2,
                ),
                transforms.RandomGrayscale(p=0.1),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.1),
                transforms.RandomSolarize(threshold=128, p=0.1),
            ])
            
            self.cloth_transform = transforms.Compose([
                transforms.RandomRotation(degrees=30),
                transforms.RandomResizedCrop(
                    224, scale=(0.9, 1.1), ratio=(0.9, 1.1), interpolation=transforms.InterpolationMode.BICUBIC
                    ),
                transforms.RandomHorizontalFlip(),
                # transforms.RandomApply(
                #     [transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)], p=0.1,
                # ),
                # transforms.RandomGrayscale(p=0.05),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.1),
                # transforms.RandomSolarize(threshold=128, p=0.05),
            ])

        else:
            self.face_transform = transforms.Compose([
                transforms.RandomResizedCrop(
                    224, scale=(1.0, 1.0), ratio=(1.0, 1.0), interpolation=transforms.InterpolationMode.BICUBIC
                    ),
            ])


        # self.clip_image_processor = CLIPImageProcessor()
        self.clip_image_processor = CLIPImageProcessor(crop_size={"height": self.clip_size, "width": self.clip_size}, size={"shortest_edge": self.clip_size})
        self.error_idx = []

        # if self.face_embedding == True:
        #     self.face_model = FaceAnalysis(providers=['CPUExecutionProvider'])
        #     self.face_model.prepare(ctx_id=-1, det_size=(640, 640))
        # face_recognition.face_encodings(image)[0]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if idx in self.error_idx:
            return self.__getitem__(random.randint(0, len(self.data) - 1))
        try:
        # if True:
            n, json_path, image_file = self.data[idx].strip().split(", ")
            
            if 'IGPair' not in json_path:
                do_box_only = False
                # if self.genal_caption_flag:
                ftype = image_file.split(".")[-1]
                caption_path = "/mnt3/wangxuekuan/data/ip_adapter_gpt4o__desc/" + image_file[:-len(ftype)] + "txt"
                if not os.path.exists(caption_path):
                    print("no caption path", caption_path)
                    self.error_idx.append(idx)
                    return self.__getitem__(random.randint(0, len(self.data) - 1))

                prompt = open(caption_path, "r").readlines()[0].strip()

                mask_sapiens_path, skeleton_sapiens_path, mask_sam_path = get_sapiens_sam_path(json_path)
                
                if "danbooru_anime" not in json_path and "anime_pictures" not in json_path:
                    if mask_sapiens_path is None or skeleton_sapiens_path is None or mask_sam_path is None:
                        return self.__getitem__(random.randint(0, len(self.data) - 1))
                
                if "danbooru_anime" in json_path or "anime_pictures" in json_path:
                    do_box_only = True

                # read json
                new_json_path = json_path.replace("/mnt2/wangxuekuan/data/MSDBv2_json/mnt/wangxuekuan/data/vcg_images_122W_1_v3_format_simple_skeleton/", \
                                                "/mnt2/huangyuqiu/share/vcg_images_122W_1/json_final_v2/")

                with open(new_json_path, 'r') as f:
                    info_dict = json.load(f)
                    
                raw_image = Image.open(image_file).convert("RGB")
                raw_size = raw_image.size

                # if self.target_size[0] == -1:
                new_w, new_h = get_new_size(raw_image, self.max_side_len)
                # print(raw_size, new_w, new_h)
                self.target_size = (new_w // 16 * 16, new_h // 16 * 16)

                # init mask image
                hint_mask_image = np.array(copy.deepcopy(raw_image))
                hint_mask = np.ones_like(hint_mask_image)
                
                # control image
                hint_skeleton = gen_control_image(raw_image, json_path, image_file, skeleton_sapiens_path, mask_sapiens_path)

                #image = cropped_image.convert("RGB").resize(self.size)
                ### == > list 
                # 读取bbox信息
                face_image_list = []
                ip_atten_mask_list = []
                face_box_list = []
                body_bbox = []

                # 每次选择一个脸
                seg_tmp_list = []
                face_idx = []
                idx = 0
                for seg in info_dict['segments']:
                    if "face_bbox" in list(seg.keys()):
                        if seg["face_bbox"] != None:
                            if (seg["face_bbox"][3] - seg["face_bbox"][1]) * (seg["face_bbox"][2] - seg["face_bbox"][0]) / (raw_size[0] * raw_size[1]) > self.face_min_area:
                            # if seg["face_bbox"][3] - seg["face_bbox"][1] > 48 and seg["face_bbox"][2] - seg["face_bbox"][0] > 48:
                                seg_tmp_list.append(seg)
                                face_idx.append(idx)
                            idx += 1
                
                if len(seg_tmp_list) == 1 and self.ip_number == 1:
                    choice_idx = random.choice([i for i in range(len(seg_tmp_list))])
                    seg_all = [seg_tmp_list[choice_idx]]
                    face_embedding_idx = face_idx[choice_idx]
                else:
                    print("small - data")
                    self.error_idx.append(idx)
                    return self.__getitem__(random.randint(0, len(self.data) - 1))
                
                for seg in seg_all:
                    # 读取face box 和 对应的 attention box （feature 注入）
                    face_bbox = np.array(seg["face_bbox"])
                    
                    # 提取脸部的embedding
                    face_image_tmp = raw_image.crop(face_bbox)
                    face_image_tmp = np.array(face_image_tmp)
                    # print(face_image_tmp.shape)
                    if True:
                        # 读取face embedding 特征
                        face_embedding_path = json_path.replace("/mnt2/wangxuekuan/data/MSDBv2_json/mnt/wangxuekuan/data/vcg_images_122W_1_v3_format_simple_skeleton/", \
                                                                "/mnt2/huangyuqiu/share/vcg_images_122W_1/face_embedding_wxk/").replace(".json", ".npz")
                        
                        try:
                            if not os.path.exists(face_embedding_path):
                                face_embedding = np.zeros(512)
                            else:
                                loaded_sparse_mask = scipy.sparse.load_npz(face_embedding_path)
                                face_embedding = loaded_sparse_mask.toarray()[face_embedding_idx]
                                face_embedding = torch.from_numpy(face_embedding)
                        except:
                            face_embedding = np.zeros(512)

                        # 计算另一个
                        # 
                        if self.face_align:
                            if "face_kps" in list(seg.keys()) and random.random() > 0.5:
                                lmk = np.array(seg['face_kps'])
                                origin_w, origin_h = info_dict['image_size']['w'], info_dict['image_size']['h']
                                now_w, now_h = raw_image.size
                                ratio = min(origin_w/now_w, origin_h/now_h)
                                lmk = lmk / ratio
                                # t = time.time()
                                # cv2.imwrite(f"test/face_image_tmp_1_{t}.jpg", face_image_tmp)
                                face_image_tmp = face_align.norm_crop(np.array(raw_image), landmark=lmk, image_size=224)
                                # print("align face 50%")
                                # cv2.imwrite(f"test/face_image_tmp_2_{t}.jpg", face_image_tmp)
                            else:
                                print("no face_kps", json_path)

                        face_image_tmp = cv2.resize(face_image_tmp, (112, 112))
                        face_image_tmp = torch.from_numpy(face_image_tmp).permute(2, 0, 1) / 255.0

                        # OPENAI_DATASET_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).reshape(-1, 1, 1)
                        # OPENAI_DATASET_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).reshape(-1, 1, 1)

                        # print()
                        # print("face_image_tmp", face_image_tmp.shape, face_image_tmp.max(), face_image_tmp.min())

                        # facerecog_image = face_image_tmp * OPENAI_DATASET_STD + OPENAI_DATASET_MEAN  # [0, 1]
                        facerecog_image = torch.clamp((face_image_tmp  * 2) - 1, -1, 1)  # [-1, 1]
                        # facerecog_image = F.interpolate(facerecog_image, size=(112, 112), mode="bilinear", align_corners=False)
                    
                        # print("facerecog_image", facerecog_image.shape, facerecog_image.max(), facerecog_image.min())


                        # 调用识别模型
                        # face_embedding_circular = facerecog_model(facerecog_image, return_mid_feats=True)[0].to(device, dtype)

                        # face_embedding_circular = drop_embeddings(face_embedding_circular, drop_image_embeds).to(image_embeds.device, image_embeds.dtype)
                        # face_embedding = face_recognition.face_encodings(face_image_tmp)[0]
                        # print(face_image_tmp.shape, face_embedding.shape)
                    # except:
                    #     # print("error:", face_image_tmp.shape)
                    #     face_embedding = np.zeros(512)
                    
                    if random.random() > 0.5 and self.face_pad:
                        face_bbox = pad_bbox(face_bbox)     # 正方形
                    atten_box = face_bbox

                    template_id = None
                    if "template_id" in seg.keys():
                        template_id = seg["template_id"]

                    expansion_factor = random.choice([1.0, 1.0, random.uniform(1.0, 1.8)])
                    face_bbox = expand_bbox(raw_image, face_bbox, expansion_factor)
                    face_bbox[face_bbox<0] = 0
                    
                    # 计算头发和脸的mask
                    all_mask = None #np.ones_like()
                    face_type = "box"
                    face_mask = None
                    hair_type = False
                    drag_flag = False
                    cloth_type = False
                    
                    left_arm_flag = random.choice([True, False])
                    right_arm_flag = random.choice([True, False])
                    left_leg_flag = random.choice([True, False])
                    right_leg_flag = random.choice([True, False])
                    torso_flag = random.choice([True, False])

                    
                    if self.atten_mask_type == "face_hair" and not do_box_only:
                        if self.have_cloth:
                            # print("have_cloth:", self.have_cloth,  "==>")
                            cloth_type = random.choice([True, True, False])
                        else:
                            cloth_type = False
                        
                        if self.do_only_cloth:
                            cloth_type = True

                        face_type = random.choice(["box", "mask"])
                        hair_type = random.choice([True, False])
                        
                        face_mask, hair_mask, cloth_mask, body_mask, mask_info = \
                            get_per_mask_region(mask_sapiens_path, mask_sam_path, int(seg['id']), left_arm_flag, right_arm_flag, left_leg_flag, right_leg_flag, torso_flag)

                        face_mask = fill_mask(face_mask)
                        hair_mask = fill_mask(hair_mask)
                        cloth_mask = fill_mask(cloth_mask)
                        body_mask = fill_mask(body_mask)
                        
                        if cloth_type and cloth_mask is not None:
                            cloth_mask_sum = cloth_mask.sum()
                            if cloth_mask_sum < (seg['bbox'][3] - seg['bbox'][1]) * (seg['bbox'][2] - seg['bbox'][0]) * 0.3:
                                cloth_type = False
                            else:
                                hair_type = False
                                face_mask = cloth_mask
                                face_type = "mask"
                        else:
                            print("XXXX ==> cloth_type:", cloth_type, ", +++>>", cloth_mask == None)
                            cloth_type = False

                        if face_mask is None:
                            drag_flag = True
                        else:
                            face_bbox, all_mask, drag_flag  = update_face_box_by_mask(face_bbox, face_mask, hair_mask, face_type, hair_type, cloth_type)
                            body_mask = body_mask.astype(bool)

                    face_image = raw_image.crop(face_bbox)

                    protrait_flag = False
                    if random.random() > 1.0 and cloth_type == False:
                        protrait_bbox = None
                        if face_type == "box" and "vcg_images_122W_1" in json_path:
                            portrait_path = json_path.replace("/mnt2/wangxuekuan/data/MSDBv2_json/mnt/wangxuekuan/data/vcg_images_122W_1_v3_format_simple_skeleton", 
                                                            "/mnt2/huangyuqiu/share/vcg_images_122W_1/img_liveportrait")[:-5]
                            if os.path.exists(portrait_path):
                                protrait_imgs = glob.glob(portrait_path + "/*.jpg")
                                if len(protrait_imgs) != 0:
                                    protrait_img_path = random.choice(protrait_imgs)
                                    protrait_img = Image.open(protrait_img_path)
                                    protrait_bbox_path = protrait_img_path.replace("/vcg_images_122W_1/img_liveportrait", "/vcg_images_122W_1_liveportrait/json_w_face").replace(".jpg", ".json")
                                    if os.path.exists(protrait_bbox_path):
                                        with open(protrait_bbox_path, "r") as f:
                                            protrait_info_dict = json.load(f)
                                            if len(protrait_info_dict['segments']) != 0:
                                                protrait_bbox = protrait_info_dict['segments'][seg['id']]['face_bbox']
                                                expansion_factor = random.choice([1.0, 1.0, random.uniform(1.0, 1.2)])
                                                protrait_bbox = expand_bbox(protrait_img, protrait_bbox, 1.0)
                        # 裁剪人脸区域
                        if protrait_bbox is not None:
                            if protrait_bbox[2] - protrait_bbox[0] > 64 and protrait_bbox[3] - protrait_bbox[1] > 64:
                                # face_bbox = protrait_bbox.tolist()
                                protrait_flag = True
                                face_image = protrait_img.crop(protrait_bbox)
                                face_type = "box"

                    tuchong_flag = False
                    if random.random() > 0.5 and "tuchong" in json_path and cloth_type==False:
                        #随机替换其他的图片中的ref image
                        json_folder_path = os.path.dirname(json_path)
                        json_all_list = glob.glob(json_folder_path + "/*.json")
                        random.shuffle(json_all_list)
                        # print(json_all_list)
                        
                        for new_json_path in json_all_list:
                            if new_json_path != json_path and not tuchong_flag:
                                # print(new_json_path)
                                with open(new_json_path, 'r') as file:
                                    new_data = json.load(file)
                                    for ss in new_data["segments"]:
                                        if ss["template_id"] == template_id:
                                            face_bbox_new = ss["face_bbox"]
                                            if face_bbox_new[2] - face_bbox_new[0] > 64 and face_bbox_new[3] - face_bbox_new[1] > 64:
                                                face_bbox_crop = face_bbox_new
                                                if random.random() > 0.5 and self.face_pad:
                                                    face_bbox_crop = pad_bbox(face_bbox_crop)     # 正方形
                                                
                                                expansion_factor = random.choice([1.0, 1.0, random.uniform(1.0, 1.8)])
                                                face_bbox_crop = expand_bbox(raw_image, face_bbox_crop, expansion_factor)
                                                face_bbox_crop[face_bbox_crop<0] = 0
                                                # print("do face bbox crop", face_bbox_crop)
                                                tuchong_flag = True
                                                # json
                                                tuchong_img_path = new_json_path.replace(".json", ".jpeg").replace("json_new", "img")
                                                tuchong_img = Image.open(tuchong_img_path)
                                                face_image = tuchong_img.crop(face_bbox_crop)
                                                print("tuchong face crop", face_bbox_crop)
                                                break

                    # 根据mask对face_image填充噪声
                    if random.random() > 0.5 and protrait_flag is False:
                        if face_mask is None:
                            face_mask, hair_mask, cloth_mask, body_mask, mask_info = get_per_mask_region(mask_sapiens_path, mask_sam_path, int(seg['id']), left_arm_flag, right_arm_flag, left_leg_flag, right_leg_flag, torso_flag)
                            face_mask = fill_mask(face_mask)
                            hair_mask = fill_mask(hair_mask)
                            cloth_mask = fill_mask(cloth_mask)
                            body_mask = fill_mask(body_mask)

                        if face_mask is not None:
                            face_image = fill_image_noise(face_image, face_mask, hair_mask, face_bbox, hair_type)

                    face_image_list.append(face_image)

                    if face_mask is None or all_mask is None:
                        face_type = "box"
                    
                    # 计算注入区域
                    # if self.atten_mask_type == "face_hair":
                        # 是否注入头发
                        # if hair_type is False:
                        #     hair_type = random.choice([True, False])
                        # if mask_info is not None and drop_tag is False:
                        #     face_bbox, all_mask, drag_flag  = update_face_box_by_mask(face_bbox, face_mask, hair_mask, face_type, hair_type, cloth_type)
                        # 

                    ip_atten_mask = np.zeros((raw_size[1], raw_size[0]))
                    if self.atten_mask_type == "face_hair" and face_type == "mask" and not do_box_only:
                        # 膨胀一下
                    
                        all_mask = fill_mask(all_mask)
                        all_mask = mask_aug(all_mask, aug_type="dilate_small")
                        
                        temp_box = calculate_mask_external_rectangle(all_mask)    # todo debug None
                        face_box_list.append(temp_box)
                        
                        # 随机替换cloth mask 为boy-mask
                        if random.random() > 0.5:
                            all_mask = fill_mask(body_mask)  
                        
                        if random.random() > 0.5:
                            hint_mask_image[all_mask] = 0
                            hint_mask[all_mask] = 0
                        else:
                            temp_box_mask = calculate_mask_external_rectangle(all_mask)
                            hint_mask_image[temp_box_mask[1]:temp_box_mask[3], temp_box_mask[0]:temp_box_mask[2]] = 0
                            hint_mask[temp_box_mask[1]:temp_box_mask[3], temp_box_mask[0]:temp_box_mask[2]] = 0

                        # 随机更新为box
                        all_mask = mask_aug(all_mask, aug_type="dilate")
                        if random.random() > 0.5:
                            ip_atten_mask[all_mask] = 1
                        else:
                            temp_box_mask = calculate_mask_external_rectangle(all_mask)
                            ip_atten_mask[temp_box_mask[1]:temp_box_mask[3], temp_box_mask[0]:temp_box_mask[2]] = 1

                    else:
                        atten_box = face_bbox
                        expansion_factor = random.choice([1.0, 1.0, random.uniform(1.0, 1.2)])
                        atten_box = expand_bbox(raw_image, atten_box, expansion_factor)
                        atten_box[atten_box<0] = 0

                        ip_atten_mask[atten_box[1]:atten_box[3], atten_box[0]:atten_box[2]] = 1
                        hint_mask_image[atten_box[1]:atten_box[3], atten_box[0]:atten_box[2]] = 0
                        hint_mask[atten_box[1]:atten_box[3], atten_box[0]:atten_box[2]] = 0

                        face_box_list.append(atten_box)
                    
                    ip_atten_mask_small = cv2.resize(ip_atten_mask, (self.target_size[0]//16, self.target_size[1]//16))
                    ip_atten_mask_list.append(ip_atten_mask_small)
            
            else:
            # if True:
                print("Got IGpair!")
                cloth_img_path = json_path
                witch_cloth_type = image_file.split(".")[0][-1]
                if witch_cloth_type != '0' and witch_cloth_type != '1':
                    print("cloth type error", image_file)
                    self.error_idx.append(idx)
                    return self.__getitem__(random.randint(0, len(self.data) - 1))
                original_img_path = image_file.split('_cloth')[0].replace('images_pair', 'images') + ".jpg"
                mask_path = original_img_path.replace('images', 'mask_sapiens').replace('.png', '.jpg').replace('.jpg', '_seg.npz')
                skeleton_path = original_img_path.replace('images', 'skeleton').replace('.png', '.jpg').replace('.jpg', '.json')
                original_img_path_list = original_img_path.split("/")
                original_img_path = os.path.join("/mnt2/shuiyunhao/data/IGPair/IGPair_seperate", original_img_path_list[-2], "IGPair/images", original_img_path_list[-1])
                ftype = original_img_path.split(".")[-1]
                caption_path = "/mnt3/wangxuekuan/data/ip_adapter_gpt4o__desc/" + original_img_path[:-len(ftype)] + "txt"
                
                if not os.path.exists(caption_path):
                    print("no caption path", caption_path)
                    self.error_idx.append(idx)
                    return self.__getitem__(random.randint(0, len(self.data) - 1))

                prompt = open(caption_path, "r").readlines()[0].strip()
                
                if not os.path.exists(mask_path):
                    print(f'{mask_path} not exists')
                    self.error_idx.append(idx)
                    return self.__getitem__(random.randint(0, len(self.data) - 1))
                
                seg_data = scipy.sparse.load_npz(mask_path).toarray()
                if witch_cloth_type == '0':
                    cloth_mask = seg_data == 22
                    if random.random() > 0.5:
                        for xxxx in [5, 6, 10]:
                            cloth_mask = np.logical_or(cloth_mask, seg_data == xxxx)
                    if random.random() > 0.5:
                        for xxxx in [14, 15, 19]:
                            cloth_mask = np.logical_or(cloth_mask, seg_data == xxxx)
                    if random.random() > 0.5:
                        cloth_mask = np.logical_or(cloth_mask, seg_data == 21)
                else:
                    cloth_mask = seg_data == 12
                    if random.random() > 0.5:
                        for xxxx in [4, 7, 11]:
                            cloth_mask = np.logical_or(cloth_mask, seg_data == xxxx)
                    if random.random() > 0.5:
                        for xxxx in [13, 16, 20]:
                            cloth_mask = np.logical_or(cloth_mask, seg_data == xxxx)
                    
                raw_image = Image.open(image_file).convert("RGB")
                raw_size = raw_image.size

                # if self.target_size[0] == -1:
                new_w, new_h = get_new_size(raw_image, self.max_side_len)
                # print(raw_size, new_w, new_h)
                self.target_size = (new_w // 16 * 16, new_h // 16 * 16)

                # init mask image
                hint_mask_image = np.array(copy.deepcopy(raw_image))
                hint_mask = np.ones_like(hint_mask_image)
                
                # control image
                hint_skeleton = gen_control_image(raw_image, json_path, image_file, skeleton_path, mask_path)
                
                reference_image = Image.open(cloth_img_path)
                face_image_list = [reference_image]
                face_box = calculate_mask_external_rectangle(cloth_mask)
                face_box_list = [face_box]
                cloth_type = True
                ip_atten_mask = np.zeros((raw_size[1], raw_size[0]))
                if random.random() > 0.5:
                    ip_atten_mask[face_box[1]:face_box[3], face_box[0]:face_box[2]] = 1
                    hint_mask_image[face_box[1]:face_box[3], face_box[0]:face_box[2]] = 0
                    hint_mask[face_box[1]:face_box[3], face_box[0]:face_box[2]] = 0
                else:
                    all_mask = fill_mask(cloth_mask)
                    all_mask = mask_aug(all_mask, aug_type="dilate_small")
                    ip_atten_mask[all_mask] = 1
                    hint_mask_image[all_mask] = 0
                    hint_mask[all_mask] = 0
                    
                # cv2.imwrite(f"test/{datetime.now().strftime('%Y%m%d_%H%M%S')}_hint_mask_image.jpg", hint_mask_image)
                # cv2.imwrite(f"test/{datetime.now().strftime('%Y%m%d_%H%M%S')}_hint_skeleton.jpg", hint_skeleton)
                # raw_image.save(f"test/{datetime.now().strftime('%Y%m%d_%H%M%S')}_raw_image.jpg")
                # reference_image.save(f"test/{datetime.now().strftime('%Y%m%d_%H%M%S')}_reference_image.jpg")
                    
                ip_atten_mask_small = cv2.resize(ip_atten_mask, (self.target_size[0]//16, self.target_size[1]//16))
                ip_atten_mask_list = []
                ip_atten_mask_list.append(ip_atten_mask_small)
                
                face_embedding = np.zeros(512)
                facerecog_image = np.array(raw_image)
                
            '''
            # 统一尺寸为16的倍数 + 随机drop prompt
            '''

            image = c_cropv2(raw_image)
            w, h = raw_image.size
            new_w, new_h = image.size
            image = image.convert("RGB").resize(self.target_size)
            ratio_w = self.target_size[0] / new_w
            ratio_h = self.target_size[1] / new_h
            tt = datetime.now().strftime('%Y%m%d_%H%M%S')

            # image.save(f"test/{tt}_all_image.jpg")
            image = np.array(image).transpose(2, 0, 1) / 127.5 - 1
            
            face_image = np.array(face_image_list[0])

            hint_mask_image = crop_norm_image(hint_mask_image, self.target_size, value_norm_flag = True)
            hint_mask = crop_norm_image(hint_mask, (self.target_size[0]//16, self.target_size[1]//16), value_norm_flag = False)[:1]
            hint_skeleton = crop_norm_image(hint_skeleton, self.target_size, value_norm_flag = True)

            dw = (w - new_w) // 2
            dh = (h - new_h) // 2
            
            for idx in range(len(face_box_list)):
                face_box_list[idx][0] = face_box_list[idx][0] - dw
                face_box_list[idx][1] = face_box_list[idx][1] - dh
                face_box_list[idx][2] = face_box_list[idx][2] - dw
                face_box_list[idx][3] = face_box_list[idx][3] - dh
                face_box_list[idx][0] = max(int(face_box_list[idx][0] * ratio_w), 0)
                face_box_list[idx][1] = max(int(face_box_list[idx][1] * ratio_h), 0)
                face_box_list[idx][2] = max(int(face_box_list[idx][2] * ratio_w), 0)
                face_box_list[idx][3] = max(int(face_box_list[idx][3] * ratio_h), 0)
                if (face_box_list[idx][2] - face_box_list[idx][0]) < 16 or (face_box_list[idx][3] - face_box_list[idx][1]) < 16:
                    del face_box_list[idx]
                face_box_list[idx] = [int(i) for i in face_box_list[idx]]


            clip_image_list = []
            idx = 0
            for face_image in face_image_list:
                if cloth_type is False:
                    face_image = self.face_transform(face_image.convert("RGB"))
                else:
                    face_image = self.cloth_transform(face_image.convert("RGB"))
                # face_image.save(f"test/{tt}_face_image_{idx}.jpg")
                clip_image = self.clip_image_processor(images=face_image, return_tensors="pt").pixel_values[0]
                clip_image_list.append(clip_image)
                idx += 1

            rand_num = random.random()
            if rand_num < self.i_drop_rate:
                drop_image_embed = 1
                for ii in range(len(clip_image_list)):
                    clip_image_list[ii] = torch.zeros_like(clip_image_list[ii])

            elif rand_num < (self.i_drop_rate + self.t_drop_rate):
                prompt = ""
            elif rand_num < (self.i_drop_rate + self.t_drop_rate + self.ti_drop_rate):
                prompt = ""
                drop_image_embed = 1
                for ii in range(len(clip_image_list)):
                    clip_image_list[ii] = torch.zeros_like(clip_image_list[ii])
                
            for ii in range(len(ip_atten_mask_list)):
                ip_atten_mask_list[ii] = torch.from_numpy(ip_atten_mask_list[ii])


            clip_image_list = torch.stack(clip_image_list, axis=0)
            ip_atten_mask_list = torch.stack(ip_atten_mask_list, axis=0)

            # 
            ip_atten_mask_list[ip_atten_mask_list <=0] = 0
            ip_atten_mask_list[ip_atten_mask_list >0] = 1
            hint_mask[hint_mask <=0] = 0
            hint_mask[hint_mask >0] = 1

            face_image_list_copy = []
            # keypoints = []
            # 衣服计算计算滑动窗口
            if cloth_type:
                for face_image in face_image_list:
                    width, height = face_image.size
                    if width > height:
                        block = math.ceil(width / height)
                        overlap = block * height - width
                        if overlap > 0:
                            overlap = overlap / (block - 1)
                        
                        block_size = height - overlap

                        for i in range(block):
                            box = (i * block_size, 0, i * block_size + height, height)
                            face_image_list_copy.append(face_image.crop(box))
                    else:
                        block = math.ceil(height / width)
                        overlap = block * width - height
                        if overlap > 0:
                            overlap = overlap / (block - 1)
                        for i in range(block):
                            face_image_list_copy.append(face_image.crop((0, i * (width - overlap), width, i * (width - overlap) + width)))

            if len(face_image_list_copy) > 0 and len(face_image_list_copy) < 4:
                for idxxx in range(len(face_image_list_copy)):
                    face_image_list_copy[idxxx] = self.cloth_transform(face_image_list_copy[idxxx].convert("RGB"))
                    face_image_list_copy[idxxx] = self.clip_image_processor(images=face_image_list_copy[idxxx], return_tensors="pt").pixel_values[0]
                clip_image_list = torch.cat((clip_image_list, torch.stack(face_image_list_copy)), dim=0)
            
            # face_embedding_list = torch.
            # if self.face_embedding:
             
                # face_embedding_list.append(face_embedding)
            # print(face_embedding)
            feature_type = torch.zeros(1)
            if cloth_type == False:
                feature_type = torch.ones(1)

            return image, prompt, hint_mask_image, hint_mask, hint_skeleton, clip_image_list, ip_atten_mask_list, face_box_list, face_embedding, feature_type, facerecog_image

        except Exception as e:
            print(e)
            self.error_idx.append(idx)
            return self.__getitem__(random.randint(0, len(self.data) - 1))

def loader(train_batch_size, num_workers, **args):
    # print(args)
    dataset = IpAdaptorImageDataset(args)
    return DataLoader(dataset, batch_size=train_batch_size, num_workers=1, shuffle=True)
