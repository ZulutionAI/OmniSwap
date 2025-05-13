import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import json
import random
import cv2

def canny_processor(image, low_threshold=100, high_threshold=200):
    image = np.array(image)
    image = cv2.Canny(image, low_threshold, high_threshold)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    canny_image = Image.fromarray(image)
    return canny_image

def draw_bodypose(canvas: np.ndarray, keypoints: np.ndarray, min_conf: float=0, color=None) -> np.ndarray:
        H, W, C = canvas.shape
        # automatically adjust the thickness of the skeletons
        if max(W, H) < 500:
            ratio = 1.0
        elif max(W, H) >= 500 and max(W, H) < 1000:
            ratio = 2.0
        elif max(W, H) >= 1000 and max(W, H) < 2000:
            ratio = 3.0
        elif max(W, H) >= 2000 and max(W, H) < 3000:
            ratio = 4.0
        elif max(W, H) >= 3000 and max(W, H) < 4000:
            ratio = 5.0
        elif max(W, H) >= 4000 and max(W, H) < 5000:
            ratio = 6.0
        else:
            ratio = 7.0

        stickwidth = 4
        # connections and colors
        limbSeq = [
            [2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10],
            [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17],
            [1, 16], [16, 18]]

        colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0],[85, 255, 0], 
                [0, 255, 0],[0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255],[0, 85, 255], 
                [0, 0, 255], [85, 0, 255], [170, 0,255], [255, 0, 255],[255, 0, 170], [255, 0, 85]]
        
        if color is not None:
            colors = [color] * len(colors)
            
        # draw the links
        for (k1, k2), color in zip(limbSeq, colors):
            cur_canvas = canvas.copy()
            keypoint1 = keypoints[k1-1, :]
            keypoint2 = keypoints[k2-1, :]

            if keypoint1[-1] < min_conf or keypoint2[-1] < min_conf:
                continue

            Y = np.array([keypoint1[0], keypoint2[0]])
            X = np.array([keypoint1[1], keypoint2[1]])
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            import math
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), int(stickwidth * ratio)), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(cur_canvas, polygon, [int(float(c)) for c in color])
            canvas = cv2.addWeighted(canvas, 0.3, cur_canvas, 0.7, 0)
            
        # draw the points
        for id, color in zip(range(keypoints.shape[0]), colors):
            keypoint = keypoints[id, :]
            if keypoint[-1]<min_conf:
                continue

            x, y = keypoint[0], keypoint[1]
            cv2.circle(canvas, (int(x), int(y)), int(4 * ratio), color, thickness=-1)

        return canvas

def convert_open_to_mmpose(keypoints: np.ndarray) -> np.ndarray:
        neck = (keypoints[5,:] + keypoints[6,:])/2
        keypoints = np.vstack((keypoints, neck))
        # transform mmpose keypoints to openpose keypoints
        openpose_idx = [15, 14, 17, 16, 2, 6, 3, 7, 4, 8, 12, 9, 13, 10, 1]
        mmpose_idx = [1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17]
        new_keypoints = keypoints[:, ...]
        new_keypoints[openpose_idx, ...] = keypoints[mmpose_idx, ...]
        # new_keypoints[:,2] = 1
        
        return new_keypoints

def draw_skeleton_on_black_bg(keypoints_list, size):
    
    image = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    # colors = [[255,0,0], [0,255,0]]
    for i in range(len(keypoints_list)):
        keypoints = keypoints_list[i]
        keypoints = convert_open_to_mmpose(keypoints)
        keypoints[:,0] = keypoints[:,0] #* size[0]
        keypoints[:,1] = keypoints[:,1] #* size[1]
        image = draw_bodypose(image, keypoints, min_conf=0.3)
    return image

def skeleton_processor(keypoints_list, img_size):

    skeleton_img = draw_skeleton_on_black_bg(keypoints_list, img_size)
    # cv2.imwrite("skeleton.jpg", skeleton_img)
    skeleton_img = Image.fromarray(skeleton_img)

    return skeleton_img
    
def c_crop(image):
    width, height = image.size
    new_size = min(width, height)
    left = (width - new_size) / 2
    top = (height - new_size) / 2
    right = (width + new_size) / 2
    bottom = (height + new_size) / 2
    return image.crop((left, top, right, bottom))

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
            hint = canny_processor(img)

            img = torch.from_numpy((np.array(img) / 127.5) - 1)
            img = img.permute(2, 0, 1)
            hint = torch.from_numpy((np.array(hint) / 127.5) - 1)
            hint = hint.permute(2, 0, 1)
            json_path = self.images[idx].split('.')[0] + '.json'
            prompt = json.load(open(json_path))['caption']
            return img, hint, prompt
        except Exception as e:
            print(e)
            return self.__getitem__(random.randint(0, len(self.images) - 1))


class CustomImageDatasetv0(Dataset):
    def __init__(self, img_dir, img_size=512):
        # self.images = [os.path.join(img_dir, i) for i in os.listdir(img_dir) if '.jpg' in i or '.png' in i]
        # path = "/mnt/wangxuekuan/code/layout/dataset_split/skeleton_trainv1.txt"

        # f = open(path, "r")
        # all_path = f.readlines()
        
        plist = [
            # "/mnt/wangxuekuan/code/layout/dataset_split/skeleton_trainv8_vcg_0.txt_new.txt",
            # "/mnt/wangxuekuan/code/layout/dataset_split/skeleton_trainv8_vcg_1.txt_new.txt",
            # "/mnt/wangxuekuan/code/layout/dataset_split/skeleton_trainv8_vcg_2.txt_new.txt",
            # "/mnt/wangxuekuan/code/layout/dataset_split/skeleton_trainv8_vcg_3.txt_new.txt",
            # "/mnt/wangxuekuan/code/layout/dataset_split/skeleton_trainv8_reelshort.txt_new.txt",
            # "/mnt/wangxuekuan/code/layout/dataset_split/skeleton_trainv8_pinterest.txt_new.txt",
            "/mnt/wangxuekuan/code/layout/dataset_split/skeleton_trainv8_flux_1.txt_new.txt",
            # "/mnt/wangxuekuan/code/layout/dataset_split/skeleton_trainv8_flux_0.txt_new.txt"
        ]
        all_path = []
        for p in plist:
            f = open(p, "r")
            all_path += f.readlines()
        
        #  
        # self.json_path_list = list(map(lambda x: x[3:].strip(), all_path))
        self.json_path_list = all_path
        # random.shuffle(self.json_path_list)

        self.json_path_list[:10000].sort()

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
            
            if "caption_detail" in prompt.keys():
                caption_type = "caption_detail"
            else:
                caption_type = random.choice(caption_list)
                
            prompt_caption = prompt[caption_type].lower()

            if caption_type in ["caption_action", "caption_character_desc"]:
                if random.choice([0, 1]) == 0 and "caption_env" in prompt.keys():
                    prompt_caption += "," + ", ".join(prompt["caption_env"])
                    
            # single caption
            prompt_single_caption = ""

            # try:
            kpt_list = []
            # if "caption" in prompt["segments"][0].keys():
            # if random.choice([0, 1]) == 0:
            if True:
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
                    
                    kpt_p = np.array(item["skeleton"]["keypoints"]).reshape(17, -1)
                    kpt_s = np.array(item["skeleton"]["keypoint_scores"]).reshape(17, -1)
                    kpt = np.concatenate([kpt_p, kpt_s], axis = -1)
                    kpt_list.append(kpt)
            # except:
            #     prompt_single_caption = ""
                
            # print(prompt)
            num_character = len(prompt["segments"])      
            prompt = f"{num_character}characters," + prompt_caption + prompt_single_caption
   
            hint = skeleton_processor(kpt_list, img_size)
            
            hint = c_crop_1344x768(hint)
            hint = hint.resize(self.img_size)
            
            img = torch.from_numpy((np.array(img) / 127.5) - 1)
            img = img.permute(2, 0, 1)[:3, ...]
            hint = torch.from_numpy((np.array(hint) / 127.5) - 1)
            hint = hint.permute(2, 0, 1)[:3, ...]

            return img, hint, prompt

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
            # "/mnt/wangxuekuan/code/layout/dataset_split/skeleton_trainv8_vcg_0.txt_new.txt",
            # "/mnt/wangxuekuan/code/layout/dataset_split/skeleton_trainv8_vcg_1.txt_new.txt",
            # "/mnt/wangxuekuan/code/layout/dataset_split/skeleton_trainv8_vcg_2.txt_new.txt",
            # "/mnt/wangxuekuan/code/layout/dataset_split/skeleton_trainv8_vcg_3.txt_new.txt",
            # "/mnt/wangxuekuan/code/layout/dataset_split/skeleton_trainv8_reelshort.txt_new.txt",
            # "/mnt/wangxuekuan/code/layout/dataset_split/skeleton_trainv8_pinterest.txt_new.txt",
            # "/mnt/wangxuekuan/code/layout/dataset_split/skeleton_trainv8_flux_1.txt_new.txt",
            "/mnt2/wangxuekuan/code/x-flux.bk/data/controlnet/train_skeleton_v0.txt"
            # "/mnt/wangxuekuan/code/layout/dataset_split/skeleton_trainv8_flux_0.txt_new.txt"
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
            json_path = self.json_path_list[idx].split(", ")[0].strip()
            image_path = self.json_path_list[idx].split(", ")[1].strip()
            skeleton_path = self.json_path_list[idx].split(", ")[2].strip()
            
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
            
            if "caption_detail" in prompt.keys():
                caption_type = "caption_detail"
            else:
                caption_type = random.choice(caption_list)
                
            prompt_caption = prompt[caption_type].lower()

            if caption_type in ["caption_action", "caption_character_desc"]:
                if random.choice([0, 1]) == 0 and "caption_env" in prompt.keys():
                    prompt_caption += "," + ", ".join(prompt["caption_env"])
                    
            # single caption
            prompt_single_caption = ""

            # try:
            kpt_list = []
            # if "caption" in prompt["segments"][0].keys():
            # if random.choice([0, 1]) == 0:
            if True:
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
                    
                    kpt_p = np.array(item["skeleton"]["keypoints"]).reshape(17, -1)
                    kpt_s = np.array(item["skeleton"]["keypoint_scores"]).reshape(17, -1)
                    kpt = np.concatenate([kpt_p, kpt_s], axis = -1)
                    kpt_list.append(kpt)
            # except:
            #     prompt_single_caption = ""
                
            # print(prompt)
            num_character = len(prompt["segments"])      
            prompt = f"{num_character}characters," + prompt_caption + prompt_single_caption
   
            # hint = skeleton_processor(kpt_list, img_size)
            hint = Image.open(skeleton_path)
            
            hint = c_crop_1344x768(hint)
            hint = hint.resize(self.img_size)
            
            img = torch.from_numpy((np.array(img) / 127.5) - 1)
            img = img.permute(2, 0, 1)[:3, ...]
            hint = torch.from_numpy((np.array(hint) / 127.5) - 1)
            hint = hint.permute(2, 0, 1)[:3, ...]

            return img, hint, prompt

        except Exception as e:
            print(e)
            return self.__getitem__(random.randint(0, len(self.json_path_list) - 1))


def loader(train_batch_size, num_workers, **args):
    dataset = CustomImageDataset(**args)
    return DataLoader(dataset, batch_size=train_batch_size, num_workers=num_workers, shuffle=True)
