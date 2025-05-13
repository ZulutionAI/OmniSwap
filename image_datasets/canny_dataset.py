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
        image = draw_bodypose(image, keypoints, min_conf=0)
    return image


def draw_handpose(canvas: np.ndarray, keypoints_righthand, keypoints_lefthand, min_conf: float=0, color=None) -> np.ndarray:
    H, W, C = canvas.shape
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

    stickwidth = 1

    limbSeq = [
        [0, 4], [1, 2], [2, 3], [3, 4],  # 拇指的连接
        [0, 8], [5, 6], [6, 7], [7, 8],  # 食指的连接
        [0, 12], [9, 10], [10, 11], [11, 12],  # 中指的连接
        [0, 16], [13, 14], [14, 15], [15, 16],  # 无名指的连接
        [0, 20], [17, 18], [18, 19], [19, 20]   # 小指的连接
    ]

    colors = [[0, 255, 255],[0, 255, 255],[0, 255, 255],[0, 255, 255],
              [0, 170, 255],[0, 170, 255],[0, 170, 255],[0, 170, 255],
              [0, 85, 255],[0, 85, 255],[0, 85, 255],[0, 85, 255],
              [85, 0, 255],[85, 0, 255],[85, 0, 255],[85, 0, 255],
              [170, 0, 255],[170, 0, 255],[170, 0, 255],[170, 0, 255]]

    for keypoints in [keypoints_righthand, keypoints_lefthand]:
        for (k1, k2), color in zip(limbSeq, colors):
            cur_canvas = canvas.copy()
            keypoint1 = keypoints[k1, :]
            keypoint2 = keypoints[k2, :]

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
        
        for keypoint, color in zip(keypoints, colors):
            if keypoint[-1]<min_conf:
                continue

            x, y = keypoint[0], keypoint[1]
            cv2.circle(canvas, (int(x), int(y)), int(1 * ratio), color, thickness=-1)

    return canvas

def draw_bodypose(canvas: np.ndarray, keypoints, min_conf: float=0, color=None) -> np.ndarray:
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

    stickwidth = 2
    # connections and colors
    limbSeq = [
        [2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10],
        [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17],
        [1, 16], [16, 18]]

    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0],[85, 255, 0], 
            [0, 255, 0],[0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255],[0, 85, 255], 
            [0, 0, 255], [85, 0, 255], [170, 0,255], [255, 0, 255],[255, 0, 170],[255, 0, 170]]


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
        
    for keypoint, color in zip(keypoints, colors):
        if keypoint[-1]<min_conf:
            continue

        x, y = keypoint[0], keypoint[1]
        cv2.circle(canvas, (int(x), int(y)), int(4 * ratio), color, thickness=-1)

    return canvas

def get_skeleton_keypoints_from_facebook308(keypoints, keypoint_scores):
    num = 0
    for i in range(len(keypoints)):
        if keypoint_scores[i] < 0.5:
            num += 1
        keypoints[i].append(keypoint_scores[i])
    if num > 154:
        return None
    keypoints_new = [keypoints[0], keypoints[1], keypoints[2], keypoints[3], keypoints[4], keypoints[5], keypoints[6], keypoints[7], keypoints[8], keypoints[62], keypoints[41], keypoints[9], keypoints[10], keypoints[11], keypoints[12], keypoints[13], keypoints[14]]
    keypoints_righthand = [
        keypoints[41],  # right_wrist
        keypoints[21],  # right_thumb4
        keypoints[22],  # right_thumb3
        keypoints[23],  # right_thumb2
        keypoints[24],  # right_thumb_third_joint
        keypoints[25],  # right_forefinger4
        keypoints[26],  # right_forefinger3
        keypoints[27],  # right_forefinger2
        keypoints[28],  # right_forefinger_third_joint
        keypoints[29],  # right_middle_finger4
        keypoints[30],  # right_middle_finger3
        keypoints[31],  # right_middle_finger2
        keypoints[32],  # right_middle_finger_third_joint
        keypoints[33],  # right_ring_finger4
        keypoints[34],  # right_ring_finger3
        keypoints[35],  # right_ring_finger2
        keypoints[36],  # right_ring_finger_third_joint
        keypoints[37],  # right_pinky_finger4
        keypoints[38],  # right_pinky_finger3
        keypoints[39],  # right_pinky_finger2
        keypoints[40]   # right_pinky_finger_third_joint
    ] # total 21
    keypoints_lefthand = [
        keypoints[62],  # left_wrist
        keypoints[42],  # left_thumb4
        keypoints[43],  # left_thumb3
        keypoints[44],  # left_thumb2
        keypoints[45],  # left_thumb_third_joint
        keypoints[46],  # left_forefinger4
        keypoints[47],  # left_forefinger3
        keypoints[48],  # left_forefinger2
        keypoints[49],  # left_forefinger_third_joint
        keypoints[50],  # left_middle_finger4
        keypoints[51],  # left_middle_finger3
        keypoints[52],  # left_middle_finger2
        keypoints[53],  # left_middle_finger_third_joint
        keypoints[54],  # left_ring_finger4
        keypoints[55],  # left_ring_finger3
        keypoints[56],  # left_ring_finger2
        keypoints[57],  # left_ring_finger_third_joint
        keypoints[58],  # left_pinky_finger4
        keypoints[59],  # left_pinky_finger3
        keypoints[60],  # left_pinky_finger2
        keypoints[61]   # left_pinky_finger_third_joint
    ] # total 21
    return keypoints_new, keypoints_righthand, keypoints_lefthand

def combine_canny_skeleton(img_path, skeleton_path, seg_path=None, is_facebook=True, add_canny=True, draw_hands_flag=True):
    try:
        img = cv2.imread(img_path)
        sharpen_kernel = np.array([[-1, -1, -1],
                           [-1,  9, -1],
                           [-1, -1, -1]])
        sharpened_image = cv2.filter2D(img, -1, sharpen_kernel)
        canny_image = canny_processor(sharpened_image)
        
        keypoints = []
        keypoints_righthand = []
        keypoints_lefthand = []
        model_type = -1
        with open(skeleton_path, "r") as f:
            skeleton_data = json.load(f)
            if not is_facebook:
                for item in skeleton_data:
                    if item['skeleton'] is None:
                        continue
                    combined_keypoints = []
                    for kp, score in zip(item['skeleton']['keypoints'], item['skeleton']['keypoint_scores']):
                        combined_keypoints.append(kp + [score])
                    keypoints.append(np.array(combined_keypoints))
            else:
                for item in skeleton_data['instance_info']:
                    if len(item['keypoints']) == 308:
                        model_type = 308
                        temp = get_skeleton_keypoints_from_facebook308(item['keypoints'], item['keypoint_scores'])
                        if temp is not None:
                            keypoint_body, keypoint_righthand, keypoint_lefthand = temp
                            keypoints.append(np.array(keypoint_body))
                            keypoints_righthand.append(np.array(keypoint_righthand))
                            keypoints_lefthand.append(np.array(keypoint_lefthand))
                    elif len(item['keypoints']) == 17:
                        model_type = 17
                        combined_keypoints = []
                        for kp, score in zip(item['keypoints'], item['keypoint_scores']):
                            combined_keypoints.append(kp + [score])
                        keypoints.append(np.array(combined_keypoints))
        keypoints = [keypoint for keypoint in keypoints if keypoint is not None]
        # canvas = np.zeros([1344, 768, 3])
        h, w = img.shape[:2]
        canvas = np.zeros([h, w, 3])
        for keypoint in keypoints:
            keypoint = convert_open_to_mmpose(keypoint)
            canvas = draw_bodypose(canvas, keypoint, min_conf=0.3)
        
        if draw_hands_flag == True:
            if is_facebook and model_type == 308:
                for keypoint_righthand, keypoint_lefthand in zip(keypoints_righthand, keypoints_lefthand):
                    canvas = draw_handpose(canvas, keypoint_righthand, keypoint_lefthand, min_conf=0.3)

        if add_canny and seg_path is not None:
            if seg_path.endswith('.npy'):
                seg_data = np.load(seg_path)
            else:
                seg_data = scipy.sparse.load_npz(seg_path).toarray()

            hand_labels = [5,14]

            for label in hand_labels:
                hand_mask = seg_data == label
                num_labels, labels_im = cv2.connectedComponents(hand_mask.astype(np.uint8))

                for label in range(1, num_labels):
                    hand_component_mask = labels_im == label

                    canny_array = np.array(canny_image)
                    cropped_hand = canny_array * hand_component_mask[:,:,None]
                    
                    # 只将白色线条粘贴到canvas上
                    white_mask = cropped_hand > 0
                    canvas[white_mask] = 255

        return canvas
    except Exception as e:
        print('error:', e)
        return img_path, e

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


class CustomImageDatasetV3(Dataset):
    def __init__(self, img_dir, img_size=512):
        # self.images = [os.path.join(img_dir, i) for i in os.listdir(img_dir) if '.jpg' in i or '.png' in i]
        # path = "/mnt/wangxuekuan/code/layout/dataset_split/skeleton_trainv1.txt"

        # f = open(path, "r")
        # all_path = f.readlines()
        
        plist = [
            # "/mnt/wangxuekuan/code/layout/dataset_split/skeleton_trainv8_vcg_0.txt_new.txt",
            # "/mnt/wangxuekuan/code/layout/dataset_split/skeleton_trainv8_vcg_1.txt_new.txt",
            # # "/mnt/wangxuekuan/code/layout/dataset_split/skeleton_trainv8_vcg_2.txt_new.txt",
            # # "/mnt/wangxuekuan/code/layout/dataset_split/skeleton_trainv8_vcg_3.txt_new.txt",
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

            hint = canny_processor(img)
            # hint.save("canny.jpg")
            hint = np.array(hint)
            h_size = hint.shape
            # print(h_size)
            # random pad mask block
            mask_type = random.choice([0, 0, 1, 1, 2, 3]) # top, bottom, left, right
            if random.choice([0, 1, 1]) == 0:
                if mask_type == 0:
                    m = random.randint(0, h_size[0] // 2)
                    hint[:m] = 0

                elif mask_type == 1:
                    m = random.randint(0, h_size[0] // 2)
                    hint[h_size[0]-m:] = 0

                elif mask_type == 2:
                    m = random.randint(0, h_size[1] // 2)
                    hint[:, :m] = 0

                elif mask_type == 3:
                    m = random.randint(0, h_size[1] // 2)
                    hint[:, h_size[1]-m:] = 0

            # cv2.imwrite("canny.jpg", hint)
            img = torch.from_numpy((np.array(img) / 127.5) - 1)
            img = img.permute(2, 0, 1)[:3, ...]
            hint = torch.from_numpy((hint / 127.5) - 1)
            hint = hint.permute(2, 0, 1)[:3, ...]


            return img, hint, prompt

        except Exception as e:
            print(e)
            return self.__getitem__(random.randint(0, len(self.json_path_list) - 1))


class CustomImageDataset_base(Dataset):
    def __init__(self, img_dir, img_size=512):
        # self.images = [os.path.join(img_dir, i) for i in os.listdir(img_dir) if '.jpg' in i or '.png' in i]
        # path = "/mnt/wangxuekuan/code/layout/dataset_split/skeleton_trainv1.txt"

        # f = open(path, "r")
        # all_path = f.readlines()
        
        plist = [
            # "/mnt/wangxuekuan/code/layout/dataset_split/skeleton_trainv8_vcg_0.txt_new.txt",
            # "/mnt/wangxuekuan/code/layout/dataset_split/skeleton_trainv8_vcg_1.txt_new.txt",
            # # "/mnt/wangxuekuan/code/layout/dataset_split/skeleton_trainv8_vcg_2.txt_new.txt",
            # # "/mnt/wangxuekuan/code/layout/dataset_split/skeleton_trainv8_vcg_3.txt_new.txt",
            # "/mnt/wangxuekuan/code/layout/dataset_split/skeleton_trainv8_reelshort.txt_new.txt",
            # "/mnt/wangxuekuan/code/layout/dataset_split/skeleton_trainv8_pinterest.txt_new.txt",
            "/mnt2/wangxuekuan/code/x-flux.bk/data/controlnet/train_canny_v0.txt",
            # "/mnt/wangxuekuan/code/layout/dataset_split/skeleton_trainv8_flux_0.txt_new.txt"
        ]
        all_path = []
        for p in plist:
            f = open(p, "r")
            all_path += f.readlines()
        
        #  
        # self.json_path_list = list(map(lambda x: x[3:].strip(), all_path))
        self.image_path_list = all_path
        # random.shuffle(self.json_path_list)
        print(len(self.image_path_list))

        # self.image_path_list.sort()

        self.img_size = (768, 1344)

    def __len__(self):
        return len(self.image_path_list)

    def __getitem__(self, idx):
        try:
            print(len(self.image_path_list))
            image_path = self.image_path_list[idx].strip()
            caption_path = "/mnt2/wangxuekuan/data/flux_controlnet_canny_v0/" + image_path[:-3] + "txt"
            # image_path = self.json_path_list[idx].split(", ")[2].strip()
            
            img = Image.open(image_path)
            # image = np.asarray(image, dtype=np.float32) # 转化为array
            # image = image[:, :, :3] 
            
            img_size = img.size # width, height
            # pad image
            img = c_crop_1344x768(img)
            img = img.resize(self.img_size)
            # hint = canny_processor(img)
            # prompt = json.load(open(json_path))
            prompt = open(caption_path, "r").readlines()[0].strip()
            '''
            # /mnt2/zhenghaoyu/prompt_data/flux_img_78_all/img/
            # /mnt2/zhenghaoyu/prompt_data/flux_img_1character_78_all_scene/img/
            # /mnt2/zhenghaoyu/prompt_data/flux_anime/img/
            '''
            if "flux_img_78_all" in image_path:
                canny_path = image_path.replace("/mnt2/zhenghaoyu/prompt_data/flux_img_78_all/img/", "/mnt2/zhenghaoyu/prompt_data/flux_img_78_all/canny_skeleton/")
                hint = Image.open(canny_path)
            elif "flux_img_1character_78_all_scene" in image_path:
                canny_path = image_path.replace("/mnt2/zhenghaoyu/prompt_data/flux_img_1character_78_all_scene/img/", "/mnt2/zhenghaoyu/prompt_data/flux_img_1character_78_all_scene/canny_skeleton/")
                hint = Image.open(canny_path)
            elif "flux_anime" in image_path:
                canny_path = image_path.replace("/mnt2/zhenghaoyu/prompt_data/flux_anime/img/", "/mnt2/zhenghaoyu/prompt_data/flux_anime/canny_skeleton_sapiens/")
                hint = Image.open(canny_path)
            else:
                hint = canny_processor(img)

            '''
            /mnt2/zhenghaoyu/prompt_data/flux_img_78_all/canny_skeleton
            /mnt2/zhenghaoyu/prompt_data/flux_img_1character_78_all_scene/canny_skeleton
            /mnt2/zhenghaoyu/prompt_data/flux_anime/canny_skeleton_sapiens
            '''

            # hint.save("canny.jpg")
            hint = np.array(hint)
            h_size = hint.shape
            # print(h_size)
            # random pad mask block
            mask_type = random.choice([0, 0, 1, 1, 2, 3]) # top, bottom, left, right
            if random.choice([0, 1, 1]) == 0:
                if mask_type == 0:
                    m = random.randint(0, h_size[0] // 2)
                    hint[:m] = 0

                elif mask_type == 1:
                    m = random.randint(0, h_size[0] // 2)
                    hint[h_size[0]-m:] = 0

                elif mask_type == 2:
                    m = random.randint(0, h_size[1] // 2)
                    hint[:, :m] = 0

                elif mask_type == 3:
                    m = random.randint(0, h_size[1] // 2)
                    hint[:, h_size[1]-m:] = 0


            # cv2.imwrite("canny.jpg", hint)
            img = torch.from_numpy((np.array(img) / 127.5) - 1)
            img = img.permute(2, 0, 1)[:3, ...]
            hint = torch.from_numpy((hint / 127.5) - 1)
            hint = hint.permute(2, 0, 1)[:3, ...]


            return img, hint, prompt

        except Exception as e:
            print(e)
            print("error, ", caption_path)
            return self.__getitem__(random.randint(0, len(self.image_path_list) - 1))



class CustomImageDataset(Dataset):
    def __init__(self, img_dir, img_size=512):

        plist = [
            "/mnt2/wangxuekuan/code/x-flux.bk/data/ip_adapter/train_ip_vcg_80w.txt",
            # "/mnt2/wangxuekuan/code/x-flux.bk/data/ip_adapter/train_ip_flux_0.txt",
            "/mnt2/wangxuekuan/code/x-flux.bk/data/ip_adapter/train_ip_flux_1.txt",
            # "/mnt2/wangxuekuan/code/x-flux.bk/data/ip_adapter/train_ip_vcg_122w.txt",
            # "/mnt2/wangxuekuan/code/x-flux.bk/data/ip_adapter/train_ip_vcg_chat_history_0_300_two_people.txt",
            # "/mnt2/wangxuekuan/code/x-flux.bk/data/ip_adapter/train_ip_vcg_reelshort_0_200.txt",
            # "/mnt2/wangxuekuan/code/x-flux.bk/data/ip_adapter/train_ip_flux_img_1character_78_all_noscene.txt",
            "/mnt2/wangxuekuan/code/x-flux.bk/data/ip_adapter/train_ip_flux_img_1character_78_all_scene.txt",
            # "/mnt2/wangxuekuan/code/x-flux.bk/data/ip_adapter/train_ip_coyo_two_people_hw_ratio_155_195.txt"
            "/mnt/wangxuekuan/code/layout/dataset_split/flux_anime_1character.txt",
            "/mnt/wangxuekuan/code/layout/dataset_split/flux_anime_2characters.txt",
        ]

        self.data = []
        for p in plist:
            self.data += open(p).readlines() 

        #glob.glob("/mnt2/huangyuqiu/crawler/anime_crawler/data_1016/*/*/*")
        
        self.image_path_list = self.data

        self.size = (448, 768)
        # self.size
        self.error_idx = []

    def __len__(self):
        return len(self.image_path_list)


    def __getitem__(self, idx):
        if idx in self.error_idx:
            return self.__getitem__(random.randint(0, len(self.data) - 1))
        # try:
        # if True:
        try:
            # item = self.data[idx] 
            #
            n, json_path, image_file = self.data[idx].strip().split(", ")

            caption_path = "/mnt2/wangxuekuan/data/ip_adapter_gpt4o__desc/" + image_file[:-3] + "txt"
            if not os.path.exists(caption_path):
                print("no caption path", caption_path)
                self.error_idx.append(idx)
                return self.__getitem__(random.randint(0, len(self.data) - 1))

            caption_new = open(caption_path, "r").readlines()[0].strip()
            prompt = caption_new

            if "vcg_images_80W" in json_path:
                mask_sapiens_path = json_path.replace("/mnt2/huangyuqiu/share/vcg_images_80W/json_final", "/mnt2/zhenghaoyu/share/vcg_images_80W/mask_sapiens")[:-5] + "_seg.npz"
                skeleton_sapiens_path = json_path.replace("/mnt2/huangyuqiu/share/vcg_images_80W/json_final", "/mnt2/zhenghaoyu/share/vcg_images_80W/skeleton_sapiens")
            elif "vcg_images_122W_1" in json_path:
                mask_sapiens_path = json_path.replace("/mnt2/wangxuekuan/data/MSDBv2_json/mnt/wangxuekuan/data/vcg_images_122W_1_v3_format_simple_skeleton", 
                                                      "/mnt2/huangyuqiu/share/vcg_images_122W_1//mask_sapiens")[:-5] + "_seg.npz"
                skeleton_sapiens_path = json_path.replace("/mnt2/wangxuekuan/data/MSDBv2_json/mnt/wangxuekuan/data/vcg_images_122W_1_v3_format_simple_skeleton", 
                                                      "/mnt2/huangyuqiu/share/vcg_images_122W_1//skeleton_sapiens")
            elif "/mnt2/huangyuqiu/share/flux_img_78_all/json_final" in json_path:
                mask_sapiens_path = json_path.replace("/mnt2/huangyuqiu/share/flux_img_78_all/json_final", 
                                                      "/mnt2/zhenghaoyu/prompt_data/flux_img_78_all/mask_sapiens")[:-5] + "_seg.npz"
                skeleton_sapiens_path = json_path.replace("/mnt2/huangyuqiu/share/flux_img_78_all/json_final", 
                                                      "/mnt2/zhenghaoyu/prompt_data/flux_img_78_all/skeleton_sapiens")
            elif "/mnt2/huangyuqiu/share/flux/json_final" in json_path:
                mask_sapiens_path = json_path.replace("/mnt2/huangyuqiu/share/flux/json_final", 
                                                      "/mnt2/huangyuqiu/share/flux/mask_sapiens")[:-5] + "_seg.npz"
                skeleton_sapiens_path = json_path.replace("/mnt2/huangyuqiu/share/flux/json_final", 
                                                      "/mnt2/huangyuqiu/share/flux/skeleton_sapiens")
            elif "flux_img_1character_78_all_noscene" in json_path:
                mask_sapiens_path = json_path.replace("/mnt2/huangyuqiu/share/flux_img_1character_78_all_noscene/json_final", 
                                                      "/mnt2/zhenghaoyu/prompt_data/flux_img_1character_78_all_noscene/mask_sapiens")[:-5] + "_seg.npz"
                skeleton_sapiens_path = json_path.replace("/mnt2/huangyuqiu/share/flux_img_1character_78_all_noscene/json_final", 
                                                      "/mnt2/zhenghaoyu/prompt_data/flux_img_1character_78_all_noscene/skeleton_sapiens")
            elif "flux_img_1character_78_all_scene" in json_path:
                mask_sapiens_path = json_path.replace("/mnt2/huangyuqiu/share/flux_img_1character_78_all_scene/json_final", 
                                                      "/mnt2/zhenghaoyu/prompt_data/flux_img_1character_78_all_scene/mask_sapiens")[:-5] + "_seg.npz"
                skeleton_sapiens_path = json_path.replace("/mnt2/huangyuqiu/share/flux_img_1character_78_all_scene/json_final", 
                                                      "/mnt2/zhenghaoyu/prompt_data/flux_img_1character_78_all_scene/skeleton_sapiens")
            elif "vcg_chat_history_0_300_two_people" in json_path:
                mask_sapiens_path = json_path.replace("/mnt2/huangyuqiu/share/vcg_chat_history_0_300_two_people/json_final", 
                                                      "/mnt2/huangyuqiu/share/vcg_chat_history_0_300_two_people/json_final/mask_sapiens")[:-5] + "_seg.npz"
                skeleton_sapiens_path = json_path.replace("/mnt2/huangyuqiu/share/vcg_chat_history_0_300_two_people/json_final", 
                                                      "/mnt2/huangyuqiu/share/vcg_chat_history_0_300_two_people/json_final/skeleton_sapiens")
            elif "vcg_chat_history_0_300_two_people" in json_path:
                mask_sapiens_path = json_path.replace("/mnt2/huangyuqiu/share/vcg_chat_history_0_300_two_people/json_final", 
                                                      "/mnt2/huangyuqiu/share/vcg_chat_history_0_300_two_people/mask_sapiens")[:-5] + "_seg.npz"
                skeleton_sapiens_path = json_path.replace("/mnt2/huangyuqiu/share/vcg_chat_history_0_300_two_people/json_final", 
                                                      "/mnt2/huangyuqiu/share/vcg_chat_history_0_300_two_people/skeleton_sapiens")
            elif "vcg_reelshort_0_200" in json_path:
                mask_sapiens_path = json_path.replace("/mnt2/huangyuqiu/share/vcg_reelshort_0_200/json_final", 
                                                      "/mnt2/huangyuqiu/share/vcg_reelshort_0_200/mask_sapiens")[:-5] + "_seg.npz"
                skeleton_sapiens_path = json_path.replace("/mnt2/huangyuqiu/share/vcg_reelshort_0_200/json_final", 
                                                      "/mnt2/huangyuqiu/share/vcg_reelshort_0_200/skeleton_sapiens")
            else:  
                print("error json path", json_path)
                return self.__getitem__(random.randint(0, len(self.data) - 1))

            if not os.path.exists(mask_sapiens_path):
                mask_sapiens_path = mask_sapiens_path[:-3] + "npy"
        
            if not os.path.exists(mask_sapiens_path) or not os.path.exists(skeleton_sapiens_path):
                print("no - data", mask_sapiens_path, skeleton_sapiens_path)
                self.error_idx.append(idx)
                return self.__getitem__(random.randint(0, len(self.data) - 1))


            # read json
            with open(json_path, 'r') as f:
                info_dict = json.load(f)

            prob = random.random()

            # read image         
            raw_image = Image.open(image_file)

            raw_image = c_crop_1344x768(raw_image)
            
            image = raw_image.convert("RGB").resize(self.size)
            
            draw_hands_flag = random.choices([True, True, True, False])
            add_canny = False
            if draw_hands_flag == False:
                add_canny = random.choices([True, False])
            
            
            hint = combine_canny_skeleton(img_path=image_file, \
                                          skeleton_path=skeleton_sapiens_path, 
                                          seg_path=mask_sapiens_path, 
                                          is_facebook=True, add_canny=add_canny, draw_hands_flag=draw_hands_flag).astype(np.uint8)

    
            hint = Image.fromarray(hint)
            hint = c_crop_1344x768(hint)
            
            hint.save("hint.jpg")

            hint = hint.convert("RGB").resize(self.size)
            hint = torch.from_numpy((np.array(hint) / 127.5) - 1)
            hint = hint.permute(2, 0, 1)
            image = np.array(image).transpose(2, 0, 1) / 127.5 - 1

            if random.random() < 0.05:
                prompt = ""
            elif random.random() < 0.1:
                hint = torch.zeros_like(hint)

            return image, hint, prompt

        except Exception as e:
            print(e)
            self.error_idx.append(idx)
            return self.__getitem__(random.randint(0, len(self.data) - 1))

def loader(train_batch_size, num_workers, **args):
    dataset = CustomImageDataset(**args)
    return DataLoader(dataset, batch_size=train_batch_size, num_workers=num_workers, shuffle=True)
