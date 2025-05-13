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

def convert_open_to_mmpose(keypoints: np.ndarray):
    neck = (keypoints[5,:] + keypoints[6,:])/2
    keypoints = np.vstack((keypoints, neck))
    openpose_idx = [15, 14, 17, 16, 2, 6, 3, 7, 4, 8, 12, 9, 13, 10, 1]
    mmpose_idx = [1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17]
    new_keypoints = keypoints[:, ...]
    new_keypoints[openpose_idx, ...] = keypoints[mmpose_idx, ...]
    # new_keypoints[:,2] = 1
    
    return new_keypoints

def canny_processor(image, low_threshold=100, high_threshold=200):
    image = np.array(image)
    image = cv2.Canny(image, low_threshold, high_threshold)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    canny_image = Image.fromarray(image)
    return canny_image

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

def combine_canny_skeleton(img_path, skeleton_path, seg_path=None, is_facebook=True, add_canny=True):
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
        self.genal_caption_flag = args.get("genal_caption_flag", False)

        self.face_embedding = args.get("face_embedding", False)
        self.atten_mask_type = args.get("atten_mask_type", "face")

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

        plist = [
            "/mnt2/wangxuekuan/code/x-flux.bk/data/ip_adapter/train_ip_vcg_80w.txt",
            # "/mnt2/wangxuekuan/code/x-flux.bk/data/ip_adapter/train_ip_flux_0.txt",
            "/mnt2/wangxuekuan/code/x-flux.bk/data/ip_adapter/train_ip_flux_1.txt",
            "/mnt2/wangxuekuan/code/x-flux.bk/data/ip_adapter/train_ip_vcg_122w.txt",
            "/mnt2/wangxuekuan/code/x-flux.bk/data/ip_adapter/train_ip_vcg_chat_history_0_300_two_people.txt",
            "/mnt2/wangxuekuan/code/x-flux.bk/data/ip_adapter/train_ip_vcg_reelshort_0_200.txt",
            "/mnt2/wangxuekuan/code/x-flux.bk/data/ip_adapter/train_ip_flux_img_1character_78_all_noscene.txt",
            # "/mnt2/wangxuekuan/code/x-flux.bk/data/ip_adapter/train_ip_flux_img_1character_78_all_scene.txt",
            # "/mnt2/wangxuekuan/code/x-flux.bk/data/ip_adapter/train_ip_coyo_two_people_hw_ratio_155_195.txt"
        ]

        self.data = []
        for p in plist:
            self.data += open(p).readlines() 
        
        if data_num != -1:
            #self.data = json.load(open(json_file))[:data_num]  # list of dict: [{"image_file": "1.png", "text": "A dog"}]
            self.data = self.data[:data_num] * 100
        # else:
        #     # self.data = json.load(open(json_file))
        #     self.data = open(txt_file).readlines()


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
    
    def get_sapiens_path(self, json_path):
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
            return None, None#self.__getitem__(random.randint(0, len(self.data) - 1))

        if not os.path.exists(mask_sapiens_path):
            mask_sapiens_path = mask_sapiens_path[:-3] + "npy"
    
        if not os.path.exists(mask_sapiens_path) or not os.path.exists(skeleton_sapiens_path):
            print("no - data", mask_sapiens_path, skeleton_sapiens_path)
            # self.error_idx.append(idx)
            return None, None
        
        return mask_sapiens_path, skeleton_sapiens_path

    def __getitem__(self, idx):
        if idx in self.error_idx:
            return self.__getitem__(random.randint(0, len(self.data) - 1))
        # try:
        # if True:
        try:
            # item = self.data[idx] 
            #
            n, json_path, image_file = self.data[idx].strip().split(", ")

            if self.genal_caption_flag:
                caption_path = "/mnt2/wangxuekuan/data/ip_adapter_gpt4o__desc/" + image_file[:-3] + "txt"
                if not os.path.exists(caption_path):
                    print("no caption path", caption_path)
                    self.error_idx.append(idx)
                    return self.__getitem__(random.randint(0, len(self.data) - 1))

                caption_new = open(caption_path, "r").readlines()[0].strip()

            
            mask_sapiens_path, skeleton_sapiens_path = self.get_sapiens_path(json_path)
            
            if mask_sapiens_path is None or skeleton_sapiens_path is None:
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

            if "caption_detail" in info_dict.keys():
                prompt = info_dict["caption_detail"]
            else:
                prompt = info_dict["caption_action"]

            if self.genal_caption_flag:
                prompt = caption_new
            
            # read image         
            raw_image = Image.open(image_file)

            # 
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

            
            cropped_image = raw_image.crop((left, top, right, bottom))

            if self.center_crop is False:
                cropped_image = c_crop_1344x768(cropped_image)
            
            # cropped_image.save("cropped_image.jpg")

            # image = self.transform(cropped_image.convert("RGB"))
        
            image = cropped_image.convert("RGB").resize(self.size)
            
            # hint = canny_processor(image)
            # hint = torch.from_numpy((np.array(hint) / 127.5) - 1)
            # hint = hint.permute(2, 0, 1)

            hint = combine_canny_skeleton(img_path=image_file, \
                                          skeleton_path=skeleton_sapiens_path, 
                                          seg_path=mask_sapiens_path, 
                                          is_facebook=True, add_canny=True).astype(np.uint8)

            # canvas = combine_canny_skeleton(img_path, skeleton_path, seg_path=mask_path, is_facebook=True, add_canny=True)

            if self.center_crop is False:
                hint = Image.fromarray(hint)
                hint = c_crop_1344x768(hint)
            
            hint = hint.convert("RGB").resize(self.size)
            
            # hint.save("hint_skeleton_canny.jpg")
            hint = torch.from_numpy((np.array(hint) / 127.5) - 1)
            hint = hint.permute(2, 0, 1)
        
            ### == > list 
            # 读取bbox信息
            face_image_list = []
            ip_atten_mask_list = []

            if self.box_type == "face" and "face_bbox" in list(info_dict['segments'][0].keys()):
                # print("face")
                for seg in info_dict['segments']:
                    bbox = np.array(seg["face_bbox"])
                    
                    atten_box = np.array(seg["bbox"])

                    if self.face_pad:
                        w = bbox[2] - bbox[0]
                        h = bbox[3] - bbox[1]
                        if w > h:
                            bbox[1] -= (w - h) // 2
                            bbox[3] += (w - h) // 2
                        else:
                            bbox[0] -= (h - w) // 2
                            bbox[2] += (h - w) // 2
                    
                    bbox[bbox<0] = 0
                    atten_box[atten_box<0] = 0

                    expansion_factor = random.choice([1.0, 1.0, random.uniform(1.0, 1.8)])
                    bbox = expand_bbox(raw_image, bbox, expansion_factor)
                    atten_box = expand_bbox(raw_image, atten_box, expansion_factor)
                     
                    # crop image
                    face_image = raw_image.crop(bbox)
                    if face_image.size[0] > 64 and face_image.size[1] > 64:
                        face_image_list.append(face_image)
                        # mask
                        size = raw_image.size
                        ip_atten_mask = np.zeros((size[1], size[0]))

                        if self.atten_mask_type == "face":
                            atten_box = bbox
                         
                         
                        ip_atten_mask[atten_box[1]:atten_box[3], atten_box[0]:atten_box[2]] = 1

                        if self.atten_mask_type == "all":
                            ip_atten_mask = 1

                        ip_atten_mask = cv2.resize(ip_atten_mask, (self.size[0]//16, self.size[1]//16))
                        ip_atten_mask_list.append(ip_atten_mask)

            elif self.box_type == "all":
                bbox = np.array([left, top, right, bottom])
            else:
                # print("body")
                for seg in info_dict['segments']:
                    bbox = np.array(seg["bbox"])
                    bbox[bbox<0] = 0

                    expansion_factor = random.choice([1.0, 1.0, random.uniform(1.0, 1.8)])
                    bbox = expand_bbox(raw_image, bbox, expansion_factor)
                     
                    # crop image
                    face_image = raw_image.crop(bbox)
                    face_image_list.append(face_image)

                    # mask
                    size = raw_image.size
                    ip_atten_mask = np.zeros((size[1], size[0]))
                    # print(ip_atten_mask.shape, bbox)
                    ip_atten_mask[bbox[1]:bbox[3], bbox[0]:bbox[2]] = 1
                    ip_atten_mask = cv2.resize(ip_atten_mask, (self.size[0]//16, self.size[1]//16))
                    ip_atten_mask_list.append(ip_atten_mask)
            
             
            #print("Can't detect face, use whole body img!")
            # face_image.save("face_image.jpg")
            clip_image_list = []
            if len(face_image_list) > 0:
                for face_image in face_image_list:
                    face_image = self.face_transform(face_image.convert("RGB"))
                    clip_image = self.clip_image_processor(images=face_image, return_tensors="pt").pixel_values[0]
                    clip_image_list.append(clip_image)

                    # ip_atten_mask_list[]
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

                # prompt = ""
                image = np.array(image).transpose(2, 0, 1) / 127.5 - 1
                # faces_embedding = torch.from_numpy(np.zeros(512))
                # ip_atten_mask = torch.from_numpy(ip_atten_mask)
                # print(hint.shape)
                clip_image_list = torch.stack(clip_image_list, axis=0)
                ip_atten_mask_list = torch.stack(ip_atten_mask_list, axis=0)
                # print(ip_atten_mask_list.sum())
                return image, prompt, clip_image_list, hint, ip_atten_mask_list #, drop_image_embed

            else:
                print("small - data")
                self.error_idx.append(idx)
                return self.__getitem__(random.randint(0, len(self.data) - 1))

            # if face_image.size[0] > 64 and face_image.size[1] > 64:
            #     face_image = self.face_transform(face_image.convert("RGB"))
            #     clip_image = self.clip_image_processor(images=face_image, return_tensors="pt").pixel_values[0]
            #     drop_image_embed = 0
            #     rand_num = random.random()
            #     if rand_num < self.i_drop_rate:
            #         drop_image_embed = 1
            #         clip_image = torch.zeros_like(clip_image)
            #     elif rand_num < (self.i_drop_rate + self.t_drop_rate):
            #         prompt = ""
            #     elif rand_num < (self.i_drop_rate + self.t_drop_rate + self.ti_drop_rate):
            #         prompt = ""
            #         drop_image_embed = 1
            #         clip_image = torch.zeros_like(clip_image)
                
            #     # prompt = ""
            #     image = np.array(image).transpose(2, 0, 1) / 127.5 - 1
            #     faces_embedding = torch.from_numpy(np.zeros(512))
            #     ip_atten_mask = torch.from_numpy(ip_atten_mask)
            #     return image, prompt, clip_image, faces_embedding, hint, ip_atten_mask #, drop_image_embed
            # else:
            #     print("small - data")
            #     self.error_idx.append(idx)
            #     return self.__getitem__(random.randint(0, len(self.data) - 1))

        except Exception as e:
            print(e)
            self.error_idx.append(idx)
            return self.__getitem__(random.randint(0, len(self.data) - 1))


def loader(train_batch_size, num_workers, **args):
    # print(args)
    dataset = IpAdaptorImageDataset(args)
    return DataLoader(dataset, batch_size=train_batch_size, num_workers=num_workers, shuffle=True)
