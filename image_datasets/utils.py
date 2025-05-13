import json
import os
import random

import numpy as np
from PIL import Image, ImageOps
import torchvision.transforms.functional as F
from torchvision import transforms
import cv2
import os
from PIL import Image
import cv2
import numpy as np
import json
import scipy

def check_bbox_overlap(body_bbox):
    if len(body_bbox) <= 1:
        return True
    
    def calculate_overlap(box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 < x1 or y2 < y1:
            return 0
        
        overlap_area = (x2 - x1) * (y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        overlap_ratio = overlap_area / min(box1_area, box2_area)
        return overlap_ratio
    
    for i in range(len(body_bbox)):
        for j in range(i+1, len(body_bbox)):
            if calculate_overlap(body_bbox[i], body_bbox[j]) > 0.2:
                return False
    
    return True

def calculate_mask_external_rectangle(mask):
    non_zero_points = np.nonzero(mask)
    if len(non_zero_points[0]) == 0 or len(non_zero_points[1]) == 0:
        return None
    x_min, x_max = np.min(non_zero_points[1]), np.max(non_zero_points[1])
    y_min, y_max = np.min(non_zero_points[0]), np.max(non_zero_points[0])
    return [x_min, y_min, x_max, y_max]

def get_new_size(image, max_side_len=1024):
    width, height = image.size
    new_width = width
    new_height = height
    if width > height:
        if width > max_side_len:
            new_width = max_side_len
            new_height =  int((height / (width / new_width)) // 16 * 16)
    else:
        if height > max_side_len:
            new_height = max_side_len
            new_width =  int((width / (height / new_height)) // 16 * 16)
    
    return new_width, new_height

def c_cropv2(image, bl = 16):
    width, height = image.size
    width_new = width // bl * bl
    height_new = height // bl * bl

    left = (width - width_new) / 2
    top = (height - height_new) / 2
    right = (width + width_new) / 2
    bottom = (height + height_new) / 2
    return image.crop((left, top, right, bottom))

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
    keypoints_new = [
        keypoints[0],   # nose
        keypoints[1],   # left_eye
        keypoints[2],   # right_eye
        keypoints[3],   # left_ear
        keypoints[4],   # right_ear
        keypoints[5],   # left_shoulder
        keypoints[6],   # right_shoulder
        keypoints[7],   # left_elbow
        keypoints[8],   # right_elbow
        keypoints[62],  # left_wrist
        keypoints[41],  # right_wrist
        keypoints[9],   # left_hip
        keypoints[10],  # right_hip
        keypoints[11],  # left_knee
        keypoints[12],  # right_knee
        keypoints[13],  # left_ankle
        keypoints[14]   # right_ankle
    ]
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

def combine_canny_skeleton(img_path, skeleton_path, seg_path=None, is_facebook=True, add_canny=True, min_conf=0.3):
    # try:
    if True:
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
                for item in skeleton_data['segments']:
                    if 'skeleton' not in item:
                        print("no skeleton", skeleton_path, "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
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
            canvas = draw_bodypose(canvas, keypoint, min_conf=min_conf)
        if is_facebook and model_type == 308:
            for keypoint_righthand, keypoint_lefthand in zip(keypoints_righthand, keypoints_lefthand):
                canvas = draw_handpose(canvas, keypoint_righthand, keypoint_lefthand, min_conf=min_conf)

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
    # except Exception as e:
    #     print('error:', e)
    #     return img_path, e
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

    return np.array([new_x1, new_y1, new_x2, new_y2])
    
def pad_bbox(bbox):
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    if w > h:
        bbox[1] -= (w - h) // 2
        bbox[3] += (w - h) // 2
    else:
        bbox[0] -= (h - w) // 2
        bbox[2] += (h - w) // 2
    return bbox
    
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
'''
/mnt2/wangxuekuan/data/ip_adapter_gpt4o__desc//mnt2/huangyuqiu/share/danbooru_anime/img/Dragon_Ball/son_goku/danbooru_8290662.txt
'''
def get_sapiens_path(json_path):
    if "vcg_images_80W" in json_path:
        mask_sapiens_path = json_path.replace("/mnt2/huangyuqiu/share/vcg_images_80W/json_final", "/mnt2/zhenghaoyu/share/vcg_images_80W/mask_sapiens")[:-5] + "_seg.npz"
        skeleton_sapiens_path = json_path.replace("/mnt2/huangyuqiu/share/vcg_images_80W/json_final", "/mnt2/zhenghaoyu/share/vcg_images_80W/skeleton_sapiens")
    elif "vcg_images_122W_1" in json_path:
        mask_sapiens_path = json_path.replace("/mnt2/wangxuekuan/data/MSDBv2_json/mnt/wangxuekuan/data/vcg_images_122W_1_v3_format_simple_skeleton", 
                                                "/mnt2/huangyuqiu/share/vcg_images_122W_1//mask_sapiens")[:-5] + "_seg.npz"
        skeleton_sapiens_path = json_path.replace("/mnt2/wangxuekuan/data/MSDBv2_json/mnt/wangxuekuan/data/vcg_images_122W_1_v3_format_simple_skeleton", 
                                                "/mnt2/huangyuqiu/share/vcg_images_122W_1//skeleton_sapiens")
    elif "flux_img_78_all" in json_path:
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
    elif "flux_anime_1character" in json_path:
        mask_sapiens_path = json_path.replace("/mnt2/zhenghaoyu/prompt_data/flux_anime_1character/json_w_face", 
                                                "/mnt2/zhenghaoyu/prompt_data/flux_anime_1character/mask_sapiens")[:-5] + "_seg.npz"
        skeleton_sapiens_path = json_path.replace("/mnt2/zhenghaoyu/prompt_data/flux_anime_1character/json_w_face", 
                                                "/mnt2/zhenghaoyu/prompt_data/flux_anime_1character/skeleton_sapiens")
    elif "flux_anime_2characters" in json_path:
        mask_sapiens_path = json_path.replace("/mnt2/zhenghaoyu/prompt_data/flux_anime_2characters/json_w_face", 
                                                "/mnt2/zhenghaoyu/prompt_data/flux_anime_2characters/mask_sapiens")[:-5] + "_seg.npz"
        skeleton_sapiens_path = json_path.replace("/mnt2/zhenghaoyu/prompt_data/flux_anime_2characters/json_w_face", 
                                                "/mnt2/zhenghaoyu/prompt_data/flux_anime_2characters/skeleton_sapiens")
    # elif "danbooru_anime" in json_path:
    #     mask_sapiens_path = json_path.replace("/mnt2/zhenghaoyu/prompt_data/flux_anime_2characters/json_w_face", 
    #                                             "/mnt2/zhenghaoyu/prompt_data/flux_anime_2characters/mask_sapiens")[:-5] + "_seg.npz"
    #     skeleton_sapiens_path = json_path.replace("/mnt2/zhenghaoyu/prompt_data/flux_anime_2characters/json_w_face", 
    #                                             "/mnt2/zhenghaoyu/prompt_data/flux_anime_2characters/skeleton_sapiens")
    else:
        # print("error json path", json_path)
        return None, None#self.__getitem__(random.randint(0, len(self.data) - 1))

    if not os.path.exists(mask_sapiens_path):
        mask_sapiens_path = mask_sapiens_path[:-3] + "npy"

    if os.path.exists(skeleton_sapiens_path.replace("skeleton_sapiens", "skeleton_sapiens_308")):
        skeleton_sapiens_path = skeleton_sapiens_path.replace("skeleton_sapiens", "skeleton_sapiens_308")

    if not os.path.exists(mask_sapiens_path) or not os.path.exists(skeleton_sapiens_path):
        print("no - data", mask_sapiens_path, skeleton_sapiens_path)
        # self.error_idx.append(idx)
        return None, None
    
    return mask_sapiens_path, skeleton_sapiens_path

def fill_mask(mask):
    if mask is None:
        return None
    # 膨胀腐蚀
    kernel_size = random.randint(3, 7)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=1)
    mask = cv2.erode(mask, kernel, iterations=1)

    # 填充内部空洞
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    for i in range(len(contours)):
        # hierarchy[0][i][3] >= 0 表示这是一个内部轮廓(孔洞)
        if hierarchy[0][i][3] >= 0:
            cv2.drawContours(mask, contours, i, 1, -1)

    return mask.astype(bool)
    
def get_sapiens_sam_path(json_path):
    if "vcg_images_80W" in json_path:
        mask_sapiens_path = json_path.replace("/mnt2/huangyuqiu/share/vcg_images_80W/json_final", "/mnt2/zhenghaoyu/share/vcg_images_80W/mask_sapiens")[:-5] + "_seg.npz"
        skeleton_sapiens_path = json_path.replace("/mnt2/huangyuqiu/share/vcg_images_80W/json_final", "/mnt2/zhenghaoyu/share/vcg_images_80W/skeleton_sapiens")
        mask_sam_path = json_path.replace("/mnt2/huangyuqiu/share/vcg_images_80W/json_final", "/mnt2/huangyuqiu/share/vcg_images_80W/mask")[:-5] + ".npz"
    elif "vcg_images_122W_1" in json_path:
        mask_sapiens_path = json_path.replace("/mnt2/wangxuekuan/data/MSDBv2_json/mnt/wangxuekuan/data/vcg_images_122W_1_v3_format_simple_skeleton", 
                                                "/mnt2/huangyuqiu/share/vcg_images_122W_1/mask_sapiens")[:-5] + "_seg.npz"
        skeleton_sapiens_path = json_path.replace("/mnt2/wangxuekuan/data/MSDBv2_json/mnt/wangxuekuan/data/vcg_images_122W_1_v3_format_simple_skeleton", 
                                                "/mnt2/huangyuqiu/share/vcg_images_122W_1/skeleton_sapiens")
        mask_sam_path = json_path.replace("/mnt2/wangxuekuan/data/MSDBv2_json/mnt/wangxuekuan/data/vcg_images_122W_1_v3_format_simple_skeleton", "/mnt2/huangyuqiu/share/vcg_images_122W_1/mask")[:-5] + ".npz"
    elif "/mnt2/huangyuqiu/share/flux_img_78_all/json_final" in json_path:
        mask_sapiens_path = json_path.replace("/mnt2/huangyuqiu/share/flux_img_78_all/json_final", 
                                                "/mnt2/zhenghaoyu/prompt_data/flux_img_78_all/mask_sapiens")[:-5] + "_seg.npz"
        skeleton_sapiens_path = json_path.replace("/mnt2/huangyuqiu/share/flux_img_78_all/json_final", 
                                                "/mnt2/zhenghaoyu/prompt_data/flux_img_78_all/skeleton_sapiens")
        mask_sam_path = json_path.replace("/mnt2/huangyuqiu/share/flux_img_78_all/json_final", "/mnt2/zhenghaoyu/prompt_data/flux_img_78_all/mask")[:-5] + ".npz"
    elif "/mnt2/huangyuqiu/share/flux/json_final" in json_path:
        mask_sapiens_path = json_path.replace("/mnt2/huangyuqiu/share/flux/json_final", 
                                                "/mnt2/huangyuqiu/share/flux/mask_sapiens")[:-5] + "_seg.npz"
        skeleton_sapiens_path = json_path.replace("/mnt2/huangyuqiu/share/flux/json_final", 
                                                "/mnt2/huangyuqiu/share/flux/skeleton_sapiens")
        mask_sam_path = json_path.replace("/mnt2/huangyuqiu/share/flux/json_final", "/mnt2/huangyuqiu/share/flux/mask")[:-5] + ".npz"
    elif "flux_img_1character_78_all_noscene" in json_path:
        mask_sapiens_path = json_path.replace("/mnt2/huangyuqiu/share/flux_img_1character_78_all_noscene/json_final", 
                                                "/mnt2/zhenghaoyu/prompt_data/flux_img_1character_78_all_noscene/mask_sapiens")[:-5] + "_seg.npz"
        skeleton_sapiens_path = json_path.replace("/mnt2/huangyuqiu/share/flux_img_1character_78_all_noscene/json_final", 
                                                "/mnt2/zhenghaoyu/prompt_data/flux_img_1character_78_all_noscene/skeleton_sapiens")
        mask_sam_path = json_path.replace("/mnt2/huangyuqiu/share/flux_img_1character_78_all_noscene/json_final", "/mnt2/zhenghaoyu/prompt_data/flux_img_1character_78_all_noscene/mask")[:-5] + ".npz"
    elif "flux_img_1character_78_all_scene" in json_path:
        mask_sapiens_path = json_path.replace("/mnt2/huangyuqiu/share/flux_img_1character_78_all_scene/json_final", 
                                                "/mnt2/zhenghaoyu/prompt_data/flux_img_1character_78_all_scene/mask_sapiens")[:-5] + "_seg.npz"
        skeleton_sapiens_path = json_path.replace("/mnt2/huangyuqiu/share/flux_img_1character_78_all_scene/json_final", 
                                                "/mnt2/zhenghaoyu/prompt_data/flux_img_1character_78_all_scene/skeleton_sapiens")
        mask_sam_path = json_path.replace("/mnt2/huangyuqiu/share/flux_img_1character_78_all_scene/json_final", "/mnt2/zhenghaoyu/prompt_data/flux_img_1character_78_all_scene/mask")[:-5] + ".npz"
    elif "vcg_chat_history_0_300_two_people" in json_path:
        mask_sapiens_path = json_path.replace("/mnt2/huangyuqiu/share/vcg_chat_history_0_300_two_people/json_final", 
                                                "/mnt2/huangyuqiu/share/vcg_chat_history_0_300_two_people/mask_sapiens")[:-5] + "_seg.npz"
        skeleton_sapiens_path = json_path.replace("/mnt2/huangyuqiu/share/vcg_chat_history_0_300_two_people/json_final", 
                                                "/mnt2/huangyuqiu/share/vcg_chat_history_0_300_two_people/skeleton_sapiens")
        mask_sam_path = json_path.replace("/mnt2/huangyuqiu/share/vcg_chat_history_0_300_two_people/json_final", "/mnt2/huangyuqiu/share/vcg_chat_history_0_300_two_people/mask")[:-5] + ".npz"
    elif "vcg_reelshort_0_200" in json_path:
        mask_sapiens_path = json_path.replace("/mnt2/huangyuqiu/share/vcg_reelshort_0_200/json_final", 
                                                "/mnt2/huangyuqiu/share/vcg_reelshort_0_200/mask_sapiens")[:-5] + "_seg.npz"
        skeleton_sapiens_path = json_path.replace("/mnt2/huangyuqiu/share/vcg_reelshort_0_200/json_final", 
                                                "/mnt2/huangyuqiu/share/vcg_reelshort_0_200/skeleton_sapiens")
        mask_sam_path = json_path.replace("/mnt2/huangyuqiu/share/vcg_reelshort_0_200/json_final", "/mnt2/huangyuqiu/share/vcg_reelshort_0_200/mask")[:-5] + ".npz"
    elif "flux_anime_1character" in json_path:
        mask_sapiens_path = json_path.replace("/mnt2/zhenghaoyu/prompt_data/flux_anime_1character/json_w_face", 
                                                "/mnt2/zhenghaoyu/prompt_data/flux_anime_1character/mask_sapiens")[:-5] + "_seg.npz"
        skeleton_sapiens_path = json_path.replace("/mnt2/zhenghaoyu/prompt_data/flux_anime_1character/json_w_face", 
                                                "/mnt2/zhenghaoyu/prompt_data/flux_anime_1character/skeleton_sapiens")
        mask_sam_path = json_path.replace("/mnt2/zhenghaoyu/prompt_data/flux_anime_1character/json_w_face", 
                                          "/mnt2/zhenghaoyu/prompt_data/flux_anime_1character/mask")[:-5] + ".npz"
    elif "flux_anime_2characters" in json_path:
        mask_sapiens_path = json_path.replace("/mnt2/zhenghaoyu/prompt_data/flux_anime_2characters/json_w_face", 
                                                "/mnt2/zhenghaoyu/prompt_data/flux_anime_2characters/mask_sapiens")[:-5] + "_seg.npz"
        skeleton_sapiens_path = json_path.replace("/mnt2/zhenghaoyu/prompt_data/flux_anime_2characters/json_w_face", 
                                                "/mnt2/zhenghaoyu/prompt_data/flux_anime_2characters/skeleton_sapiens")
        mask_sam_path = json_path.replace("/mnt2/zhenghaoyu/prompt_data/flux_anime_2characters/json_w_face", 
                                          "/mnt2/zhenghaoyu/prompt_data/flux_anime_2characters/mask")[:-5] + ".npz"
    elif "tuchong" in json_path:
        mask_sapiens_path = json_path.replace("/mnt2/huangyuqiu/share/tuchong/json_new", 
                                                "/mnt2/huangyuqiu/share/tuchong_full/mask_sapiens")[:-5] + "_seg.npz"
        skeleton_sapiens_path = json_path.replace("/mnt2/huangyuqiu/share/tuchong/json_new", 
                                                "/mnt2/huangyuqiu/share/tuchong_full/skeleton_sapiens")
        mask_sam_path = json_path.replace("/mnt2/huangyuqiu/share/tuchong/json_new", "/mnt2/huangyuqiu/share/tuchong_full/mask")[:-5] + ".npz"
    elif "anime_pictures" in json_path:
        mask_sapiens_path = json_path.replace("/mnt2/huangyuqiu/share/anime_pictures/json_w_face", 
                                                "/mnt2/huangyuqiu/share/anime_pictures/mask_sapiens")[:-5] + "_seg.npz"
        skeleton_sapiens_path = json_path.replace("/mnt2/huangyuqiu/share/anime_pictures/json_w_face", 
                                                "/mnt2/huangyuqiu/share/anime_pictures/skeleton_sapiens")
        mask_sam_path = json_path.replace("/mnt2/huangyuqiu/share/anime_pictures/json_w_face", "/mnt2/huangyuqiu/share/anime_pictures/mask")[:-5] + ".npz"
    elif "ponyXL" in json_path:
        mask_sapiens_path = json_path.replace("/mnt2/zhenghaoyu/share/ponyXL/json_final", "/mnt2/zhenghaoyu/share/ponyXL/mask_sapiens")[:-5] + "_seg.npz"
        skeleton_sapiens_path = json_path.replace("/mnt2/zhenghaoyu/share/ponyXL/json_final", "/mnt2/zhenghaoyu/share/ponyXL/skeleton_sapiens")
        mask_sam_path = json_path.replace("/mnt2/zhenghaoyu/share/ponyXL/json_final", "/mnt2/zhenghaoyu/share/ponyXL/mask")[:-5] + ".npz"
    else:  
        # print("error json path", json_path)
        return None, None, None#self.__getitem__(random.randint(0, len(self.data) - 1))

    if not os.path.exists(mask_sapiens_path):
        mask_sapiens_path = mask_sapiens_path[:-3] + "npy"
    
    if not os.path.exists(mask_sam_path):
        mask_sam_path = mask_sam_path[:-3] + "npy"
    
    if os.path.exists(skeleton_sapiens_path.replace("skeleton_sapiens", "skeleton_sapiens_308")):
        skeleton_sapiens_path = skeleton_sapiens_path.replace("skeleton_sapiens", "skeleton_sapiens_308")
        
    if not os.path.exists(skeleton_sapiens_path):
        skeleton_sapiens_path = skeleton_sapiens_path.replace("skeleton_sapiens", "skeleton")

    print(mask_sapiens_path, skeleton_sapiens_path, mask_sam_path)
    if not os.path.exists(mask_sapiens_path) or not os.path.exists(skeleton_sapiens_path) or not os.path.exists(mask_sam_path):
        print("no - data", mask_sapiens_path, skeleton_sapiens_path, mask_sam_path)
        # self.error_idx.append(idx)
        return None, None, None
    
    return mask_sapiens_path, skeleton_sapiens_path, mask_sam_path

class FastVisualizer:
    """MMPose Fast Visualizer.

    A simple yet fast visualizer for video/webcam inference.

    Args:
        metainfo (dict): pose meta information
        radius (int, optional)): Keypoint radius for visualization.
            Defaults to 6.
        line_width (int, optional): Link width for visualization.
            Defaults to 3.
        kpt_thr (float, optional): Threshold for keypoints' confidence score,
            keypoints with score below this value will not be drawn.
            Defaults to 0.3.
    """

    def __init__(self, metainfo=None, radius=1, line_width=1, kpt_thr=0.3):
        if metainfo is None:
            metainfo = json.load(open('/ucloud/mnt-98T/zhenghaoyu/prompt_data/flux_img_78_all/skeleton_sapiens_308/0001/999_1725459878_Riley_hitting_Amber_with_a_pillow_2.json', 'r'))['meta_info']
        self.radius = radius
        self.line_width = line_width
        self.kpt_thr = kpt_thr

        self.keypoint_id2name = metainfo['keypoint_id2name']
        self.keypoint_name2id = metainfo['keypoint_name2id']
        self.keypoint_colors = metainfo['keypoint_colors']['__ndarray__']
        self.skeleton_links = metainfo['skeleton_links']
        self.skeleton_link_colors = metainfo['skeleton_link_colors']['__ndarray__']
        
        self.face_keypoints_idx = [0, 1, 2, 3, 4, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271]

        self.face_keypoints_idx += [5, 6] #肩膀
        
    def draw_pose(self, raw_image, skeleton_path, dropout=0.0, only_face=False) -> np.ndarray:
        # print("skeleton_path", skeleton_path)qq
        if skeleton_path is None or not os.path.exists(skeleton_path):
            # print("No skeleton_path!1111111111111111111111111111111111", skeleton_path)
            return np.zeros((raw_image.height, raw_image.width, 3)).astype(np.uint8)
        if random.random() < dropout:
            return np.zeros((raw_image.height, raw_image.width, 3)).astype(np.uint8)
        with open(skeleton_path, 'r') as f:
            data = json.load(f)
        if 'instance_info' not in data:
            # print("No instance_info!2222222222222222222222222222222222", skeleton_path)
            return np.zeros((raw_image.height, raw_image.width, 3)).astype(np.uint8)
        instances_list = data['instance_info']
        
        canvas = np.zeros((raw_image.height, raw_image.width, 3)).astype(np.uint8)
        for instances in instances_list:
            if instances is None:
                print('no instance detected')
                return

            keypoints = instances['keypoints']
            scores = instances['keypoint_scores']

            # for kpts, score in zip(keypoints, scores):
            kpts = keypoints
            score = scores
            for sk_id, sk in enumerate(self.skeleton_links):
                if only_face and (sk[0] not in self.face_keypoints_idx or sk[1] not in self.face_keypoints_idx):
                    continue
                if score[sk[0]] < self.kpt_thr or score[sk[1]] < self.kpt_thr:
                    # skip the link that should not be drawn
                    continue

                pos1 = (int(kpts[sk[0]][0]), int(kpts[sk[0]][1]))
                pos2 = (int(kpts[sk[1]][0]), int(kpts[sk[1]][1]))

                color = self.skeleton_link_colors[sk_id]
                cv2.line(canvas, pos1, pos2, color, thickness=self.line_width)

            for kid, kpt in enumerate(kpts):
                if only_face and kid not in self.face_keypoints_idx:
                    continue
                if score[kid] < self.kpt_thr:
                    # skip the point that should not be drawn
                    continue

                x_coord, y_coord = int(kpt[0]), int(kpt[1])

                color = self.keypoint_colors[kid]
                cv2.circle(canvas, (int(x_coord), int(y_coord)), self.radius,
                            color, -1)
                cv2.circle(canvas, (int(x_coord), int(y_coord)), self.radius,
                            (255, 255, 255))
                
        return canvas
    
def get_face_keypoints(skeleton_sapiens_path, face_bbox_o, dw, dh, ratio_w, ratio_h):
    if skeleton_sapiens_path is None or not os.path.exists(skeleton_sapiens_path):
        return []
    face_keyppoints_idx = [0, 1, 2, 3, 4, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271]
    all_points_len = len(face_keyppoints_idx)
    with open(skeleton_sapiens_path, "r") as f:
        skeleton_data = json.load(f)
    flag = True
    keypoints = []
    for item in skeleton_data['instance_info']:
        if len(item['keypoints']) == 308:
            all_keypoints = item['keypoints']
            all_scores = item['keypoint_scores']
            face_keypoints = []
            for i in face_keyppoints_idx:
                face_keypoints.append(all_keypoints[i])
            out_of_bbox_count = 0
            for keypoint in face_keypoints:
                if keypoint[0] < face_bbox_o[0] or keypoint[0] > face_bbox_o[2] or keypoint[1] < face_bbox_o[1] or keypoint[1] > face_bbox_o[3]:
                    out_of_bbox_count += 1
                if out_of_bbox_count >= 3 * all_points_len / 4:
                    print("keypoints out of face bbox!")
                    flag = False
                    break
            if flag:
                keypoints += face_keypoints
                print("FOUND keypoints!")
                break
    if len(keypoints) == all_points_len:
        for keypoint in keypoints:
            keypoint[0] = max(int((keypoint[0] - dw) * ratio_w), 0)
            keypoint[1] = max(int((keypoint[1] - dh) * ratio_h), 0)
    else:
        print(len(keypoints), "no keypoints found!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    return keypoints

def apply_same_transforms(img1, img2):
    # 随机旋转
    angle = random.uniform(-30, 30)
    img1 = F.rotate(img1, angle)
    img2 = F.rotate(img2, angle)

    # 随机裁剪和调整大小
    scale = random.uniform(0.8, 1.2)
    ratio = random.uniform(0.8, 1.2)
    i, j, h, w = transforms.RandomResizedCrop.get_params(img1, scale=(scale, scale), ratio=(ratio, ratio))
    img1 = F.resized_crop(img1, i, j, h, w, size=(224, 224), interpolation=F.InterpolationMode.BICUBIC)
    img2 = F.resized_crop(img2, i, j, h, w, size=(224, 224), interpolation=F.InterpolationMode.BICUBIC)

    # 随机水平翻转
    if random.random() > 0.5:
        img1 = F.hflip(img1)
        img2 = F.hflip(img2)

    return img1, img2