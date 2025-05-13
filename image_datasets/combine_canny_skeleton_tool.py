import glob
import os
from PIL import Image
import cv2
import numpy as np
import json
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

if __name__ == "__main__":
    # canvas = combine_canny_skeleton(img_path='/mnt2/zhenghaoyu/prompt_data/flux_img_78_all/img/0003/2003_1725490362_Mason_offering_a_cigarette_to_Lola_2.jpg', skeleton_path='/mnt2/zhenghaoyu/prompt_data/flux_img_78_all/skeleton_sapiens/0003/2003_1725490362_Mason_offering_a_cigarette_to_Lola_2.json', seg_path='/mnt2/zhenghaoyu/prompt_data/flux_img_78_all/mask_sapiens/0003/2003_1725490362_Mason_offering_a_cigarette_to_Lola_2_seg.npy', is_facebook=True, add_canny=False)
    # cv2.imwrite('/mnt2/zhenghaoyu/prompt_data/flux_img_78_all/vis/2003_1725490362_Mason_offering_a_cigarette_to_Lola_2.jpg', canvas)
    # canvas = combine_canny_skeleton(img_path='/mnt2/zhenghaoyu/prompt_data/flux_img_78_all/img/0001/26_1725429539_Tigerstar_directing_a_training_session_0.jpg', skeleton_path='/mnt2/zhenghaoyu/prompt_data/flux_img_78_all/skeleton_sapiens/0001/26_1725429539_Tigerstar_directing_a_training_session_0.json', seg_path='/mnt2/zhenghaoyu/prompt_data/flux_img_78_all/mask_sapiens/0001/26_1725429539_Tigerstar_directing_a_training_session_0_seg.npy', is_facebook=True, add_canny=False)
    # cv2.imwrite('/mnt2/zhenghaoyu/prompt_data/flux_img_78_all/vis/26_1725429539_Tigerstar_directing_a_training_session_0.jpg', canvas)
    path_list = glob.glob('/mnt2/zhenghaoyu/prompt_data/flux_img_1character_78_all_scene/img/0001/*.jpg')
    folder_name = '/mnt2/zhenghaoyu/prompt_data/flux_img_1character_78_all_scene'
    for img_path in tqdm(path_list):
        skeleton_path = img_path.replace('.jpg', '.json').replace('/img', '/skeleton_sapiens_308')
        if not os.path.exists(skeleton_path):
            continue
        mask_path = img_path.replace('.jpg', '_seg.npy').replace('/img', '/mask_sapiens')
        canvas = combine_canny_skeleton(img_path, skeleton_path, seg_path=mask_path, is_facebook=True, add_canny=True)
        cv2.imwrite(os.path.join('/mnt2/zhenghaoyu/prompt_data/flux_img_1character_78_all_scene/vis', os.path.basename(img_path)), canvas)