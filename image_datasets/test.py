from PIL import Image
import cv2
import numpy as np
import json
import os

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
            metainfo = json.load(open('/mnt2/zhenghaoyu/prompt_data/flux_img_78_all/skeleton_sapiens_308/0001/999_1725459878_Riley_hitting_Amber_with_a_pillow_2.json', 'r'))['meta_info']
        self.radius = radius
        self.line_width = line_width
        self.kpt_thr = kpt_thr

        self.keypoint_id2name = metainfo['keypoint_id2name']
        self.keypoint_name2id = metainfo['keypoint_name2id']
        self.keypoint_colors = metainfo['keypoint_colors']['__ndarray__']
        self.skeleton_links = metainfo['skeleton_links']
        self.skeleton_link_colors = metainfo['skeleton_link_colors']['__ndarray__']
        
    def draw_pose(self, raw_image, skeleton_path):
        if skeleton_path is None or not os.path.exists(skeleton_path):
            return np.zeros((raw_image.height, raw_image.width, 3)).astype(np.uint8)
        with open(skeleton_path, 'r') as f:
            data = json.load(f)
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
                # if sk[0] not in self.face_keypoint_ids or sk[1] not in self.face_keypoint_ids:
                #     continue
                if score[sk[0]] < self.kpt_thr or score[sk[1]] < self.kpt_thr:
                    # skip the link that should not be drawn
                    continue

                pos1 = (int(kpts[sk[0]][0]), int(kpts[sk[0]][1]))
                pos2 = (int(kpts[sk[1]][0]), int(kpts[sk[1]][1]))

                color = self.skeleton_link_colors[sk_id]
                cv2.line(canvas, pos1, pos2, color, thickness=self.line_width)

            for kid, kpt in enumerate(kpts):
                # if kid not in self.face_keypoint_ids:
                #     continue
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
    

VISUALIZER = FastVisualizer()

path = "/mnt2/zhenghaoyu/prompt_data/flux_img_78_all/img/0005/4073_1725553697_Sophie_nudging_her_brother_2.jpg"
skeleton_path = "/mnt2/zhenghaoyu/prompt_data/flux_img_78_all/skeleton_sapiens_308/0005/4073_1725553697_Sophie_nudging_her_brother_2.json"
image = Image.open(path)

image = VISUALIZER.draw_pose(image, skeleton_path)
image = Image.fromarray(image)
image = image.resize((image.width // 2, image.height // 2))
image.save("test.png")
