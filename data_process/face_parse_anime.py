import sys
from multi_threads_w_model import MultiThreadsWithModelBase
import os
import numpy as np
import copy
import cv2
import time
from tqdm import tqdm

import matplotlib.pyplot as plt

import IPython
from IPython.display import Image as img
from IPython.display import display

from PIL import Image

import torch
import torch.nn as nn
import torchvision

from dass_det.models.yolox import YOLOX
from dass_det.models.yolo_head import YOLOXHead
from dass_det.models.yolo_head_stem import YOLOXHeadStem
from dass_det.models.yolo_pafpn import YOLOPAFPN

from dass_det.data.data_augment import ValTransform

from dass_det.evaluators.comic_evaluator import ComicEvaluator

from dass_det.utils import (
    postprocess,
    xyxy2xywh,
    vis
)
import json
import glob

def is_contained_with_threshold(A, B, threshold=0.8):
    """
    判断边界框A是否被边界框B包含，基于面积比例的阈值。
    
    参数:
    A: 子框,边界框A的坐标，格式为(x_min, y_min, x_max, y_max)。
    B: 父框,边界框B的坐标，格式为(x_min, y_min, x_max, y_max)。
    threshold: 包含的阈值，表示A在B内部的面积比例。
    
    返回:
    如果A在B内部的面积比例超过阈值，则返回True；否则返回False。
    """
    # 解包边界框A和B的坐标
    Ax_min, Ay_min, Ax_max, Ay_max = A
    Bx_min, By_min, Bx_max, By_max = B
    
    # 计算A和B的交集区域的坐标
    Ix_min = max(Ax_min, Bx_min)
    Iy_min = max(Ay_min, By_min)
    Ix_max = min(Ax_max, Bx_max)
    Iy_max = min(Ay_max, By_max)
    
    # 确保交集区域是有效的（即存在交集）
    if Ix_min < Ix_max and Iy_min < Iy_max:
        # 计算A的面积和交集区域的面积
        A_area = (Ax_max - Ax_min) * (Ay_max - Ay_min)
        I_area = (Ix_max - Ix_min) * (Iy_max - Iy_min)
        
        # 计算交集区域面积占A面积的比例
        overlap_ratio = I_area / A_area
        
        # 判断是否满足阈值条件
        return overlap_ratio >= threshold
    else:
        # 如果没有交集，显然不满足包含条件
        return False

class DetectAnimeFace(MultiThreadsWithModelBase):
    def __init__(self, json_path, dataset_root_path, total_segments=1, which_segment=0, gpu_list=[0,1,2,3,4,5,6,7], queue_max_size=200):
        super().__init__(dataset_root_path, total_segments, which_segment, gpu_list,queue_max_size)
        self.nms_thold   = 0.4
        self.conf_thold  = 0.65
        self.resize_size = (512, 512)
        self.json_path = json_path
    def model_init(self, gpu_id):
        """
        实现模型初始化逻辑。
        """
        model_path  = "weights/xs_icf_finetuned_stage3.pth" # None # "weights/..."
        model_size  = "xs" # "xl"
        model_mode  = 1    # 1 for only face, 2 for only body
        
        models = []
        transform = ValTransform()
        models.append(transform)
        if model_size == "xs":
            depth, width = 0.33, 0.375
        elif model_size == "xl":
            depth, width = 1.33, 1.25

        model = YOLOX(backbone=YOLOPAFPN(depth=depth, width=width),
                    head_stem=YOLOXHeadStem(width=width),
                    face_head=YOLOXHead(1, width=width),
                    body_head=YOLOXHead(1, width=width))

        d = torch.load(model_path, map_location=torch.device('cpu'))

        if "teacher_model" in d.keys():
            model.load_state_dict(d["teacher_model"])
        else:
            model.load_state_dict(d["model"])
        model = model.eval().cuda().to(f'cuda:{gpu_id}')
        models.append(model)
        del d
        return models

    def add_task(self):
        file_list = glob.glob(os.path.join(self.json_path, '**/*.json'), recursive=True)
        total = len(file_list)
        print('数据量:',total)
        self.total_dataset = total // self.total_segments
        for file_path in tqdm(file_list[int(total*self.which_segment/self.total_segments):int(total*(self.which_segment+1)/self.total_segments)]):
            self.task_queue.put(file_path)

    def single_thread_infer(self, model, data):
        try:
            image_path = data.replace('/json','/img').replace('.json','.jpg')
            imgs = cv2.imread(image_path)
            h, w, c = imgs.shape

            imgs, labels = model[0](imgs, None, self.resize_size)
            scale = min(self.resize_size[0] / h, self.resize_size[1] / w)
            import time
            start = time.time()
            self.predict_and_draw(model[1], copy.deepcopy(imgs), data, scale, [h, w], self.conf_thold, self.nms_thold)
            print('cost:',time.time()-start)
        except Exception as e:
            print('error:',e)


    def predict_and_draw(self, model, imgs, path, scale, sizes, conf_thold, nms_thold):
        json_save_path = path.replace('/json', '/json_w_face')
        if os.path.exists(json_save_path):
            print(f'重复:{json_save_path}')
            return
        if not os.path.exists(path):
            print(f'文件不存在:{path}')
            return
        with open(path, "r") as f:
            info_dict = json.load(f)
        
        segments = info_dict['segments']

        img_cu = torch.Tensor(imgs).unsqueeze(0).cuda()
        
        with torch.no_grad():
            face_preds, _ = model(img_cu, mode=0)
            face_preds = postprocess(face_preds, 1, conf_thold, nms_thold)[0]

        del img_cu
        
        if face_preds is not None:
            preds = face_preds
        else:
            print("No faces or bodies are found!")
            return

        preds[:,:4] /= scale
        preds[:,0]  = torch.max(preds[:,0], torch.zeros(preds.shape[0]).cuda())
        preds[:,1]  = torch.max(preds[:,1], torch.zeros(preds.shape[0]).cuda())
        preds[:,2]  = torch.min(preds[:,2], torch.zeros(preds.shape[0]).fill_(sizes[1]).cuda())
        preds[:,3]  = torch.min(preds[:,3], torch.zeros(preds.shape[0]).fill_(sizes[0]).cuda())
        scores      = preds[:,4]
        
        # import pdb;pdb.set_trace()
        for bbox, score in zip(preds[:,:4],scores):
            if score < 0.9:
                continue
            bbox = [int(item) for item in bbox]
            face_area_ratio = (bbox[3]-bbox[1])*(bbox[2]-bbox[0])/(sizes[0]*sizes[1])
            print('bbox:',bbox,'score:',score, 'face_ratio:',round(face_area_ratio,4))
            flag = False
            for segment in segments:
                id = segment['id']
                body_bbox = segment['bbox']
                if is_contained_with_threshold(bbox, body_bbox):
                    flag = True
                    segment['face_bbox'] = bbox
                    segment['face_area_ratio'] = round(face_area_ratio,4)
                    break
            if not flag: 
                segment['face_bbox'] = None
                segment['face_area_ratio'] = None

        with self.lock:
            os.makedirs(os.path.dirname(json_save_path), exist_ok=True)
            with open(json_save_path, 'w') as json_file:
                json.dump(info_dict, json_file, indent=4)
            return


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", type=str, default='./')
    parser.add_argument("--slice_num", type=str, default=1)
    parser.add_argument("--which_piece", type=str, default=0)
    parser.add_argument("--gpu_list", type=str, default="0,1,2,3,4,5,6,7")

    
    args = parser.parse_args()
    json_path = args.json_path
    total_segments = args.slice_num
    which_segment = args.which_piece
    gpu_list = args.gpu_list.split(',')

    skeleton_multi_threads = DetectAnimeFace(json_path, './', total_segments, which_segment,gpu_list=gpu_list)
    skeleton_multi_threads.producer()
    skeleton_multi_threads.start_threads()
    skeleton_multi_threads.wait_completion()
