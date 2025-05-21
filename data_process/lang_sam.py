import sys
from multi_threads_w_model import MultiThreadsWithModelBase
import glob
import time
import json
import os
import numpy as np
import copy
import cv2
from PIL import Image
from utils.iou import *
from utils.filter import *
from scipy.sparse import csr_matrix
import scipy
import torch
from lang_sam import LangSAM
from tqdm import tqdm


template_json = {
    "image_id": None,
    "segments": [],
}
template_segment = {
    "id": None,
    "word": None,
    "bbox": [],
    "score": None,
    "coco_label": None,
}

class GenBboxMask(MultiThreadsWithModelBase):    
    def model_init(self):
        """
        实现模型初始化逻辑。
        """
        models = []
        for i in self.gpu_list:
            device = torch.device(f"cuda:{i}")
            model = LangSAM(device = device)
            models.append(model)
        return models
    
    def add_task(self):
        i = 0
        file_list = []
        for root,_,files in os.walk(self.img_dir_path, followlinks=True):
            relative_path = os.path.relpath(root, self.img_dir_path)
            for file in files:
                if file.endswith('.jpg') or file.endswith('.png'):
                    file_path = os.path.join(root, file)
                    json_path = os.path.join(self.json_dir_path, relative_path, file.replace('.jpg','.json'))
                    mask_path = os.path.join(self.mask_dir_path, relative_path, file.replace('.jpg','.npz'))
                    if os.path.exists(json_path) and os.path.exists(mask_path):
                        # print(f'重复：{json_path}')
                        continue
                    i += 1
                    file_list.append(file_path)
        print('数据量:', i)
        self.total_dataset = i // self.total_segments
        for file in file_list[int(i*self.which_segment/self.total_segments):int(i*(self.which_segment+1)/self.total_segments)]:
            self.task_queue.put(file)
        for _ in range(self.thread_count):
            self.task_queue.put(None)
        
    def single_thread_infer(self, model, image_path):
        """
        image_path: 图片文件完整路径包含图片名称
        """
        time_start = time.time()
        text_prompt = 'person'
        result_dict = copy.deepcopy(template_json)
        relative_path = os.path.relpath(image_path, self.img_dir_path).rsplit('/',1)[0]
        image_name = image_path.rsplit('/', 1)[1].rsplit('.', 1)[0]
        mask_save_path = os.path.join(self.mask_dir_path, relative_path, image_name + '.npz')
        json_save_path = os.path.join(self.json_dir_path, relative_path, image_name + '.json')
        if os.path.exists(json_save_path) and os.path.exists(mask_save_path):
            return

        result_dict['image_id'] = image_name
        img = cv2.imread(image_path)
        image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) 
        img_h, img_w = image.size
        result_dict['image_size']={}
        result_dict['image_size']['h'] = img_h
        result_dict['image_size']['w'] = img_w

        masks, boxes, phrases, logits = model.predict(image, text_prompt)
        masks = np.array(masks,np.uint8)
        #寻找子bbox
        sub_bboxes = find_sub_bboxes(boxes)
        #剔除母bbox区域中，子bbox的mask为1的区域
        for key,sub_bbox in sub_bboxes.items():
            if sub_bbox:
                for index in sub_bbox:
                    masks[key] = subtract_masks(masks[key],masks[index])
        # #过滤小区域mask
        indexs = mask_area_filter(masks)
        
        id = 0
        res_numpy = np.ones(img.shape[:2]) * (255)
        for i, (bbox, mask, logit) in enumerate(zip(boxes, masks, logits)):
            if indexs[i]:
                segment = copy.deepcopy(template_segment)
                segment['id'] = id
                segment['word'] = 'Subject' + str(id)
                segment['bbox'] = [int(x) for x in bbox]
                segment['score'] = float(logit)
                segment['coco_label'] = text_prompt
                #可视化
                # cv2.rectangle(img, (int(bbox[0]),int(bbox[1])), (int(bbox[2]),int(bbox[3])), (255,0,0), 2)
                # cv2.putText(img,f'{logit}',(int(bbox[0])-1,int(bbox[1])+20),cv2.FONT_HERSHEY_COMPLEX,0.7,(0,255,0),1)
                # img = overlay(img, mask, color=colors[id%len(colors)], alpha=0.8)
                result_dict['segments'].append(segment)
                res_mask = np.array(mask,np.uint8)
                res_numpy = np.where(res_mask, res_mask * id, res_numpy)
                id += 1
        res_numpy = res_numpy.astype(np.uint8)
        sparse_mask = csr_matrix(res_numpy)
        if len(result_dict["segments"]) > 0:
            os.makedirs(os.path.dirname(json_save_path), exist_ok=True)
            os.makedirs(os.path.dirname(mask_save_path), exist_ok=True)
            scipy.sparse.save_npz(mask_save_path, sparse_mask)
            with open(json_save_path, 'w') as json_file:
               json.dump(result_dict, json_file, indent=4)
        else:
            print('detect nothing!')
                
        with self.lock:
            cost_time = time.time() - time_start
            self.process_count += 1
            print(f"{self.process_count} ==>> {self.total_dataset},cost: {cost_time}.{image_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root_path", type=str)
    parser.add_argument("--total_segments", type=str, default=1)
    parser.add_argument("--which_segment", type=str, default=0)
    parser.add_argument("--gpu_list", type=str, default="0,1,2,3,4,5,6,7")
    
    args = parser.parse_args()
    dataset_root_path = args.dataset_root_path
    total_segments = args.total_segments
    which_segment = args.which_segment
    gpu_list = args.gpu_list.split(',')
    print(dataset_root_path)

    processor = GenBboxMask(dataset_root_path, total_segments, which_segment, gpu_list=gpu_list, queue_max_size=50)
    processor.producer()
    processor.start_threads()
    processor.wait_completion()
