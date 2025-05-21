from multi_threads_w_model import MultiThreadsWithModelBase
import os
import glob
import cv2
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import torch
from PIL import Image
import torch.multiprocessing as mp
from mmpose.apis import MMPoseInferencer
import json
import numpy as np
import scipy
import logging
import time


def convert_float32_to_float(obj):
    if isinstance(obj, dict):
        return {k: convert_float32_to_float(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_float32_to_float(i) for i in obj]
    elif isinstance(obj, np.float32):
        return float(obj)
    else:
        return obj


class SkeletonMultiThreads(MultiThreadsWithModelBase):
    def __init__(self, dataset_root_path, total_segments=1, which_segment=0, gpu_list=[0,1,2,3,4,5,6,7], queue_max_size=200):
        super().__init__(dataset_root_path, total_segments, which_segment, gpu_list,queue_max_size)

    def model_init(self, gpu_id):
        """
        实现模型初始化逻辑。
        """
        # 假设这里加载了一个模型
        device = torch.device(f'cuda:{gpu_id}')
        model = MMPoseInferencer('rtmo-l_16xb16-600e_body7-640x640', device=device)
        return model
    
    def add_task(self):
        i = 0
        file_list = []
        # import pdb;pdb.set_trace()
        for root,_,files in os.walk(self.json_dir_path, followlinks=True):
            relative_path = os.path.relpath(root, self.json_dir_path)
            for file in files:
                if file.endswith('.json'):
                    file_path = os.path.join(root, file)
                    skeleton_path = os.path.join(self.skeleton_dir_path, relative_path, file)
                    if os.path.exists(skeleton_path):
                        print(f'重复：{skeleton_path}')
                        continue
                    i += 1
                    file_list.append(file_path)
        print('数据量:', i)
        self.total_dataset = i // self.total_segments
        total = i
        for file in file_list[int(total*self.which_segment/self.total_segments):int(total*(self.which_segment+1)/self.total_segments)]:
            self.task_queue.put(file)

    def single_thread_infer(self, model, data):
        # import pdb;pdb.set_trace
        print('data:',data)
        time_start = time.time()
        save_path = data.replace(self.json_dir_path, self.skeleton_dir_path)
        if os.path.exists(save_path):
            print(f'重复：{save_path}')
        # import pdb;pdb.set_trace
        os.makedirs(os.path.dirname(save_path),exist_ok=True)
        with open(data) as f:
            info_dict = json.load(f)
        segments = info_dict['segments']
        seg_path = data.replace(self.json_dir_path, self.mask_dir_path).replace('.json','.npz')
        if not os.path.exists(seg_path):
            print(f'文件不存在：{seg_path}')
            return
        loaded_sparse_mask = scipy.sparse.load_npz(seg_path)
        # 将稀疏矩阵转换回正常的numpy数组
        seg_map = loaded_sparse_mask.toarray()
        
        image_path = data.replace(self.json_dir_path, self.img_dir_path).replace('.json','.jpg')
        if not os.path.exists(image_path):
            image_path = image_path.replace('.jpg', '.png')
        img = Image.open(image_path)
        img = np.array(img)
        if len(img.shape) < 3:
            img = img[...,np.newaxis].repeat(3, axis=2)
        img = img[:,:,:3]
        new_segments = []
        save_flag = False
        for segment in segments:
            other_idx = list(np.unique(seg_map)) # 获得id
            try:
                other_idx.remove(segment['id']) 
            except Exception as e:
                print(f"{e}",segment['id'],data)
            other_idx.remove(255)

            combined_seg_map = np.zeros_like(seg_map)
            for ele in other_idx:
                combined_seg_map += (seg_map == ele)

            background = np.random.randint(0, 255, img.shape)

            seg_img_map = img * (1- combined_seg_map[...,np.newaxis])
            combined_seg_map = combined_seg_map.astype(bool)
            seg_img_map[combined_seg_map] = background[combined_seg_map] 

            bbox = np.array(segment['bbox'])
            bbox[bbox<0] = 0
            mask = np.zeros_like(seg_img_map)
            mask[bbox[1]:bbox[3], bbox[0]:bbox[2],:] = 1
            
            background[~mask.astype(bool)] = 0
            seg_img_map[~mask.astype(bool)] = 0
             
            result_generator = model(
                seg_img_map,
                return_vis=True,
                black_background=True,
                skeleton_style='openpose',
                radius=6,
                # nms_thr=0.70,
                thickness=6
            )
            pred = next(result_generator)
            pred_json = convert_float32_to_float(pred['predictions'])
            items = pred_json[0]
            if len(items) > 0: 
                items.sort(reverse=True, key=lambda person: abs(abs((person['bbox'][0][0] - person['bbox'][0][2]) * (person['bbox'][0][1] - person['bbox'][0][3]))))
            if len(items) == 0:
                skeleton_dict = None
            else:
                skeleton_dict = items[0]
                save_flag = True

            segment['skeleton'] = skeleton_dict
            # new_segments.append(segment)
        if save_flag:
            with open(save_path, 'w') as f:
                json.dump(info_dict, f, indent=4)
        with self.lock:
            cost_time = time.time() - time_start
            self.process_count += 1
            print(f"{self.process_count} ==>> {self.total_dataset},cost: {cost_time}.{save_path}")

if __name__ == '__main__':
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

    skeleton_multi_threads = SkeletonMultiThreads(dataset_root_path,total_segments=total_segments,which_segment=which_segment,gpu_list=gpu_list)
    skeleton_multi_threads.producer()
    skeleton_multi_threads.start_threads()
    skeleton_multi_threads.wait_completion()
