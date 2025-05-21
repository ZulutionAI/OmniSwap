import torch
import facer
from PIL import Image
import matplotlib.pyplot as plt
import json
import os
import cv2
import numpy as np
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
import time
import threading
from scipy.sparse import csr_matrix
import scipy

def check_mask(mask, w_img, h_img):
    # 找到非零元素的索引
    rows, cols = np.where(mask != 0)

    # 计算最小外接矩形的边界
    min_row, max_row = rows.min(), rows.max()
    min_col, max_col = cols.min(), cols.max()
    w = max_row - min_row
    h = max_col - min_col
    if w > 0.3 * w_img or h > 0.3 * h_img:
        kernel = np.ones((15,15), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        # 找到非零元素的索引
        rows, cols = np.where(mask != 0)
        # 计算最小外接矩形的边界
        min_row, max_row = rows.min(), rows.max()
        min_col, max_col = cols.min(), cols.max()
        w = max_row - min_row
        h = max_col - min_col
        if w > 0.8 * w_img or h > 0.8 * h_img:
            return None
    return mask

def process_single_img(face_detector, face_parser, img_path, ori_mask_dir, face_mask_save_dir, hair_mask_save_dir, face_bbox_path, device):
    image_name = img_path.rsplit('/', 1)[1].rsplit('.', 1)[0]
    # if image_name not in ['2548','3071','3904']:
    #     return
    ori_mask_path = os.path.join(ori_mask_dir, image_name + '.npy')
    face_mask_save_path = os.path.join(face_mask_save_dir, image_name + '.npz')
    hair_mask_save_path = os.path.join(hair_mask_save_dir, image_name + '.npz')
    if os.path.exists(face_mask_save_path) and os.path.exists(hair_mask_save_path):
        return
    with open(face_bbox_path,'r') as f:
        face_bboxes = json.load(f) 
    if len(face_bboxes) == 0:
        return
    img = cv2.imread(img_path)
    h_img,w_img,_ = img.shape
    ori_mask = np.load(ori_mask_path)
    face_res_numpy = np.ones(img.shape[:2]) * (-1)
    hair_res_numpy = np.ones(img.shape[:2]) * (-1)
    save_flag = 0
    for id,face_bbox in face_bboxes.items():
        sub_mask = np.where(ori_mask == int(id), 255, 0).astype(np.uint8)
        image = cv2.bitwise_and(img, img, mask=sub_mask)
        image = torch.from_numpy(np.array(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))
        # image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        image = facer.hwc2bchw(image).to(device=device)
        with torch.inference_mode():
            faces = face_detector(image)
        if len(faces['rects']) == 0:
            print('undetect face')
            continue

        with torch.inference_mode():
            faces = face_parser(image, faces)

        seg_logits = faces['seg']['logits']
        seg_probs = seg_logits.softmax(dim=1)  # nfaces x nclasses x h x w
        n_classes = seg_probs.size(1)
        vis_seg_probs = seg_probs.argmax(dim=1).float()/n_classes*255
        vis_img = vis_seg_probs.sum(0, keepdim=True)
        a = facer.bchw2hwc(vis_img.unsqueeze(1))
        b = a.to(torch.uint8)[:,:,0].cpu()
        face_list = [23,46,69,92,115,139,162,185,208]
        mask = np.isin(b.numpy(), face_list).astype(np.uint8)
        mask = check_mask(mask, w_img, h_img)
        if mask is None:
            return

        face_res_numpy = np.where(mask, int(id), face_res_numpy)

        # mask = np.repeat(mask,3,2).astype(np.uint8)
        # pimage = Image.fromarray(mask)
        # pimage.save(f'./face_{id}.png')

        hair_mask = np.where(b.numpy() == 231,255,0).astype(np.uint8)
        # import pdb;pdb.set_trace()
        hair_mask = check_mask(hair_mask, w_img, h_img)
        if hair_mask is None:
            return

            # import pdb;pdb.set_trace()

        hair_res_numpy = np.where(hair_mask == 255, int(id), hair_res_numpy)
        save_flag = 1
    if save_flag:
        os.makedirs(hair_mask_save_dir, exist_ok=True)
        os.makedirs(face_mask_save_dir, exist_ok=True)
        face_res_numpy = face_res_numpy.astype(np.uint8)
        hair_res_numpy = hair_res_numpy.astype(np.uint8)
        face_sparse_mask = csr_matrix(face_res_numpy)
        hair_sparse_mask = csr_matrix(hair_res_numpy)
        scipy.sparse.save_npz(face_mask_save_path, face_sparse_mask)
        scipy.sparse.save_npz(hair_mask_save_path, hair_sparse_mask)

def fill_queue(face_bbox_root_dir, task_queue, hair_mask_root_dir, face_mask_root_dir, slice_num, which_piece):
    i = 0
    file_list = []
    for root,_,files in os.walk(face_bbox_root_dir):
        relative_path = os.path.relpath(root, face_bbox_root_dir)
        # if i>1600:
        #     break
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                head_mask_path = os.path.join(hair_mask_root_dir, relative_path, file.replace('.txt','.npz'))
                face_mask_path = os.path.join(face_mask_root_dir, relative_path, file.replace('.txt','.npz'))
                if os.path.exists(head_mask_path) and os.path.exists(face_mask_path):
                    print(f'重复：{head_mask_path}')
                    continue
                i += 1
                file_list.append(file_path)
                # if i > 1600:
                #     break
    print(f'total:{i}')
    for file in file_list[int(i*which_piece/slice_num):int(i*(which_piece+1)/slice_num)]:
        task_queue.put(file)

def init_models(gpu_num):
    # device = torch.device(f"cuda:{1}")
    # models = LangSAM(device = device)
    face_detector_models = []
    face_parser_models = []
    for i in range(gpu_num):
        device = f'cuda:{i}' if torch.cuda.is_available() else 'cpu'
        face_detector = facer.face_detector('retinaface/mobilenet', device=device)
        face_parser = facer.face_parser('farl/lapa/448', device=device) # optional "farl/celebm/448"
        face_detector_models.append(face_detector)
        face_parser_models.append(face_parser)
    return face_detector_models,face_parser_models

def worker(i, face_detector, face_parser, task_queue, ori_mask_dir, image_root_dir, face_bbox_root_dir, hair_mask_root_dir, lock, total, text_prompt="person"):
    torch.cuda.set_device(i)
    device = f'cuda:{i}' if torch.cuda.is_available() else 'cpu'
    global process_count
    while True:
        try:
            face_bbox_path = task_queue.get()
            if face_bbox_path is None:
                # 停止信号
                break
            t1 = time.time()
            relative_path = os.path.relpath(face_bbox_path, face_bbox_root_dir).rsplit('/', 1)[0]
            img_path = face_bbox_path.replace(face_bbox_root_dir,image_root_dir)[:-4]+'.jpg'
            hair_mask_save_dir = os.path.join(hair_mask_root_dir, relative_path)#.replace('vcg_images_122W', 'vcg_images_122W_1')
            face_mask_save_dir = hair_mask_save_dir.replace('hair_mask', 'face_mask')
            ori_mask_path = os.path.join(ori_mask_dir,relative_path)
            # os.makedirs(hair_mask_save_dir, exist_ok=True)
            # os.makedirs(face_mask_save_dir, exist_ok=True)
            # import pdb;pdb.set_trace()

            process_single_img(face_detector, face_parser, img_path, ori_mask_path, face_mask_save_dir, hair_mask_save_dir, face_bbox_path, device)
            cost_time = time.time() - t1
            with lock:
                process_count += 1
                print(f"{process_count} ==>> {total},cost: {cost_time}.{img_path}")
        except Exception as e:
            print(f"An error occurred: {e}")
            # logging.error(f"处理项目 {img_path} 时发生错误: {e}")
        finally:
            task_queue.task_done()
    task_queue.task_done()

if __name__ == '__main__':
    import argparse
    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description='facer')

    # 添加参数
    parser.add_argument('--slice_num', default=1, type=int, help='slice num')
    parser.add_argument('--which_piece', default=0, type=int, help='choose which piece')
    parser.add_argument("--dataset_root_path", type=str, required=True)
    args = parser.parse_args()
    slice_num = args.slice_num
    which_piece = args.which_piece
    
    image_root_dir = args.dataset_root_path
    ori_mask_dir = os.path.join(image_root_dir, 'mask')
    face_mask_root_dir = os.path.join(image_root_dir, 'face_mask_npz')
    hair_mask_root_dir = os.path.join(image_root_dir, 'hair_mask_npz')
    face_bbox_root_dir = os.path.join(image_root_dir, 'face_bbox_json')

    process_count = 0  # 初始化处理次数计数器
    # 初始化锁
    counter_lock = threading.Lock()
    task_queue = Queue()
    fill_queue(face_bbox_root_dir, task_queue, hair_mask_root_dir, face_mask_root_dir, slice_num, which_piece)
    total_num = task_queue.qsize()
    # import pdb;pdb.set_trace()
    print(total_num)
    max_workers = 8
    face_detector_models,face_parser_models = init_models(max_workers)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 启动工作线程
        for i,(face_detector, face_parser) in enumerate(zip(face_detector_models,face_parser_models)):
            executor.submit(worker, i, face_detector, face_parser, task_queue, ori_mask_dir, image_root_dir, face_bbox_root_dir, hair_mask_root_dir, counter_lock, total_num)

        # 等待所有文件被处理
        task_queue.join()
        
        # 发送停止信号
        for _ in range(max_workers):
            task_queue.put(None)
    
    # logging.info("所有任务已完成。")

print('all done')