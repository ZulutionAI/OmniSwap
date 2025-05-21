import threading
from queue import Queue
import json
import os
import time
import glob
import torch

class MultiThreadsWithModelBase:
    def __init__(self, dataset_root_path, total_segments, which_segment, gpu_list=[0,1,2,3,4,5,6,7], queue_max_size=2000):
        self.thread_count = len(gpu_list)
        self.task_queue = Queue(maxsize=queue_max_size)
        self.threads = []
        self.producer_thread = None
        self.lock = threading.Lock()
        self.process_count = 0
        self.total_segments = int(total_segments)
        self.which_segment = int(which_segment)
        self.total_dataset = 0
        self.gpu_list = gpu_list

        #path
        self.dataset_root_path = dataset_root_path
        self.img_dir_path = os.path.join(dataset_root_path, 'img')
        self.json_dir_path = os.path.join(dataset_root_path, 'json')
        self.mask_dir_path = os.path.join(dataset_root_path, 'mask')
        self.filter_json_dir_path = os.path.join(dataset_root_path, 'filter_json')
        self.delete_json_dir_path = os.path.join(dataset_root_path, 'delete_json')
        self.json_w_gender_age_dir_path = os.path.join(dataset_root_path, 'json_w_gender_age')
        self.skeleton_dir_path = os.path.join(dataset_root_path, 'skeleton')
        self.face_mask_dir_path = os.path.join(dataset_root_path,'face_mask')
        self.hair_mask_dir_path = os.path.join(dataset_root_path,'hair_mask')

    def model_init(self, gpu_id):
        """
        用户应该在子类中重写这个方法来初始化模型。
        """
        raise NotImplementedError("Must implement model_init method.")

    def single_thread_infer(self, model, data):
        """
        用户应该在子类中重写这个方法，定义单线程中的推理逻辑。
        """
        raise NotImplementedError("Must implement single_thread_infer method.")

    def worker(self, i):
        """
        工作线程的执行逻辑。
        """
        torch.cuda.set_device(int(i))
        model = self.model_init(i)

        while True:
            # 从任务队列中获取任务
            data = self.task_queue.get()
            if data is None:
                # None作为任务的结束信号
                self.task_queue.task_done()
                break
            # 执行单线程推理
            try:
                # import pdb;pdb.set_trace()
                self.single_thread_infer(model, data)
            except Exception as e:
                print(f"An error occurred: {e}")
            finally:
                self.task_queue.task_done()

    def start_threads(self):
        """
        启动所有工作线程。
        """
        # import pdb;pdb.set_trace()
        for i in self.gpu_list:
            thread = threading.Thread(target=self.worker, args=(i,))
            thread.start()
            self.threads.append(thread)
    
    def producer(self):
        """
        生产者线程
        """
        self.producer_thread = threading.Thread(target=self.add_task)
        self.producer_thread.start()

    def add_task(self):
        """
        向任务队列中添加任务。
        """
        raise NotImplementedError("Must implement add_task method.")

    def wait_completion(self):
        """
        等待所有任务完成。
        """
        self.producer_thread.join()
        self.task_queue.join()
        # 发送结束信号
        for _ in self.threads:
            self.task_queue.put(None)
        for thread in self.threads:
            thread.join()

class MyDataProcessor(MultiThreadsWithModelBase):
    def model_init(self, gpu_id):
        """
        实现模型初始化逻辑。
        """
        # 假设这里加载了一个模型
        model = "my_model"
        return model
    
    # def add_task(self):
    #     i = 0
    #     for root,_,files in os.walk(self.delete_json_path):
    #         if i > 100:
    #             break
    #         for file in files:
    #             if not file.endswith('.json'):
    #                 continue
    #             file_path = os.path.join(root, file)
    #             self.task_queue.put(file_path)
    #             i+=1
    #             print(f'producer:{i}')
    #             if i > 100:
    #                 break
    #     for _ in range(self.thread_count):  # 添加结束信号
    #         self.task_queue.put(None)
    def add_task(self):
        file_lists = glob.glob(self.json_dir_path+'/**/*.json',recursive=True)
        total = len(file_lists)
        print(f'total:{total}')
        i = 0
        for file in file_lists[int(total*self.which_segment/self.total_segments):int(total*(self.which_segment+1)/self.total_segments)]:
            i+=1
            if i>10:
                break
            self.task_queue.put(file)
        

    def single_thread_infer(self, file_path):
        """
        实现单线程推理逻辑。
        """
        time.sleep(0.2)
        f = open(file_path,'r')
        dict_people = json.load(f)
        segments = dict_people['segments']
        for segment in segments:
            # import pdb;pdb.set_trace()
            if 'is_clear' in segment:
                if segment['is_clear'] == 'no':
                    # print(segment)
                    with self.lock:
                        queue_size = self.task_queue.qsize()
                        self.process_count += 1
                        print(f'{self.process_count} ==> {queue_size} ==> {file_path}')
                        return
               
        with self.lock:
            queue_size = self.task_queue.qsize()
            self.process_count += 1
            print(f'{self.process_count} ==> {queue_size} {file_path}')
            return

# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--model-path", type=str, default="./models/llava-v1.6-34b/")
#     parser.add_argument("--dataset_root_path", type=str)
#     parser.add_argument("--total_segments", type=str, default=1)
#     parser.add_argument("--which_segment", type=str, default=0)
    

#     args = parser.parse_args()
#     dataset_root_path = args.dataset_root_path
#     total_segments = args.total_segments
#     which_segment = args.which_segment
#     print(dataset_root_path)

#     processor = MyDataProcessor(dataset_root_path, total_segments, which_segment,thread_count=4, queue_max_size=50)
#     processor.producer()
#     processor.start_threads()
#     processor.wait_completion()
