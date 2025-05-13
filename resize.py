import glob
import cv2
import os
from tqdm import tqdm
import json
import itertools
import pydantic
import httpx
import os
import json
import subprocess
import random
import re
import logging
from dotenv import load_dotenv
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import argparse

def resize_image_opencv(input_path, output_path, max_size):
    img = cv2.imread(input_path)
    height, width = img.shape[:2]
    # 计算缩放比例
    scale = max_size / max(width, height)
    if scale < 1:
        new_width = int(width * scale)
        new_height = int(height * scale)
        resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
        cv2.imwrite(output_path, resized_img)
    #     print(f"图片已缩放至 {new_width}x{new_height}")
    # else:
    #     print("图片尺寸无需调整")
    # print(f"{img.shape} ==>> {resized_img.shape}")
all_path = glob.glob("/cv/wangxuekuan/code/final/flux-inpainting-ipa/data/anime_pictures_46w_new/img/*/*/*")
print(len(all_path))


# for path in tqdm(all_path):

def do_task(path):
    # im_data = cv2.imread(path)
    output_path = path.replace("anime_pictures_46w", "anime_pictures_46w_new")
    os.makedirs(os.path.dirname(output_path), exist_ok=True) 
    resize_image_opencv(path, output_path, 1080)

# with ThreadPoolExecutor(max_workers=100) as executor:
#         futures = [executor.submit(do_task, url.strip()) for idx, url in enumerate(all_path)]
#         with tqdm(total=len(futures), desc="Processing:") as pbar:
#             for future in as_completed(futures):
#                 try:
#                     future.result()
#                 except Exception as e:
#                     print(f"Error processing: {e}")
#                 pbar.update(1)

