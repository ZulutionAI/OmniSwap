import argparse
import glob
import os
from tqdm import tqdm
import concurrent.futures

parser = argparse.ArgumentParser()
parser.add_argument('--input', help='Input image dir')
parser.add_argument('--output_root', '--output-root', default=None, help='Path to output dir')
parser.add_argument('--process_type', default="mask", help='Process type')

args = parser.parse_args()

def find_images(input_path):
    img_list = []
    for dirpath, _, filenames in os.walk(input_path, followlinks=True):
        for filename in filenames:
            if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
                img_list.append(os.path.join(dirpath, filename))
    return img_list

img_list = find_images(args.input)
if len(img_list) == 0:
    img_list = glob.glob(args.input + "/*.jpg", recursive=True) + glob.glob(args.input + "/*.png", recursive=True)

exist_num = 0
not_exist_num = 0

def process_img(img_path):
    output_dir = os.path.join(args.output_root, os.path.dirname(img_path.split('/img/')[-1]))
    if args.process_type == "mask":
        output_file = os.path.join(output_dir, os.path.basename(img_path).replace('.jpg', '.png').replace('.jpeg', '.png').replace('.png', '_seg.npz'))
    elif args.process_type == "pose":
        output_file = os.path.join(output_dir, os.path.basename(img_path).replace('.jpg', '.json').replace('.jpeg', '.json').replace('.png', '.json'))
    else:
        raise ValueError("Invalid type")
    if os.path.exists(output_file):
        return None
    return img_path

with concurrent.futures.ThreadPoolExecutor(max_workers=30) as executor:
    results = list(tqdm(executor.map(process_img, img_list), total=len(img_list)))

with open(args.output_root + "/img_list.txt", "w") as f:
    for result in tqdm(results):
        if result is not None:
            not_exist_num += 1
            f.write(result + "\n")
        else:
            exist_num += 1
print(f"exist_num: {exist_num}, not_exist_num: {not_exist_num}, total: {exist_num + not_exist_num}")