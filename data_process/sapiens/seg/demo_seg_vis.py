# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from argparse import ArgumentParser

import torch
import json
import copy
from mmengine.model import revert_sync_batchnorm
from mmseg.apis import inference_model, init_model, show_result_pyplot
import os
from tqdm import tqdm
import cv2
import numpy as np
import torchvision
from scipy.sparse import csr_matrix
import scipy.sparse
torchvision.disable_beta_transforms_warning()

def main():
    parser = ArgumentParser()
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--input', help='Input image dir')
    parser.add_argument('--output_root', '--output-root', default=None, help='Path to output dir')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--opacity',
        type=float,
        default=0.5,
        help='Opacity of painted segmentation map. In (0, 1] range.')
    parser.add_argument(
        '--title', default='result', help='The image identifier.')
    parser.add_argument('--vis', action='store_true')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device=args.device)
    if args.device == 'cpu':
        model = revert_sync_batchnorm(model)

    input = args.input
    image_names = []

    # Check if the input is a directory or a text file
    if os.path.isdir(input):
        input_dir = input  # Set input_dir to the directory specified in input
        image_names = [image_name for image_name in sorted(os.listdir(input_dir))
                    if image_name.endswith('.jpg') or image_name.endswith('.png') or image_name.endswith('.jpeg')]
    elif os.path.isfile(input) and input.endswith('.txt'):
        # If the input is a text file, read the paths from it and set input_dir to the directory of the first image
        with open(input, 'r') as file:
            image_paths = [line.strip() for line in file if line.strip()]
        print(f"Got image paths from txt file{input[-5]}!")
        image_names = [os.path.basename(path) for path in image_paths]  # Extract base names for image processing
        input_dir = os.path.dirname(image_paths[0]) if image_paths else ''  # Use the directory of the first image path

    if not os.path.exists(args.output_root):
        os.makedirs(args.output_root)

    for i, image_path in tqdm(enumerate(image_paths), total=len(image_paths)):
        torch.cuda.empty_cache()
        try:
            if image_path[0]!='/':
                continue
            # image_path = os.path.join(input_dir, image_name)
            image_path = image_path.replace('.json', '.jpg').replace('/json/', '/img/')
            if not os.path.exists(image_path):
                image_path = image_path.replace('.jpg', '.jpeg')
            if not os.path.exists(image_path):
                image_path = image_path.replace('.jpeg', '.png')
            output_dir = os.path.join(args.output_root, os.path.dirname(image_path.split('/img/')[-1]))
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, os.path.basename(image_path).replace('.jpg', '.png').replace('.jpeg', '.png').replace('.png', '.npz'))
            output_seg_file = os.path.join(output_dir, os.path.basename(image_path).replace('.jpg', '.png').replace('.jpeg', '.png').replace('.png', '_seg.npz'))

            if os.path.exists(output_file):
                continue

            result = inference_model(model, image_path)
            pred_sem_seg = result.pred_sem_seg.data[0].cpu().numpy() ## H x W. seg ids.
            mask = (pred_sem_seg > 0).astype(np.uint8)

            if args.vis:
                image = cv2.imread(image_path)
                vis_image = show_result_pyplot(
                    model,
                    image_path,
                    result,
                    title=args.title,
                    opacity=args.opacity,
                    draw_gt=False,
                    show=False,
                    out_file=None)
                vis_image = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
                vis_image = np.concatenate([image, vis_image], axis=1)
                output_file_vis = os.path.join(output_dir, os.path.basename(image_path))
                cv2.imwrite(output_file_vis, vis_image)

            sparse_mask = csr_matrix(mask)
            sparse_pred_sem_seg = csr_matrix(pred_sem_seg)
            
            scipy.sparse.save_npz(output_file, sparse_mask)
            scipy.sparse.save_npz(output_seg_file, sparse_pred_sem_seg)
        except Exception as e:
            print(image_path, e)
            continue

if __name__ == '__main__':
    main()
