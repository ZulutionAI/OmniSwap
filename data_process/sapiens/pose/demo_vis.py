# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import mimetypes
import os
import time
from argparse import ArgumentParser

import cv2
import json_tricks as json
import mmcv
import mmengine
import numpy as np
import scipy
import copy
import traceback
from tqdm import tqdm
import warnings
from mmpose.apis import inference_topdown
from mmpose.apis import init_model as init_pose_estimator
from mmpose.evaluation.functional import nms
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples, split_instances
from mmpose.utils import adapt_mmdet_pipeline

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False

warnings.filterwarnings("ignore", category=UserWarning, module='torchvision')
warnings.filterwarnings("ignore", category=UserWarning, module='mmengine')
warnings.filterwarnings("ignore", category=UserWarning, module='torch.functional')
warnings.filterwarnings("ignore", category=UserWarning, module='json_tricks.encoders')

def process_one_image(args,
                      img,
                      detector,
                      pose_estimator,
                      bboxes=None,
                      visualizer=None,
                      show_interval=0):
    """Visualize predicted keypoints (and heatmaps) of one image."""
    if bboxes is None:
        det_result = inference_detector(detector, img)
        pred_instance = det_result.pred_instances.cpu().numpy()
        bboxes = np.concatenate(
            (pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
        bboxes = bboxes[np.logical_and(pred_instance.labels == args.det_cat_id,
                                    pred_instance.scores > args.bbox_thr)]
        # args.nms_thr = 0.8
        bboxes = bboxes[nms(bboxes, args.nms_thr), :4]

    # predict keypoints
    pose_results = inference_topdown(pose_estimator, img, bboxes)
    data_samples = merge_data_samples(pose_results)

    # show the results
    if isinstance(img, str):
        img = mmcv.imread(img, channel_order='rgb')
    elif isinstance(img, np.ndarray):
        img = mmcv.bgr2rgb(img)

    if visualizer is not None:
        visualizer.add_datasample(
            'result',
            img,
            data_sample=data_samples,
            draw_gt=False,
            draw_heatmap=args.draw_heatmap,
            draw_bbox=args.draw_bbox,
            show_kpt_idx=args.show_kpt_idx,
            skeleton_style=args.skeleton_style,
            show=args.show,
            wait_time=show_interval,
            kpt_thr=args.kpt_thr)

    # if there is no instance detected, return None
    return data_samples.get('pred_instances', None)


def main():
    """Visualize the demo images.

    Using mmdet to detect the human.
    """
    parser = ArgumentParser()
    parser.add_argument('det_config', help='Config file for detection')
    parser.add_argument('det_checkpoint', help='Checkpoint file for detection')
    parser.add_argument('pose_config', help='Config file for pose')
    parser.add_argument('pose_checkpoint', help='Checkpoint file for pose')
    parser.add_argument(
        '--input', type=str, default='', help='Image/Video file')
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='whether to show img')
    parser.add_argument(
        '--output-root',
        type=str,
        default='',
        help='root of the output img file. '
        'Default not saving the visualization images.')
    parser.add_argument(
        '--save-predictions',
        action='store_true',
        default=False,
        help='whether to save predicted results')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--det-cat-id',
        type=int,
        default=0,
        help='Category id for bounding box detection model')
    parser.add_argument(
        '--bbox-thr',
        type=float,
        default=0.3,
        help='Bounding box score threshold')
    parser.add_argument(
        '--nms-thr',
        type=float,
        default=0.3,
        help='IoU threshold for bounding box NMS')
    parser.add_argument(
        '--kpt-thr',
        type=float,
        default=0.3,
        help='Visualizing keypoint thresholds')
    parser.add_argument(
        '--draw-heatmap',
        action='store_true',
        default=False,
        help='Draw heatmap predicted by the model')
    parser.add_argument(
        '--show-kpt-idx',
        action='store_true',
        default=False,
        help='Whether to show the index of keypoints')
    parser.add_argument(
        '--skeleton-style',
        default='mmpose',
        type=str,
        choices=['mmpose', 'openpose'],
        help='Skeleton style selection')
    parser.add_argument(
        '--radius',
        type=int,
        default=3,
        help='Keypoint radius for visualization')
    parser.add_argument(
        '--thickness',
        type=int,
        default=1,
        help='Link thickness for visualization')
    parser.add_argument(
        '--show-interval', type=int, default=0, help='Sleep seconds per frame')
    parser.add_argument(
        '--alpha', type=float, default=0.8, help='The transparency of bboxes')
    parser.add_argument(
        '--draw-bbox', action='store_true', help='Draw bboxes of instances')

    assert has_mmdet, 'Please install mmdet to run the demo.'

    args = parser.parse_args()

    assert args.show or (args.output_root != '')
    assert args.input != ''
    assert args.det_config is not None
    assert args.det_checkpoint is not None

    output_file = None
    if args.output_root:
        mmengine.mkdir_or_exist(args.output_root)
        output_file = os.path.join(args.output_root,
                                   os.path.basename(args.input))
        if args.input == 'webcam':
            output_file += '.mp4'

    if args.save_predictions:
        assert args.output_root != ''
        args.pred_save_path = f'{args.output_root}/results_' \
            f'{os.path.splitext(os.path.basename(args.input))[0]}.json'

    # build detector
    detector = init_detector(
        args.det_config, args.det_checkpoint, device=args.device)
    detector.cfg = adapt_mmdet_pipeline(detector.cfg)

    # build pose estimator
    pose_estimator = init_pose_estimator(
        args.pose_config,
        args.pose_checkpoint,
        override_ckpt_meta=True, # dont load the checkpoint meta data, load from config file
        device=args.device,
        cfg_options=dict(
            model=dict(test_cfg=dict(output_heatmaps=args.draw_heatmap))))

    # build visualizer
    pose_estimator.cfg.visualizer.radius = args.radius
    pose_estimator.cfg.visualizer.alpha = args.alpha
    pose_estimator.cfg.visualizer.line_width = args.thickness
    visualizer = VISUALIZERS.build(pose_estimator.cfg.visualizer)
    # the dataset_meta is loaded from the checkpoint and
    # then pass to the model in init_pose_estimator
    visualizer.set_dataset_meta(pose_estimator.dataset_meta, skeleton_style=args.skeleton_style)

    input = args.input
    image_names = []

    # Check if the input is a directory or a text file
    if os.path.isdir(input):
        input_dir = input  # Set input_dir to the directory specified in input
        # image_names = [image_name for image_name in sorted(os.listdir(input_dir))
        #             if image_name.endswith('.jpg') or image_name.endswith('.png')]
    elif os.path.isfile(input) and input.endswith('.txt'):
        # If the input is a text file, read the paths from it and set input_dir to the directory of the first image
        with open(input, 'r') as file:
            image_paths = [line.strip() for line in file if line.strip()]
        # image_names = [os.path.basename(path) for path in image_paths]  # Extract base names for image processing
        # input_dir = os.path.dirname(image_paths[0]) if image_paths else ''  # Use the directory of the first image path

    for i, image_path in tqdm(enumerate(image_paths), total=len(image_paths)):
        try:
            if image_path[0]!='/':
                continue
            output_file = os.path.join(args.output_root, image_path.split('/img/')[-1])
            if os.path.exists(output_file):
                continue
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            if not os.path.exists(image_path):
                image_path = image_path.replace('.jpg', '.png')
            if not os.path.exists(image_path):
                image_path = image_path.replace('.png', '.jpeg')
            if not os.path.exists(image_path):
                image_path = image_path.replace('.jpeg', '.webp')
            img = cv2.imread(image_path)

            bboxes_path = output_file.replace(args.output_root.split('/')[-1], 'json').replace('.jpg', '.json').replace('.png', '.json').replace('.jpeg', '.json')
            # bboxes_path = image_path.replace('/img/', '/json/').replace('.jpg', '.json').replace('.png', '.json')
            with open(bboxes_path, 'r') as f:
                data = json.load(f)
                bboxes = np.array([data['segments'][i]['bbox'] for i in range(len(data['segments']))], dtype=np.float32)

            mask_path = output_file.replace(args.output_root.split('/')[-1], 'mask').replace('.jpg','.npz').replace('.png','.npz').replace('.jpeg', '.npz')
            # mask_path = image_path.replace('/img/', '/mask/').replace('.jpg','.npz').replace('.png','.npz')
            if os.path.exists(mask_path):
                loaded_sparse_mask = scipy.sparse.load_npz(mask_path)
                npy = loaded_sparse_mask.toarray()
            else:
                npy = np.load(mask_path.replace('.npz', '.npy'))

            all_pred_instances = []
            vis_pics = []
            for i, bbox in enumerate(bboxes):
                img_person = copy.deepcopy(img)
                people_mask = npy == i
                people_mask = ~people_mask
                img_person[people_mask] = 0

                pred_instances = process_one_image(args, img_person, detector, pose_estimator, [bbox], visualizer)
                all_pred_instances.append(pred_instances)

                img_vis = visualizer.get_image()
                vis_pics.append(mmcv.rgb2bgr(img_vis))
                # mmcv.imwrite(mmcv.rgb2bgr(img_vis), output_file.replace('.jpg', f'_{i}.jpg').replace('.png', f'_{i}.png'))

            # img_vis = np.concatenate([img] + vis_pics, axis=1)
            # mmcv.imwrite(img_vis, output_file)

            if args.save_predictions:
                pred_instances_list = []
                for pred_instances in all_pred_instances:
                    pred_instances_list.extend(split_instances(pred_instances))
                pred_save_path = os.path.join(output_file.replace('.jpg', '.json').replace('.png', '.json').replace('.jpeg', '.json'))

                for i, pred_instances in enumerate(pred_instances_list):
                    pred_instances['id'] = i

                with open(pred_save_path, 'w') as f:
                    json.dump(
                        dict(
                            meta_info=pose_estimator.dataset_meta,
                            instance_info=pred_instances_list),
                        f,
                        indent='\t')
        except Exception as e:
            # traceback.print_exc()
            print(f"Error processing image {image_path}: {e}")

if __name__ == '__main__':
    main()
