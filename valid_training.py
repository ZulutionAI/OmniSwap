import os
import time
import pickle
import traceback
import torch
import numpy as np
from src.flux.xflux_pipelinev2 import XFluxPipeline
from PIL import Image, ImageOps
import cv2
from einops import rearrange
import math 
import glob 
import random
from omegaconf import OmegaConf
import face_recognition
import copy
import json
import argparse
from torchvision import models
import torchvision.transforms as transforms

from src.flux.modules.facerecog_model import IR_101
from insightface.app import FaceAnalysis
import insightface.model_zoo
from transformers import CLIPImageProcessor

from valid_utils import draw_bodypose, convert_open_to_mmpose, calculate_mask_external_rectangle, get_new_size, canny_processor, pad_bbox, combine_canny_skeleton, prepare_videos

clip_image_processor = CLIPImageProcessor(crop_size={"height": 336, "width": 336}, size={"shortest_edge": 336})

def init_face_model():
    print("Loading face recognition model")
    facerecog_model = IR_101([112, 112])
    facerecog_model.load_state_dict(torch.load("checkpoints/CurricularFace/CurricularFace_Backbone.pth"))
    facerecog_model.requires_grad_(False)
    facerecog_model.eval()
    
    print("Loading handler_ante model")
    handler_ante = insightface.model_zoo.get_model('checkpoints/antelopev2/glintr100.onnx')
    handler_ante.prepare(ctx_id=0)
    return facerecog_model, handler_ante

def split_clothes(reference_image):
    clothes_image_list = [reference_image]
    w, h = reference_image.size
    if w > h:
        block = math.ceil(w / h)
        overlap = block * h - w
        if overlap > 0:
            overlap = overlap / (block - 1)
        
        block_size = h - overlap


        for i in range(block):
            box = (i * block_size, 0, i * block_size + h, h)
            clothes_image_list.append(reference_image.crop(box))
    else:
        block = math.ceil(h / w)
        overlap = block * w - h
        if overlap > 0:
            overlap = overlap / (block - 1)
        for i in range(block):
            clothes_image_list.append(reference_image.crop((0, i * (w - overlap), w, i * (w - overlap) + w)))
    print("Do clothes_image_list:", len(clothes_image_list))
    return clothes_image_list

def calculate_faces_embedding(facerecog_model, handler_ante, reference_image):
    # feture 1
    reference_image_np = np.array(reference_image)
    if reference_image_np.shape[2] == 4:
        reference_image_np = reference_image_np[:, :, :3]
    face_embeddings = torch.from_numpy(handler_ante.get_feat(reference_image_np))
    # feature 2
    facerecog_image = reference_image_np
    facerecog_image = cv2.resize(facerecog_image, (112, 112))
    facerecog_image = torch.from_numpy(facerecog_image).permute(2, 0, 1) / 255
    facerecog_image = torch.clamp((facerecog_image * 2) - 1, -1, 1)  # [-1, 1]
    id_embedding_circular = facerecog_model(facerecog_image.unsqueeze(0), return_mid_feats=True)[0]
    if id_embedding_circular is None:
        id_embedding_circular = torch.zeros(1, 512)
    
    faces_embedding = [[face_embeddings], [id_embedding_circular]]
    return faces_embedding

def load_checkpoint(checkpoint_path, args):
    global global_args
    xflux_pipeline = XFluxPipeline('flux-dev', 'cuda', False)

    ip_local_path = os.path.join(checkpoint_path, "ip_adaptor.safetensors")
    ip_name = "flux-ip-adapter.safetensors"
    feature_type = args.feature_type
    if "ip_type" in args:
        ip_type = args.ip_type
    else:
        ip_type = "v3"
    if "image_encoder_path" in args:
        image_encoder_path = args.image_encoder_path
    else:
        image_encoder_path = None
    if "clip_image_size" in args.data_config:
        clip_image_size = args.data_config.clip_image_size
    else:
        clip_image_size = 224

    xflux_pipeline.set_ipv5(ip_local_path, None, ip_name, feature_type, ip_type, image_encoder_path, clip_image_size, drop_local=False)
    # xflux_pipeline.set_ipv6(repo_id=None, name=ip_name, feature_type=feature_type, model_type=ip_type, image_encoder_path=image_encoder_path, clip_image_size=clip_image_size, flag=flag)
    control_local_path = os.path.join(checkpoint_path, "ip_adaptor_controlnet.safetensors")
    # control_local_path = os.path.join(checkpoint_path, "controlnet.safetensors")
    control_type = "canny"
    xflux_pipeline.set_controlnet(control_type, control_local_path, None, None, True, controlnet_depth=args.control_depth) # type: ignore
    
    return xflux_pipeline

def run_inference(xflux_pipeline, args):
    global clip_image_processor
    facerecog_model, handler_ante = init_face_model()
                
    face_list = glob.glob("valid_data/origin/face/**/*.jpg", recursive=True) + glob.glob("valid_data/origin/face/**/*.png", recursive=True)
    face_list = sorted(face_list)
    clothes_list = glob.glob("valid_data/origin/clothes/**/*.jpg", recursive=True) + glob.glob("valid_data/origin/clothes/**/*.png", recursive=True)
    clothes_list = sorted(clothes_list)
    if args.data_config.do_only_cloth:
        img_list = clothes_list
    elif args.feature_type == "clip_pooling_local_mix_face":
        img_list = face_list
    else:
        img_list = face_list + clothes_list
    print("img_list:", len(img_list), "face_list:", len(face_list), "clothes_list:", len(clothes_list))
    
    results = []
    for img_path in img_list:
        original_image = img_path
        img = Image.open(original_image)
        img = img.convert("RGB")
        raw_image = img.copy()
        o_width = img.width
        o_height = img.height
        width, height = get_new_size(o_width, o_height)
        print(img_path, o_width, o_height, width, height)
        img = img.resize((width, height))
        
        # control img
        skeleotn_path = original_image.replace('/origin/', '/process_img/skeleton/').replace('.png', '.jpg').replace('.jpg', '.json')
        if os.path.exists(skeleotn_path):
            keypoints = []
            with open(skeleotn_path, "r") as f:
                data = json.load(f)
            for segment in data["segments"]:
                if segment['skeleton'] is None:
                    continue
                combined_keypoints = []
                for kp, score in zip(segment['skeleton']['keypoints'], segment['skeleton']['keypoint_scores']):
                    combined_keypoints.append([kp[0] / o_width * width, kp[1] / o_height * height] + [score])
                keypoints.append(np.array(combined_keypoints))
            keypoints = [keypoint for keypoint in keypoints if keypoint is not None]
            canvas = np.zeros((height, width, 3))
            for keypoint in keypoints:
                keypoint = convert_open_to_mmpose(keypoint)
                canvas = draw_bodypose(canvas, keypoint, min_conf=0.3)
            canvas = Image.fromarray(canvas.astype(np.uint8))
            control_image = canvas
        else:
            control_image = canny_processor(img)
        control_weight = 0.8

        mask = original_image.replace("origin", "mask").replace(".jpg", ".png")
        mask = Image.open(mask)
        o_mask = mask.copy()
        o_mask = np.array(o_mask).astype(bool)
        mask = mask.resize((width, height))
        mask = np.array(mask)
        mask = mask.astype(bool)

        mask_image = np.array(img)
        mask_image[mask] = 0
        mask_image = Image.fromarray(mask_image)
        mask_image = mask_image.resize((width, height))

        ip_atten_mask_list = []
        ip_atten_mask = cv2.resize(np.array(mask).astype(np.uint8), (width//16, height//16))
        ip_atten_mask_list.append(torch.from_numpy(ip_atten_mask))
        ip_atten_mask = torch.stack(ip_atten_mask_list, dim=0)
        if ip_atten_mask is not None:
            ip_atten_mask[ip_atten_mask > 0] = 1
            ip_atten_mask[ip_atten_mask <= 0.1] = 0
            print(ip_atten_mask.shape, ip_atten_mask.sum())
            ip_atten_mask = ip_atten_mask.to(torch.bool)
            print("ip_atten_mask:", ip_atten_mask.shape)
            ip_atten_mask = [rearrange(ip_atten_mask[i], "h w -> (h w)").reshape(1, -1) for i in range(ip_atten_mask.shape[0])]
        
        if "face" in original_image:
            reference_img_list = glob.glob(os.path.dirname(original_image).replace("origin", "reference") + "/*.jpg") + glob.glob(os.path.dirname(original_image).replace("origin", "reference") + "/*.png")
        else:
            reference_img_list = glob.glob(os.path.dirname(original_image).replace("origin", "reference").replace("/anime", "").replace("/real", "") + "/*.jpg") + glob.glob(os.path.dirname(original_image).replace("origin", "reference").replace("/anime", "").replace("/real", "") + "/*.png")
        
        for reference_image_path in reference_img_list:
            prompt = ""
            
            reference_image = Image.open(reference_image_path)
            
            # if "face" in reference_image_path:
            #     reference_face_bbox_path = reference_image_path.replace('/origin/', '/process_img/face_bbox/').replace('.png', '.json').replace('.jpg', '.json')
            #     with open(reference_face_bbox_path, "r") as f:
            #         reference_face_bbox = json.load(f)
            #     reference_face_bbox = pad_bbox(reference_face_bbox[0])
            #     reference_image = reference_image.crop((reference_face_bbox[0], reference_face_bbox[1], reference_face_bbox[2], reference_face_bbox[3]))
                
            reference_image = reference_image.convert("RGB")
            noise = np.random.randint(0, 256, size=(int(reference_image.height+reference_image.height*0.1), int(reference_image.width+reference_image.width*0.1), 3), dtype=np.uint8)
            noise = Image.fromarray(noise)
            noise.paste(reference_image, (int(reference_image.width*0.05), int(reference_image.height*0.05)))
            image_prompt_list = [noise]
            
            # 衣服滑窗
            if args.data_config.do_only_cloth and "clothes" in reference_image_path:
                image_prompt_list = split_clothes(image_prompt_list[0])

            faces_embedding = calculate_faces_embedding(facerecog_model, handler_ante, reference_image)

            # face_type
            if "face" in reference_image_path:
                face_type = torch.ones(1)
            else:
                face_type = torch.zeros(1)
            
            if args.data_config.do_only_cloth:
                do_dinov2 = True         
            else:
                do_dinov2 = False
            
            result = xflux_pipeline(
                prompt=prompt,
                controlnet_image=control_image,
                controlnet_mask_image=mask_image, 
                width=width,
                height=height,
                guidance=4,
                num_steps=25,
                seed=123456789,
                true_gs=args.guidance_vec,
                control_weight=control_weight,
                neg_prompt="lowres, text, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, username, watermark, signature",
                timestep_to_start_cfg=1 if args.guidance_vec != 1 else 100,
                image_prompt_list=image_prompt_list,
                neg_image_prompt=None,
                ip_scale=1.0,
                neg_ip_scale=1.0,
                feature_type=args.feature_type,
                faces_embedding=faces_embedding,
                ip_atten_mask=ip_atten_mask,
                do_vae=True,
                do_inpainting=True,
                face_type=face_type,
                do_dinov2=do_dinov2,
            ) # type: ignore

            result = np.array(result)
            img_array = np.array(img)
            mask_array = np.array(mask_image)
            control_array = np.array(control_image)

            target_height = result.shape[0]
            if img_array.shape[0] != target_height:
                img_array = cv2.resize(img_array, (img_array.shape[1], target_height))
            if mask_array.shape[0] != target_height:
                mask_array = cv2.resize(mask_array, (mask_array.shape[1], target_height))
            if control_array.shape[0] != target_height:
                control_array = cv2.resize(control_array, (control_array.shape[1], target_height))

            combined = np.hstack([result, img_array, mask_array, control_array])
            # if skeleton_image is not None:
            #     skeleton_image = np.array(skeleton_image)
            #     skeleton_image = cv2.resize(skeleton_image, (control_array.shape[1], target_height))
            #     combined = np.concatenate([combined, skeleton_image], axis=1)
            # 高相同，宽等比例放缩
            reference_array = np.array(image_prompt_list[0])
            reference_array = cv2.resize(reference_array, (int(combined.shape[0] * reference_array.shape[1] / reference_array.shape[0]), combined.shape[0]))
            if reference_array.shape[2] != combined.shape[2]:
                reference_array = reference_array[:, :, :combined.shape[2]]
            combined = np.concatenate([reference_array, combined], axis=1)
            result = combined
            result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            o_img_name = os.path.basename(original_image)
            r_img_name = os.path.basename(reference_image_path)
            os.makedirs(os.path.join(args.save_dir, "valid_img"), exist_ok=True)
            cv2.imwrite(os.path.join(args.save_dir, "valid_img", f"{o_img_name}_{r_img_name}.webp"), result)
            # cv2.imwrite(f"/mnt2/zhenghaoyu/code/x-flux.bk/valid_datav2/result/{o_img_name}_{r_img_name}.jpg", result)
            # [2,6], [1,0], [0,9], [2,2], [6,6], [15,12], [12,12]
            # if o_img_name == "origin_2.jpg" and r_img_name == "reference_6.jpg" or o_img_name == "origin_1.jpg" and r_img_name == "reference_0.jpg" or o_img_name == "origin_0.jpg" and r_img_name == "reference_9.jpg" or o_img_name == "origin_2.jpg" and r_img_name == "reference_2.jpg" or o_img_name == "origin_6.jpg" and r_img_name == "reference_6.jpg" or o_img_name == "origin_15.jpg" and r_img_name == "reference_12.jpg" or o_img_name == "origin_12.jpg" and r_img_name == "reference_12.jpg":
            results.append(result)

    return results

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints/checkpoint-cloth")
    parser.add_argument("--config_path", type=str, default="train_configs/inpaint_cloth.yaml")
    parser.add_argument("--save_dir", type=str, default="save")
    return parser.parse_args()

if __name__ == '__main__':
    global_args = parse_args()
    args = OmegaConf.load(global_args.config_path)
    pipeline = load_checkpoint(global_args.checkpoint_path, args)
    args.save_dir = global_args.save_dir
    run_inference(pipeline, args)
