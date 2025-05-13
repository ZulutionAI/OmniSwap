import argparse
import logging
import math
import os
import re
import random
from PIL import Image
import shutil
from safetensors.torch import save_file
import datasets
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from tqdm.auto import tqdm
from omegaconf import OmegaConf
from copy import deepcopy
import diffusers
from diffusers.optimization import get_scheduler
from diffusers.utils import is_wandb_available
from einops import rearrange
from src.flux.sampling import prepare, unpack
from src.flux.util import (configs, load_ae, load_clip,
                       load_flow_model2, load_t5, load_controlnet, load_checkpoint)

from src.flux.modules.layers import (
    SingleStreamBlockProcessor,
    DoubleStreamBlockProcessor,
    SingleStreamBlockLoraProcessor,
    DoubleStreamBlockLoraProcessor,
    IPDoubleStreamBlockProcessor,
    IPDoubleStreamBlockProcessorV2,
    IPDoubleStreamBlockProcessorV3,
    IPDoubleStreamBlockProcessorV4,
    IPSingleStreamBlockProcessor,
    IPSingleStreamBlockProcessorV3,
    IPSingleStreamBlockProcessorV4,
    ImageProjModel,
    ImageProjModelLocal,
    ImageProjModelLocalB1,
    ImageProjModelLocalB0,
    ImageProjModelFaceLocal,
    ImageProjModelLocalB0_face,
    ImageProjModelLocalB2_face,
    ImageProjModelLocalB3_face
)
from src.flux.modules.projection_models import MLPMixIDModel, MLPMixIDModel_v2, MLPMixIDModelQFormer, MLPMixIDModelTransformer

from src.flux.ip_flux import IPFluxModelv2, IPFluxModelv3, IPFluxModelv7

from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
from src.flux.modules.facerecog_model import IR_101

from eva_clip import create_model_and_transforms


# from ip_adapter.facerecog_model import IR_101
# /mnt2/wangxuekuan/code/x-flux.bk/src/flux/modules/facerecog_model.py

##### from image_datasets.dataset import loader
# from image_datasets.ip_dataset_mask import loader
from image_datasets.ip_dataset_inpaintingv4 import loader

if is_wandb_available():
    import wandb
logger = get_logger(__name__, log_level="INFO")
 

def get_clip_embeddings(image_encoder, clip_images):
    # print(clip_images.shape, clip_images.dtype, clip_images.device)
    if image_encoder.__class__.__name__ == 'EVAVisionTransformer':
        with torch.no_grad():
            id_cond_vit, id_vit_hidden = image_encoder(
                clip_images, return_all_features=False, return_hidden=True, shuffle=False
            )
            image_embeds_pooling = id_cond_vit
            image_embeds = id_vit_hidden[-2]

    elif image_encoder.__class__.__name__ == 'CLIPVisionModelWithProjection':
        with torch.no_grad():
            image_embeds = image_encoder(clip_images, output_hidden_states=True).hidden_states[-2]
            image_embeds_pooling = image_encoder(clip_images, output_hidden_states=True).image_embeds

    elif image_encoder.__class__.__name__== 'SiglipVisionModel':
        with torch.no_grad():
            image_embeds = image_encoder(clip_images, output_hidden_states=True).hidden_states[-2]
            image_embeds_pooling = image_encoder(clip_images, output_hidden_states=True).pooler_output

    # print(image_embeds.shape, image_embeds_pooling.shape)
    return image_embeds, image_embeds_pooling

def get_models(name: str, device, offload: bool, is_schnell: bool):
    t5 = load_t5(device, max_length=256 if is_schnell else 512)
    clip = load_clip(device)
    clip.requires_grad_(False)
    model = load_flow_model2(name, device="cpu")
    vae = load_ae(name, device="cpu" if offload else device)

    return model, vae, t5, clip

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        required=True,
        help="path to config",
    )
    args = parser.parse_args()


    return args.config

def get_image_proj(
        image_prompt, image_encoder, feature_type="clip_pooling_local"
    ):
    return get_clip_embeddings(image_encoder, image_prompt)
#     # encode image-prompt embeds

#     if feature_type == "clip_pooling":
#         image_prompt_embeds_pooling = image_encoder(
#             image_prompt
#         ).image_embeds
#         image_prompt_embeds = None
        
#         #image_proj = improj(image_prompt_embeds)

#     #elif feature_type in ["clip_pooling_local", "clip_face", "clip_pooling_local_b0_prompt_face", "clip_pooling_local_b1", "clip_pooling_local_b0_prompt", "clip_pooling_local_b0", "clip_h_pooling_local_b1", "clip_h_pooling_local"]:
#     else:
#         output_encoder = image_encoder(image_prompt, output_hidden_states=True)
#         image_prompt_embeds = output_encoder.hidden_states[-2]
#         image_prompt_embeds_pooling = output_encoder.image_embeds

    
#     return image_prompt_embeds, image_prompt_embeds_pooling
def canny_processor(image, low_threshold=100, high_threshold=200):
    image = np.array(image)
    image = cv2.Canny(image, low_threshold, high_threshold)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    # canny_image = Image.fromarray(image)
    return image

def main():
    args = OmegaConf.load(parse_args())
    is_schnell = args.model_name == "flux-schnell"
    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()


    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        args.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        args.mixed_precision = accelerator.mixed_precision
    
    from image_datasets.ip_dataset_inpaintingv10 import loader
    print("data --> v10")

    dit, vae, t5, clip = get_models(name=args.model_name, device=accelerator.device, offload=False, is_schnell=is_schnell)
    
    dinov2_vitg14 = torch.hub.load('/mnt/wangxuekuan/huggingface/hub/hub/facebookresearch_dinov2_main/', 'dinov2_vitg14', source='local')
    # dinov2_vitg14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14', source='github')
    # dinov2_vitg14 = torch.load('/mnt/wangxuekuan/huggingface/hub/hub/checkpoints/dinov2_vitg14_pretrain.pth', map_location='cpu')
    dinov2_vitg14 = dinov2_vitg14.to(accelerator.device)
    dinov2_vitg14.requires_grad_(False).eval()
    # VGG
    # vgg_model = models.vgg16(pretrained=True)
    # vgg_model.to(accelerator.device, dtype=weight_dtype)
    # vgg_model.eval()

    # controlnet = load_controlnet("flux-dev", accelerator.device, is_inpainting=is_inpainting)


    is_inpainting = True
    control_weight = args.control_weight
    print(args.control_weight)
    
    controlnet = load_controlnet(name=args.model_name, device=accelerator.device, transformer=dit, controlnet_depth=args.control_depth, is_inpainting=is_inpainting)

    if args.control_weight:
        checkpoint = load_checkpoint(args.control_weight, None, None)
        controlnet.load_state_dict(checkpoint, strict=False)
        print("load weight ==>> \n" * 10)

    controlnet = controlnet.to(torch.float32)
    # controlnet.train()
    # 140000=canny-exp2
    # checkpoint = load_checkpoint("/mnt2/wangxuekuan/code/flux/saves_canny_exp4/checkpoint-90000/controlnet.bin", None, None)
    
    # controlnet.load_state_dict(checkpoint, strict=False)
    
    controlnet = controlnet.to(
        accelerator.device, dtype=weight_dtype
    ) #.requires_grad_(False)
 
    # lora_attn_procs = {}
    # clip_image_processor = CLIPImageProcessor()
    # load image encoder
    ip_scale = 1.0
    # image_encoder_path = ""
    

    if 'EVA02-CLIP-L-14-336' in args.image_encoder_path:
        image_encoder, _, _ = create_model_and_transforms('EVA02-CLIP-L-14-336', 'eva_clip', force_custom_clip=True)
        image_encoder = image_encoder.visual
    else: 
        image_encoder_path = "openai/clip-vit-large-patch14"
        if args.feature_type == "clip_h_pooling_local":
            image_encoder_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
            
        # image_encoder_path = "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"
        # image_encoder_path = '/mnt2/share/huggingface_models/CLIP-ViT-bigG-14-laion2B-39B-b160k'
        
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(image_encoder_path).to(
            accelerator.device#, dtype=weight_dtype
        )

    image_encoder = image_encoder.to(accelerator.device).requires_grad_(False)

    if args.use_arc_face:
        # arc_face_path = "/mnt2/wangxuekuan/pretrain/arc_face_weight/backbone.pth"
        # print("load ==> iresnet100", arc_face_path)
        # arc_face_model = iresnet100()
        # arc_face_model.load_state_dict(torch.load(arc_face_path))
        # arc_face_model.requires_grad_(False)
        facerecog_model = IR_101([112, 112])
        facerecog_model.load_state_dict(torch.load("/mnt2/wangxuekuan/pretrain/CurricularFace/CurricularFace_Backbone.pth"))
        facerecog_model.requires_grad_(False)
        facerecog_model.eval()
        # facerecog_model = facerecog_model


    # image_encoder = CLIPModel.from_pretrained(clip_model_name).to(
    #     accelerator.device, dtype=weight_dtype
    # ).requires_grad_(False)

    # self.processor = CLIPProcessor.from_pretrained(clip_model_name)

    # clip_image_processor = CLIPImageProcessor()

    # setup image embedding projection model
    print("XX:", args.feature_type)
    print("clip_pooling_local_mix_face")
    if 'EVA02-CLIP-L-14-336' in args.image_encoder_path:
        if 'big' in args.image_encoder_path:
            improj = MLPMixIDModel_v2(4096, 1024, 768, 512, max_seq_len=578).to(accelerator.device, dtype=weight_dtype)
        else:
            improj = MLPMixIDModel(4096, 1024, 768, 512).to(accelerator.device, dtype=weight_dtype)
    else:
        improj = MLPMixIDModel(4096, 1024, 768, 512).to(accelerator.device, dtype=weight_dtype)

    if args.improj_weight:
        print(args.improj_weight)
        improj_checkpoint = load_checkpoint(args.improj_weight, None, None)
        improj.load_state_dict(improj_checkpoint, strict=False)

    # print(improj.shape)
    ip_attn_procs = {}

    for name, attn_processor in dit.attn_processors.items():
        match = re.search(r'\.(\d+)\.', name)
        if match:
            layer_index = int(match.group(1))

        if name.startswith("double_blocks"): #and layer_index in double_blocks_idx:
            print("setting IP Processor for", name)
            if args.ip_type == "v0":
                ip_attn_procs[name] = IPDoubleStreamBlockProcessor(4096, 3072)
            else:
                ip_attn_procs[name] = IPDoubleStreamBlockProcessorV3(4096, 3072)
            
        elif name.startswith("single_blocks"):
            if args.ip_type == "v0":
                ip_attn_procs[name] = IPSingleStreamBlockProcessor(4096, 3072)
                print("v0")
            else:
                ip_attn_procs[name] = IPSingleStreamBlockProcessorV3(4096, 3072)
                print("v3")
                
        else:
            ip_attn_procs[name] = attn_processor

    dit.set_attn_processor(ip_attn_procs)

    if args.dit_weight:
        print(args.dit_weight)
        dit_checkpoint = load_checkpoint(args.dit_weight, None, None)
        dit.load_state_dict(dit_checkpoint, strict=False)

    vae.requires_grad_(False)
    t5.requires_grad_(False)
    clip.requires_grad_(False)
    dit = dit.to(torch.float32)
    # controlnet = controlnet.requires_grad_(False)

    if "clip_pooling_local_mix_face" in args.feature_type:
        ip_flux_model = IPFluxModelv7(dit, improj, controlnet)
    else:
        ip_flux_model = IPFluxModelv3(dit, improj, controlnet)

    ip_flux_model.train()

    optimizer_cls = torch.optim.AdamW

    for n, param in ip_flux_model.dit.named_parameters():
        flag = False
        for block_name in args.finetune_blocks:
            if block_name in n:
                param.requires_grad = True
                flag = True
        if flag == False:
            param.requires_grad = False
    
    print(sum([p.numel() for p in ip_flux_model.parameters() if p.requires_grad]) / 1000000, 'parameters')
    optimizer = optimizer_cls(
        [p for p in ip_flux_model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    train_dataloader = loader(**args.data_config)
    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )
    global_step = 0
    first_epoch = 0

    ip_flux_model, optimizer, _, lr_scheduler = accelerator.prepare(
        ip_flux_model, optimizer, deepcopy(train_dataloader), lr_scheduler
    )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    if accelerator.is_main_process:
        accelerator.init_trackers(args.tracker_project_name, {"test": None})

    timesteps = list(torch.linspace(1, 0, 1000).numpy())
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(first_epoch, args.num_train_epochs):
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            torch.cuda.empty_cache()
            with accelerator.accumulate(ip_flux_model):
                # img, image_prompt, prompts = batch
                # print(batch)
                with torch.no_grad():
                    #img, prompts, image_prompts_list, control_image, ip_atten_mask_list = batch
                    # if len(batch)
                    img, prompts, control_image, control_image_mask, control_skeleton, \
                        image_prompts_list, ip_atten_mask_list, face_box_list, \
                        face_embedding_list, feature_type, facerecog_image, _ = batch

                    control_image = control_image.to(accelerator.device)
                    control_skeleton = control_skeleton.to(accelerator.device)

                    ip_atten_mask_list[ip_atten_mask_list > 0] = 1
                    ip_atten_mask_list[ip_atten_mask_list <= 0.1] = 0

                    # if args.use_arc_face:
                    #     print("XXX do use_arc_face")
                    #     id_embedding_circular = facerecog_model(facerecog_image, return_mid_feats=True)[0].to(accelerator.device, weight_dtype)
                    #     if id_embedding_circular is None:
                    #         id_embedding_circular = torch.zeros(1, 512)
                    # else:
                    #     id_embedding_circular = None

                     
                    # print(ip_atten_mask_list.shape, ip_atten_mask_list.sum())
                    feature_type = feature_type.to(accelerator.device).to(weight_dtype).reshape(-1)

                    if "prompt" not in args.feature_type:
                        feature_type = None
                    
                    print("XXXXX:", feature_type)

                    ip_atten_mask_list = ip_atten_mask_list.to(accelerator.device).to(torch.bool)

                    # print(ip_atten_mask_list.sum())
                    # ip_atten_mask_list = [ip_atten_mask.to(accelerator.device).to(torch.bool) for ip_atten_mask in ip_atten_mask_list]

                    # faces_embedding = faces_embedding.to(accelerator.device).to(torch.float32)

                    x_1 = vae.encode(img.to(accelerator.device).to(torch.float32))

                    inp = prepare(t5=t5, clip=clip, img=x_1, prompt=prompts)
                    x_1 = rearrange(x_1, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)

                ip_atten_mask_list = [rearrange(ip_atten_mask_list[:, i], "b h w -> b (h w)") for i in range(ip_atten_mask_list.shape[1])]

                bs = img.shape[0]
                t = torch.sigmoid(torch.randn((bs,), device=accelerator.device))
                
                # t[1] = t[0]
                # t[2] = t[0]
                # t[3] = t[0]

                x_0 = torch.randn_like(x_1).to(accelerator.device)

                # x_t = (1 - t) * x_1 + t * x_0
                x_t = torch.randn_like(x_1).to(accelerator.device)
                
                
                # x_t = (1 - t[0]) * x_1 + t[0] * x_0

                for iidx in range(bs):
                    x_t[iidx] = (1 - t[iidx]) * x_1[iidx] + t[iidx] * x_0[iidx]

                bsz = x_1.shape[0]
                guidance_vec = torch.full((x_t.shape[0],), args.guidance_vec, device=x_t.device, dtype=x_t.dtype)

                image_embedding_list = []
                image_embedding_pooling_list = []
                dinov2_feature_list = []
                for idx in range(image_prompts_list.shape[1]):
                    image_prompts = image_prompts_list[:, idx]
                    image_prompts = image_prompts.to(x_t.device).to(torch.float32)
                    # 检查x的数值范围
                    # min_val_image_prompts = image_prompts.min().item()
                    # max_val_image_prompts = image_prompts.max().item()
                    # if accelerator.is_main_process and global_step % 100 == 0:
                    #     print(f"image_prompts值的范围: min={min_val_image_prompts:.4f}, max={max_val_image_prompts:.4f}")
                    with torch.no_grad():
                        dinov2_feature = dinov2_vitg14(image_prompts)
                        # print("dinov2_feature.shape!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", dinov2_feature.shape)
                        dinov2_feature_list.append(dinov2_feature.to(accelerator.device).to(weight_dtype))
                    image_embedding, image_embedding_pooling = get_image_proj(image_prompts, image_encoder, feature_type=args.feature_type)
                    image_embedding = image_embedding.to(x_t.device).to(weight_dtype)

                    if args.only_id_feature:
                        image_embedding = torch.zeros_like(image_embedding)[:, :1, ].to(x_t.device).to(weight_dtype)
                        print("image_embedding==>>", image_embedding.shape)

                    image_embedding_pooling = image_embedding_pooling.to(x_t.device).to(weight_dtype)
                    
                    # if args.only_id_feature:
                    #     image_embedding_pooling = torch.zeros_like(image_embedding_pooling)[:, :1, ].to(x_t.device).to(weight_dtype)

                    image_embedding_list.append(image_embedding)
                    image_embedding_pooling_list.append(image_embedding_pooling)
                
                #
                if len(image_embedding_list) > 1:
                    image_embedding_list = [torch.cat(image_embedding_list, -2)]
                    image_embedding_pooling_list = [torch.cat(image_embedding_pooling_list, -2)]
                    dinov2_feature_list = [torch.cat(dinov2_feature_list, -2)]

                # for _ in range(100):
                #     print(image_embedding_list[0].shape, image_embedding_pooling_list[0].shape)
                # x_t, img_ids, txt, txt_ids, y, timesteps, guidance, ip_scale, image_prompt_local_embeds, weight_dtype, image_prompt_pool_embeds=None):
                    # neg_image_proj = self.get_image_proj(neg_image_prompt)
                # if not(args.feature_type == "clip_face"):
                #     faces_embedding = None
                # elif faces_embedding is not None:
                #     faces_embedding = faces_embedding.to(weight_dtype)
                # face_embedding_list = None
                # if "face" in args.feature_type:
                # print(face_embedding_list.shape)
                # print(id_embedding_circular.shape)

                # print(id_embedding_circular)
                # print(face_embedding_list)
                if face_embedding_list is None:
                    face_embedding_list = torch.zeros(1, 512)
                face_embedding_list = [torch.zeros_like(face_embedding).to(accelerator.device).to(weight_dtype) for face_embedding in face_embedding_list]

                # id_embedding_circular_list = None
                # if id_embedding_circular is not None:
                #     print("XXX, id_embedding_circular")
                #     id_embedding_circular_list = [torch.zeros_like(face_embedding).to(accelerator.device).to(weight_dtype) for face_embedding in id_embedding_circular]

                # print(face_embedding_list)
                
                control_image = vae.encode(control_image.to(accelerator.device).to(torch.float32))
                control_skeleton = vae.encode(control_skeleton.to(accelerator.device).to(torch.float32))

                control_image = rearrange(control_image, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
                control_skeleton = rearrange(control_skeleton, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)

                if is_inpainting is True:
                    control_image_mask = rearrange(control_image_mask, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=1, pw=1).to(accelerator.device)
                    # print(x_t.shape, control_image.shape, control_image_mask.shape)
                    # control_image = torch.cat([control_image, control_image_mask], axis=-1)
                    control_image = torch.cat([control_image, control_skeleton, control_image_mask], axis=-1)

                if len(dinov2_feature_list) > 0:
                    face_embedding_list = [[None], dinov2_feature_list]
                
                if "face" not in args.feature_type:
                    face_embedding_list = None

                model_pred = ip_flux_model(
                                x_t=x_t.to(weight_dtype),
                                img_ids=inp['img_ids'].to(weight_dtype),
                                txt=inp['txt'].to(weight_dtype),
                                txt_ids=inp['txt_ids'].to(weight_dtype),
                                y=inp['vec'].to(weight_dtype),
                                timesteps=t.to(weight_dtype),
                                guidance=guidance_vec.to(weight_dtype),
                                image_prompt_local_embeds=image_embedding_list,
                                image_prompt_pool_embeds=image_embedding_pooling_list,
                                faces_embedding=face_embedding_list,
                                ip_scale=ip_scale, 
                                ip_atten_mask=ip_atten_mask_list,
                                controlnet_cond=control_image.to(weight_dtype),
                                do_vae=True,
                                feature_type=feature_type,
                                weight_dtype=weight_dtype)
                
                # model_pred = x_1
                
                loss = 0

                x = model_pred.clone().detach()
                x = x_0 - x
                x = unpack(x.float(), img.shape[2], img.shape[3])
                x = vae.decode(x)
                x = x.clamp(-1, 1)
                # 检查x的数值范围
                # min_val = x.min().item()
                # max_val = x.max().item()
                # if accelerator.is_main_process and global_step % 100 == 0:
                #     print(f"x值的范围: min={min_val:.4f}, max={max_val:.4f}")
                
                # print("face_box_list:", len(face_box_list))

                for i, box in enumerate(face_box_list):
                    x_temp = x.clone().detach()
                    try:
                        x1, y1, x2, y2 = box
                    except:
                        x1, y1, x2, y2 = box.tolist()
                    # print("x_temp.shape", x_temp.shape, "x1", x1, "y1", y1, "x2", x2, "y2", y2)
                    x_temp = x_temp[:, :, y1:y2, x1:x2]
                    x_temp = F.interpolate(x_temp, size=(image_prompts.shape[2], image_prompts.shape[3]), mode='bilinear', align_corners=False)

                    with torch.no_grad():
                        x_temp_r = rearrange(x_temp[-1], "c h w -> h w c")
                        pred_output_img = Image.fromarray((127.5 * (x_temp_r + 1.0)).cpu().byte().numpy())
                        # output_img.save(f"test/temp_{t[0]}_x.jpg")
                    # pred
                    
                    # face-image
                    if args.clip_loss_weight > 0:
                        image_embedding_temp, image_embedding_pooling_temp = get_image_proj(x_temp.to(torch.float32), image_encoder, feature_type=args.feature_type)
                        image_embedding_temp = image_embedding_temp.to(accelerator.device).to(weight_dtype).detach()
                        image_embedding_pooling_temp = image_embedding_pooling_temp.to(accelerator.device).to(weight_dtype).detach()

                    if args.id_loss_weight > 0:
                        x_temp = F.interpolate(x_temp, size=(112, 112), mode='bilinear', align_corners=False)
                        id_embedding_circular_pred = facerecog_model(x_temp.cpu().to(torch.float), return_mid_feats=True)[0].to(accelerator.device, weight_dtype)
                         
                    #gt
    
                    img_clone = img.clone().detach().to(weight_dtype).to(accelerator.device)
                    img_clone = img_clone[:, :, y1:y2, x1:x2]
                    img_clone = F.interpolate(img_clone, size=(image_prompts.shape[2], image_prompts.shape[3]), mode='bilinear', align_corners=False)
                    with torch.no_grad():
                        x_temp_r = rearrange(img_clone[-1], "c h w -> h w c")
                        gt_output_img = Image.fromarray((127.5 * (x_temp_r + 1.0)).cpu().byte().numpy())
                        # output_img.save(f"test/temp_{t[0]}_gt.jpg")

                    if args.clip_loss_weight > 0:
                        image_embedding_gt, image_embedding_pooling_gt = get_image_proj(img_clone.to(torch.float32), image_encoder, feature_type=args.feature_type) 
                        image_embedding_gt = image_embedding_gt.to(accelerator.device).to(weight_dtype).detach()
                        image_embedding_pooling_gt = image_embedding_pooling_gt.to(accelerator.device).to(weight_dtype).detach()

                    if args.id_loss_weight > 0:
                        img_clone = F.interpolate(img_clone, size=(112, 112), mode='bilinear', align_corners=False)
                        id_embedding_circular_gt = facerecog_model(img_clone.cpu().to(torch.float), return_mid_feats=True)[0].to(accelerator.device, weight_dtype)
                    
                    if args.id_loss_weight > 0 and id_embedding_circular_pred is not None and id_embedding_circular_gt is not None:
                        id_embedding_loss = F.mse_loss(id_embedding_circular_pred.float(), id_embedding_circular_gt.float(), reduction="mean")
                    else:
                        id_embedding_loss = 0

                    if args.clip_loss_weight > 0:
                        clip_local_loss = F.mse_loss(image_embedding_temp.float(), image_embedding_gt.float(), reduction="mean")
                        clip_pool_loss = F.mse_loss(image_embedding_pooling_temp.float(), image_embedding_pooling_gt.float(), reduction="mean")
                    else:
                        clip_local_loss = 0
                        clip_pool_loss = 0

                image_loss = F.mse_loss(model_pred.float(), (x_0 - x_1).float(), reduction="mean")
                
                if args.id_loss_weight > 0 and id_embedding_circular_pred is not None and id_embedding_circular_gt is not None:
                    print("loss is:", image_loss, id_embedding_loss, clip_local_loss, clip_pool_loss)

                    loss = image_loss + args.id_loss_weight * id_embedding_loss + args.clip_loss_weight * (clip_local_loss + clip_pool_loss)
                else:
                    print("loss is:", image_loss, args.clip_loss_weight, args.id_loss_weight)
                    loss = image_loss + args.clip_loss_weight * (clip_local_loss + clip_pool_loss)

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(ip_flux_model.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    os.makedirs(save_path, exist_ok=True)

                    ### accelerator.save_state(save_path)
                    unwrapped_model_state = accelerator.unwrap_model(ip_flux_model.dit).state_dict()

                    unwrapped_improj_state = accelerator.unwrap_model(ip_flux_model.image_proj).state_dict()

                    unwrapped_controlnet = accelerator.unwrap_model(ip_flux_model.controlnet).state_dict()

                    save_file(
                        unwrapped_controlnet,
                        os.path.join(save_path, "ip_adaptor_controlnet.safetensors")
                    )
                    
                    save_file(
                        unwrapped_improj_state,
                        os.path.join(save_path, "ip_adaptor_project.safetensors")
                    )

                    # save checkpoint in safetensors format
                    # ip_state_dict = {k:unwrapped_model_state[k] for k in unwrapped_model_state.keys() if 'double_blocks' in k}
                    ip_state_dict =  unwrapped_model_state #{k:unwrapped_model_state[k] for k in unwrapped_model_state.keys() if 'double_blocks' in k}
                    save_file(
                        ip_state_dict,
                        os.path.join(save_path, "ip_adaptor.safetensors")
                    )
                    logger.info(f"Saved state to {save_path}")

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()
