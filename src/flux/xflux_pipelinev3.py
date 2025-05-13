from PIL import Image, ExifTags
import numpy as np
import torch
from torch import Tensor

from einops import rearrange
import uuid
import os

from src.flux.modules.layers import (
    SingleStreamBlockProcessor,
    DoubleStreamBlockProcessor,
    SingleStreamBlockLoraProcessor,
    DoubleStreamBlockLoraProcessor,
    IPDoubleStreamBlockProcessor,
    IPDoubleStreamBlockProcessorV2,
    IPDoubleStreamBlockProcessorV3,
    IPDoubleStreamBlockProcessorV4,
    IPDoubleStreamBlockProcessorV5,
    IPSingleStreamBlockProcessor,
    IPSingleStreamBlockProcessorV3,
    IPSingleStreamBlockProcessorV4,
    IPSingleStreamBlockProcessorV5,
    ImageProjModel,
    ImageProjModelLocal,
    ImageProjModelFaceLocal,
    ImageProjModelLocalB1,
    ImageProjModelLocalB0,
    ImageProjModelLocalB0_face,
    ImageProjModelLocalB2_face,
    ImageProjModelLocalB3_face,
    InputProjModel,
    SkeletonMLP
)
from src.flux.modules.projection_models import MLPMixIDModel
from src.flux.samplingv3 import denoise, denoise_controlnet, get_noise, get_schedule, prepare, unpack, prepare_v2
from src.flux.utilv2 import (
    load_ae,
    load_clip,
    load_flow_model,
    load_t5,
    load_controlnet,
    load_flow_model_quintized,
    Annotator,
    get_lora_rank,
    load_checkpoint
)

from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor

from eva_clip import create_model_and_transforms

class XFluxPipeline:
    def __init__(self, model_type, device, offload: bool = False):
        self.device = torch.device(device)
        self.offload = offload
        self.model_type = model_type

        self.clip = load_clip(self.device)
        self.t5 = load_t5(self.device, max_length=512)
        self.ae = load_ae(model_type, device="cpu" if offload else self.device)
        if "fp8" in model_type:
            self.model = load_flow_model_quintized(model_type, device="cpu" if offload else self.device)
        else:
            self.model = load_flow_model(model_type, device="cpu" if offload else self.device)

        self.image_encoder_path = "openai/clip-vit-large-patch14"
        self.hf_lora_collection = "XLabs-AI/flux-lora-collection"
        self.lora_types_to_names = {
            "realism": "lora.safetensors",
        }
        self.controlnet_loaded = False
        self.ip_loaded = False
        self.input_proj_loaded = False
    
    def set_skeleton_encoder(self, local_path: str = None):
        print(f"Loading skeleton encoder from {local_path}")
        checkpoint = load_checkpoint(local_path, None, None)
        self.skeleton_encoder = SkeletonMLP(15, 4096 * 4)
        self.skeleton_encoder.load_state_dict(checkpoint)
        self.skeleton_encoder = self.skeleton_encoder.to(self.device, dtype=torch.bfloat16)

    def set_dit(self, local_path: str = None,):
        print(f"Resuming from checkpoint {local_path}")
        dit_state = torch.load(local_path, map_location='cpu')
        dit_state2 = {}
        for k in dit_state.keys():
            dit_state2[k[len('module.'):]] = dit_state[k]
        self.model.load_state_dict(dit_state2)
        self.model.to(self.device)

    def set_ipv2(self, local_path: str = None, repo_id = None, name: str = None, feature_type: str = "clip_pooling"):
        self.model.to(self.device)
        # unpack checkpoint
        import os
        checkpoint = load_checkpoint(local_path, repo_id, name)
        checkpoint_proj = load_checkpoint(os.path.dirname(local_path) + "/ip_adaptor_project.safetensors", repo_id, name)

        prefix = "double_blocks."
        blocks = {}
        proj = {}

        for key, value in checkpoint.items():
            if key.startswith(prefix):
                blocks[key[len(prefix):].replace('.processor.', '.')] = value
            if key.startswith("ip_adapter_proj_model"):
                proj[key[len("ip_adapter_proj_model."):]] = value

        # load image encoder
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(self.image_encoder_path).to(
            self.device, dtype=torch.float16
        )
        self.clip_image_processor = CLIPImageProcessor()

        for k in checkpoint_proj.keys():
            print(k)

        # setup image embedding projection model
        # setup image embedding projection model
        if feature_type == "clip_pooling":
            self.improj = ImageProjModel(4096, 768, 4)
        elif feature_type == "clip_pooling_local":
            self.improj = ImageProjModelLocal(4096, 1024, 768)
        elif feature_type == "clip_h_pooling_local":
            self.improj = ImageProjModelLocal(4096, 1280, 1024)
        elif feature_type == "clip_face":
            self.improj = ImageProjModelFaceLocal(4096, 1024, 768, 512)
        elif feature_type == "clip_pooling_local_b1":
            self.improj = ImageProjModelLocalB1(4096, 1024, 768)
        elif feature_type == "clip_pooling_local_b0":
            self.improj = ImageProjModelLocalB0(4096, 1024, 768)
        elif feature_type == "clip_h_pooling_local_b1":
            self.improj = ImageProjModelLocalB1(4096, 1280, 1024)

        
        # self.input_proj = InputProjModel(129, 64)

        
        self.improj.load_state_dict(checkpoint_proj)
        self.improj = self.improj.to(self.device, dtype=torch.bfloat16)

        ip_attn_procs = {}
        # ip_attn_procs = {}

        for name, attn_processor in self.model.attn_processors.items():
            # match = re.search(r'\.(\d+)\.', name)
            # if match:
            #     layer_index = int(match.group(1))

            if name.startswith("double_blocks"): #and layer_index in double_blocks_idx:
                print("setting IP Processor for", name)
                ip_attn_procs[name] = IPDoubleStreamBlockProcessor(4096, 3072)
            elif name.startswith("single_blocks"):
                ip_attn_procs[name] = IPSingleStreamBlockProcessor(4096, 3072)
            else:
                ip_attn_procs[name] = attn_processor

        # for name, _ in self.model.attn_processors.items():
        #     if "double_blocks" in name:
        #         print(f"name: {name}")
        #     ip_state_dict = {}
        #     for k in checkpoint.keys():
        #         if name in k:
        #             ip_state_dict[k.replace(f'{name}.', '')] = checkpoint[k]
        #     if ip_state_dict:
        #         ip_attn_procs[name] = IPDoubleStreamBlockProcessor(4096, 3072)
        #         ip_attn_procs[name].load_state_dict(ip_state_dict)
        #         ip_attn_procs[name].to(self.device, dtype=torch.bfloat16)
        #         print(f"do insert IPDoubleStreamBlockProcessor, {name}")
        #     else:
        #         ip_attn_procs[name] = self.model.attn_processors[name]

        self.model.set_attn_processor(ip_attn_procs)
        self.ip_loaded = True

        self.model.load_state_dict(checkpoint)
        self.model = self.model.to(self.device, dtype=torch.bfloat16)

        # print(self.model)
        # with torch.no_grad():
        #     for n, param in self.model.named_parameters():
        #         # if 'double_blocks' in n:
        #         #     param.copy_(checkpoint[n])
        #         # elif "single_block" in n:
        #         param.copy_(checkpoint[n])
                
        #         print("name update", n)
  
    def set_ipv4(self, local_path: str = None, repo_id = None, name: str = None, feature_type: str = "clip_pooling"):
        self.model.to(self.device)
        # unpack checkpoint
        import os
        checkpoint = load_checkpoint(local_path, repo_id, name)
        checkpoint_proj = load_checkpoint(os.path.dirname(local_path) + "/ip_adaptor_project.safetensors", repo_id, name)

        prefix = "double_blocks."
        blocks = {}
        proj = {}

        for key, value in checkpoint.items():
            if key.startswith(prefix):
                blocks[key[len(prefix):].replace('.processor.', '.')] = value
            if key.startswith("ip_adapter_proj_model"):
                proj[key[len("ip_adapter_proj_model."):]] = value

        # load image encoder
        # self.image_encoder_path = ""
        if feature_type in ["clip_h_pooling_local_b1", "clip_h_pooling_local"]:
            self.image_encoder_path = '/mnt2/share/huggingface_models/CLIP-ViT-bigG-14-laion2B-39B-b160k'

        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(self.image_encoder_path).to(
            self.device, dtype=torch.float16
        )
        self.clip_image_processor = CLIPImageProcessor()

        for k in checkpoint_proj.keys():
            print(k)

        # setup image embedding projection model
        # setup image embedding projection model
        if feature_type == "clip_pooling":
            self.improj = ImageProjModel(4096, 768, 4)
        elif feature_type == "clip_pooling_local":
            self.improj = ImageProjModelLocal(4096, 1664, 1280)
        elif feature_type == "clip_face":
            self.improj = ImageProjModelFaceLocal(4096, 1024, 768, 512)

        
        self.improj.load_state_dict(checkpoint_proj)
        self.improj = self.improj.to(self.device, dtype=torch.bfloat16)

        ip_attn_procs = {}
        # ip_attn_procs = {}

        for name, attn_processor in self.model.attn_processors.items():
            # match = re.search(r'\.(\d+)\.', name)
            # if match:
            #     layer_index = int(match.group(1))

            if name.startswith("double_blocks"): #and layer_index in double_blocks_idx:
                print("setting IP Processor for", name)
                ip_attn_procs[name] = IPDoubleStreamBlockProcessor(4096, 3072)
            elif name.startswith("single_blocks"):
                ip_attn_procs[name] = IPSingleStreamBlockProcessor(4096, 3072)
            else:
                ip_attn_procs[name] = attn_processor

        # for name, _ in self.model.attn_processors.items():
        #     if "double_blocks" in name:
        #         print(f"name: {name}")
        #     ip_state_dict = {}
        #     for k in checkpoint.keys():
        #         if name in k:
        #             ip_state_dict[k.replace(f'{name}.', '')] = checkpoint[k]
        #     if ip_state_dict:
        #         ip_attn_procs[name] = IPDoubleStreamBlockProcessor(4096, 3072)
        #         ip_attn_procs[name].load_state_dict(ip_state_dict)
        #         ip_attn_procs[name].to(self.device, dtype=torch.bfloat16)
        #         print(f"do insert IPDoubleStreamBlockProcessor, {name}")
        #     else:
        #         ip_attn_procs[name] = self.model.attn_processors[name]

        self.model.set_attn_processor(ip_attn_procs)
        self.ip_loaded = True

        self.model.load_state_dict(checkpoint)
        self.model = self.model.to(self.device, dtype=torch.bfloat16)

        # print(self.model)
        # with torch.no_grad():
        #     for n, param in self.model.named_parameters():
        #         # if 'double_blocks' in n:
        #         #     param.copy_(checkpoint[n])
        #         # elif "single_block" in n:
        #         param.copy_(checkpoint[n])
                
        #         print("name update", n)
    def set_ipv3(self, local_path: str = None, repo_id = None, name: str = None, feature_type: str = "clip_pooling"):
        self.model.to(self.device)
        # unpack checkpoint
        import os
        checkpoint = load_checkpoint(local_path, repo_id, name)
        checkpoint_proj = load_checkpoint(os.path.dirname(local_path) + "/ip_adaptor_project.safetensors", repo_id, name)

        prefix = "double_blocks."
        blocks = {}
        proj = {}

        for key, value in checkpoint.items():
            if key.startswith(prefix):
                blocks[key[len(prefix):].replace('.processor.', '.')] = value
            if key.startswith("ip_adapter_proj_model"):
                proj[key[len("ip_adapter_proj_model."):]] = value

        # load image encoder
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(self.image_encoder_path).to(
            self.device, dtype=torch.float16
        )
        self.clip_image_processor = CLIPImageProcessor()

        for k in checkpoint_proj.keys():
            print(k)

        # setup image embedding projection model
        # setup image embedding projection model
        if feature_type == "clip_pooling":
            self.improj = ImageProjModel(4096, 768, 4)
        elif feature_type == "clip_pooling_local":
            self.improj = ImageProjModelLocal(4096, 1024, 768)
        elif feature_type == "clip_face":
            self.improj = ImageProjModelFaceLocal(4096, 1024, 768, 512)

        
        self.improj.load_state_dict(checkpoint_proj)
        self.improj = self.improj.to(self.device, dtype=torch.bfloat16)

        ip_attn_procs = {}
        # ip_attn_procs = {}

        for name, attn_processor in self.model.attn_processors.items():
            # match = re.search(r'\.(\d+)\.', name)
            # if match:
            #     layer_index = int(match.group(1))

            if name.startswith("double_blocks"): #and layer_index in double_blocks_idx:
                print("setting IP Processor for", name)
                ip_attn_procs[name] = IPDoubleStreamBlockProcessorV2(4096, 3072)
            elif name.startswith("single_blocks"):
                ip_attn_procs[name] = IPSingleStreamBlockProcessor(4096, 3072)
            else:
                ip_attn_procs[name] = attn_processor

        # for name, _ in self.model.attn_processors.items():
        #     if "double_blocks" in name:
        #         print(f"name: {name}")
        #     ip_state_dict = {}
        #     for k in checkpoint.keys():
        #         if name in k:
        #             ip_state_dict[k.replace(f'{name}.', '')] = checkpoint[k]
        #     if ip_state_dict:
        #         ip_attn_procs[name] = IPDoubleStreamBlockProcessor(4096, 3072)
        #         ip_attn_procs[name].load_state_dict(ip_state_dict)
        #         ip_attn_procs[name].to(self.device, dtype=torch.bfloat16)
        #         print(f"do insert IPDoubleStreamBlockProcessor, {name}")
        #     else:
        #         ip_attn_procs[name] = self.model.attn_processors[name]

        self.model.set_attn_processor(ip_attn_procs)
        self.ip_loaded = True

        self.model.load_state_dict(checkpoint)
        self.model = self.model.to(self.device, dtype=torch.bfloat16)

        # print(self.model)
        # with torch.no_grad():
        #     for n, param in self.model.named_parameters():
        #         # if 'double_blocks' in n:
        #         #     param.copy_(checkpoint[n])
        #         # elif "single_block" in n:
        #         param.copy_(checkpoint[n])
                
        #         print("name update", n)
    
    def set_ipv5(self, local_path: str = None, repo_id = None, name: str = None, feature_type: str = "clip_pooling", model_type="v0", image_encoder_path: str = None, clip_image_size: int = 224):
        self.model.to(self.device)
        # unpack checkpoint
        import os
        checkpoint = load_checkpoint(local_path, repo_id, name)
        checkpoint_proj = load_checkpoint(os.path.dirname(local_path) + "/ip_adaptor_project.safetensors", repo_id, name)

        if os.path.exists(os.path.dirname(local_path) + "/ip_adaptor_input_proj.safetensors"):
            checkpoint_input_proj = load_checkpoint(os.path.dirname(local_path) + "/ip_adaptor_input_proj.safetensors", repo_id, name)
            self.input_proj = InputProjModel(129, 64)
            self.input_proj.load_state_dict(checkpoint_input_proj)
            self.input_proj = self.input_proj.to(self.device, dtype=torch.bfloat16)
            self.input_proj_loaded = True

        prefix = "double_blocks."
        blocks = {}
        proj = {}

        for key, value in checkpoint.items():
            if key.startswith(prefix):
                blocks[key[len(prefix):].replace('.processor.', '.')] = value
            if key.startswith("ip_adapter_proj_model"):
                proj[key[len("ip_adapter_proj_model."):]] = value

        # load image encoder
        if image_encoder_path is not None and 'EVA02-CLIP-L-14-336' in image_encoder_path:
            image_encoder, _, _ = create_model_and_transforms('EVA02-CLIP-L-14-336', 'eva_clip', force_custom_clip=True)
            self.image_encoder = image_encoder.visual.to(self.device, dtype=torch.float32)
        else:
            if feature_type in ["clip_h_pooling_local_b1", "clip_h_pooling_local"]:
                self.image_encoder_path = '/mnt2/share/huggingface_models/CLIP-ViT-H-14-laion2B-s32B-b79K' 
            #'/mnt2/share/huggingface_models/CLIP-ViT-bigG-14-laion2B-39B-b160k'
            self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(self.image_encoder_path).to(
                self.device, dtype=torch.bfloat16
            )
        # self.clip_image_processor = CLIPImageProcessor()
        self.clip_image_processor = CLIPImageProcessor(crop_size={"height": clip_image_size, "width": clip_image_size}, size={"shortest_edge": clip_image_size})
        for k in checkpoint_proj.keys():
            print(k)

        # setup image embedding projection model
        # setup image embedding projection model
        print("feature_type:", feature_type)
        if feature_type == "clip_pooling":
            self.improj = ImageProjModel(4096, 768, 4)
        elif feature_type == "clip_pooling_local":
            self.improj = ImageProjModelLocal(4096, 1024, 768)
        elif feature_type == "clip_face":
            self.improj = ImageProjModelFaceLocal(4096, 1024, 768, 512)
        elif feature_type == "clip_pooling_local_b1":
            print("clip_pooling_local_b1 XXX")
            self.improj = ImageProjModelLocalB1(4096,  1024, 768)
        elif feature_type == "clip_h_pooling_local_b1":
            print("clip_h_pooling_local_b1 XXX")
            self.improj = ImageProjModelLocalB1(4096, 1280, 1024)
        elif feature_type == "clip_pooling_local_b0":
            print("clip_h_pooling_local_b0 XXX")
            self.improj = ImageProjModelLocalB0(4096, 1024, 768, 128)
        elif feature_type == "clip_h_pooling_local":
            self.improj = ImageProjModelLocal(4096, 1280, 1024)
        elif feature_type =="clip_pooling_local_b0_prompt":
            self.improj = ImageProjModelLocalB0(4096 - 32, 1024, 768, 128)
        elif feature_type =="clip_pooling_local_b0_prompt_face":
            self.improj = ImageProjModelLocalB0_face(4096 - 32, 1024, 768, 128)
        elif feature_type == "clip_pooling_local_b2_prompt_face":
            self.improj = ImageProjModelLocalB2_face(4096 - 32, 1024, 768, 128)
        elif feature_type == "clip_pooling_local_b3_prompt_face":
            self.improj = ImageProjModelLocalB3_face(4096 - 32, 1024, 768, 128)
        elif feature_type == "clip_pooling_local_mix_face":
            if 'EVA02-CLIP-L-14-336' in image_encoder_path:
                print("EVA02-CLIP-L-14-336")
                self.improj = MLPMixIDModel(4096, 1024, 768, 512, max_seq_len=578)
            else:
                self.improj = MLPMixIDModel(4096, 1024, 768, 512)
        
        self.improj.load_state_dict(checkpoint_proj)
        self.improj = self.improj.to(self.device, dtype=torch.bfloat16)

        ip_attn_procs = {}
        # ip_attn_procs = {}

        for name, attn_processor in self.model.attn_processors.items():
            # match = re.search(r'\.(\d+)\.', name)
            # if match:
            #     layer_index = int(match.group(1))
            if name.startswith("double_blocks"): #and layer_index in double_blocks_idx:
                print("setting IP Processor for", model_type, name)
                if model_type == "v3":
                    ip_attn_procs[name] = IPDoubleStreamBlockProcessorV3(4096, 3072)
                elif model_type == "v5":
                    ip_attn_procs[name] = IPDoubleStreamBlockProcessorV5(4096, 3072)
                else:
                    ip_attn_procs[name] = IPDoubleStreamBlockProcessor(4096, 3072)
            elif name.startswith("single_blocks"):
                if model_type == "v3":
                    ip_attn_procs[name] = IPSingleStreamBlockProcessorV3(4096, 3072)
                elif model_type == "v5":
                    ip_attn_procs[name] = IPSingleStreamBlockProcessorV5(4096, 3072)
                else:
                    ip_attn_procs[name] = IPSingleStreamBlockProcessor(4096, 3072)
            else:
                ip_attn_procs[name] = attn_processor

        self.model.set_attn_processor(ip_attn_procs)
        self.ip_loaded = True

        self.model.load_state_dict(checkpoint)
        self.model = self.model.to(self.device, dtype=torch.bfloat16)


    def set_ip(self, local_path: str = None, repo_id = None, name: str = None):
        self.model.to(self.device)
        # unpack checkpoint
        checkpoint = load_checkpoint(local_path, repo_id, name)
        prefix = "double_blocks."
        blocks = {}
        proj = {}

        for key, value in checkpoint.items():
            if key.startswith(prefix):
                blocks[key[len(prefix):].replace('.processor.', '.')] = value
            if key.startswith("ip_adapter_proj_model"):
                proj[key[len("ip_adapter_proj_model."):]] = value

        # load image encoder
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(self.image_encoder_path).to(
            self.device, dtype=torch.float16
        )
        self.clip_image_processor = CLIPImageProcessor()

        # setup image embedding projection model
        self.improj = ImageProjModel(4096, 768, 4)
        self.improj.load_state_dict(proj)
        
        self.improj = self.improj.to(self.device, dtype=torch.bfloat16)

        ip_attn_procs = {}

        for name, _ in self.model.attn_processors.items():
            if "double_blocks" in name:
                print(f"name: {name}")
            ip_state_dict = {}
            for k in checkpoint.keys():
                if name in k:
                    ip_state_dict[k.replace(f'{name}.', '')] = checkpoint[k]
            if ip_state_dict:
                ip_attn_procs[name] = IPDoubleStreamBlockProcessor(4096, 3072)
                ip_attn_procs[name].load_state_dict(ip_state_dict)
                ip_attn_procs[name].to(self.device, dtype=torch.bfloat16)
            else:
                ip_attn_procs[name] = self.model.attn_processors[name]

        self.model.set_attn_processor(ip_attn_procs)
        self.ip_loaded = True

    def set_lora(self, local_path: str = None, repo_id: str = None,
                 name: str = None, lora_weight: int = 0.7):
        checkpoint = load_checkpoint(local_path, repo_id, name)
        self.update_model_with_lora(checkpoint, lora_weight)

    def set_lora_from_collection(self, lora_type: str = "realism", lora_weight: int = 0.7):
        checkpoint = load_checkpoint(
            None, self.hf_lora_collection, self.lora_types_to_names[lora_type]
        )
        self.update_model_with_lora(checkpoint, lora_weight)

    def update_model_with_lora(self, checkpoint, lora_weight):
        rank = get_lora_rank(checkpoint)
        lora_attn_procs = {}

        for name, _ in self.model.attn_processors.items():
            lora_state_dict = {}
            for k in checkpoint.keys():
                if name in k:
                    lora_state_dict[k[len(name) + 1:]] = checkpoint[k] * lora_weight

            if len(lora_state_dict):
                if name.startswith("single_blocks"):
                    lora_attn_procs[name] = SingleStreamBlockLoraProcessor(dim=3072, rank=rank)
                else:
                    lora_attn_procs[name] = DoubleStreamBlockLoraProcessor(dim=3072, rank=rank)
                lora_attn_procs[name].load_state_dict(lora_state_dict)
                lora_attn_procs[name].to(self.device)
            else:
                if name.startswith("single_blocks"):
                    lora_attn_procs[name] = SingleStreamBlockProcessor()
                else:
                    lora_attn_procs[name] = DoubleStreamBlockProcessor()

        self.model.set_attn_processor(lora_attn_procs)

    def set_controlnet(self, control_type: str, local_path: str = None, repo_id: str = None, name: str = None, is_inpainting: bool = False, controlnet_depth: int = 6):
        self.model.to(self.device)
        self.controlnet = load_controlnet(self.model_type, self.device, controlnet_depth=controlnet_depth, is_inpainting=is_inpainting).to(torch.bfloat16)
        
        # print(local_path)
        checkpoint = load_checkpoint(local_path, repo_id, name)
        self.controlnet.load_state_dict(checkpoint, strict=False)
        self.annotator = Annotator(control_type, self.device)
        self.controlnet_loaded = True
        self.control_type = control_type

        if os.path.exists(os.path.dirname(local_path) + "/ip_adaptor_input_proj.safetensors"):
            print("load ip_adaptor_input_proj")
            checkpoint_input_proj = load_checkpoint(os.path.dirname(local_path) + "/ip_adaptor_input_proj.safetensors", repo_id, name)
            self.input_proj = InputProjModel(129, 64)
            self.input_proj.load_state_dict(checkpoint_input_proj)
            self.input_proj = self.input_proj.to(self.device, dtype=torch.bfloat16)
            self.input_proj_loaded = True
    
    def get_clip_embeddings(self, clip_images):
        # print(clip_images.shape, clip_images.dtype, clip_images.device)
        if self.image_encoder.__class__.__name__ == 'EVAVisionTransformer':
            clip_images = clip_images.to(self.device, dtype=torch.float32)
            with torch.no_grad():
                id_cond_vit, id_vit_hidden = self.image_encoder(
                    clip_images, return_all_features=False, return_hidden=True, shuffle=False
                )
                image_embeds_pooling = id_cond_vit
                image_embeds = id_vit_hidden[-2]

        elif self.image_encoder.__class__.__name__ == 'CLIPVisionModelWithProjection':
            clip_images = clip_images.to(self.device, dtype=torch.bfloat16)
            with torch.no_grad():
                image_embeds = self.image_encoder(clip_images, output_hidden_states=True).hidden_states[-2]
                image_embeds_pooling = self.image_encoder(clip_images, output_hidden_states=True).image_embeds

        elif self.image_encoder.__class__.__name__== 'SiglipVisionModel':
            clip_images = clip_images.to(self.device, dtype=torch.bfloat16)
            with torch.no_grad():
                image_embeds = self.image_encoder(clip_images, output_hidden_states=True).hidden_states[-2]
                image_embeds_pooling = self.image_encoder(clip_images, output_hidden_states=True).pooler_output
                
        image_embeds = image_embeds.to(self.device, dtype=torch.bfloat16)
        image_embeds_pooling = image_embeds_pooling.to(self.device, dtype=torch.bfloat16)

        # print(image_embeds.shape, image_embeds_pooling.shape)
        return image_embeds, image_embeds_pooling

    def get_image_proj(
        self,
        image_prompt: Tensor,
        face_embedding: Tensor = None,
        feature_type: str = "clip_pooling",
        face_type = None,
        keypoints = None,
        faces_embedding = None
    ):
        # encode image-prompt embeds
        image_prompt = self.clip_image_processor(
            images=image_prompt,
            return_tensors="pt"
        ).pixel_values

        
        image_prompt = image_prompt.to(self.device)
        

        # if feature_type == "clip_pooling":
        #     image_prompt_embeds_pooling = self.image_encoder(
        #         image_prompt
        #     ).image_embeds.to(
        #         device=self.device, dtype=torch.bfloat16,
        #     )
        #     image_prompt_embeds = None
            
        #     #image_proj = improj(image_prompt_embeds)
        # # elif feature_type in ["clip_pooling_local", "clip_face", "clip_pooling_local_b1", "clip_h_pooling_local_b1", "clip_h_pooling_local"]:
        # else:
        #     output_encoder = self.image_encoder(image_prompt, output_hidden_states=True)
        #     image_prompt_embeds = output_encoder.hidden_states[-2].to(
        #         device=self.device, dtype=torch.bfloat16,
        #     )
        #     image_prompt_embeds_pooling = output_encoder.image_embeds.to(
        #         device=self.device, dtype=torch.bfloat16,
        #     )
        image_prompt_embeds, image_prompt_embeds_pooling = self.get_clip_embeddings(image_prompt)

        if feature_type == "clip_face":
            image_embedding = self.improj(image_prompt_embeds, image_prompt_embeds_pooling, face_embedding)
        elif feature_type in ["clip_pooling_local", "clip_pooling_local_b1", "clip_h_pooling_local_b1", "clip_h_pooling_local"]:
            image_embedding = self.improj(image_prompt_embeds, image_prompt_embeds_pooling)
        elif feature_type == "clip_pooling":
            image_embedding = self.improj(image_prompt_embeds_pooling)
        elif feature_type == "clip_pooling_local_b0_prompt" or feature_type == "clip_pooling_local_b0_prompt_face" or feature_type == "clip_pooling_local_b0":
            image_embedding = self.improj(image_prompt_embeds, image_prompt_embeds_pooling, face_embedding, face_type)
        elif feature_type == "clip_pooling_local_mix_face":
            image_embedding = self.improj(image_prompt_embeds, image_prompt_embeds_pooling, faces_embedding[0][0], faces_embedding[1][0])
        else:
            image_embedding = self.improj(image_prompt_embeds, image_prompt_embeds_pooling, face_embedding, face_type)
        
        if keypoints is not None:
            keypoints = keypoints.view(1, -1)
            keypoints_embedding = self.skeleton_encoder(keypoints)
            keypoints_embedding = keypoints_embedding.view(1, -1, 4096)
            print("keypoints_embedding:", keypoints_embedding)
            image_embedding = torch.cat((keypoints_embedding, image_embedding), dim=1)

        return image_embedding

    def __call__(self,
                 prompt: str,
                 image_prompt_list: list = None,
                 controlnet_image: Image = None,
                 controlnet_mask_image: Image = None,
                 width: int = 512,
                 height: int = 512,
                 guidance: float = 4,
                 num_steps: int = 50,
                 seed: int = 123456789,
                 true_gs: float = 3,
                 control_weight: float = 0.9,
                 ip_scale: float = 1.0,
                 neg_ip_scale: float = 1.0,
                 neg_prompt: str = '',
                 neg_image_prompt: Image = None,
                 timestep_to_start_cfg: int = 0,
                 do_vae: bool = False,
                 face_embedding = None,
                 feature_type = "clip_pooling",
                 faces_embedding=None,
                 ip_atten_mask=None,
                 do_inpainting=False,
                 face_type=None,
                 keypoints=None,
                 prompt_cloth=None
                 ):
        width = 16 * (width // 16)
        height = 16 * (height // 16)
        image_proj = None
        neg_image_proj = None
        
        if face_type is not None:
            face_type = face_type.to(torch.bfloat16).to(self.device)

        image_embedding_list = []
        image_embedding_neg_list = []
        if image_prompt_list is None:
            image_embedding_list = None
            image_embedding_neg_list = None
        else:
            if face_embedding is not None:
                face_embedding = torch.from_numpy(face_embedding).to(torch.bfloat16).to(self.device)
            if faces_embedding is not None:
                faces_embedding = [[faces_embedding[0][0].to(torch.bfloat16).to(self.device)], [faces_embedding[1][0].to(torch.bfloat16).to(self.device)]]
            if keypoints is not None:
                if len(keypoints) > 0:
                    keypoints = torch.tensor(keypoints, dtype=torch.bfloat16).to(self.device)
                else:
                    keypoints = torch.zeros(15).to(self.device).to(torch.bfloat16)
            for image_prompt in image_prompt_list:
                if not (image_prompt is None and neg_image_prompt is None) :
                    assert self.ip_loaded, 'You must setup IP-Adapter to add image prompt as input'

                    if image_prompt is None:
                        image_prompt = np.zeros((width, height, 3), dtype=np.uint8)
                    if neg_image_prompt is None:
                        neg_image_prompt = np.zeros((width, height, 3), dtype=np.uint8)
                    
                    image_proj = self.get_image_proj(image_prompt, face_embedding=face_embedding, feature_type=feature_type, face_type=face_type, keypoints=keypoints, faces_embedding=faces_embedding)
                    neg_image_proj = self.get_image_proj(neg_image_prompt, face_embedding=face_embedding, feature_type=feature_type, face_type=face_type, keypoints=keypoints, faces_embedding=faces_embedding)
                    image_embedding_list.append(image_proj)
                    image_embedding_neg_list.append(neg_image_proj)
        if len(image_embedding_list) > 1:
            # print("image_embedding_list:",[emb.shape for emb in image_embedding_list])
            image_embedding_list = [torch.cat(image_embedding_list, axis=1)]
            # print("image_embedding_list after cat:", image_embedding_list[0].shape)
            image_embedding_neg_list = [image_embedding_neg_list[0]]

        
        print("ip_atten_mask is not None:", ip_atten_mask is not None)
        if ip_atten_mask is not None:
            ip_atten_mask = [i.to(self.device) for i in ip_atten_mask]
        
        if self.controlnet_loaded:
        # if True:
            # controlnet_image = self.annotator(controlnet_image, width, height)
            print("processing controlnet_image", controlnet_image.convert("RGB").save("controlnet_image.jpg"))
            controlnet_image = torch.from_numpy((np.array(controlnet_image) / 127.5) - 1)
            controlnet_image = controlnet_image.permute(
                2, 0, 1).unsqueeze(0).to(torch.bfloat16).to(self.device)
            
            # do 
            # print("do --vae encoder \n" * 100)
            # if do_vae:
            controlnet_image = self.ae.encode(controlnet_image.to(torch.float)).to(torch.bfloat16)
            # print(controlnet_image.shape)
            controlnet_image = rearrange(controlnet_image, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
            # print(controlnet_image.shape)
        
            if controlnet_mask_image:
                print("processing controlnet_mask_image", controlnet_mask_image.convert("RGB").save("controlnet_mask_image.jpg"))
                controlnet_mask_image = torch.from_numpy((np.array(controlnet_mask_image) / 127.5) - 1)
                controlnet_mask_image = controlnet_mask_image.permute(
                    2, 0, 1).unsqueeze(0).to(torch.bfloat16).to(self.device)
                
                # do 
                # print("do --vae encoder \n" * 100)
                # if do_vae:
                controlnet_mask_image = self.ae.encode(controlnet_mask_image.to(torch.float)).to(torch.bfloat16)
                # print(controlnet_image.shape)
                controlnet_mask_image = rearrange(controlnet_mask_image, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
                # print(controlnet_image.shape)
                 
        if do_inpainting == True:
             
            inpainting_mask = ip_atten_mask[0]

            inpainting_mask = inpainting_mask.to(torch.bfloat16)
            # print(inpainting_mask.shape)
            for xx in ip_atten_mask:
                # print("xx", xx.sum())
                inpainting_mask += xx.to(torch.bfloat16)

            inpainting_mask[inpainting_mask <= 0.2] = 255.0
            inpainting_mask[inpainting_mask < 10] = 0
            inpainting_mask = inpainting_mask / 255.0

            # inpainting_mask[inpainting_mask <= 0] = 0
            # inpainting_mask[inpainting_mask > 0] = 1
            # inpainting_mask = inpainting_mask / 255

            # inpainting_mask = inpainting_mask + 1
            # inpainting_mask = (inpainting_mask / 127.5) - 1
            inpainting_mask = inpainting_mask.reshape(1, -1, 1).to(torch.bfloat16) #rearrange(inpainting_mask, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=1, pw=1).to(torch.bfloat16)
            # print(inpainting_mask.shape)
            # print("inpainting_mask:", inpainting_mask.sum())

            # print(controlnet_mask_image.shape, controlnet_image.shape, inpainting_mask.shape)
            # ip_atten_mask = None
            # controlnet_image = torch.cat([controlnet_mask_image, inpainting_mask], axis=-1)
            controlnet_image = torch.cat([controlnet_mask_image, controlnet_image, inpainting_mask], axis=-1)
            # print(controlnet_mask_image.shape, controlnet_image.shape, inpainting_mask.shape, controlnet_image.shape)
        else:
            controlnet_mask_image = None
            inpainting_mask = None

        return self.forward(
            prompt,
            width,
            height,
            guidance,
            num_steps,
            seed,
            controlnet_image,
            timestep_to_start_cfg=timestep_to_start_cfg,
            true_gs=true_gs,
            control_weight=control_weight,
            neg_prompt=neg_prompt,
            image_proj=image_embedding_list,
            neg_image_proj=image_embedding_neg_list,
            ip_scale=ip_scale,
            neg_ip_scale=neg_ip_scale,
            do_vae=do_vae,
            ip_atten_mask=ip_atten_mask,
            do_inpainting=do_inpainting,
            mask_image=controlnet_mask_image,
            inpainting_mask=inpainting_mask,
            prompt_cloth=prompt_cloth
        )

    @torch.inference_mode()
    def gradio_generate(self, prompt, image_prompt, controlnet_image, width, height, guidance,
                        num_steps, seed, true_gs, ip_scale, neg_ip_scale, neg_prompt,
                        neg_image_prompt, timestep_to_start_cfg, control_type, control_weight,
                        lora_weight, local_path, lora_local_path, ip_local_path):
        if controlnet_image is not None:
            controlnet_image = Image.fromarray(controlnet_image)
            if ((self.controlnet_loaded and control_type != self.control_type)
                or not self.controlnet_loaded):
                if local_path is not None:
                    self.set_controlnet(control_type, local_path=local_path)
                else:
                    self.set_controlnet(control_type, local_path=None,
                                        repo_id=f"xlabs-ai/flux-controlnet-{control_type}-v3",
                                        name=f"flux-{control_type}-controlnet-v3.safetensors")
        if lora_local_path is not None:
            self.set_lora(local_path=lora_local_path, lora_weight=lora_weight)
        if image_prompt is not None:
            image_prompt = Image.fromarray(image_prompt)
            if neg_image_prompt is not None:
                neg_image_prompt = Image.fromarray(neg_image_prompt)
            if not self.ip_loaded:
                if ip_local_path is not None:
                    self.set_ip(local_path=ip_local_path)
                else:
                    self.set_ip(repo_id="xlabs-ai/flux-ip-adapter",
                                name="flux-ip-adapter.safetensors")
        seed = int(seed)
        if seed == -1:
            seed = torch.Generator(device="cpu").seed()

        img = self(prompt, image_prompt, controlnet_image, width, height, guidance,
                   num_steps, seed, true_gs, control_weight, ip_scale, neg_ip_scale, neg_prompt,
                   neg_image_prompt, timestep_to_start_cfg)

        filename = f"output/gradio/{uuid.uuid4()}.jpg"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        exif_data = Image.Exif()
        exif_data[ExifTags.Base.Make] = "XLabs AI"
        exif_data[ExifTags.Base.Model] = self.model_type
        img.save(filename, format="jpeg", exif=exif_data, quality=95, subsampling=0)
        return img, filename

    def forward(
        self,
        prompt,
        width,
        height,
        guidance,
        num_steps,
        seed,
        controlnet_image = None,
        timestep_to_start_cfg = 0,
        true_gs = 3.5,
        control_weight = 0.9,
        neg_prompt="",
        image_proj=None,
        neg_image_proj=None,
        ip_scale=1.0,
        neg_ip_scale=1.0,
        do_vae=False,
        ip_atten_mask=None, 
        do_inpainting=False,
        mask_image=None,
        inpainting_mask=None,
        prompt_cloth=None
    ):
        x = get_noise(
            1, height, width, device=self.device,
            dtype=torch.bfloat16, seed=seed
        )
        # print(controlnet_image.shape)
        # x = 0.99*x + 0.01 * controlnet_image
        # print(x.shape)
        #image = np.array(controlnet_image).shape
        #x = torch.from_numpy().to(torch.bfloat16).to(accelerator.device)
        # x = xtorch.randn_like(x).to(accelerator.device)
        # print(x.shape)

        timesteps = get_schedule(
            num_steps,
            (width // 8) * (height // 8) // (16 * 16),
            shift=True,
        )
        torch.manual_seed(seed)
        with torch.no_grad():
            if self.offload:
                self.t5, self.clip = self.t5.to(self.device), self.clip.to(self.device)
            # inp_cond = prepare(t5=self.t5, clip=self.clip, img=x, prompt=prompt)
            inp_cond = prepare_v2(t5=self.t5, clip=self.clip, img=x, prompt=prompt, cloth_prompt=prompt_cloth)

            # if do_inpainting is True:
            #     controlnet_image = torch.cat([inp_cond["img"], controlnet_image], axis=-1)

            if not self.input_proj_loaded:
                self.input_proj = None
               
            print("neg_prompt:", neg_prompt)
            neg_inp_cond = prepare(t5=self.t5, clip=self.clip, img=x, prompt=neg_prompt)

            if self.offload:
                self.offload_model_to_cpu(self.t5, self.clip)
                self.model = self.model.to(self.device)
            if self.controlnet_loaded:
                x = denoise_controlnet(
                    self.model,
                    **inp_cond,
                    controlnet=self.controlnet,
                    timesteps=timesteps,
                    guidance=guidance,
                    controlnet_cond=controlnet_image,
                    timestep_to_start_cfg=timestep_to_start_cfg,
                    neg_txt=neg_inp_cond['txt'],
                    neg_txt_ids=neg_inp_cond['txt_ids'],
                    neg_vec=neg_inp_cond['vec'],
                    true_gs=true_gs,
                    controlnet_gs=control_weight,
                    image_proj=image_proj,
                    neg_image_proj=neg_image_proj,
                    ip_scale=ip_scale,
                    neg_ip_scale=neg_ip_scale,
                    do_vae=do_vae,
                    ip_atten_mask=ip_atten_mask,
                    do_inpainting=do_inpainting,
                    mask_image=mask_image,
                    inpainting_mask=inpainting_mask,
                    input_proj=self.input_proj
                )
            else:
                # import pdb; pdb.set_trace()
                x = denoise(
                    self.model,
                    **inp_cond,
                    timesteps=timesteps,
                    guidance=guidance,
                    timestep_to_start_cfg=timestep_to_start_cfg,
                    neg_txt=neg_inp_cond['txt'],
                    neg_txt_ids=neg_inp_cond['txt_ids'],
                    neg_vec=neg_inp_cond['vec'],
                    true_gs=true_gs,
                    image_proj=image_proj,
                    neg_image_proj=neg_image_proj,
                    ip_scale=ip_scale,
                    neg_ip_scale=neg_ip_scale,
                    ip_atten_mask=ip_atten_mask
                )

            if self.offload:
                self.offload_model_to_cpu(self.model)
                self.ae.decoder.to(x.device)
            x = unpack(x.float(), height, width)
            x = self.ae.decode(x)
            self.offload_model_to_cpu(self.ae.decoder)

        x1 = x.clamp(-1, 1)
        x1 = rearrange(x1[-1], "c h w -> h w c")
        output_img = Image.fromarray((127.5 * (x1 + 1.0)).cpu().byte().numpy())
        return output_img

    def offload_model_to_cpu(self, *models):
        if not self.offload: return
        for model in models:
            model.cpu()
            torch.cuda.empty_cache()
