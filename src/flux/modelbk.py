from dataclasses import dataclass

import torch
from torch import Tensor, nn
from einops import rearrange

from .modules.layers import (DoubleStreamBlock, EmbedND, LastLayer,
                                 MLPEmbedder, SingleStreamBlock,
                                 timestep_embedding)


@dataclass
class FluxParams:
    in_channels: int
    vec_in_dim: int
    context_in_dim: int
    hidden_size: int
    mlp_ratio: float
    num_heads: int
    depth: int
    depth_single_blocks: int
    axes_dim: list[int]
    theta: int
    qkv_bias: bool
    guidance_embed: bool

def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module

class Flux(nn.Module):
    """
    Transformer model for flow matching on sequences.
    """
    _supports_gradient_checkpointing = True

    def __init__(self, params: FluxParams, inpainting_flag=False):
        super().__init__()

        self.params = params
        self.in_channels = params.in_channels
        self.out_channels = self.in_channels
        if params.hidden_size % params.num_heads != 0:
            raise ValueError(
                f"Hidden size {params.hidden_size} must be divisible by num_heads {params.num_heads}"
            )
        pe_dim = params.hidden_size // params.num_heads
        if sum(params.axes_dim) != pe_dim:
            raise ValueError(f"Got {params.axes_dim} but expected positional dim {pe_dim}")
        self.hidden_size = params.hidden_size
        self.num_heads = params.num_heads
        self.pe_embedder = EmbedND(dim=pe_dim, theta=params.theta, axes_dim=params.axes_dim)
        self.img_in = nn.Linear(self.in_channels, self.hidden_size, bias=True)

        # self.inpainting_flag = inpainting_flag
        # self.inpainting_in = None
        # if self.inpainting_flag is True:
        #     self.inpainting_in = zero_module(nn.Linear(self.in_channels * 2 + 1, self.in_channels, bias=True))
             

        self.time_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size)
        self.vector_in = MLPEmbedder(params.vec_in_dim, self.hidden_size)
        self.guidance_in = (
            MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size) if params.guidance_embed else nn.Identity()
        )
        self.txt_in = nn.Linear(params.context_in_dim, self.hidden_size)

        self.double_blocks = nn.ModuleList(
            [
                DoubleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=params.mlp_ratio,
                    qkv_bias=params.qkv_bias,
                )
                for _ in range(params.depth)
            ]
        )

        self.single_blocks = nn.ModuleList(
            [
                SingleStreamBlock(self.hidden_size, self.num_heads, mlp_ratio=params.mlp_ratio)
                for _ in range(params.depth_single_blocks)
            ]
        )

        self.final_layer = LastLayer(self.hidden_size, 1, self.out_channels)
        self.gradient_checkpointing = False

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    # @property
    # def set_inpainting_proj(self):
    #     self.inpainting_flag = True
    #     self.inpainting_in = zero_module(nn.Linear(self.in_channels * 2 + 1, self.in_channels, bias=True))

    @property
    def attn_processors(self):
        # set recursively
        processors = {}

        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors):
            if hasattr(module, "set_processor"):
                processors[f"{name}.processor"] = module.processor

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    def set_attn_processor(self, processor):
        r"""
        Sets the attention processor to use to compute attention.

        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.

                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.

        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    def forward(
        self,
        img: Tensor,
        img_ids: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        timesteps: Tensor,
        y: Tensor,
        block_controlnet_hidden_states=None,
        guidance: Tensor | None = None,
        image_proj: Tensor | None = None, 
        ip_scale: Tensor | float = 1.0, 
        ip_atten_mask: Tensor | None = None,
        lora_weight_list = None,
        control_depth=None,
        c_inp=None,
        **kwargs
    ) -> Tensor:
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")
        # running on sequences img

        img = self.img_in(img)
        vec = self.time_in(timestep_embedding(timesteps, 256))
        if self.params.guidance_embed:
            if guidance is None:
                raise ValueError("Didn't get guidance strength for guidance distilled model.")
            vec = vec + self.guidance_in(timestep_embedding(guidance, 256))
        vec = vec + self.vector_in(y)
        txt = self.txt_in(txt)

        ids = torch.cat((txt_ids, img_ids), dim=1)
        pe = self.pe_embedder(ids)
        pe2 = None
        # if image_proj is None:
        #     pe2 = None
        # else:
        #     image_proj_ids = torch.zeros(image_proj.shape[0], image_proj.shape[1], 3).to(img_ids.device).to(img_ids.dtype)
            
        #     ids = torch.cat((image_proj_ids, img_ids), dim=1)
        #     pe2 = self.pe_embedder(ids)
        if block_controlnet_hidden_states is not None:
            controlnet_depth = len(block_controlnet_hidden_states)
        for index_block, block in enumerate(self.double_blocks):
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                encoder_hidden_states, hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    img,
                    txt,
                    vec,
                    pe,
                    pe2,
                    image_proj,
                    ip_scale,
                    ip_atten_mask
                )
            else:
                img, txt = block(
                    img=img, 
                    txt=txt, 
                    vec=vec, 
                    pe=pe, 
                    pe2=pe2,
                    image_proj=image_proj,
                    ip_scale=ip_scale, 
                    ip_atten_mask=ip_atten_mask,
                    lora_weight_list=lora_weight_list,
                    **kwargs
                )
                
            # controlnet residual
            if block_controlnet_hidden_states is not None:
                img = img + block_controlnet_hidden_states[index_block % 2]
            # if block_controlnet_hidden_states is not None:
            #     if control_depth is None:
            #         img = img + block_controlnet_hidden_states[index_block % 2]
            #     else:
            #         if control_depth == -1:
            #             if index_block < len(block_controlnet_hidden_states):
            #                 img = img + block_controlnet_hidden_states[index_block]
            #         else:
            #             # print(ip_atten_mask[0].shape, block_controlnet_hidden_states[index_block % control_depth].shape)
            #             '''
            #             ip_query = []
            #             for bi in range(img_q.shape[0]):
            #                 ip_query.append(img_q[bi][ip_atten_mask_tmp[bi]])
            #             ip_query = torch.stack(ip_query, axis=0)
            #             '''
            #             mask_tmp = torch.stack([ip_atten_mask[0]] * block_controlnet_hidden_states[index_block % control_depth].shape[-1], axis=-1)
            #             img = img + block_controlnet_hidden_states[index_block % control_depth] * (1 - mask_tmp.to(img.dtype))
            # control_depth = 2
            # if block_controlnet_hidden_states is not None:
            #     mask_tmp = torch.stack([ip_atten_mask[0]] * block_controlnet_hidden_states[index_block % control_depth].shape[-1], axis=-1)
            #     img = img + block_controlnet_hidden_states[index_block % control_depth] * (1 - mask_tmp.to(img.dtype))
                
        img = torch.cat((txt, img), 1)
 
        for block in self.single_blocks:
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                encoder_hidden_states, hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    img,
                    vec,
                    pe,
                    image_proj,
                    ip_scale,
                    ip_atten_mask
                )
            else:
                img = block(img, vec=vec, pe=pe,
                            image_proj=image_proj,
                            ip_scale=ip_scale, 
                            ip_atten_mask=ip_atten_mask,
                            lora_weight_list=lora_weight_list,
                            **kwargs
                            )

        img = img[:, txt.shape[1] :, ...]

        img = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)
        return img


class Flux_v2(nn.Module):
    """
    Transformer model for flow matching on sequences.
    """
    _supports_gradient_checkpointing = True

    def __init__(self, params: FluxParams, inpainting_flag=False):
        super().__init__()

        self.params = params
        self.in_channels = params.in_channels
        self.out_channels = self.in_channels
        if params.hidden_size % params.num_heads != 0:
            raise ValueError(
                f"Hidden size {params.hidden_size} must be divisible by num_heads {params.num_heads}"
            )
        pe_dim = params.hidden_size // params.num_heads
        if sum(params.axes_dim) != pe_dim:
            raise ValueError(f"Got {params.axes_dim} but expected positional dim {pe_dim}")
        self.hidden_size = params.hidden_size
        self.num_heads = params.num_heads
        self.pe_embedder = EmbedND(dim=pe_dim, theta=params.theta, axes_dim=params.axes_dim)
        self.img_in = nn.Linear(self.in_channels, self.hidden_size, bias=True)

        # self.inpainting_flag = inpainting_flag
        # self.inpainting_in = None
        # if self.inpainting_flag is True:
        #     self.inpainting_in = zero_module(nn.Linear(self.in_channels * 2 + 1, self.in_channels, bias=True))
             

        self.time_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size)
        self.vector_in = MLPEmbedder(params.vec_in_dim, self.hidden_size)
        self.guidance_in = (
            MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size) if params.guidance_embed else nn.Identity()
        )
        self.txt_in = nn.Linear(params.context_in_dim, self.hidden_size)

        self.double_blocks = nn.ModuleList(
            [
                DoubleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=params.mlp_ratio,
                    qkv_bias=params.qkv_bias,
                )
                for _ in range(params.depth)
            ]
        )

        self.single_blocks = nn.ModuleList(
            [
                SingleStreamBlock(self.hidden_size, self.num_heads, mlp_ratio=params.mlp_ratio)
                for _ in range(params.depth_single_blocks)
            ]
        )

        self.final_layer = LastLayer(self.hidden_size, 1, self.out_channels)
        self.gradient_checkpointing = False

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    # @property
    # def set_inpainting_proj(self):
    #     self.inpainting_flag = True
    #     self.inpainting_in = zero_module(nn.Linear(self.in_channels * 2 + 1, self.in_channels, bias=True))

    @property
    def attn_processors(self):
        # set recursively
        processors = {}

        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors):
            if hasattr(module, "set_processor"):
                processors[f"{name}.processor"] = module.processor

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    def set_attn_processor(self, processor):
        r"""
        Sets the attention processor to use to compute attention.

        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.

                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.

        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    def forward(
        self,
        img: Tensor,
        img_ids: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        timesteps: Tensor,
        y: Tensor,
        txt_cloth: Tensor | None = None,
        txt_cloth_ids: Tensor | None = None,
        y_cloth: Tensor | None = None,
        block_controlnet_hidden_states=None,
        guidance: Tensor | None = None,
        image_proj: Tensor | None = None, 
        ip_scale: Tensor | float = 1.0, 
        ip_atten_mask: Tensor | None = None,
    ) -> Tensor:
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")
        # running on sequences img

        img = self.img_in(img)
        vec = self.time_in(timestep_embedding(timesteps, 256))
        if self.params.guidance_embed:
            if guidance is None:
                raise ValueError("Didn't get guidance strength for guidance distilled model.")
            vec = vec + self.guidance_in(timestep_embedding(guidance, 256))
        vec = vec + self.vector_in(y)
        txt = self.txt_in(txt)
        
        vec_cloth = self.time_in(timestep_embedding(timesteps, 256))
        if self.params.guidance_embed:
            vec_cloth = vec_cloth + self.guidance_in(timestep_embedding(guidance, 256))
        vec_cloth = vec_cloth + self.vector_in(y_cloth)
        txt_cloth = self.txt_in(txt_cloth)

        ids = torch.cat((txt_ids, img_ids), dim=1)
        pe = self.pe_embedder(ids)
        ids_cloth = torch.cat((txt_cloth_ids, img_ids), dim=1)
        pe_cloth = self.pe_embedder(ids_cloth)
        print("txt_cloth_ids", txt_cloth_ids.shape, "img_ids", img_ids.shape, "pe_cloth", pe_cloth.shape)
        pe2 = None
        # if image_proj is None:
        #     pe2 = None
        # else:
        #     image_proj_ids = torch.zeros(image_proj.shape[0], image_proj.shape[1], 3).to(img_ids.device).to(img_ids.dtype)
            
        #     ids = torch.cat((image_proj_ids, img_ids), dim=1)
        #     pe2 = self.pe_embedder(ids)
        if block_controlnet_hidden_states is not None:
            controlnet_depth = len(block_controlnet_hidden_states)
        for index_block, block in enumerate(self.double_blocks):
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                encoder_hidden_states, hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    img,
                    txt,
                    vec,
                    pe,
                    pe2,
                    image_proj,
                    ip_scale,
                    ip_atten_mask
                )
            else:
                print("ip_atten_mask-model", ip_atten_mask)
                img, txt, txt_cloth = block(
                    img=img, 
                    txt=txt, 
                    vec=vec, 
                    pe=pe, 
                    pe2=pe2,
                    image_proj=image_proj,
                    ip_scale=ip_scale, 
                    ip_atten_mask=ip_atten_mask,
                    txt_cloth=txt_cloth,
                    vec_cloth=vec_cloth,
                    pe_cloth=pe_cloth
                )
                
            # controlnet residual
            if block_controlnet_hidden_states is not None:
                img = img + block_controlnet_hidden_states[index_block % 2]
                
        # img = torch.cat((txt, img), 1)
        for block in self.single_blocks:
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                encoder_hidden_states, hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    img,
                    vec,
                    pe,
                    image_proj,
                    ip_scale,
                    ip_atten_mask
                )
            else:
                txt, txt_cloth, img = block(img, txt=txt, vec=vec, pe=pe,
                            image_proj=image_proj,
                            ip_scale=ip_scale, 
                            ip_atten_mask=ip_atten_mask,
                            txt_cloth=txt_cloth,
                            vec_cloth=vec_cloth,
                            pe_cloth=pe_cloth,
                            )

        # img = img[:, txt.shape[1] :, ...]

        img = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)
        return img
