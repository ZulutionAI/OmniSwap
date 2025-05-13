import math
from dataclasses import dataclass

import torch
from einops import rearrange
from torch import Tensor, nn

from ..math import attention, rope
import torch.nn.functional as F
from typing import Tuple
from diffusers.models.modeling_utils import ModelMixin

class EmbedND(nn.Module):
    def __init__(self, dim: int, theta: int, axes_dim: list[int]):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: Tensor) -> Tensor:
        n_axes = ids.shape[-1]
        emb = torch.cat(
            [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)],
            dim=-3,
        )

        return emb.unsqueeze(1)


def timestep_embedding(t: Tensor, dim, max_period=10000, time_factor: float = 1000.0):
    """
    Create sinusoidal timestep embeddings.
    :param t: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an (N, D) Tensor of positional embeddings.
    """
    t = time_factor * t
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
        t.device
    )

    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    if torch.is_floating_point(t):
        embedding = embedding.to(t)
    return embedding


class MLPEmbedder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim, bias=True)
        self.silu = nn.SiLU()
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.out_layer(self.silu(self.in_layer(x)))


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor):
        x_dtype = x.dtype
        x = x.float()
        rrms = torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + 1e-6)
        return (x * rrms).to(dtype=x_dtype) * self.scale


class QKNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.query_norm = RMSNorm(dim)
        self.key_norm = RMSNorm(dim)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        q = self.query_norm(q)
        k = self.key_norm(k)
        return q.to(v), k.to(v)

class LoRALinearLayer(nn.Module):
    def __init__(self, in_features, out_features, rank=4, network_alpha=None, device=None, dtype=None):
        super().__init__()

        self.down = nn.Linear(in_features, rank, bias=False, device=device, dtype=dtype)
        self.up = nn.Linear(rank, out_features, bias=False, device=device, dtype=dtype)
        # This value has the same meaning as the `--network_alpha` option in the kohya-ss trainer script.
        # See https://github.com/darkstorm2150/sd-scripts/blob/main/docs/train_network_README-en.md#execute-learning
        self.network_alpha = network_alpha
        self.rank = rank

        nn.init.normal_(self.down.weight, std=1 / rank)
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden_states):
        orig_dtype = hidden_states.dtype
        dtype = self.down.weight.dtype

        down_hidden_states = self.down(hidden_states.to(dtype))
        up_hidden_states = self.up(down_hidden_states)

        if self.network_alpha is not None:
            up_hidden_states *= self.network_alpha / self.rank

        return up_hidden_states.to(orig_dtype)

class FLuxSelfAttnProcessor:
    def __call__(self, attn, x, pe, **attention_kwargs):
        print('2' * 30)

        qkv = attn.qkv(x)
        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = attn.norm(q, k, v)
        x = attention(q, k, v, pe=pe)
        x = attn.proj(x)
        return x

class LoraFluxAttnProcessor(nn.Module):

    def __init__(self, dim: int, rank=4, network_alpha=None, lora_weight=1):
        super().__init__()
        self.qkv_lora = LoRALinearLayer(dim, dim * 3, rank, network_alpha)
        self.proj_lora = LoRALinearLayer(dim, dim, rank, network_alpha)
        self.lora_weight = lora_weight


    def __call__(self, attn, x, pe, **attention_kwargs):
        qkv = attn.qkv(x) + self.qkv_lora(x) * self.lora_weight
        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = attn.norm(q, k, v)
        x = attention(q, k, v, pe=pe)
        x = attn.proj(x) + self.proj_lora(x) * self.lora_weight
        # print('1' * 30)
        # print(x.norm(), (self.proj_lora(x) * self.lora_weight).norm(), 'norm')
        return x

class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.norm = QKNorm(head_dim)
        self.proj = nn.Linear(dim, dim)
    def forward():
        pass


@dataclass
class ModulationOut:
    shift: Tensor
    scale: Tensor
    gate: Tensor


class Modulation(nn.Module):
    def __init__(self, dim: int, double: bool):
        super().__init__()
        self.is_double = double
        self.multiplier = 6 if double else 3
        self.lin = nn.Linear(dim, self.multiplier * dim, bias=True)

    def forward(self, vec: Tensor) -> tuple[ModulationOut, ModulationOut | None]:
        out = self.lin(nn.functional.silu(vec))[:, None, :].chunk(self.multiplier, dim=-1)

        return (
            ModulationOut(*out[:3]),
            ModulationOut(*out[3:]) if self.is_double else None,
        )

class DoubleStreamBlockMultiLoraProcessor(nn.Module):
    def __init__(self, dim: int, ranks=[], network_alpha=None, lora_weights = [], lora_state_dicts = [], device=None):
        super().__init__()
        # self.qkv_lora1 = LoRALinearLayer(dim, dim * 3, rank, network_alpha)
        # self.proj_lora1 = LoRALinearLayer(dim, dim, rank, network_alpha)
        # self.qkv_lora2 = LoRALinearLayer(dim, dim * 3, rank, network_alpha)
        # self.proj_lora2 = LoRALinearLayer(dim, dim, rank, network_alpha)
        self.lora_num = len(lora_weights)
        self.qkv_lora1s = []
        self.proj_lora1s = []
        self.qkv_lora2s = []
        self.proj_lora2s = []
        self.lora_weights = lora_weights

        for i in range(self.lora_num):
            temp = DoubleStreamBlockLoraProcessor(dim=3072, rank=ranks[i])
            temp.load_state_dict(lora_state_dicts[i])
            temp.to(device)

            qkv1 = temp.qkv_lora1
            proj1 = temp.proj_lora1
            qkv2 = temp.qkv_lora2
            proj2 = temp.proj_lora2

            self.qkv_lora1s.append(qkv1)
            self.proj_lora1s.append(proj1)
            self.qkv_lora2s.append(qkv2)
            self.proj_lora2s.append(proj2)

    def forward(self, attn, img, txt, vec, pe, **attention_kwargs):
        img_mod1, img_mod2 = attn.img_mod(vec)
        txt_mod1, txt_mod2 = attn.txt_mod(vec)

        # prepare image for attention
        img_modulated = attn.img_norm1(img)
        img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
        img_qkv = attn.img_attn.qkv(img_modulated)
        for i in range(self.lora_num):
            img_qkv += self.qkv_lora1s[i](img_modulated) * self.lora_weights[i]
        img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads)
        img_q, img_k = attn.img_attn.norm(img_q, img_k, img_v)

        # prepare txt for attention
        txt_modulated = attn.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        txt_qkv = attn.txt_attn.qkv(txt_modulated)
        for i in range(self.lora_num):
            txt_qkv += self.qkv_lora2s[i](txt_modulated) * self.lora_weights[i]
        txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads)
        txt_q, txt_k = attn.txt_attn.norm(txt_q, txt_k, txt_v)

        # run actual attention
        q = torch.cat((txt_q, img_q), dim=2)
        k = torch.cat((txt_k, img_k), dim=2)
        v = torch.cat((txt_v, img_v), dim=2)

        attn1 = attention(q, k, v, pe=pe)
        txt_attn, img_attn = attn1[:, : txt.shape[1]], attn1[:, txt.shape[1] :]

        # calculate the img bloks
        img = img + img_mod1.gate * attn.img_attn.proj(img_attn)
        for i in range(self.lora_num):
            img += img_mod1.gate * self.proj_lora1s[i](img_attn) * self.lora_weights[i]
        img = img + img_mod2.gate * attn.img_mlp((1 + img_mod2.scale) * attn.img_norm2(img) + img_mod2.shift)

        # calculate the txt bloks
        txt = txt + txt_mod1.gate * attn.txt_attn.proj(txt_attn)
        for i in range(self.lora_num):
            txt += txt_mod1.gate * self.proj_lora2s[i](txt_attn) * self.lora_weights[i]
        txt = txt + txt_mod2.gate * attn.txt_mlp((1 + txt_mod2.scale) * attn.txt_norm2(txt) + txt_mod2.shift)
        return img, txt

        
class DoubleStreamBlockLoraProcessorV2(nn.Module):
    def __init__(self, dim: int, rank=4, network_alpha=None, lora_weight=1):
        super().__init__()
        self.qkv_lora1 = LoRALinearLayer(dim, dim * 3, rank, network_alpha)
        self.proj_lora1 = LoRALinearLayer(dim, dim, rank, network_alpha)
        self.qkv_lora2 = LoRALinearLayer(dim, dim * 3, rank, network_alpha)
        self.proj_lora2 = LoRALinearLayer(dim, dim, rank, network_alpha)
        self.lora_weight = lora_weight

    def forward(self, attn, img, txt, vec, pe, lora_weight=1, **attention_kwargs):
        self.lora_weight = lora_weight
        img_mod1, img_mod2 = attn.img_mod(vec)
        txt_mod1, txt_mod2 = attn.txt_mod(vec)

        # prepare image for attention
        img_modulated = attn.img_norm1(img)
        img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
        img_qkv = attn.img_attn.qkv(img_modulated)
        q, k, v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads)
        self.KQQ = F.scaled_dot_product_attention(k[:,:,:512,:], q[:,:,512:,:], q[:,:,512:,:], dropout_p=0.0, is_causal=False)
        self.Q = q[:,:,512:,:]
        img_qkv = img_qkv + self.qkv_lora1(img_modulated) * self.lora_weight
        img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads)
        img_q, img_k = attn.img_attn.norm(img_q, img_k, img_v)

        # prepare txt for attention
        txt_modulated = attn.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        txt_qkv = attn.txt_attn.qkv(txt_modulated) + self.qkv_lora2(txt_modulated) * self.lora_weight
        txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads)
        txt_q, txt_k = attn.txt_attn.norm(txt_q, txt_k, txt_v)

        # run actual attention
        q = torch.cat((txt_q, img_q), dim=2)
        k = torch.cat((txt_k, img_k), dim=2)
        v = torch.cat((txt_v, img_v), dim=2)

        attn1 = attention(q, k, v, pe=pe)
        txt_attn, img_attn = attn1[:, : txt.shape[1]], attn1[:, txt.shape[1] :]

        # calculate the img bloks
        img = img + img_mod1.gate * attn.img_attn.proj(img_attn) + img_mod1.gate * self.proj_lora1(img_attn) * self.lora_weight
        img = img + img_mod2.gate * attn.img_mlp((1 + img_mod2.scale) * attn.img_norm2(img) + img_mod2.shift)

        # calculate the txt bloks
        txt = txt + txt_mod1.gate * attn.txt_attn.proj(txt_attn) + txt_mod1.gate * self.proj_lora2(txt_attn) * self.lora_weight
        txt = txt + txt_mod2.gate * attn.txt_mlp((1 + txt_mod2.scale) * attn.txt_norm2(txt) + txt_mod2.shift)
        return img, txt, self.KQQ, self.Q

class DoubleStreamBlockLoraProcessor(nn.Module):
    def __init__(self, dim: int, rank=4, network_alpha=None, lora_weight=1):
        super().__init__()
        self.qkv_lora1 = LoRALinearLayer(dim, dim * 3, rank, network_alpha)
        self.proj_lora1 = LoRALinearLayer(dim, dim, rank, network_alpha)
        self.qkv_lora2 = LoRALinearLayer(dim, dim * 3, rank, network_alpha)
        self.proj_lora2 = LoRALinearLayer(dim, dim, rank, network_alpha)
        self.lora_weight = lora_weight

    def forward(self, attn, img, txt, vec, pe, **attention_kwargs):
        img_mod1, img_mod2 = attn.img_mod(vec)
        txt_mod1, txt_mod2 = attn.txt_mod(vec)

        # prepare image for attention
        img_modulated = attn.img_norm1(img)
        img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
        img_qkv = attn.img_attn.qkv(img_modulated) + self.qkv_lora1(img_modulated) * self.lora_weight
        img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads)
        img_q, img_k = attn.img_attn.norm(img_q, img_k, img_v)

        # prepare txt for attention
        txt_modulated = attn.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        txt_qkv = attn.txt_attn.qkv(txt_modulated) + self.qkv_lora2(txt_modulated) * self.lora_weight
        txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads)
        txt_q, txt_k = attn.txt_attn.norm(txt_q, txt_k, txt_v)

        # run actual attention
        q = torch.cat((txt_q, img_q), dim=2)
        k = torch.cat((txt_k, img_k), dim=2)
        v = torch.cat((txt_v, img_v), dim=2)

        attn1 = attention(q, k, v, pe=pe)
        txt_attn, img_attn = attn1[:, : txt.shape[1]], attn1[:, txt.shape[1] :]

        # calculate the img bloks
        img = img + img_mod1.gate * attn.img_attn.proj(img_attn) + img_mod1.gate * self.proj_lora1(img_attn) * self.lora_weight
        img = img + img_mod2.gate * attn.img_mlp((1 + img_mod2.scale) * attn.img_norm2(img) + img_mod2.shift)

        # calculate the txt bloks
        txt = txt + txt_mod1.gate * attn.txt_attn.proj(txt_attn) + txt_mod1.gate * self.proj_lora2(txt_attn) * self.lora_weight
        txt = txt + txt_mod2.gate * attn.txt_mlp((1 + txt_mod2.scale) * attn.txt_norm2(txt) + txt_mod2.shift)
        return img, txt

class IPDoubleStreamBlockProcessorV2(nn.Module):
    """Attention processor for handling IP-adapter with double stream block."""

    def __init__(self, context_dim, hidden_dim, num_heads=24):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "IPDoubleStreamBlockProcessor requires PyTorch 2.0 or higher. Please upgrade PyTorch."
            )

        # Ensure context_dim matches the dimension of image_proj
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim
        self.mlp_hidden_dim = hidden_dim * 2

        self.head_dim = hidden_dim // num_heads
         
        # Initialize projections for IP-adapter 4096 3072
        
        self.ip_adapter_double_stream_proj = nn.Linear(context_dim, hidden_dim, bias=True)

        self.ip_adapter_double_stream_k_proj = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.ip_adapter_double_stream_v_proj = nn.Linear(context_dim, hidden_dim, bias=True)

        nn.init.zeros_(self.ip_adapter_double_stream_k_proj.weight)
        nn.init.zeros_(self.ip_adapter_double_stream_k_proj.bias)

        nn.init.zeros_(self.ip_adapter_double_stream_v_proj.weight)
        nn.init.zeros_(self.ip_adapter_double_stream_v_proj.bias)

        nn.init.zeros_(self.ip_adapter_double_stream_proj.weight)
        nn.init.zeros_(self.ip_adapter_double_stream_proj.weight)

        self.ip_adapter_double_mod = Modulation(hidden_dim, double=True)
        self.ip_adapter_double_norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.ip_adapter_double_attn = SelfAttention(dim=hidden_dim, num_heads=num_heads, qkv_bias=True)

        self.ip_adapter_double_norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.ip_adapter_double_mlp = nn.Sequential(
            nn.Linear(hidden_dim, self.mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(self.mlp_hidden_dim, hidden_dim, bias=True),
        )

        self.ip_adapter_double_stream_proj_2 = nn.Linear(hidden_dim, context_dim, bias=True)
        nn.init.zeros_(self.ip_adapter_double_stream_proj_2.weight)
        nn.init.zeros_(self.ip_adapter_double_stream_proj_2.weight)

    def __call__(self, attn, img, txt, vec, pe, pe2=None, image_proj=None, ip_scale=1.0, ip_atten_mask=None, **attention_kwargs):

        # Prepare image for attention
        # import pdb; pdb.set_trace()
        ip_img = img.clone()
        #
        ip_adapter_vec = self.ip_adapter_double_stream_proj(image_proj)

        img_mod1, img_mod2 = attn.img_mod(vec)
        txt_mod1, txt_mod2 = attn.txt_mod(vec)
        ip_adapter_mod1, ip_adapter_mod2 = self.ip_adapter_double_mod(vec)

        img_modulated = attn.img_norm1(img)
        img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
        img_qkv = attn.img_attn.qkv(img_modulated)
        img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads, D=attn.head_dim)
        img_q, img_k = attn.img_attn.norm(img_q, img_k, img_v)

        txt_modulated = attn.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        txt_qkv = attn.txt_attn.qkv(txt_modulated)
        txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads, D=attn.head_dim)
        txt_q, txt_k = attn.txt_attn.norm(txt_q, txt_k, txt_v)

        q = torch.cat((txt_q, img_q), dim=2)
        k = torch.cat((txt_k, img_k), dim=2)
        v = torch.cat((txt_v, img_v), dim=2)

        # print(q.shape, txt_q.shape, img_q.shape)
        attn1 = attention(q, k, v, pe=pe)

        txt_attn, img_attn = attn1[:, :txt.shape[1]], attn1[:, txt.shape[1]:]

        # print(f"txt_attn shape: {txt_attn.size()}")
        # print(f"img_attn shape: {img_attn.size()}")

        img = img + img_mod1.gate * attn.img_attn.proj(img_attn)
        img = img + img_mod2.gate * attn.img_mlp((1 + img_mod2.scale) * attn.img_norm2(img) + img_mod2.shift)

        txt = txt + txt_mod1.gate * attn.txt_attn.proj(txt_attn)
        txt = txt + txt_mod2.gate * attn.txt_mlp((1 + txt_mod2.scale) * attn.txt_norm2(txt) + txt_mod2.shift)

        # print(ip_scale, image_proj)
        # IP-adapter processing
        # ip_query = img_q  # latent sample query

        ip_adapter_modulated = self.ip_adapter_double_norm1(ip_adapter_vec)
        ip_adapter_modulated = (1 + ip_adapter_mod1.scale) * ip_adapter_modulated + ip_adapter_mod2.shift

        ip_adapter_qkv = self.ip_adapter_double_attn.qkv(ip_adapter_modulated)

        ip_adapter_q, ip_adapter_k, ip_adapter_v = rearrange(ip_adapter_qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads, D=attn.head_dim)
        ip_adapter_q, ip_adapter_k = self.ip_adapter_double_attn.norm(ip_adapter_q, ip_adapter_k, ip_adapter_v)

        q = torch.cat((ip_adapter_q, img_q), dim=2)
        k = torch.cat((ip_adapter_k, img_k), dim=2)
        v = torch.cat((ip_adapter_v, img_v), dim=2)

        # print(ip_adapter_q.shape, img_q.shape)
        # print(q.shape, v.shape, k.shape)
        # print(pe.shape)
        attn1 = attention(q, k, v, pe=pe2)

        ip_adapter_attn, img_ip_attn = attn1[:, :ip_adapter_vec.shape[1]], attn1[:, ip_adapter_vec.shape[1]:]

        ip_img = ip_img + img_mod1.gate * attn.img_attn.proj(img_ip_attn)
        ip_img = ip_img + img_mod2.gate * attn.img_mlp((1 + img_mod2.scale) * attn.img_norm2(ip_img) + img_mod2.shift)

        ip_adapter_vec = ip_adapter_vec + ip_adapter_mod1.gate * self.ip_adapter_double_attn.proj(ip_adapter_attn)
        ip_adapter_vec = ip_adapter_vec + ip_adapter_mod2.gate * \
                         self.ip_adapter_double_mlp((1 + ip_adapter_mod2.scale) * \
                         self.ip_adapter_double_norm2(ip_adapter_vec) + ip_adapter_mod2.shift)

        ip_adapter_vec = self.ip_adapter_double_stream_proj_2(ip_adapter_vec)

        img = img + ip_scale * ip_img
        # print("img, txt, ip_adapter_vec")
        return img, txt, ip_adapter_vec


class IPDoubleStreamBlockProcessorV3WithMultiLora(nn.Module):
    """Attention processor for handling IP-adapter with double stream block."""

    def __init__(self, context_dim, hidden_dim, dim: int, ranks=[], network_alpha=None, lora_weights = [], lora_state_dicts = [], device=None):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "IPDoubleStreamBlockProcessor requires PyTorch 2.0 or higher. Please upgrade PyTorch."
            )

        # Ensure context_dim matches the dimension of image_proj
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim

        # Initialize projections for IP-adapter 4096 
        self.ip_adapter_double_stream_k_proj = nn.Linear(context_dim, hidden_dim, bias=True)
        self.ip_adapter_double_stream_v_proj = nn.Linear(context_dim, hidden_dim, bias=True)

        nn.init.zeros_(self.ip_adapter_double_stream_k_proj.weight)
        nn.init.zeros_(self.ip_adapter_double_stream_k_proj.bias)

        nn.init.zeros_(self.ip_adapter_double_stream_v_proj.weight)
        nn.init.zeros_(self.ip_adapter_double_stream_v_proj.bias)
        
        self.lora_num = len(lora_weights)
        self.qkv_lora1s = []
        self.proj_lora1s = []
        self.qkv_lora2s = []
        self.proj_lora2s = []
        self.lora_weights = lora_weights

        for i in range(self.lora_num):
            temp = DoubleStreamBlockLoraProcessor(dim=3072, rank=ranks[i])
            temp.load_state_dict(lora_state_dicts[i], strict=False)
            temp.to(device)

            qkv1 = temp.qkv_lora1
            proj1 = temp.proj_lora1
            qkv2 = temp.qkv_lora2
            proj2 = temp.proj_lora2

            self.qkv_lora1s.append(qkv1)
            self.proj_lora1s.append(proj1)
            self.qkv_lora2s.append(qkv2)
            self.proj_lora2s.append(proj2)

    def __call__(self, attn, img, txt, vec, pe, pe2=None, image_proj=None, ip_scale=1.0, ip_atten_mask=None, lora_weight_list=None, **attention_kwargs):
        if lora_weight_list is not None:
            # print("==>lora_weight_list", lora_weight_list)
            self.lora_weights = lora_weight_list

        # Prepare image for attention
        # print(img.shape, txt.shape, vec.shape)
        img_mod1, img_mod2 = attn.img_mod(vec)
        txt_mod1, txt_mod2 = attn.txt_mod(vec)

        img_modulated = attn.img_norm1(img)
        img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
        img_qkv = attn.img_attn.qkv(img_modulated)
        for i in range(self.lora_num):
            img_qkv += self.qkv_lora1s[i](img_modulated) * self.lora_weights[i]
        img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads, D=attn.head_dim)
        img_q, img_k = attn.img_attn.norm(img_q, img_k, img_v)

        txt_modulated = attn.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        txt_qkv = attn.txt_attn.qkv(txt_modulated)
        for i in range(self.lora_num):
            txt_qkv += self.qkv_lora2s[i](txt_modulated) * self.lora_weights[i]
        txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads, D=attn.head_dim)
        txt_q, txt_k = attn.txt_attn.norm(txt_q, txt_k, txt_v)

        q = torch.cat((txt_q, img_q), dim=2)
        k = torch.cat((txt_k, img_k), dim=2)
        v = torch.cat((txt_v, img_v), dim=2)

        attn1 = attention(q, k, v, pe=pe)

        # print(attn1.shape, q.shape)

        txt_attn, img_attn = attn1[:, :txt.shape[1]], attn1[:, txt.shape[1]:]

        # print(f"txt_attn shape: {txt_attn.size()}")
        # print(f"img_attn shape: {img_attn.size()}")

        img = img + img_mod1.gate * attn.img_attn.proj(img_attn)
        for i in range(self.lora_num):
            img += img_mod1.gate * self.proj_lora1s[i](img_attn) * self.lora_weights[i]
        img = img + img_mod2.gate * attn.img_mlp((1 + img_mod2.scale) * attn.img_norm2(img) + img_mod2.shift)

        txt = txt + txt_mod1.gate * attn.txt_attn.proj(txt_attn)
        for i in range(self.lora_num):
            txt += txt_mod1.gate * self.proj_lora2s[i](txt_attn) * self.lora_weights[i]
        txt = txt + txt_mod2.gate * attn.txt_mlp((1 + txt_mod2.scale) * attn.txt_norm2(txt) + txt_mod2.shift)

        # print(ip_scale, image_proj)
        # IP-adapter processing
        # print(img_q.shape, img.shape, txt.shape)
        # print(image_proj.shape)

        '''
 
        torch.Size([1, 24, 1372, 128]) torch.Size([1, 1372, 3072]) torch.Size([1, 512, 3072])
        torch.Size([1, 4, 4096])
        torch.Size([1, 24, 4, 128]) torch.Size([1, 24, 4, 128])
 
        '''

        img_q = rearrange(img_q, 'B H L D -> B L H D')
        

        for i in range(len(ip_atten_mask)):
            # ip_atten_mask: bs x L
            
            ip_key = self.ip_adapter_double_stream_k_proj(image_proj[i])
            ip_value = self.ip_adapter_double_stream_v_proj(image_proj[i])

            # Reshape projections for multi-head attention
            ip_key = rearrange(ip_key, 'B L (H D) -> B H L D', H=attn.num_heads, D=attn.head_dim)
            ip_value = rearrange(ip_value, 'B L (H D) -> B H L D', H=attn.num_heads, D=attn.head_dim)

            ip_atten_mask_tmp = ip_atten_mask[i]
            # print(ip_atten_mask_tmp.shape, img_q.shape)
            # ip_query = img_q[ip_atten_mask_tmp].reshape()
            ip_query = []
            for bi in range(img_q.shape[0]):
                ip_query.append(img_q[bi][ip_atten_mask_tmp[bi]])
            ip_query = torch.stack(ip_query, axis=0)
            # print(ip_atten_mask_tmp.sum(), ip_atten_mask_tmp.shape, img_q.shape)
            ip_query = rearrange(ip_query, 'B L H D -> B H L D')
            # print(ip_key.shape, ip_value.shape)
            # Compute attention between IP projections and the latent query
            ip_attention = F.scaled_dot_product_attention(
                ip_query,
                ip_key,
                ip_value,
                dropout_p=0.0,
                is_causal=False
            )
            # print(ip_attention.shape, ip_attention.shape)
            ip_attention = rearrange(ip_attention, "B H L D -> B L (H D)", H=attn.num_heads, D=attn.head_dim)
            img[ip_atten_mask_tmp] = img[ip_atten_mask_tmp] + ip_scale * ip_attention
    

        return img, txt 


class IPDoubleStreamBlockProcessorV3(nn.Module):
    """Attention processor for handling IP-adapter with double stream block."""

    def __init__(self, context_dim, hidden_dim):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "IPDoubleStreamBlockProcessor requires PyTorch 2.0 or higher. Please upgrade PyTorch."
            )

        # Ensure context_dim matches the dimension of image_proj
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim

        # Initialize projections for IP-adapter 4096 
        self.ip_adapter_double_stream_k_proj = nn.Linear(context_dim, hidden_dim, bias=True)
        self.ip_adapter_double_stream_v_proj = nn.Linear(context_dim, hidden_dim, bias=True)

        nn.init.zeros_(self.ip_adapter_double_stream_k_proj.weight)
        nn.init.zeros_(self.ip_adapter_double_stream_k_proj.bias)

        nn.init.zeros_(self.ip_adapter_double_stream_v_proj.weight)
        nn.init.zeros_(self.ip_adapter_double_stream_v_proj.bias)

    def __call__(self, attn, img, txt, vec, pe, pe2=None, image_proj=None, ip_scale=1.0, ip_atten_mask=None, **attention_kwargs):

        # Prepare image for attention
        # print(img.shape, txt.shape, vec.shape)
        img_mod1, img_mod2 = attn.img_mod(vec)
        txt_mod1, txt_mod2 = attn.txt_mod(vec)

        img_modulated = attn.img_norm1(img)
        img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
        img_qkv = attn.img_attn.qkv(img_modulated)
        img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads, D=attn.head_dim)
        img_q, img_k = attn.img_attn.norm(img_q, img_k, img_v)

        txt_modulated = attn.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        txt_qkv = attn.txt_attn.qkv(txt_modulated)
        txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads, D=attn.head_dim)
        txt_q, txt_k = attn.txt_attn.norm(txt_q, txt_k, txt_v)

        q = torch.cat((txt_q, img_q), dim=2)
        k = torch.cat((txt_k, img_k), dim=2)
        v = torch.cat((txt_v, img_v), dim=2)

        attn1 = attention(q, k, v, pe=pe)

        # print(attn1.shape, q.shape)

        txt_attn, img_attn = attn1[:, :txt.shape[1]], attn1[:, txt.shape[1]:]

        # print(f"txt_attn shape: {txt_attn.size()}")
        # print(f"img_attn shape: {img_attn.size()}")

        img = img + img_mod1.gate * attn.img_attn.proj(img_attn)
        img = img + img_mod2.gate * attn.img_mlp((1 + img_mod2.scale) * attn.img_norm2(img) + img_mod2.shift)

        txt = txt + txt_mod1.gate * attn.txt_attn.proj(txt_attn)
        txt = txt + txt_mod2.gate * attn.txt_mlp((1 + txt_mod2.scale) * attn.txt_norm2(txt) + txt_mod2.shift)

        # print(ip_scale, image_proj)
        # IP-adapter processing
        # print(img_q.shape, img.shape, txt.shape)
        # print(image_proj.shape)

        '''
 
        torch.Size([1, 24, 1372, 128]) torch.Size([1, 1372, 3072]) torch.Size([1, 512, 3072])
        torch.Size([1, 4, 4096])
        torch.Size([1, 24, 4, 128]) torch.Size([1, 24, 4, 128])
 
        '''

        img_q = rearrange(img_q, 'B H L D -> B L H D')
        

        for i in range(len(ip_atten_mask)):
            # ip_atten_mask: bs x L
            
            ip_key = self.ip_adapter_double_stream_k_proj(image_proj[i])
            ip_value = self.ip_adapter_double_stream_v_proj(image_proj[i])

            # Reshape projections for multi-head attention
            ip_key = rearrange(ip_key, 'B L (H D) -> B H L D', H=attn.num_heads, D=attn.head_dim)
            ip_value = rearrange(ip_value, 'B L (H D) -> B H L D', H=attn.num_heads, D=attn.head_dim)

            ip_atten_mask_tmp = ip_atten_mask[i]
            # print(ip_atten_mask_tmp.shape, img_q.shape)
            # ip_query = img_q[ip_atten_mask_tmp].reshape()
            ip_query = []
            for bi in range(img_q.shape[0]):
                ip_query.append(img_q[bi][ip_atten_mask_tmp[bi]])
            ip_query = torch.stack(ip_query, axis=0)
            # print(ip_atten_mask_tmp.sum(), ip_atten_mask_tmp.shape, img_q.shape)
            ip_query = rearrange(ip_query, 'B L H D -> B H L D')
            # print(ip_key.shape, ip_value.shape)
            # Compute attention between IP projections and the latent query
            ip_attention = F.scaled_dot_product_attention(
                ip_query,
                ip_key,
                ip_value,
                dropout_p=0.0,
                is_causal=False
            )
            # print(ip_attention.shape, ip_attention.shape)
            ip_attention = rearrange(ip_attention, "B H L D -> B L (H D)", H=attn.num_heads, D=attn.head_dim)
            img[ip_atten_mask_tmp] = img[ip_atten_mask_tmp] + ip_scale * ip_attention
    

        return img, txt 
    
class IPDoubleStreamBlockProcessorV6(nn.Module):
    """Attention processor for handling IP-adapter with double stream block."""

    def __init__(self, context_dim, hidden_dim, portrait_dim=64):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "IPDoubleStreamBlockProcessor requires PyTorch 2.0 or higher. Please upgrade PyTorch."
            )

        # Ensure context_dim matches the dimension of image_proj
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim

        # Initialize projections for IP-adapter 4096 
        self.ip_adapter_double_stream_k_proj = nn.Linear(context_dim, hidden_dim, bias=True)
        self.ip_adapter_double_stream_v_proj = nn.Linear(context_dim, hidden_dim, bias=True)

        nn.init.zeros_(self.ip_adapter_double_stream_k_proj.weight)
        nn.init.zeros_(self.ip_adapter_double_stream_k_proj.bias)

        nn.init.zeros_(self.ip_adapter_double_stream_v_proj.weight)
        nn.init.zeros_(self.ip_adapter_double_stream_v_proj.bias)
        
        self.ip_adapter_double_stream_k_proj_portrait = nn.Linear(portrait_dim, hidden_dim, bias=True)
        self.ip_adapter_double_stream_v_proj_portrait = nn.Linear(portrait_dim, hidden_dim, bias=True)

        nn.init.zeros_(self.ip_adapter_double_stream_k_proj_portrait.weight)
        nn.init.zeros_(self.ip_adapter_double_stream_k_proj_portrait.bias)

        nn.init.zeros_(self.ip_adapter_double_stream_v_proj_portrait.weight)
        nn.init.zeros_(self.ip_adapter_double_stream_v_proj_portrait.bias)

    def __call__(self, attn, img, txt, vec, pe, pe2=None, image_proj=None, ip_scale=1.0, ip_atten_mask=None, portrait_scale=1.0, portrait_prompt=None, **attention_kwargs):
        
        # portrait_scale = 0.0
        # print("portrait_scale", portrait_scale)
        if portrait_prompt is None:
            print("portrait_prompt is None!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        # Prepare image for attention
        # print(img.shape, txt.shape, vec.shape)
        img_mod1, img_mod2 = attn.img_mod(vec)
        txt_mod1, txt_mod2 = attn.txt_mod(vec)

        img_modulated = attn.img_norm1(img)
        img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
        img_qkv = attn.img_attn.qkv(img_modulated)
        img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads, D=attn.head_dim)
        img_q, img_k = attn.img_attn.norm(img_q, img_k, img_v)

        txt_modulated = attn.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        txt_qkv = attn.txt_attn.qkv(txt_modulated)
        txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads, D=attn.head_dim)
        txt_q, txt_k = attn.txt_attn.norm(txt_q, txt_k, txt_v)

        q = torch.cat((txt_q, img_q), dim=2)
        k = torch.cat((txt_k, img_k), dim=2)
        v = torch.cat((txt_v, img_v), dim=2)

        attn1 = attention(q, k, v, pe=pe)

        # print(attn1.shape, q.shape)

        txt_attn, img_attn = attn1[:, :txt.shape[1]], attn1[:, txt.shape[1]:]

        # print(f"txt_attn shape: {txt_attn.size()}")
        # print(f"img_attn shape: {img_attn.size()}")

        img = img + img_mod1.gate * attn.img_attn.proj(img_attn)
        img = img + img_mod2.gate * attn.img_mlp((1 + img_mod2.scale) * attn.img_norm2(img) + img_mod2.shift)

        txt = txt + txt_mod1.gate * attn.txt_attn.proj(txt_attn)
        txt = txt + txt_mod2.gate * attn.txt_mlp((1 + txt_mod2.scale) * attn.txt_norm2(txt) + txt_mod2.shift)

        # print(ip_scale, image_proj)
        # IP-adapter processing
        # print(img_q.shape, img.shape, txt.shape)
        # print(image_proj.shape)

        '''
 
        torch.Size([1, 24, 1372, 128]) torch.Size([1, 1372, 3072]) torch.Size([1, 512, 3072])
        torch.Size([1, 4, 4096])
        torch.Size([1, 24, 4, 128]) torch.Size([1, 24, 4, 128])
 
        '''

        img_q = rearrange(img_q, 'B H L D -> B L H D')
        
        # print("image_proj", image_proj[0].shape, "portrait_prompt", portrait_prompt[0].shape)
        for i in range(len(ip_atten_mask)):
            # ip_atten_mask: bs x L
            
            ip_key = self.ip_adapter_double_stream_k_proj(image_proj[i])
            ip_value = self.ip_adapter_double_stream_v_proj(image_proj[i])

            # Reshape projections for multi-head attention
            ip_key = rearrange(ip_key, 'B L (H D) -> B H L D', H=attn.num_heads, D=attn.head_dim)
            ip_value = rearrange(ip_value, 'B L (H D) -> B H L D', H=attn.num_heads, D=attn.head_dim)

            ip_atten_mask_tmp = ip_atten_mask[i]
            # print(ip_atten_mask_tmp.shape, img_q.shape)
            # ip_query = img_q[ip_atten_mask_tmp].reshape()
            ip_query = []
            for bi in range(img_q.shape[0]):
                ip_query.append(img_q[bi][ip_atten_mask_tmp[bi]])
            ip_query = torch.stack(ip_query, axis=0)
            # print(ip_atten_mask_tmp.sum(), ip_atten_mask_tmp.shape, img_q.shape)
            ip_query = rearrange(ip_query, 'B L H D -> B H L D')
            # print(ip_key.shape, ip_value.shape)
            # Compute attention between IP projections and the latent query
            ip_attention = F.scaled_dot_product_attention(
                ip_query,
                ip_key,
                ip_value,
                dropout_p=0.0,
                is_causal=False
            )
            # print(ip_attention.shape, ip_attention.shape)
            ip_attention = rearrange(ip_attention, "B H L D -> B L (H D)", H=attn.num_heads, D=attn.head_dim)
            # print("face ip_attention", ip_attention, "ip_scale", ip_scale)
            img[ip_atten_mask_tmp] = img[ip_atten_mask_tmp] + ip_scale * ip_attention
            
        for i in range(len(ip_atten_mask)):
            # print("Do portrait prompt")
            # print("portrait_prompt", portrait_prompt[0])
            ip_key = self.ip_adapter_double_stream_k_proj_portrait(portrait_prompt[0])
            ip_value = self.ip_adapter_double_stream_v_proj_portrait(portrait_prompt[0])

            # Reshape projections for multi-head attention
            ip_key = rearrange(ip_key, 'B L (H D) -> B H L D', H=attn.num_heads, D=attn.head_dim)
            ip_value = rearrange(ip_value, 'B L (H D) -> B H L D', H=attn.num_heads, D=attn.head_dim)

            ip_atten_mask_tmp = ip_atten_mask[i]
            # print(ip_atten_mask_tmp.shape, img_q.shape)
            # ip_query = img_q[ip_atten_mask_tmp].reshape()
            ip_query = []
            for bi in range(img_q.shape[0]):
                ip_query.append(img_q[bi][ip_atten_mask_tmp[bi]])
            ip_query = torch.stack(ip_query, axis=0)
            # print(ip_atten_mask_tmp.sum(), ip_atten_mask_tmp.shape, img_q.shape)
            ip_query = rearrange(ip_query, 'B L H D -> B H L D')
            # print(ip_key.shape, ip_value.shape)
            # Compute attention between IP projections and the latent query
            ip_attention = F.scaled_dot_product_attention(
                ip_query,
                ip_key,
                ip_value,
                dropout_p=0.0,
                is_causal=False
            )
            # print(ip_attention.shape, ip_attention.shape)
            ip_attention_1 = rearrange(ip_attention, "B H L D -> B L (H D)", H=attn.num_heads, D=attn.head_dim)
            # print("portrait ip_attention", ip_attention, "portrait_scale", portrait_scale)
            # img[ip_atten_mask_tmp] = img[ip_atten_mask_tmp] + portrait_scale * ip_attention
            
        for i in range(len(ip_atten_mask)):
            # print("Do portrait prompt")
            # print("portrait_prompt", portrait_prompt)
            ip_key = self.ip_adapter_double_stream_k_proj_portrait(portrait_prompt[1])
            ip_value = self.ip_adapter_double_stream_v_proj_portrait(portrait_prompt[1])

            # Reshape projections for multi-head attention
            ip_key = rearrange(ip_key, 'B L (H D) -> B H L D', H=attn.num_heads, D=attn.head_dim)
            ip_value = rearrange(ip_value, 'B L (H D) -> B H L D', H=attn.num_heads, D=attn.head_dim)

            ip_atten_mask_tmp = ip_atten_mask[i]
            # print(ip_atten_mask_tmp.shape, img_q.shape)
            # ip_query = img_q[ip_atten_mask_tmp].reshape()
            ip_query = []
            for bi in range(img_q.shape[0]):
                ip_query.append(img_q[bi][ip_atten_mask_tmp[bi]])
            ip_query = torch.stack(ip_query, axis=0)
            # print(ip_atten_mask_tmp.sum(), ip_atten_mask_tmp.shape, img_q.shape)
            ip_query = rearrange(ip_query, 'B L H D -> B H L D')
            # print(ip_key.shape, ip_value.shape)
            # Compute attention between IP projections and the latent query
            ip_attention = F.scaled_dot_product_attention(
                ip_query,
                ip_key,
                ip_value,
                dropout_p=0.0,
                is_causal=False
            )
            # print(ip_attention.shape, ip_attention.shape)
            ip_attention_2 = rearrange(ip_attention, "B H L D -> B L (H D)", H=attn.num_heads, D=attn.head_dim)
            
        ip_attention = ip_attention_1 - ip_attention_2
        if ip_attention.device == torch.device('cuda:1'):
            print("ip_attention", ip_attention,"ip_attention_1", ip_attention_1,"ip_attention_2", ip_attention_2)

        img[ip_atten_mask_tmp] = img[ip_atten_mask_tmp] + portrait_scale * ip_attention

        return img, txt 

    
class IPDoubleStreamBlockProcessorV7(nn.Module):
    """Attention processor for handling IP-adapter with double stream block."""

    def __init__(self, context_dim, hidden_dim, feature_in_dim=3072):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "IPDoubleStreamBlockProcessor requires PyTorch 2.0 or higher. Please upgrade PyTorch."
            )

        # Ensure context_dim matches the dimension of image_proj
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim

        # Initialize projections for IP-adapter 4096 
        self.ip_adapter_double_stream_k_proj = nn.Linear(context_dim, hidden_dim, bias=True)
        self.ip_adapter_double_stream_v_proj = nn.Linear(context_dim, hidden_dim, bias=True)

        nn.init.zeros_(self.ip_adapter_double_stream_k_proj.weight)
        nn.init.zeros_(self.ip_adapter_double_stream_k_proj.bias)

        nn.init.zeros_(self.ip_adapter_double_stream_v_proj.weight)
        nn.init.zeros_(self.ip_adapter_double_stream_v_proj.bias)

        self.ip_adapter_double_stream_k_proj_vae = nn.Linear(feature_in_dim, hidden_dim, bias=True)
        self.ip_adapter_double_stream_v_proj_vae = nn.Linear(feature_in_dim, hidden_dim, bias=True)

        nn.init.zeros_(self.ip_adapter_double_stream_k_proj_vae.weight)
        nn.init.zeros_(self.ip_adapter_double_stream_k_proj_vae.bias)

        nn.init.zeros_(self.ip_adapter_double_stream_v_proj_vae.weight)
        nn.init.zeros_(self.ip_adapter_double_stream_v_proj_vae.bias)

    def __call__(self, attn, img, txt, vec, pe, pe2=None, image_proj=None, ip_scale=1.0, ip_atten_mask=None, vae_controlnet_hidden_states=None, use_ipa=True, **attention_kwargs):
        
        # portrait_scale = 0.0
        if vae_controlnet_hidden_states is None and use_ipa:
            print("vae_controlnet_hidden_states is None!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        # Prepare image for attention
        # print(img.shape, txt.shape, vec.shape)
        img_mod1, img_mod2 = attn.img_mod(vec)
        txt_mod1, txt_mod2 = attn.txt_mod(vec)

        img_modulated = attn.img_norm1(img)
        img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
        img_qkv = attn.img_attn.qkv(img_modulated)
        img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads, D=attn.head_dim)
        img_q, img_k = attn.img_attn.norm(img_q, img_k, img_v)

        txt_modulated = attn.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        txt_qkv = attn.txt_attn.qkv(txt_modulated)
        txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads, D=attn.head_dim)
        txt_q, txt_k = attn.txt_attn.norm(txt_q, txt_k, txt_v)

        q = torch.cat((txt_q, img_q), dim=2)
        k = torch.cat((txt_k, img_k), dim=2)
        v = torch.cat((txt_v, img_v), dim=2)

        attn1 = attention(q, k, v, pe=pe)

        # print(attn1.shape, q.shape)

        txt_attn, img_attn = attn1[:, :txt.shape[1]], attn1[:, txt.shape[1]:]

        # print(f"txt_attn shape: {txt_attn.size()}")
        # print(f"img_attn shape: {img_attn.size()}")

        img = img + img_mod1.gate * attn.img_attn.proj(img_attn)
        img = img + img_mod2.gate * attn.img_mlp((1 + img_mod2.scale) * attn.img_norm2(img) + img_mod2.shift)

        txt = txt + txt_mod1.gate * attn.txt_attn.proj(txt_attn)
        txt = txt + txt_mod2.gate * attn.txt_mlp((1 + txt_mod2.scale) * attn.txt_norm2(txt) + txt_mod2.shift)

        # print(ip_scale, image_proj)
        # IP-adapter processing
        # print(img_q.shape, img.shape, txt.shape)
        # print(image_proj.shape)

        '''
 
        torch.Size([1, 24, 1372, 128]) torch.Size([1, 1372, 3072]) torch.Size([1, 512, 3072])
        torch.Size([1, 4, 4096])
        torch.Size([1, 24, 4, 128]) torch.Size([1, 24, 4, 128])
 
        '''

        if use_ipa:
            img_q = rearrange(img_q, 'B H L D -> B L H D')
            
            for i in range(len(ip_atten_mask)):
                # ip_atten_mask: bs x L
                
                ip_key = self.ip_adapter_double_stream_k_proj(image_proj[i])
                ip_value = self.ip_adapter_double_stream_v_proj(image_proj[i])

                # Reshape projections for multi-head attention
                ip_key = rearrange(ip_key, 'B L (H D) -> B H L D', H=attn.num_heads, D=attn.head_dim)
                ip_value = rearrange(ip_value, 'B L (H D) -> B H L D', H=attn.num_heads, D=attn.head_dim)

                ip_atten_mask_tmp = ip_atten_mask[i]
                # print(ip_atten_mask_tmp.shape, img_q.shape)
                # ip_query = img_q[ip_atten_mask_tmp].reshape()
                ip_query = []
                for bi in range(img_q.shape[0]):
                    ip_query.append(img_q[bi][ip_atten_mask_tmp[bi]])
                ip_query = torch.stack(ip_query, axis=0)
                # print(ip_atten_mask_tmp.sum(), ip_atten_mask_tmp.shape, img_q.shape)
                ip_query = rearrange(ip_query, 'B L H D -> B H L D')
                # print(ip_key.shape, ip_value.shape)
                # Compute attention between IP projections and the latent query
                ip_attention = F.scaled_dot_product_attention(
                    ip_query,
                    ip_key,
                    ip_value,
                    dropout_p=0.0,
                    is_causal=False
                )
                # print(ip_attention.shape, ip_attention.shape)
                ip_attention = rearrange(ip_attention, "B H L D -> B L (H D)", H=attn.num_heads, D=attn.head_dim)
                # print("face ip_attention", ip_attention, "ip_scale", ip_scale)
                img[ip_atten_mask_tmp] = img[ip_atten_mask_tmp] + ip_scale * ip_attention
                
            for i in range(len(ip_atten_mask)):
                # print("Do portrait prompt")
                # print("portrait_prompt", portrait_prompt)
                ip_key = self.ip_adapter_double_stream_k_proj_vae(vae_controlnet_hidden_states[i])
                ip_value = self.ip_adapter_double_stream_v_proj_vae(vae_controlnet_hidden_states[i])

                # Reshape projections for multi-head attention
                ip_key = rearrange(ip_key, 'B L (H D) -> B H L D', H=attn.num_heads, D=attn.head_dim)
                ip_value = rearrange(ip_value, 'B L (H D) -> B H L D', H=attn.num_heads, D=attn.head_dim)

                ip_atten_mask_tmp = ip_atten_mask[i]
                # print(ip_atten_mask_tmp.shape, img_q.shape)
                # ip_query = img_q[ip_atten_mask_tmp].reshape()
                ip_query = []
                for bi in range(img_q.shape[0]):
                    ip_query.append(img_q[bi][ip_atten_mask_tmp[bi]])
                ip_query = torch.stack(ip_query, axis=0)
                # print(ip_atten_mask_tmp.sum(), ip_atten_mask_tmp.shape, img_q.shape)
                ip_query = rearrange(ip_query, 'B L H D -> B H L D')
                # print(ip_key.shape, ip_value.shape)
                # Compute attention between IP projections and the latent query
                ip_attention = F.scaled_dot_product_attention(
                    ip_query,
                    ip_key,
                    ip_value,
                    dropout_p=0.0,
                    is_causal=False
                )
                # print(ip_attention.shape, ip_attention.shape)
                ip_attention = rearrange(ip_attention, "B H L D -> B L (H D)", H=attn.num_heads, D=attn.head_dim)
                if ip_attention.device == torch.device('cuda:1'):
                    print("vae ip_attention", ip_attention, "vae_scale", 1.0)
                img[ip_atten_mask_tmp] = img[ip_atten_mask_tmp] + 1.0 * ip_attention
                # img[ip_atten_mask_tmp] = img[ip_atten_mask_tmp] + 0.0 * ip_attention

        return img, txt

    
class IPDoubleStreamBlockProcessorV8(nn.Module):
    """Attention processor for handling IP-adapter with double stream block."""

    def __init__(self, context_dim, hidden_dim, feature_in_dim=3072, lora_dim=3072, rank=32):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "IPDoubleStreamBlockProcessor requires PyTorch 2.0 or higher. Please upgrade PyTorch."
            )

        # Ensure context_dim matches the dimension of image_proj
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim

        # Initialize projections for IP-adapter 4096 
        self.ip_adapter_double_stream_k_proj = nn.Linear(context_dim, hidden_dim, bias=True)
        self.ip_adapter_double_stream_v_proj = nn.Linear(context_dim, hidden_dim, bias=True)

        nn.init.zeros_(self.ip_adapter_double_stream_k_proj.weight)
        nn.init.zeros_(self.ip_adapter_double_stream_k_proj.bias)

        nn.init.zeros_(self.ip_adapter_double_stream_v_proj.weight)
        nn.init.zeros_(self.ip_adapter_double_stream_v_proj.bias)

        self.ip_adapter_double_stream_k_proj_vae = nn.Linear(feature_in_dim, hidden_dim, bias=True)
        self.ip_adapter_double_stream_v_proj_vae = nn.Linear(feature_in_dim, hidden_dim, bias=True)

        nn.init.zeros_(self.ip_adapter_double_stream_k_proj_vae.weight)
        nn.init.zeros_(self.ip_adapter_double_stream_k_proj_vae.bias)

        nn.init.zeros_(self.ip_adapter_double_stream_v_proj_vae.weight)
        nn.init.zeros_(self.ip_adapter_double_stream_v_proj_vae.bias)
        
        self.ip_adapter_qkv_lora1 = LoRALinearLayer(lora_dim, lora_dim * 3, rank)
        self.ip_adapter_proj_lora1 = LoRALinearLayer(lora_dim, lora_dim, rank)
        self.ip_adapter_qkv_lora2 = LoRALinearLayer(lora_dim, lora_dim * 3, rank)
        self.ip_adapter_proj_lora2 = LoRALinearLayer(lora_dim, lora_dim, rank)

    def __call__(self, attn, img, txt, vec, pe, pe2=None, image_proj=None, ip_scale=1.0, ip_atten_mask=None, vae_controlnet_hidden_states=None, use_ipa=True, **attention_kwargs):
        
        # portrait_scale = 0.0
        if vae_controlnet_hidden_states is None and use_ipa:
            print("vae_controlnet_hidden_states is None!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        # Prepare image for attention
        # print(img.shape, txt.shape, vec.shape)
        img_mod1, img_mod2 = attn.img_mod(vec)
        txt_mod1, txt_mod2 = attn.txt_mod(vec)

        img_modulated = attn.img_norm1(img)
        img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
        img_qkv = attn.img_attn.qkv(img_modulated)
        if not use_ipa:
            img_qkv += self.ip_adapter_qkv_lora1(img_modulated)
        img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads, D=attn.head_dim)
        img_q, img_k = attn.img_attn.norm(img_q, img_k, img_v)

        txt_modulated = attn.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        txt_qkv = attn.txt_attn.qkv(txt_modulated)
        if not use_ipa:
            txt_qkv += self.ip_adapter_qkv_lora2(txt_modulated)
        txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads, D=attn.head_dim)
        txt_q, txt_k = attn.txt_attn.norm(txt_q, txt_k, txt_v)

        q = torch.cat((txt_q, img_q), dim=2)
        k = torch.cat((txt_k, img_k), dim=2)
        v = torch.cat((txt_v, img_v), dim=2)

        attn1 = attention(q, k, v, pe=pe)

        # print(attn1.shape, q.shape)

        txt_attn, img_attn = attn1[:, :txt.shape[1]], attn1[:, txt.shape[1]:]

        # print(f"txt_attn shape: {txt_attn.size()}")
        # print(f"img_attn shape: {img_attn.size()}")

        img = img + img_mod1.gate * attn.img_attn.proj(img_attn)
        if not use_ipa:
            img += img_mod1.gate * self.ip_adapter_proj_lora1(img_attn)
        img = img + img_mod2.gate * attn.img_mlp((1 + img_mod2.scale) * attn.img_norm2(img) + img_mod2.shift)

        txt = txt + txt_mod1.gate * attn.txt_attn.proj(txt_attn)
        if not use_ipa:
            txt += txt_mod1.gate * self.ip_adapter_proj_lora1(txt_attn)
        txt = txt + txt_mod2.gate * attn.txt_mlp((1 + txt_mod2.scale) * attn.txt_norm2(txt) + txt_mod2.shift)

        # print(ip_scale, image_proj)
        # IP-adapter processing
        # print(img_q.shape, img.shape, txt.shape)
        # print(image_proj.shape)

        '''
 
        torch.Size([1, 24, 1372, 128]) torch.Size([1, 1372, 3072]) torch.Size([1, 512, 3072])
        torch.Size([1, 4, 4096])
        torch.Size([1, 24, 4, 128]) torch.Size([1, 24, 4, 128])
 
        '''

        if use_ipa:
            img_q = rearrange(img_q, 'B H L D -> B L H D')
            
            for i in range(len(ip_atten_mask)):
                # ip_atten_mask: bs x L
                
                ip_key = self.ip_adapter_double_stream_k_proj(image_proj[i])
                ip_value = self.ip_adapter_double_stream_v_proj(image_proj[i])

                # Reshape projections for multi-head attention
                ip_key = rearrange(ip_key, 'B L (H D) -> B H L D', H=attn.num_heads, D=attn.head_dim)
                ip_value = rearrange(ip_value, 'B L (H D) -> B H L D', H=attn.num_heads, D=attn.head_dim)

                ip_atten_mask_tmp = ip_atten_mask[i]
                # print(ip_atten_mask_tmp.shape, img_q.shape)
                # ip_query = img_q[ip_atten_mask_tmp].reshape()
                ip_query = []
                for bi in range(img_q.shape[0]):
                    ip_query.append(img_q[bi][ip_atten_mask_tmp[bi]])
                ip_query = torch.stack(ip_query, axis=0)
                # print(ip_atten_mask_tmp.sum(), ip_atten_mask_tmp.shape, img_q.shape)
                ip_query = rearrange(ip_query, 'B L H D -> B H L D')
                # print(ip_key.shape, ip_value.shape)
                # Compute attention between IP projections and the latent query
                ip_attention = F.scaled_dot_product_attention(
                    ip_query,
                    ip_key,
                    ip_value,
                    dropout_p=0.0,
                    is_causal=False
                )
                # print(ip_attention.shape, ip_attention.shape)
                ip_attention = rearrange(ip_attention, "B H L D -> B L (H D)", H=attn.num_heads, D=attn.head_dim)
                # print("face ip_attention", ip_attention, "ip_scale", ip_scale)
                img[ip_atten_mask_tmp] = img[ip_atten_mask_tmp] + ip_scale * ip_attention
                
            for i in range(len(ip_atten_mask)):
                # print("Do portrait prompt")
                # print("portrait_prompt", portrait_prompt)
                ip_key = self.ip_adapter_double_stream_k_proj_vae(vae_controlnet_hidden_states[i])
                ip_value = self.ip_adapter_double_stream_v_proj_vae(vae_controlnet_hidden_states[i])

                # Reshape projections for multi-head attention
                ip_key = rearrange(ip_key, 'B L (H D) -> B H L D', H=attn.num_heads, D=attn.head_dim)
                ip_value = rearrange(ip_value, 'B L (H D) -> B H L D', H=attn.num_heads, D=attn.head_dim)

                ip_atten_mask_tmp = ip_atten_mask[i]
                # print(ip_atten_mask_tmp.shape, img_q.shape)
                # ip_query = img_q[ip_atten_mask_tmp].reshape()
                ip_query = []
                for bi in range(img_q.shape[0]):
                    ip_query.append(img_q[bi][ip_atten_mask_tmp[bi]])
                ip_query = torch.stack(ip_query, axis=0)
                # print(ip_atten_mask_tmp.sum(), ip_atten_mask_tmp.shape, img_q.shape)
                ip_query = rearrange(ip_query, 'B L H D -> B H L D')
                # print(ip_key.shape, ip_value.shape)
                # Compute attention between IP projections and the latent query
                ip_attention = F.scaled_dot_product_attention(
                    ip_query,
                    ip_key,
                    ip_value,
                    dropout_p=0.0,
                    is_causal=False
                )
                # print(ip_attention.shape, ip_attention.shape)
                ip_attention = rearrange(ip_attention, "B H L D -> B L (H D)", H=attn.num_heads, D=attn.head_dim)
                if ip_attention.device == torch.device('cuda:1'):
                    print("vae ip_attention", ip_attention, "vae_scale", 1.0)
                img[ip_atten_mask_tmp] = img[ip_atten_mask_tmp] + 1.0 * ip_attention
                # img[ip_atten_mask_tmp] = img[ip_atten_mask_tmp] + 0.0 * ip_attention

        return img, txt



class IPDoubleStreamBlockProcessorV5(nn.Module):
    """Attention processor for handling IP-adapter with double stream block."""

    def __init__(self, context_dim, hidden_dim):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "IPDoubleStreamBlockProcessor requires PyTorch 2.0 or higher. Please upgrade PyTorch."
            )

        # Ensure context_dim matches the dimension of image_proj
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim

        # Initialize projections for IP-adapter 4096 
        self.ip_adapter_double_stream_k_proj = nn.Linear(context_dim, hidden_dim, bias=True)
        self.ip_adapter_double_stream_v_proj = nn.Linear(context_dim, hidden_dim, bias=True)

        nn.init.zeros_(self.ip_adapter_double_stream_k_proj.weight)
        nn.init.zeros_(self.ip_adapter_double_stream_k_proj.bias)

        nn.init.zeros_(self.ip_adapter_double_stream_v_proj.weight)
        nn.init.zeros_(self.ip_adapter_double_stream_v_proj.bias)

    def __call__(self, attn, img, txt, vec, pe, pe2=None, image_proj=None, ip_scale=1.0, ip_atten_mask=None, ip_scale_cloth=1.0, txt_cloth=None, vec_cloth=None, pe_cloth=None, **attention_kwargs):

        # Prepare image for attention
        # print(img.shape, txt.shape, vec.shape)
        img_mod1, img_mod2 = attn.img_mod(vec)
        txt_mod1, txt_mod2 = attn.txt_mod(vec)

        img_modulated = attn.img_norm1(img)
        img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
        img_qkv = attn.img_attn.qkv(img_modulated)
        img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads, D=attn.head_dim)
        img_q, img_k = attn.img_attn.norm(img_q, img_k, img_v)

        txt_modulated = attn.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        txt_qkv = attn.txt_attn.qkv(txt_modulated)
        txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads, D=attn.head_dim)
        txt_q, txt_k = attn.txt_attn.norm(txt_q, txt_k, txt_v)

        q = torch.cat((txt_q, img_q), dim=2)
        k = torch.cat((txt_k, img_k), dim=2)
        v = torch.cat((txt_v, img_v), dim=2)
        
        attn1 = attention(q, k, v, pe=pe)
        txt_attn, img_attn = attn1[:, :txt.shape[1]], attn1[:, txt.shape[1]:]
        
        if txt_cloth is not None and ip_atten_mask is not None:
            img_cloth_mod1, img_cloth_mod2 = attn.img_mod(vec_cloth)
            txt_cloth_mod1, txt_cloth_mod2 = attn.txt_mod(vec_cloth)
            
            txt_cloth_modulated = attn.txt_norm1(txt_cloth)
            txt_cloth_modulated = (1 + txt_cloth_mod1.scale) * txt_cloth_modulated + txt_cloth_mod1.shift
            txt_cloth_qkv = attn.txt_attn.qkv(txt_cloth_modulated)
            txt_cloth_q, txt_cloth_k, txt_cloth_v = rearrange(txt_cloth_qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads, D=attn.head_dim)
            txt_cloth_q, txt_cloth_k = attn.txt_attn.norm(txt_cloth_q, txt_cloth_k, txt_cloth_v)
            
            ip_atten_mask_tmp = ip_atten_mask[0]
            img_mask_q = []
            img_mask_k = []
            img_mask_v = []
            img_q_rearrange = rearrange(img_q, 'B H L D -> B L H D')
            img_k_rearrange = rearrange(img_k, 'B H L D -> B L H D')
            img_v_rearrange = rearrange(img_v, 'B H L D -> B L H D')
            for bi in range(img_q_rearrange.shape[0]):
                img_mask_q.append(img_q_rearrange[bi][ip_atten_mask_tmp[bi]])
                img_mask_k.append(img_k_rearrange[bi][ip_atten_mask_tmp[bi]])
                img_mask_v.append(img_v_rearrange[bi][ip_atten_mask_tmp[bi]])
            img_mask_q = torch.stack(img_mask_q, axis=0)
            img_mask_k = torch.stack(img_mask_k, axis=0)
            img_mask_v = torch.stack(img_mask_v, axis=0)
            img_mask_q = rearrange(img_mask_q, 'B L H D -> B H L D')
            img_mask_k = rearrange(img_mask_k, 'B L H D -> B H L D')
            img_mask_v = rearrange(img_mask_v, 'B L H D -> B H L D')
            q_cloth = torch.cat((txt_cloth_q, img_mask_q), dim=2)
            k_cloth = torch.cat((txt_cloth_k, img_mask_k), dim=2)
            v_cloth = torch.cat((txt_cloth_v, img_mask_v), dim=2)
            
            txt_cloth_pe = pe_cloth[:,:,:txt_cloth.shape[1]]
            img_cloth_pe = pe_cloth[:,:,txt_cloth.shape[1]:]
            img_cloth_pe = img_cloth_pe[:,:,ip_atten_mask_tmp[0]]
            pe_cloth = torch.cat((txt_cloth_pe, img_cloth_pe), dim=2)
            
            attn2 = attention(q_cloth, k_cloth, v_cloth, pe=pe_cloth)

            # print(attn1.shape, q.shape)
            txt_cloth_attn, img_cloth_attn = attn2[:, :txt_cloth.shape[1]], attn2[:, txt_cloth.shape[1]:]
            
            img_cloth = img[ip_atten_mask_tmp] + img_cloth_mod1.gate * attn.img_attn.proj(img_cloth_attn)
            img_cloth = img_cloth + img_cloth_mod2.gate * attn.img_mlp((1 + img_cloth_mod2.scale) * attn.img_norm2(img_cloth) + img_cloth_mod2.shift)

            txt_cloth = txt_cloth + txt_cloth_mod1.gate * attn.txt_attn.proj(txt_cloth_attn)
            txt_cloth = txt_cloth + txt_cloth_mod2.gate * attn.txt_mlp((1 + txt_cloth_mod2.scale) * attn.txt_norm2(txt_cloth) + txt_cloth_mod2.shift)
        else:
            img_cloth=0

        # print(f"txt_attn shape: {txt_attn.size()}")
        # print(f"img_attn shape: {img_attn.size()}")

        img = img + img_mod1.gate * attn.img_attn.proj(img_attn)
        img = img + img_mod2.gate * attn.img_mlp((1 + img_mod2.scale) * attn.img_norm2(img) + img_mod2.shift)

        txt = txt + txt_mod1.gate * attn.txt_attn.proj(txt_attn)
        txt = txt + txt_mod2.gate * attn.txt_mlp((1 + txt_mod2.scale) * attn.txt_norm2(txt) + txt_mod2.shift)

        # print(ip_scale, image_proj)
        # IP-adapter processing
        # print(img_q.shape, img.shape, txt.shape)
        # print(image_proj.shape)

        '''
 
        torch.Size([1, 24, 1372, 128]) torch.Size([1, 1372, 3072]) torch.Size([1, 512, 3072])
        torch.Size([1, 4, 4096])
        torch.Size([1, 24, 4, 128]) torch.Size([1, 24, 4, 128])
 
        '''

        img_q = rearrange(img_q, 'B H L D -> B L H D')
        

        for i in range(len(ip_atten_mask)):
            # ip_atten_mask: bs x L
            
            ip_key = self.ip_adapter_double_stream_k_proj(image_proj[i])
            ip_value = self.ip_adapter_double_stream_v_proj(image_proj[i])

            # Reshape projections for multi-head attention
            ip_key = rearrange(ip_key, 'B L (H D) -> B H L D', H=attn.num_heads, D=attn.head_dim)
            ip_value = rearrange(ip_value, 'B L (H D) -> B H L D', H=attn.num_heads, D=attn.head_dim)

            ip_atten_mask_tmp = ip_atten_mask[i]
            # print(ip_atten_mask_tmp.shape, img_q.shape)
            # ip_query = img_q[ip_atten_mask_tmp].reshape()
            ip_query = []
            for bi in range(img_q.shape[0]):
                ip_query.append(img_q[bi][ip_atten_mask_tmp[bi]])
            ip_query = torch.stack(ip_query, axis=0)
            # print(ip_atten_mask_tmp.sum(), ip_atten_mask_tmp.shape, img_q.shape)
            ip_query = rearrange(ip_query, 'B L H D -> B H L D')
            # print(ip_key.shape, ip_value.shape)
            # Compute attention between IP projections and the latent query
            ip_attention = F.scaled_dot_product_attention(
                ip_query,
                ip_key,
                ip_value,
                dropout_p=0.0,
                is_causal=False
            )
            # print(ip_attention.shape, ip_attention.shape)
            ip_attention = rearrange(ip_attention, "B H L D -> B L (H D)", H=attn.num_heads, D=attn.head_dim)
            
            img[ip_atten_mask_tmp] = img[ip_atten_mask_tmp] + ip_scale * ip_attention + ip_scale_cloth * img_cloth
    

        return img, txt, txt_cloth

class IPDoubleStreamBlockProcessor(nn.Module):
    """Attention processor for handling IP-adapter with double stream block."""

    def __init__(self, context_dim, hidden_dim):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "IPDoubleStreamBlockProcessor requires PyTorch 2.0 or higher. Please upgrade PyTorch."
            )

        # Ensure context_dim matches the dimension of image_proj
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim

        # Initialize projections for IP-adapter 4096 
        self.ip_adapter_double_stream_k_proj = nn.Linear(context_dim, hidden_dim, bias=True)
        self.ip_adapter_double_stream_v_proj = nn.Linear(context_dim, hidden_dim, bias=True)

        nn.init.zeros_(self.ip_adapter_double_stream_k_proj.weight)
        nn.init.zeros_(self.ip_adapter_double_stream_k_proj.bias)

        nn.init.zeros_(self.ip_adapter_double_stream_v_proj.weight)
        nn.init.zeros_(self.ip_adapter_double_stream_v_proj.bias)

    def __call__(self, attn, img, txt, vec, pe, pe2=None, image_proj=None, ip_scale=1.0, ip_atten_mask=None, **attention_kwargs):

        if isinstance(image_proj, list):
            image_proj = image_proj[0]
        # Prepare image for attention
        img_mod1, img_mod2 = attn.img_mod(vec)
        txt_mod1, txt_mod2 = attn.txt_mod(vec)

        img_modulated = attn.img_norm1(img)
        img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
        img_qkv = attn.img_attn.qkv(img_modulated)
        img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads, D=attn.head_dim)
        img_q, img_k = attn.img_attn.norm(img_q, img_k, img_v)

        txt_modulated = attn.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        txt_qkv = attn.txt_attn.qkv(txt_modulated)
        txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads, D=attn.head_dim)
        txt_q, txt_k = attn.txt_attn.norm(txt_q, txt_k, txt_v)

        q = torch.cat((txt_q, img_q), dim=2)
        k = torch.cat((txt_k, img_k), dim=2)
        v = torch.cat((txt_v, img_v), dim=2)

        attn1 = attention(q, k, v, pe=pe)

        txt_attn, img_attn = attn1[:, :txt.shape[1]], attn1[:, txt.shape[1]:]

        # print(f"txt_attn shape: {txt_attn.size()}")
        # print(f"img_attn shape: {img_attn.size()}")

        img = img + img_mod1.gate * attn.img_attn.proj(img_attn)
        img = img + img_mod2.gate * attn.img_mlp((1 + img_mod2.scale) * attn.img_norm2(img) + img_mod2.shift)

        txt = txt + txt_mod1.gate * attn.txt_attn.proj(txt_attn)
        txt = txt + txt_mod2.gate * attn.txt_mlp((1 + txt_mod2.scale) * attn.txt_norm2(txt) + txt_mod2.shift)

        # print(ip_scale, image_proj)
        # IP-adapter processing
        # print(img_q.shape, img.shape, txt.shape)
        # print(image_proj.shape)

        '''
 
        torch.Size([1, 24, 1372, 128]) torch.Size([1, 1372, 3072]) torch.Size([1, 512, 3072])
        torch.Size([1, 4, 4096])
        torch.Size([1, 24, 4, 128]) torch.Size([1, 24, 4, 128])
 
        '''
        # ip_atten_mask: bs x L

        ip_query = img_q  # latent sample query
        ip_key = self.ip_adapter_double_stream_k_proj(image_proj)
        ip_value = self.ip_adapter_double_stream_v_proj(image_proj)

        # Reshape projections for multi-head attention
        ip_key = rearrange(ip_key, 'B L (H D) -> B H L D', H=attn.num_heads, D=attn.head_dim)
        ip_value = rearrange(ip_value, 'B L (H D) -> B H L D', H=attn.num_heads, D=attn.head_dim)

        # print(ip_key.shape, ip_value.shape)

        # Compute attention between IP projections and the latent query
        ip_attention = F.scaled_dot_product_attention(
            ip_query,
            ip_key,
            ip_value,
            dropout_p=0.0,
            is_causal=False
        )
        
        # print(ip_attention.shape, ip_attention.shape)

        ip_attention = rearrange(ip_attention, "B H L D -> B L (H D)", H=attn.num_heads, D=attn.head_dim)
        
        # print(ip_scale, ip_attention)
        # print("img", img)
        
        img = img + ip_scale * ip_attention

        return img, txt

def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module
    
class IPDoubleStreamBlockProcessorV4(nn.Module):
    """Attention processor for handling IP-adapter with double stream block."""

    def __init__(self, context_dim, hidden_dim):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "IPDoubleStreamBlockProcessor requires PyTorch 2.0 or higher. Please upgrade PyTorch."
            )

        # Ensure context_dim matches the dimension of image_proj
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim

        # Initialize projections for IP-adapter 4096 
        self.ip_adapter_double_stream_k_proj = nn.Linear(context_dim, hidden_dim, bias=True)
        self.ip_adapter_double_stream_v_proj = nn.Linear(context_dim, hidden_dim, bias=True)

        # self.ip_adapter_double_stream_mask_proj = nn.Linear(1, hidden_dim, bias=True)
        # self.ip_adapter_double_stream_mask_proj = nn.Linear(1, hidden_dim, bias=True)

        self.ip_adapter_double_conv_proj_block = nn.Sequential(
            zero_module(nn.Conv2d(1, 8, 3, padding=1)),
            nn.SiLU(),
            #zero_module(nn.Conv2d(8, 16, 3, padding=1)),
        )

        self.ip_adapter_double_mask_block = nn.Sequential(
            zero_module(nn.Conv2d(hidden_dim + 8, hidden_dim//8, 3, padding=1)),
            nn.SiLU(),
            zero_module(nn.Conv2d(hidden_dim//8, hidden_dim, 3, padding=1)),
            nn.SiLU(),
            # zero_module(nn.Conv2d(hidden_dim * 2, hidden_dim, 3, padding=1)),
        )

        nn.init.zeros_(self.ip_adapter_double_stream_k_proj.weight)
        nn.init.zeros_(self.ip_adapter_double_stream_k_proj.bias)

        nn.init.zeros_(self.ip_adapter_double_stream_v_proj.weight)
        nn.init.zeros_(self.ip_adapter_double_stream_v_proj.bias)

    def __call__(self, attn, img, txt, vec, pe, pe2=None, image_proj=None, ip_scale=1.0, ip_atten_mask=None, **attention_kwargs):

        # Prepare image for attention
        # print(img.shape, txt.shape, vec.shape)
        img_mod1, img_mod2 = attn.img_mod(vec)
        txt_mod1, txt_mod2 = attn.txt_mod(vec)

        img_modulated = attn.img_norm1(img)
        img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
        img_qkv = attn.img_attn.qkv(img_modulated)
        img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads, D=attn.head_dim)
        img_q, img_k = attn.img_attn.norm(img_q, img_k, img_v)

        txt_modulated = attn.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        txt_qkv = attn.txt_attn.qkv(txt_modulated)
        txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads, D=attn.head_dim)
        txt_q, txt_k = attn.txt_attn.norm(txt_q, txt_k, txt_v)

        q = torch.cat((txt_q, img_q), dim=2)
        k = torch.cat((txt_k, img_k), dim=2)
        v = torch.cat((txt_v, img_v), dim=2)

        attn1 = attention(q, k, v, pe=pe)

        # print(attn1.shape, q.shape)

        txt_attn, img_attn = attn1[:, :txt.shape[1]], attn1[:, txt.shape[1]:]

        # print(f"txt_attn shape: {txt_attn.size()}")
        # print(f"img_attn shape: {img_attn.size()}")

        img = img + img_mod1.gate * attn.img_attn.proj(img_attn)
        img = img + img_mod2.gate * attn.img_mlp((1 + img_mod2.scale) * attn.img_norm2(img) + img_mod2.shift)

        txt = txt + txt_mod1.gate * attn.txt_attn.proj(txt_attn)
        txt = txt + txt_mod2.gate * attn.txt_mlp((1 + txt_mod2.scale) * attn.txt_norm2(txt) + txt_mod2.shift)

        # print(ip_scale, image_proj)
        # IP-adapter processing
        # print(img_q.shape, img.shape, txt.shape)
        # print(image_proj.shape)

        '''
 
        torch.Size([1, 24, 1372, 128]) torch.Size([1, 1372, 3072]) torch.Size([1, 512, 3072])
        torch.Size([1, 4, 4096])
        torch.Size([1, 24, 4, 128]) torch.Size([1, 24, 4, 128])
 
        '''

        # img_q = rearrange(img_q, 'B H L D -> B L H D')
        ip_query = img_q

        for i in range(len(ip_atten_mask)):
            # ip_atten_mask: bs x L
            
            ip_key = self.ip_adapter_double_stream_k_proj(image_proj[i])
            ip_value = self.ip_adapter_double_stream_v_proj(image_proj[i])

            # Reshape projections for multi-head attention
            ip_key = rearrange(ip_key, 'B L (H D) -> B H L D', H=attn.num_heads, D=attn.head_dim)
            ip_value = rearrange(ip_value, 'B L (H D) -> B H L D', H=attn.num_heads, D=attn.head_dim)
            
            # ip_query ==> B x H x L x D
            ip_atten_mask_tmp = ip_atten_mask[i].reshape(1, 1, 1344//16, 768//16) # B x L x 1 >> B x L (HxD)
            # ip_atten_mask_tmp = rearrange(ip_atten_mask_tmp, 'B L -> B C H W', C=1, H=768//16, W=448//16)

            ip_atten_mask_tmp = self.ip_adapter_double_conv_proj_block(ip_atten_mask_tmp.to(ip_query.dtype))
            # print(ip_atten_mask_tmp.shape)
            ip_query_tmp = rearrange(ip_query, 'B H L D -> B (H D) L').reshape(1, -1, 1344//16, 768//16)

            ip_query_tmp_bk = ip_query_tmp
            ip_query_tmp = torch.cat([ip_query_tmp, ip_atten_mask_tmp], axis=1)

            ip_query_tmp = ip_query_tmp_bk + self.ip_adapter_double_mask_block(ip_query_tmp)
            
            ip_query_tmp = rearrange(ip_query_tmp, 'B (H D) sH sW -> B H (sH sW) D', H=attn.num_heads, D=attn.head_dim)

            # print(ip_atten_mask_tmp.shape, img_q.shape)
            # ip_query = img_q[ip_atten_mask_tmp].reshape()
            # ip_query = []
            # for bi in range(img_q.shape[0]):
            #     ip_query.append(img_q[bi][ip_atten_mask_tmp[bi]])
            # ip_query = torch.stack(ip_query, axis=0)
            # # print(ip_atten_mask_tmp.sum(), ip_atten_mask_tmp.shape, img_q.shape)
            # ip_query = rearrange(ip_query, 'B L H D -> B H L D')
            # print(ip_key.shape, ip_value.shape)
            # Compute attention between IP projections and the latent query
            ip_attention = F.scaled_dot_product_attention(
                ip_query_tmp,
                ip_key,
                ip_value,
                dropout_p=0.0,
                is_causal=False
            )
            # print(ip_attention.shape, ip_attention.shape)
            ip_attention = rearrange(ip_attention, "B H L D -> B L (H D)", H=attn.num_heads, D=attn.head_dim)
            # print(ip_scale, ip_attention)
            # print("img", img)
            
            #mask = torch.stack([ip_atten_mask_tmp] * ip_attention.shape[-1], axis=-1)
            #attn_1_img = attn_1_img + ip_scale * (ip_attention * mask)

            img = img + ip_scale * ip_attention
    

        return img, txt, image_proj

class DoubleStreamBlockProcessor:
    def __call__(self, attn, img, txt, vec, pe, **attention_kwargs):
        img_mod1, img_mod2 = attn.img_mod(vec)
        txt_mod1, txt_mod2 = attn.txt_mod(vec)

        # prepare image for attention
        img_modulated = attn.img_norm1(img)
        img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
        img_qkv = attn.img_attn.qkv(img_modulated)
        img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads, D=attn.head_dim)
        img_q, img_k = attn.img_attn.norm(img_q, img_k, img_v)

        # prepare txt for attention
        txt_modulated = attn.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        txt_qkv = attn.txt_attn.qkv(txt_modulated)
        txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads, D=attn.head_dim)
        txt_q, txt_k = attn.txt_attn.norm(txt_q, txt_k, txt_v)

        # run actual attention
        q = torch.cat((txt_q, img_q), dim=2)
        k = torch.cat((txt_k, img_k), dim=2)
        v = torch.cat((txt_v, img_v), dim=2)

        attn1 = attention(q, k, v, pe=pe)
        
        txt_attn, img_attn = attn1[:, : txt.shape[1]], attn1[:, txt.shape[1] :]

        # calculate the img bloks
        img = img + img_mod1.gate * attn.img_attn.proj(img_attn)
        img = img + img_mod2.gate * attn.img_mlp((1 + img_mod2.scale) * attn.img_norm2(img) + img_mod2.shift)

        # calculate the txt bloks
        txt = txt + txt_mod1.gate * attn.txt_attn.proj(txt_attn)
        txt = txt + txt_mod2.gate * attn.txt_mlp((1 + txt_mod2.scale) * attn.txt_norm2(txt) + txt_mod2.shift)
        return img, txt

class DoubleStreamBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float, qkv_bias: bool = False):
        super().__init__()
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.head_dim = hidden_size // num_heads

        self.img_mod = Modulation(hidden_size, double=True)
        self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)

        self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

        self.txt_mod = Modulation(hidden_size, double=True)
        self.txt_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)

        self.txt_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

        processor = DoubleStreamBlockProcessor()
        self.set_processor(processor)

    def set_processor(self, processor) -> None:
        self.processor = processor

    def get_processor(self):
        return self.processor

    def forward(
        self,
        img: Tensor,
        txt: Tensor,
        vec: Tensor,
        pe: Tensor,
        image_proj: Tensor = None,
        ip_scale: float =1.0,
        ip_scale_cloth: float =1.0,
        pe2: Tensor = None,
        ip_atten_mask: Tensor = None,
        txt_cloth=None,
        vec_cloth=None,
        pe_cloth=None,
        lora_weight=1,
        lora_weight_list=None,
        **kwargs
    ) -> tuple[Tensor, Tensor]:
        if image_proj is None:
            # print("none image_proj")
            return self.processor(self, img, txt, vec, pe, lora_weight=lora_weight)
        else:
            # print("do image_proj")
            # print(self.processor)
            return self.processor(self, img, txt, vec, pe, pe2, image_proj, ip_scale, ip_atten_mask, ip_scale_cloth=ip_scale_cloth, txt_cloth=txt_cloth, vec_cloth=vec_cloth, pe_cloth=pe_cloth, lora_weight_list=lora_weight_list, **kwargs)


class IPSingleStreamBlockProcessorV5(nn.Module):
    """Attention processor for handling IP-adapter with single stream block."""
    def __init__(self, context_dim, hidden_dim):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "IPSingleStreamBlockProcessor requires PyTorch 2.0 or higher. Please upgrade PyTorch."
            )

        # Ensure context_dim matches the dimension of image_proj
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim

        # Initialize projections for IP-adapter
        self.ip_adapter_single_stream_k_proj = nn.Linear(context_dim, hidden_dim, bias=False)
        self.ip_adapter_single_stream_v_proj = nn.Linear(context_dim, hidden_dim, bias=False)

        nn.init.zeros_(self.ip_adapter_single_stream_k_proj.weight)
        nn.init.zeros_(self.ip_adapter_single_stream_v_proj.weight)

    def __call__(
        self,
        attn: nn.Module,
        img: Tensor,
        vec: Tensor,
        pe: Tensor,
        image_proj: Tensor | None = None,
        ip_scale: float = 1.0,
        ip_atten_mask=None,
        txt=None,
        txt_cloth=None,
        vec_cloth=None,
        pe_cloth=None,
        ip_scale_cloth: float = 1.0,
    ) -> Tensor:
        
        x = torch.cat((txt, img), dim=1)
        x_cloth = torch.cat((txt_cloth, img), dim=1)
        x_all = torch.cat((txt, txt_cloth, img), dim=1)
        mod, _ = attn.modulation(vec)
        mod_cloth, _ = attn.modulation(vec_cloth)
        x_mod = (1 + mod.scale) * attn.pre_norm(x) + mod.shift
        x_cloth_mod = (1 + mod_cloth.scale) * attn.pre_norm(x_cloth) + mod_cloth.shift
        qkv, mlp = torch.split(attn.linear1(x_mod), [3 * attn.hidden_size, attn.mlp_hidden_dim], dim=-1)
        qkv_cloth, mlp_cloth = torch.split(attn.linear1(x_cloth_mod), [3 * attn.hidden_size, attn.mlp_hidden_dim], dim=-1)
        
        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads, D=attn.head_dim)
        q_cloth, k_cloth, v_cloth = rearrange(qkv_cloth, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads, D=attn.head_dim)
        # print(" ===>>>> 1", q.shape)

        q, k = attn.norm(q, k, v)
        q_cloth, k_cloth = attn.norm(q_cloth, k_cloth, v_cloth)
        # print(" ===>>>> 2", q.shape)

        # compute attention
        attn_1 = attention(q, k, v, pe=pe)
        # txt_cloth_pe = pe_cloth[:,:,:txt_cloth.shape[1]]
        # img_cloth_pe = pe_cloth[:,:,txt_cloth.shape[1]:]
        # img_cloth_pe = img_cloth_pe[:,:,ip_atten_mask[0][0]]
        # pe_cloth = torch.cat((txt_cloth_pe, img_cloth_pe), dim=2)
        attn_1_cloth = attention(q_cloth, k_cloth, v_cloth, pe=pe_cloth)
        # IP-adapter processing 
        # attn_out = attn_1
        
        # print(" ===>>>> 3", q.shape)
        # tmp_q = rearrange(q, "B H L D -> B L (H D)")

        # ip_query = rearrange(ip_query, "B L (H D) -> B H L D", K=3, H=attn.num_heads, D=attn.head_dim)
        # ip_key = self.ip_adapter_single_stream_k_proj(image_proj)
        # ip_value = self.ip_adapter_single_stream_v_proj(image_proj)
        # ip_attention_list = []
        ip_query = q[:, :, -ip_atten_mask[0].shape[1]:]
        attn_1_img = attn_1[:, -ip_atten_mask[0].shape[1]:, ]
        attn_1_txt = attn_1[:, :-ip_atten_mask[0].shape[1], ]
        
        attn_1_cloth_img = attn_1_cloth[:, -ip_atten_mask[0].shape[1]:, ]
        attn_1_cloth_txt = attn_1_cloth[:, :-ip_atten_mask[0].shape[1], ]
        # base_atten = attn_1[:, -ip_atten_mask[0].shape[1]:, ]
        for i in range(len(ip_atten_mask)):
            # ip_query = q[:, :, -ip_atten_mask[0].shape[1]:]
            ip_key = self.ip_adapter_single_stream_k_proj(image_proj[i])
            ip_value = self.ip_adapter_single_stream_v_proj(image_proj[i])

            ip_atten_mask_tmp = ip_atten_mask[i]
            # ip_query = img_q[ip_atten_mask_tmp].reshape()
            # ip_query_list = []
            # 实际上只能是1，不然需要pad

            # print("==>1", ip_query.shape, ip_atten_mask_tmp.shape)
            # ip_query_tmp = rearrange(ip_query, "B H L D -> B L (H D)")
            # # print("==>2",ip_query.shape, ip_atten_mask_tmp.shape)
            # ip_query_tmp = ip_query_tmp[ip_atten_mask_tmp].reshape(ip_query_tmp.shape[0], -1, ip_query_tmp.shape(-1))

            # for bi in range(ip_query.shape[0]):
            #     # print()
            #     ip_query_list.append(ip_query[ip_atten_mask_tmp])
            # ip_query = torch.stack(ip_query_list, axis=0)
            # print("==>3",ip_query.shape, ip_atten_mask_tmp.shape)
            # ip_query_tmp = rearrange(ip_query_tmp, "B L (H D) -> B H L D", H=attn.num_heads, D=attn.head_dim)

            # Reshape projections for multi-head attention
            ip_key = rearrange(ip_key, 'B L (H D) -> B H L D', H=attn.num_heads, D=attn.head_dim)
            ip_value = rearrange(ip_value, 'B L (H D) -> B H L D', H=attn.num_heads, D=attn.head_dim)

            # attention()
            # Compute attention between IP projections and the latent query
            ip_attention = F.scaled_dot_product_attention(
                ip_query.clone(),
                ip_key,
                ip_value,
                dropout_p=0.0,
                is_causal=False
            )
            ip_attention = rearrange(ip_attention, "B H L D -> B L (H D)")
            
            # print("++>ip_attention", ip_attention.shape, ip_atten_mask_tmp.shape)
            # print("")

            #base_atten[:, ip_atten_mask_tmp[0]] = base_atten[:, ip_atten_mask_tmp[0]] + ip_scale * ip_attention
            mask = torch.stack([ip_atten_mask_tmp] * ip_attention.shape[-1], axis=-1)
            attn_1_img = attn_1_img + ip_scale * (ip_attention * mask) + ip_scale_cloth * (attn_1_cloth_img * mask)
            
        # print("attn_out", attn_1.shape)
        # compute activation in mlp stream, cat again and run second linear layer
        attn_1 = torch.cat([attn_1_txt, attn_1_img], axis=1)
        output = attn.linear2(torch.cat((attn_1, attn.mlp_act(mlp)), 2))
        out = x + mod.gate * output
        
        out_txt = out[:, :txt.shape[1], ]
        out_img = out[:, txt.shape[1]:, ]
        
        attn_1_cloth = torch.cat([attn_1_cloth_txt, out_img], axis=1)
        output_cloth = attn.linear2(torch.cat((attn_1_cloth, attn.mlp_act(mlp_cloth)), 2))
        out_txt_cloth = x_cloth + mod_cloth.gate * output_cloth
        
        out_img = out_txt_cloth[:, txt_cloth.shape[1]:, ]
        out_txt_cloth = out_txt_cloth[:, :txt_cloth.shape[1], ]
        # print("out==>", out.shape)

        return out_txt, out_txt_cloth, out_img


class IPSingleStreamBlockProcessorV4(nn.Module):
    """Attention processor for handling IP-adapter with single stream block."""
    def __init__(self, context_dim, hidden_dim):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "IPSingleStreamBlockProcessor requires PyTorch 2.0 or higher. Please upgrade PyTorch."
            )

        # Ensure context_dim matches the dimension of image_proj
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim

        # Initialize projections for IP-adapter
        self.ip_adapter_single_stream_k_proj = nn.Linear(context_dim, hidden_dim, bias=False)
        self.ip_adapter_single_stream_v_proj = nn.Linear(context_dim, hidden_dim, bias=False)

        nn.init.zeros_(self.ip_adapter_single_stream_k_proj.weight)
        nn.init.zeros_(self.ip_adapter_single_stream_v_proj.weight)

        self.ip_adapter_single_conv_proj_block = nn.Sequential(
            zero_module(nn.Conv2d(1, 8, 3, padding=1)),
            nn.SiLU(),
            # zero_module(nn.Conv2d(8, 16, 3, padding=1)),
        )

        self.ip_adapter_single_mask_block = nn.Sequential(
            zero_module(nn.Conv2d(hidden_dim + 8, hidden_dim//8, 3, padding=1)),
            nn.SiLU(),
            zero_module(nn.Conv2d(hidden_dim//8, hidden_dim, 3, padding=1)),
            nn.SiLU(),
            # zero_module(nn.Conv2d(hidden_dim * 2, hidden_dim, 3, padding=1)),
        )


    def __call__(
        self,
        attn: nn.Module,
        x: Tensor,
        vec: Tensor,
        pe: Tensor,
        image_proj: Tensor | None = None,
        ip_scale: float = 1.0,
        ip_atten_mask=None,
    ) -> Tensor:

        mod, _ = attn.modulation(vec)
        x_mod = (1 + mod.scale) * attn.pre_norm(x) + mod.shift
        qkv, mlp = torch.split(attn.linear1(x_mod), [3 * attn.hidden_size, attn.mlp_hidden_dim], dim=-1)

        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads, D=attn.head_dim)

        # print(" ===>>>> 1", q.shape)

        q, k = attn.norm(q, k, v)

        # print(" ===>>>> 2", q.shape)

        # compute attention
        attn_1 = attention(q, k, v, pe=pe)

        # IP-adapter processing 
        # attn_out = attn_1
        
        # print(" ===>>>> 3", q.shape)
        # tmp_q = rearrange(q, "B H L D -> B L (H D)")

        # ip_query = rearrange(ip_query, "B L (H D) -> B H L D", K=3, H=attn.num_heads, D=attn.head_dim)
        # ip_key = self.ip_adapter_single_stream_k_proj(image_proj)
        # ip_value = self.ip_adapter_single_stream_v_proj(image_proj)
        # ip_attention_list = []
        ip_query = q[:, :, -ip_atten_mask[0].shape[1]:]
        attn_1_img = attn_1[:, -ip_atten_mask[0].shape[1]:, ]
        attn_1_txt = attn_1[:, :-ip_atten_mask[0].shape[1], ]
        # base_atten = attn_1[:, -ip_atten_mask[0].shape[1]:, ]
        for i in range(len(ip_atten_mask)):
            # ip_query = q[:, :, -ip_atten_mask[0].shape[1]:]
            ip_key = self.ip_adapter_single_stream_k_proj(image_proj[i])
            ip_value = self.ip_adapter_single_stream_v_proj(image_proj[i])

            # ip_atten_mask_tmp = ip_atten_mask[i]
            # ip_query = img_q[ip_atten_mask_tmp].reshape()
            # ip_query_list = []
            # 实际上只能是1，不然需要pad

            # print("==>1", ip_query.shape, ip_atten_mask_tmp.shape)
            # ip_query_tmp = rearrange(ip_query, "B H L D -> B L (H D)")
            # # print("==>2",ip_query.shape, ip_atten_mask_tmp.shape)
            # ip_query_tmp = ip_query_tmp[ip_atten_mask_tmp].reshape(ip_query_tmp.shape[0], -1, ip_query_tmp.shape(-1))

            # for bi in range(ip_query.shape[0]):
            #     # print()
            #     ip_query_list.append(ip_query[ip_atten_mask_tmp])
            # ip_query = torch.stack(ip_query_list, axis=0)
            # print("==>3",ip_query.shape, ip_atten_mask_tmp.shape)
            # ip_query_tmp = rearrange(ip_query_tmp, "B L (H D) -> B H L D", H=attn.num_heads, D=attn.head_dim)

            # Reshape projections for multi-head attention
            ip_key = rearrange(ip_key, 'B L (H D) -> B H L D', H=attn.num_heads, D=attn.head_dim)
            ip_value = rearrange(ip_value, 'B L (H D) -> B H L D', H=attn.num_heads, D=attn.head_dim)

             # ip_query ==> B x H x L x D
            ip_atten_mask_tmp = ip_atten_mask[i].reshape(1, 1, 1344//16, 768//16) # B x L x 1 >> B x L (HxD)
            # ip_atten_mask_tmp = rearrange(ip_atten_mask_tmp, 'B L -> B C H W', C=1, H=768//16, W=448//16)

            ip_atten_mask_tmp = self.ip_adapter_single_conv_proj_block(ip_atten_mask_tmp.to(ip_query.dtype))
            # print(ip_atten_mask_tmp.shape)
            ip_query_tmp = rearrange(ip_query, 'B H L D -> B (H D) L').reshape(1, -1, 1344//16, 768//16)

            ip_query_tmp_bk = ip_query_tmp
            ip_query_tmp = torch.cat([ip_query_tmp, ip_atten_mask_tmp], axis=1)

            ip_query_tmp = ip_query_tmp_bk + self.ip_adapter_single_mask_block(ip_query_tmp)
            
            ip_query_tmp = rearrange(ip_query_tmp, 'B (H D) sH sW -> B H (sH sW) D', H=attn.num_heads, D=attn.head_dim)


            # attention()
            # Compute attention between IP projections and the latent query
            ip_attention = F.scaled_dot_product_attention(
                ip_query_tmp.clone(),
                ip_key,
                ip_value,
                dropout_p=0.0,
                is_causal=False
            )
            ip_attention = rearrange(ip_attention, "B H L D -> B L (H D)")
            
            # print("++>ip_attention", ip_attention.shape, ip_atten_mask_tmp.shape)
            # print("")

            #base_atten[:, ip_atten_mask_tmp[0]] = base_atten[:, ip_atten_mask_tmp[0]] + ip_scale * ip_attention
            # mask = torch.stack([ip_atten_mask_tmp] * ip_attention.shape[-1], axis=-1)
            attn_1_img = attn_1_img + ip_scale * (ip_attention)
            
        # print("attn_out", attn_1.shape)
        # compute activation in mlp stream, cat again and run second linear layer
        attn_1 = torch.cat([attn_1_txt, attn_1_img], axis=1)
        output = attn.linear2(torch.cat((attn_1, attn.mlp_act(mlp)), 2))
        out = x + mod.gate * output

        # print("out==>", out.shape)

        return out

class IPSingleStreamBlockProcessorV3(nn.Module):
    """Attention processor for handling IP-adapter with single stream block."""
    def __init__(self, context_dim, hidden_dim):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "IPSingleStreamBlockProcessor requires PyTorch 2.0 or higher. Please upgrade PyTorch."
            )

        # Ensure context_dim matches the dimension of image_proj
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim

        # Initialize projections for IP-adapter
        self.ip_adapter_single_stream_k_proj = nn.Linear(context_dim, hidden_dim, bias=False)
        self.ip_adapter_single_stream_v_proj = nn.Linear(context_dim, hidden_dim, bias=False)

        nn.init.zeros_(self.ip_adapter_single_stream_k_proj.weight)
        nn.init.zeros_(self.ip_adapter_single_stream_v_proj.weight)

    def __call__(
        self,
        attn: nn.Module,
        x: Tensor,
        vec: Tensor,
        pe: Tensor,
        image_proj: Tensor | None = None,
        ip_scale: float = 1.0,
        ip_atten_mask=None,
        **kwargs,
    ) -> Tensor:

        mod, _ = attn.modulation(vec)
        x_mod = (1 + mod.scale) * attn.pre_norm(x) + mod.shift
        qkv, mlp = torch.split(attn.linear1(x_mod), [3 * attn.hidden_size, attn.mlp_hidden_dim], dim=-1)

        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads, D=attn.head_dim)

        # print(" ===>>>> 1", q.shape)

        q, k = attn.norm(q, k, v)

        # print(" ===>>>> 2", q.shape)

        # compute attention
        attn_1 = attention(q, k, v, pe=pe)

        # IP-adapter processing 
        # attn_out = attn_1
        
        # print(" ===>>>> 3", q.shape)
        # tmp_q = rearrange(q, "B H L D -> B L (H D)")

        # ip_query = rearrange(ip_query, "B L (H D) -> B H L D", K=3, H=attn.num_heads, D=attn.head_dim)
        # ip_key = self.ip_adapter_single_stream_k_proj(image_proj)
        # ip_value = self.ip_adapter_single_stream_v_proj(image_proj)
        # ip_attention_list = []
        ip_query = q[:, :, -ip_atten_mask[0].shape[1]:]
        attn_1_img = attn_1[:, -ip_atten_mask[0].shape[1]:, ]
        attn_1_txt = attn_1[:, :-ip_atten_mask[0].shape[1], ]
        # base_atten = attn_1[:, -ip_atten_mask[0].shape[1]:, ]
        for i in range(len(ip_atten_mask)):
            # ip_query = q[:, :, -ip_atten_mask[0].shape[1]:]
            ip_key = self.ip_adapter_single_stream_k_proj(image_proj[i])
            ip_value = self.ip_adapter_single_stream_v_proj(image_proj[i])

            ip_atten_mask_tmp = ip_atten_mask[i]
            # ip_query = img_q[ip_atten_mask_tmp].reshape()
            # ip_query_list = []
            # 实际上只能是1，不然需要pad

            # print("==>1", ip_query.shape, ip_atten_mask_tmp.shape)
            # ip_query_tmp = rearrange(ip_query, "B H L D -> B L (H D)")
            # # print("==>2",ip_query.shape, ip_atten_mask_tmp.shape)
            # ip_query_tmp = ip_query_tmp[ip_atten_mask_tmp].reshape(ip_query_tmp.shape[0], -1, ip_query_tmp.shape(-1))

            # for bi in range(ip_query.shape[0]):
            #     # print()
            #     ip_query_list.append(ip_query[ip_atten_mask_tmp])
            # ip_query = torch.stack(ip_query_list, axis=0)
            # print("==>3",ip_query.shape, ip_atten_mask_tmp.shape)
            # ip_query_tmp = rearrange(ip_query_tmp, "B L (H D) -> B H L D", H=attn.num_heads, D=attn.head_dim)

            # Reshape projections for multi-head attention
            ip_key = rearrange(ip_key, 'B L (H D) -> B H L D', H=attn.num_heads, D=attn.head_dim)
            ip_value = rearrange(ip_value, 'B L (H D) -> B H L D', H=attn.num_heads, D=attn.head_dim)

            # attention()
            # Compute attention between IP projections and the latent query
            ip_attention = F.scaled_dot_product_attention(
                ip_query.clone(),
                ip_key,
                ip_value,
                dropout_p=0.0,
                is_causal=False
            )
            ip_attention = rearrange(ip_attention, "B H L D -> B L (H D)")
            
            # print("++>ip_attention", ip_attention.shape, ip_atten_mask_tmp.shape)
            # print("")

            #base_atten[:, ip_atten_mask_tmp[0]] = base_atten[:, ip_atten_mask_tmp[0]] + ip_scale * ip_attention
            mask = torch.stack([ip_atten_mask_tmp] * ip_attention.shape[-1], axis=-1)
            attn_1_img = attn_1_img + ip_scale * (ip_attention * mask)
            
        # print("attn_out", attn_1.shape)
        # compute activation in mlp stream, cat again and run second linear layer
        attn_1 = torch.cat([attn_1_txt, attn_1_img], axis=1)
        output = attn.linear2(torch.cat((attn_1, attn.mlp_act(mlp)), 2))
        out = x + mod.gate * output

        # print("out==>", out.shape)

        return out


class IPSingleStreamBlockProcessorV6(nn.Module):
    """Attention processor for handling IP-adapter with single stream block."""
    def __init__(self, context_dim, hidden_dim, portrait_dim=64):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "IPSingleStreamBlockProcessor requires PyTorch 2.0 or higher. Please upgrade PyTorch."
            )

        # Ensure context_dim matches the dimension of image_proj
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim

        # Initialize projections for IP-adapter
        self.ip_adapter_single_stream_k_proj = nn.Linear(context_dim, hidden_dim, bias=False)
        self.ip_adapter_single_stream_v_proj = nn.Linear(context_dim, hidden_dim, bias=False)

        nn.init.zeros_(self.ip_adapter_single_stream_k_proj.weight)
        nn.init.zeros_(self.ip_adapter_single_stream_v_proj.weight)
        
        self.ip_adapter_single_stream_k_proj_portrait = nn.Linear(portrait_dim, hidden_dim, bias=False)
        self.ip_adapter_single_stream_v_proj_portrait = nn.Linear(portrait_dim, hidden_dim, bias=False)

        nn.init.zeros_(self.ip_adapter_single_stream_k_proj_portrait.weight)
        nn.init.zeros_(self.ip_adapter_single_stream_v_proj_portrait.weight)

    def __call__(
        self,
        attn: nn.Module,
        x: Tensor,
        vec: Tensor,
        pe: Tensor,
        image_proj: Tensor | None = None,
        ip_scale: float = 1.0,
        ip_atten_mask=None,
        portrait_prompt=None,
        portrait_scale=1.0,
        **kwargs,
    ) -> Tensor:
        # portrait_scale = 0.0
        if portrait_prompt is None:
            print("portrait_prompt is None!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

        mod, _ = attn.modulation(vec)
        x_mod = (1 + mod.scale) * attn.pre_norm(x) + mod.shift
        qkv, mlp = torch.split(attn.linear1(x_mod), [3 * attn.hidden_size, attn.mlp_hidden_dim], dim=-1)

        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads, D=attn.head_dim)

        # print(" ===>>>> 1", q.shape)

        q, k = attn.norm(q, k, v)

        # print(" ===>>>> 2", q.shape)

        # compute attention
        attn_1 = attention(q, k, v, pe=pe)

        # IP-adapter processing 
        # attn_out = attn_1
        
        # print(" ===>>>> 3", q.shape)
        # tmp_q = rearrange(q, "B H L D -> B L (H D)")

        # ip_query = rearrange(ip_query, "B L (H D) -> B H L D", K=3, H=attn.num_heads, D=attn.head_dim)
        # ip_key = self.ip_adapter_single_stream_k_proj(image_proj)
        # ip_value = self.ip_adapter_single_stream_v_proj(image_proj)
        # ip_attention_list = []
        ip_query = q[:, :, -ip_atten_mask[0].shape[1]:]
        attn_1_img = attn_1[:, -ip_atten_mask[0].shape[1]:, ]
        attn_1_txt = attn_1[:, :-ip_atten_mask[0].shape[1], ]
        # base_atten = attn_1[:, -ip_atten_mask[0].shape[1]:, ]
        for i in range(len(ip_atten_mask)):
            # ip_query = q[:, :, -ip_atten_mask[0].shape[1]:]
            ip_key = self.ip_adapter_single_stream_k_proj(image_proj[i])
            ip_value = self.ip_adapter_single_stream_v_proj(image_proj[i])

            ip_atten_mask_tmp = ip_atten_mask[i]
            # ip_query = img_q[ip_atten_mask_tmp].reshape()
            # ip_query_list = []
            # 实际上只能是1，不然需要pad

            # print("==>1", ip_query.shape, ip_atten_mask_tmp.shape)
            # ip_query_tmp = rearrange(ip_query, "B H L D -> B L (H D)")
            # # print("==>2",ip_query.shape, ip_atten_mask_tmp.shape)
            # ip_query_tmp = ip_query_tmp[ip_atten_mask_tmp].reshape(ip_query_tmp.shape[0], -1, ip_query_tmp.shape(-1))

            # for bi in range(ip_query.shape[0]):
            #     # print()
            #     ip_query_list.append(ip_query[ip_atten_mask_tmp])
            # ip_query = torch.stack(ip_query_list, axis=0)
            # print("==>3",ip_query.shape, ip_atten_mask_tmp.shape)
            # ip_query_tmp = rearrange(ip_query_tmp, "B L (H D) -> B H L D", H=attn.num_heads, D=attn.head_dim)

            # Reshape projections for multi-head attention
            ip_key = rearrange(ip_key, 'B L (H D) -> B H L D', H=attn.num_heads, D=attn.head_dim)
            ip_value = rearrange(ip_value, 'B L (H D) -> B H L D', H=attn.num_heads, D=attn.head_dim)

            # attention()
            # Compute attention between IP projections and the latent query
            ip_attention = F.scaled_dot_product_attention(
                ip_query.clone(),
                ip_key,
                ip_value,
                dropout_p=0.0,
                is_causal=False
            )
            ip_attention = rearrange(ip_attention, "B H L D -> B L (H D)")

            # print("++>ip_attention", ip_attention.shape, ip_atten_mask_tmp.shape)
            # print("")

            #base_atten[:, ip_atten_mask_tmp[0]] = base_atten[:, ip_atten_mask_tmp[0]] + ip_scale * ip_attention
            mask = torch.stack([ip_atten_mask_tmp] * ip_attention.shape[-1], axis=-1)
            attn_1_img = attn_1_img + ip_scale * (ip_attention * mask)

        for i in range(len(ip_atten_mask)):
            # print("Do portrait prompt")
            ip_key = self.ip_adapter_single_stream_k_proj_portrait(portrait_prompt[0])
            ip_value = self.ip_adapter_single_stream_v_proj_portrait(portrait_prompt[0])

            ip_atten_mask_tmp = ip_atten_mask[i]
            # Reshape projections for multi-head attention
            ip_key = rearrange(ip_key, 'B L (H D) -> B H L D', H=attn.num_heads, D=attn.head_dim)
            ip_value = rearrange(ip_value, 'B L (H D) -> B H L D', H=attn.num_heads, D=attn.head_dim)
            # attention()
            # Compute attention between IP projections and the latent query
            ip_attention = F.scaled_dot_product_attention(
                ip_query.clone(),
                ip_key,
                ip_value,
                dropout_p=0.0,
                is_causal=False
            )
            ip_attention_1 = rearrange(ip_attention, "B H L D -> B L (H D)")
            # mask = torch.stack([ip_atten_mask_tmp] * ip_attention.shape[-1], axis=-1)
            # attn_1_img = attn_1_img + portrait_scale * (ip_attention * mask)
            
        for i in range(len(ip_atten_mask)):
            # print("Do portrait prompt")
            ip_key = self.ip_adapter_single_stream_k_proj_portrait(portrait_prompt[1])
            ip_value = self.ip_adapter_single_stream_v_proj_portrait(portrait_prompt[1])

            ip_atten_mask_tmp = ip_atten_mask[i]
            # Reshape projections for multi-head attention
            ip_key = rearrange(ip_key, 'B L (H D) -> B H L D', H=attn.num_heads, D=attn.head_dim)
            ip_value = rearrange(ip_value, 'B L (H D) -> B H L D', H=attn.num_heads, D=attn.head_dim)
            # attention()
            # Compute attention between IP projections and the latent query
            ip_attention = F.scaled_dot_product_attention(
                ip_query.clone(),
                ip_key,
                ip_value,
                dropout_p=0.0,
                is_causal=False
            )
            ip_attention_2 = rearrange(ip_attention, "B H L D -> B L (H D)")
            # mask = torch.stack([ip_atten_mask_tmp] * ip_attention.shape[-1], axis=-1)
            # attn_1_img = attn_1_img + portrait_scale * (ip_attention * mask)
            
        ip_attention = ip_attention_1 - ip_attention_2
        mask = torch.stack([ip_atten_mask_tmp] * ip_attention.shape[-1], axis=-1)
        attn_1_img = attn_1_img + portrait_scale * (ip_attention * mask)
        
        # print("attn_out", attn_1.shape)
        # compute activation in mlp stream, cat again and run second linear layer
        attn_1 = torch.cat([attn_1_txt, attn_1_img], axis=1)
        output = attn.linear2(torch.cat((attn_1, attn.mlp_act(mlp)), 2))
        out = x + mod.gate * output

        # print("out==>", out.shape)

        return out


class IPSingleStreamBlockProcessorV7(nn.Module):
    """Attention processor for handling IP-adapter with single stream block."""
    def __init__(self, context_dim, hidden_dim, feature_in_dim=3072):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "IPSingleStreamBlockProcessor requires PyTorch 2.0 or higher. Please upgrade PyTorch."
            )

        # Ensure context_dim matches the dimension of image_proj
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim

        # Initialize projections for IP-adapter
        self.ip_adapter_single_stream_k_proj = nn.Linear(context_dim, hidden_dim, bias=False)
        self.ip_adapter_single_stream_v_proj = nn.Linear(context_dim, hidden_dim, bias=False)

        nn.init.zeros_(self.ip_adapter_single_stream_k_proj.weight)
        nn.init.zeros_(self.ip_adapter_single_stream_v_proj.weight)
        
        self.ip_adapter_single_stream_k_proj_vae = nn.Linear(feature_in_dim, hidden_dim, bias=False)
        self.ip_adapter_single_stream_v_proj_vae = nn.Linear(feature_in_dim, hidden_dim, bias=False)

        nn.init.zeros_(self.ip_adapter_single_stream_k_proj_vae.weight)
        nn.init.zeros_(self.ip_adapter_single_stream_v_proj_vae.weight)

    def __call__(
        self,
        attn: nn.Module,
        x: Tensor,
        vec: Tensor,
        pe: Tensor,
        image_proj: Tensor | None = None,
        ip_scale: float = 1.0,
        ip_atten_mask=None,
        vae_controlnet_hidden_states=None,
        use_ipa=True,
        **kwargs,
    ) -> Tensor:
        # portrait_scale = 0.0
        if vae_controlnet_hidden_states is None and use_ipa:
            print("vae_controlnet_hidden_states is None!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

        mod, _ = attn.modulation(vec)
        x_mod = (1 + mod.scale) * attn.pre_norm(x) + mod.shift
        qkv, mlp = torch.split(attn.linear1(x_mod), [3 * attn.hidden_size, attn.mlp_hidden_dim], dim=-1)

        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads, D=attn.head_dim)

        # print(" ===>>>> 1", q.shape)

        q, k = attn.norm(q, k, v)

        # print(" ===>>>> 2", q.shape)

        # compute attention
        attn_1 = attention(q, k, v, pe=pe)

        # IP-adapter processing 
        # attn_out = attn_1
        
        # print(" ===>>>> 3", q.shape)
        # tmp_q = rearrange(q, "B H L D -> B L (H D)")

        # ip_query = rearrange(ip_query, "B L (H D) -> B H L D", K=3, H=attn.num_heads, D=attn.head_dim)
        # ip_key = self.ip_adapter_single_stream_k_proj(image_proj)
        # ip_value = self.ip_adapter_single_stream_v_proj(image_proj)
        # ip_attention_list = []
        ip_query = q[:, :, -ip_atten_mask[0].shape[1]:]
        attn_1_img = attn_1[:, -ip_atten_mask[0].shape[1]:, ]
        attn_1_txt = attn_1[:, :-ip_atten_mask[0].shape[1], ]
        # base_atten = attn_1[:, -ip_atten_mask[0].shape[1]:, ]
        if use_ipa:
            for i in range(len(ip_atten_mask)):
                # ip_query = q[:, :, -ip_atten_mask[0].shape[1]:]
                ip_key = self.ip_adapter_single_stream_k_proj(image_proj[i])
                ip_value = self.ip_adapter_single_stream_v_proj(image_proj[i])

                ip_atten_mask_tmp = ip_atten_mask[i]
                # ip_query = img_q[ip_atten_mask_tmp].reshape()
                # ip_query_list = []
                # 实际上只能是1，不然需要pad

                # print("==>1", ip_query.shape, ip_atten_mask_tmp.shape)
                # ip_query_tmp = rearrange(ip_query, "B H L D -> B L (H D)")
                # # print("==>2",ip_query.shape, ip_atten_mask_tmp.shape)
                # ip_query_tmp = ip_query_tmp[ip_atten_mask_tmp].reshape(ip_query_tmp.shape[0], -1, ip_query_tmp.shape(-1))

                # for bi in range(ip_query.shape[0]):
                #     # print()
                #     ip_query_list.append(ip_query[ip_atten_mask_tmp])
                # ip_query = torch.stack(ip_query_list, axis=0)
                # print("==>3",ip_query.shape, ip_atten_mask_tmp.shape)
                # ip_query_tmp = rearrange(ip_query_tmp, "B L (H D) -> B H L D", H=attn.num_heads, D=attn.head_dim)

                # Reshape projections for multi-head attention
                ip_key = rearrange(ip_key, 'B L (H D) -> B H L D', H=attn.num_heads, D=attn.head_dim)
                ip_value = rearrange(ip_value, 'B L (H D) -> B H L D', H=attn.num_heads, D=attn.head_dim)

                # attention()
                # Compute attention between IP projections and the latent query
                ip_attention = F.scaled_dot_product_attention(
                    ip_query.clone(),
                    ip_key,
                    ip_value,
                    dropout_p=0.0,
                    is_causal=False
                )
                ip_attention = rearrange(ip_attention, "B H L D -> B L (H D)")

                # print("++>ip_attention", ip_attention.shape, ip_atten_mask_tmp.shape)
                # print("")

                #base_atten[:, ip_atten_mask_tmp[0]] = base_atten[:, ip_atten_mask_tmp[0]] + ip_scale * ip_attention
                mask = torch.stack([ip_atten_mask_tmp] * ip_attention.shape[-1], axis=-1)
                attn_1_img = attn_1_img + ip_scale * (ip_attention * mask)

            for i in range(len(ip_atten_mask)):
                # print("Do portrait prompt")
                ip_key = self.ip_adapter_single_stream_k_proj_vae(vae_controlnet_hidden_states[i])
                ip_value = self.ip_adapter_single_stream_v_proj_vae(vae_controlnet_hidden_states[i])

                ip_atten_mask_tmp = ip_atten_mask[i]
                # Reshape projections for multi-head attention
                ip_key = rearrange(ip_key, 'B L (H D) -> B H L D', H=attn.num_heads, D=attn.head_dim)
                ip_value = rearrange(ip_value, 'B L (H D) -> B H L D', H=attn.num_heads, D=attn.head_dim)
                # attention()
                # Compute attention between IP projections and the latent query
                ip_attention = F.scaled_dot_product_attention(
                    ip_query.clone(),
                    ip_key,
                    ip_value,
                    dropout_p=0.0,
                    is_causal=False
                )
                ip_attention = rearrange(ip_attention, "B H L D -> B L (H D)")
                mask = torch.stack([ip_atten_mask_tmp] * ip_attention.shape[-1], axis=-1)
                attn_1_img = attn_1_img + 1.0 * (ip_attention * mask)
                # attn_1_img = attn_1_img + 0.0 * (ip_attention * mask)
            
        # print("attn_out", attn_1.shape)
        # compute activation in mlp stream, cat again and run second linear layer
        attn_1 = torch.cat([attn_1_txt, attn_1_img], axis=1)
        output = attn.linear2(torch.cat((attn_1, attn.mlp_act(mlp)), 2))
        out = x + mod.gate * output

        # print("out==>", out.shape)

        return out


class IPSingleStreamBlockProcessorV8(nn.Module):
    """Attention processor for handling IP-adapter with single stream block."""
    def __init__(self, context_dim, hidden_dim, feature_in_dim=3072, lora_dim=3072, rank=32):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "IPSingleStreamBlockProcessor requires PyTorch 2.0 or higher. Please upgrade PyTorch."
            )

        # Ensure context_dim matches the dimension of image_proj
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim

        # Initialize projections for IP-adapter
        self.ip_adapter_single_stream_k_proj = nn.Linear(context_dim, hidden_dim, bias=False)
        self.ip_adapter_single_stream_v_proj = nn.Linear(context_dim, hidden_dim, bias=False)

        nn.init.zeros_(self.ip_adapter_single_stream_k_proj.weight)
        nn.init.zeros_(self.ip_adapter_single_stream_v_proj.weight)
        
        self.ip_adapter_single_stream_k_proj_vae = nn.Linear(feature_in_dim, hidden_dim, bias=False)
        self.ip_adapter_single_stream_v_proj_vae = nn.Linear(feature_in_dim, hidden_dim, bias=False)

        nn.init.zeros_(self.ip_adapter_single_stream_k_proj_vae.weight)
        nn.init.zeros_(self.ip_adapter_single_stream_v_proj_vae.weight)
        
        self.ip_adapter_qkv_lora = LoRALinearLayer(lora_dim, lora_dim * 3, rank)
        self.ip_adapter_proj_lora = LoRALinearLayer(15360, lora_dim, rank)

    def __call__(
        self,
        attn: nn.Module,
        x: Tensor,
        vec: Tensor,
        pe: Tensor,
        image_proj: Tensor | None = None,
        ip_scale: float = 1.0,
        ip_atten_mask=None,
        vae_controlnet_hidden_states=None,
        use_ipa=True,
        **kwargs,
    ) -> Tensor:
        # portrait_scale = 0.0
        if vae_controlnet_hidden_states is None and use_ipa:
            print("vae_controlnet_hidden_states is None!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

        mod, _ = attn.modulation(vec)
        x_mod = (1 + mod.scale) * attn.pre_norm(x) + mod.shift
        qkv, mlp = torch.split(attn.linear1(x_mod), [3 * attn.hidden_size, attn.mlp_hidden_dim], dim=-1)
        if not use_ipa:
            # qkv += self.ip_adapter_qkv_lora(x_mod)
            qkv = qkv + self.ip_adapter_qkv_lora(x_mod)
        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads, D=attn.head_dim)

        # print(" ===>>>> 1", q.shape)

        q, k = attn.norm(q, k, v)

        # print(" ===>>>> 2", q.shape)

        # compute attention
        attn_1 = attention(q, k, v, pe=pe)

        # IP-adapter processing 
        # attn_out = attn_1
        
        # print(" ===>>>> 3", q.shape)
        # tmp_q = rearrange(q, "B H L D -> B L (H D)")

        # ip_query = rearrange(ip_query, "B L (H D) -> B H L D", K=3, H=attn.num_heads, D=attn.head_dim)
        # ip_key = self.ip_adapter_single_stream_k_proj(image_proj)
        # ip_value = self.ip_adapter_single_stream_v_proj(image_proj)
        # ip_attention_list = []
        ip_query = q[:, :, -ip_atten_mask[0].shape[1]:]
        attn_1_img = attn_1[:, -ip_atten_mask[0].shape[1]:, ]
        attn_1_txt = attn_1[:, :-ip_atten_mask[0].shape[1], ]
        # base_atten = attn_1[:, -ip_atten_mask[0].shape[1]:, ]
        if use_ipa:
            for i in range(len(ip_atten_mask)):
                # ip_query = q[:, :, -ip_atten_mask[0].shape[1]:]
                ip_key = self.ip_adapter_single_stream_k_proj(image_proj[i])
                ip_value = self.ip_adapter_single_stream_v_proj(image_proj[i])

                ip_atten_mask_tmp = ip_atten_mask[i]
                # ip_query = img_q[ip_atten_mask_tmp].reshape()
                # ip_query_list = []
                # 实际上只能是1，不然需要pad

                # print("==>1", ip_query.shape, ip_atten_mask_tmp.shape)
                # ip_query_tmp = rearrange(ip_query, "B H L D -> B L (H D)")
                # # print("==>2",ip_query.shape, ip_atten_mask_tmp.shape)
                # ip_query_tmp = ip_query_tmp[ip_atten_mask_tmp].reshape(ip_query_tmp.shape[0], -1, ip_query_tmp.shape(-1))

                # for bi in range(ip_query.shape[0]):
                #     # print()
                #     ip_query_list.append(ip_query[ip_atten_mask_tmp])
                # ip_query = torch.stack(ip_query_list, axis=0)
                # print("==>3",ip_query.shape, ip_atten_mask_tmp.shape)
                # ip_query_tmp = rearrange(ip_query_tmp, "B L (H D) -> B H L D", H=attn.num_heads, D=attn.head_dim)

                # Reshape projections for multi-head attention
                ip_key = rearrange(ip_key, 'B L (H D) -> B H L D', H=attn.num_heads, D=attn.head_dim)
                ip_value = rearrange(ip_value, 'B L (H D) -> B H L D', H=attn.num_heads, D=attn.head_dim)

                # attention()
                # Compute attention between IP projections and the latent query
                ip_attention = F.scaled_dot_product_attention(
                    ip_query.clone(),
                    ip_key,
                    ip_value,
                    dropout_p=0.0,
                    is_causal=False
                )
                ip_attention = rearrange(ip_attention, "B H L D -> B L (H D)")

                # print("++>ip_attention", ip_attention.shape, ip_atten_mask_tmp.shape)
                # print("")

                #base_atten[:, ip_atten_mask_tmp[0]] = base_atten[:, ip_atten_mask_tmp[0]] + ip_scale * ip_attention
                mask = torch.stack([ip_atten_mask_tmp] * ip_attention.shape[-1], axis=-1)
                attn_1_img = attn_1_img + ip_scale * (ip_attention * mask)

            for i in range(len(ip_atten_mask)):
                # print("Do portrait prompt")
                ip_key = self.ip_adapter_single_stream_k_proj_vae(vae_controlnet_hidden_states[i])
                ip_value = self.ip_adapter_single_stream_v_proj_vae(vae_controlnet_hidden_states[i])

                ip_atten_mask_tmp = ip_atten_mask[i]
                # Reshape projections for multi-head attention
                ip_key = rearrange(ip_key, 'B L (H D) -> B H L D', H=attn.num_heads, D=attn.head_dim)
                ip_value = rearrange(ip_value, 'B L (H D) -> B H L D', H=attn.num_heads, D=attn.head_dim)
                # attention()
                # Compute attention between IP projections and the latent query
                ip_attention = F.scaled_dot_product_attention(
                    ip_query.clone(),
                    ip_key,
                    ip_value,
                    dropout_p=0.0,
                    is_causal=False
                )
                ip_attention = rearrange(ip_attention, "B H L D -> B L (H D)")
                mask = torch.stack([ip_atten_mask_tmp] * ip_attention.shape[-1], axis=-1)
                attn_1_img = attn_1_img + 1.0 * (ip_attention * mask)
                # attn_1_img = attn_1_img + 0.0 * (ip_attention * mask)
            
        # print("attn_out", attn_1.shape)
        # compute activation in mlp stream, cat again and run second linear layer
        attn_1 = torch.cat([attn_1_txt, attn_1_img], axis=1)
        output = attn.linear2(torch.cat((attn_1, attn.mlp_act(mlp)), 2))
        if not use_ipa:
            output = output + self.ip_adapter_proj_lora(torch.cat((attn_1, attn.mlp_act(mlp)), 2))
        out = x + mod.gate * output

        # print("out==>", out.shape)

        return out


class IPSingleStreamBlockProcessorV3WithMultiLora(nn.Module):
    """Attention processor for handling IP-adapter with single stream block."""
    def __init__(self, context_dim, hidden_dim, dim: int, ranks = [], network_alpha = None, lora_weights = [], lora_state_dicts = [], device=None):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "IPSingleStreamBlockProcessor requires PyTorch 2.0 or higher. Please upgrade PyTorch."
            )

        # Ensure context_dim matches the dimension of image_proj
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim

        # Initialize projections for IP-adapter
        self.ip_adapter_single_stream_k_proj = nn.Linear(context_dim, hidden_dim, bias=False)
        self.ip_adapter_single_stream_v_proj = nn.Linear(context_dim, hidden_dim, bias=False)

        nn.init.zeros_(self.ip_adapter_single_stream_k_proj.weight)
        nn.init.zeros_(self.ip_adapter_single_stream_v_proj.weight)
        
        self.lora_num = len(lora_weights)
        self.qkv_loras = []
        self.proj_loras = []
        self.lora_weights = lora_weights

        for i in range(self.lora_num):
            temp = SingleStreamBlockLoraProcessor(dim=3072, rank=ranks[i])
            temp.load_state_dict(lora_state_dicts[i])
            temp.to(device)

            qkv = temp.qkv_lora
            proj = temp.proj_lora

            self.qkv_loras.append(qkv)
            self.proj_loras.append(proj)

    def __call__(
        self,
        attn: nn.Module,
        x: Tensor,
        vec: Tensor,
        pe: Tensor,
        image_proj: Tensor | None = None,
        ip_scale: float = 1.0,
        ip_atten_mask=None,
        lora_weight_list=None,
        **kwargs,
    ) -> Tensor:
        if lora_weight_list is not None:
            # print("==>lora_weight_list", lora_weight_list)
            self.lora_weights = lora_weight_list

        mod, _ = attn.modulation(vec)
        x_mod = (1 + mod.scale) * attn.pre_norm(x) + mod.shift
        qkv, mlp = torch.split(attn.linear1(x_mod), [3 * attn.hidden_size, attn.mlp_hidden_dim], dim=-1)
        for i in range(len(self.lora_weights)):
            qkv = qkv + self.qkv_loras[i](x_mod) * self.lora_weights[i]

        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads, D=attn.head_dim)

        # print(" ===>>>> 1", q.shape)

        q, k = attn.norm(q, k, v)

        # print(" ===>>>> 2", q.shape)

        # compute attention
        attn_1 = attention(q, k, v, pe=pe)

        # IP-adapter processing 
        # attn_out = attn_1
        
        # print(" ===>>>> 3", q.shape)
        # tmp_q = rearrange(q, "B H L D -> B L (H D)")

        # ip_query = rearrange(ip_query, "B L (H D) -> B H L D", K=3, H=attn.num_heads, D=attn.head_dim)
        # ip_key = self.ip_adapter_single_stream_k_proj(image_proj)
        # ip_value = self.ip_adapter_single_stream_v_proj(image_proj)
        # ip_attention_list = []
        ip_query = q[:, :, -ip_atten_mask[0].shape[1]:]
        attn_1_img = attn_1[:, -ip_atten_mask[0].shape[1]:, ]
        attn_1_txt = attn_1[:, :-ip_atten_mask[0].shape[1], ]
        # base_atten = attn_1[:, -ip_atten_mask[0].shape[1]:, ]
        for i in range(len(ip_atten_mask)):
            # ip_query = q[:, :, -ip_atten_mask[0].shape[1]:]
            ip_key = self.ip_adapter_single_stream_k_proj(image_proj[i])
            ip_value = self.ip_adapter_single_stream_v_proj(image_proj[i])

            ip_atten_mask_tmp = ip_atten_mask[i]
            # ip_query = img_q[ip_atten_mask_tmp].reshape()
            # ip_query_list = []
            # 实际上只能是1，不然需要pad

            # print("==>1", ip_query.shape, ip_atten_mask_tmp.shape)
            # ip_query_tmp = rearrange(ip_query, "B H L D -> B L (H D)")
            # # print("==>2",ip_query.shape, ip_atten_mask_tmp.shape)
            # ip_query_tmp = ip_query_tmp[ip_atten_mask_tmp].reshape(ip_query_tmp.shape[0], -1, ip_query_tmp.shape(-1))

            # for bi in range(ip_query.shape[0]):
            #     # print()
            #     ip_query_list.append(ip_query[ip_atten_mask_tmp])
            # ip_query = torch.stack(ip_query_list, axis=0)
            # print("==>3",ip_query.shape, ip_atten_mask_tmp.shape)
            # ip_query_tmp = rearrange(ip_query_tmp, "B L (H D) -> B H L D", H=attn.num_heads, D=attn.head_dim)

            # Reshape projections for multi-head attention
            ip_key = rearrange(ip_key, 'B L (H D) -> B H L D', H=attn.num_heads, D=attn.head_dim)
            ip_value = rearrange(ip_value, 'B L (H D) -> B H L D', H=attn.num_heads, D=attn.head_dim)

            # attention()
            # Compute attention between IP projections and the latent query
            ip_attention = F.scaled_dot_product_attention(
                ip_query.clone(),
                ip_key,
                ip_value,
                dropout_p=0.0,
                is_causal=False
            )
            ip_attention = rearrange(ip_attention, "B H L D -> B L (H D)")
            
            # print("++>ip_attention", ip_attention.shape, ip_atten_mask_tmp.shape)
            # print("")

            #base_atten[:, ip_atten_mask_tmp[0]] = base_atten[:, ip_atten_mask_tmp[0]] + ip_scale * ip_attention
            mask = torch.stack([ip_atten_mask_tmp] * ip_attention.shape[-1], axis=-1)
            attn_1_img = attn_1_img + ip_scale * (ip_attention * mask)
            
        # print("attn_out", attn_1.shape)
        # compute activation in mlp stream, cat again and run second linear layer
        attn_1 = torch.cat([attn_1_txt, attn_1_img], axis=1)
        output = attn.linear2(torch.cat((attn_1, attn.mlp_act(mlp)), 2))
        for i in range(len(self.lora_weights)):
            output = output + self.proj_loras[i](torch.cat((attn_1, attn.mlp_act(mlp)), 2)) * self.lora_weights[i]
        out = x + mod.gate * output

        # print("out==>", out.shape)

        return out


class IPSingleStreamBlockProcessor(nn.Module):
    """Attention processor for handling IP-adapter with single stream block."""
    def __init__(self, context_dim, hidden_dim):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "IPSingleStreamBlockProcessor requires PyTorch 2.0 or higher. Please upgrade PyTorch."
            )

        # Ensure context_dim matches the dimension of image_proj
        self.context_dim = context_dim
        self.hidden_dim = hidden_dim

        # Initialize projections for IP-adapter
        self.ip_adapter_single_stream_k_proj = nn.Linear(context_dim, hidden_dim, bias=False)
        self.ip_adapter_single_stream_v_proj = nn.Linear(context_dim, hidden_dim, bias=False)

        nn.init.zeros_(self.ip_adapter_single_stream_k_proj.weight)
        nn.init.zeros_(self.ip_adapter_single_stream_v_proj.weight)

    def __call__(
        self,
        attn: nn.Module,
        x: Tensor,
        vec: Tensor,
        pe: Tensor,
        image_proj: Tensor | None = None,
        ip_scale: float = 1.0,
        ip_atten_mask: Tensor | None = None,
        **kwargs,
    ) -> Tensor:

        if isinstance(image_proj, list):
            image_proj = image_proj[0]
            
        mod, _ = attn.modulation(vec)
        x_mod = (1 + mod.scale) * attn.pre_norm(x) + mod.shift
        qkv, mlp = torch.split(attn.linear1(x_mod), [3 * attn.hidden_size, attn.mlp_hidden_dim], dim=-1)

        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads, D=attn.head_dim)
        q, k = attn.norm(q, k, v)

        # compute attention
        attn_1 = attention(q, k, v, pe=pe)

        # IP-adapter processing
        ip_query = q
        ip_key = self.ip_adapter_single_stream_k_proj(image_proj)
        ip_value = self.ip_adapter_single_stream_v_proj(image_proj)

        # Reshape projections for multi-head attention
        ip_key = rearrange(ip_key, 'B L (H D) -> B H L D', H=attn.num_heads, D=attn.head_dim)
        ip_value = rearrange(ip_value, 'B L (H D) -> B H L D', H=attn.num_heads, D=attn.head_dim)

        # attention()
        # Compute attention between IP projections and the latent query
        ip_attention = F.scaled_dot_product_attention(
            ip_query,
            ip_key,
            ip_value
        )
        ip_attention = rearrange(ip_attention, "B H L D -> B L (H D)")

        attn_out = attn_1 + ip_scale * ip_attention

        # compute activation in mlp stream, cat again and run second linear layer
        output = attn.linear2(torch.cat((attn_out, attn.mlp_act(mlp)), 2))
        out = x + mod.gate * output

        return out
    

class SingleStreamBlockMultiLoraProcessor(nn.Module):
    def __init__(self, dim: int, ranks = [], network_alpha = None, lora_weights = [], lora_state_dicts = [], device=None):
        super().__init__()
        self.lora_num = len(lora_weights)
        self.qkv_loras = []
        self.proj_loras = []
        self.lora_weights = lora_weights

        for i in range(self.lora_num):
            temp = SingleStreamBlockLoraProcessor(dim=3072, rank=ranks[i])
            temp.load_state_dict(lora_state_dicts[i])
            temp.to(device)

            qkv = temp.qkv_lora
            proj = temp.proj_lora

            self.qkv_loras.append(qkv)
            self.proj_loras.append(proj)

    def forward(self, attn: nn.Module, x: Tensor, vec: Tensor, pe: Tensor) -> Tensor:

        mod, _ = attn.modulation(vec)
        x_mod = (1 + mod.scale) * attn.pre_norm(x) + mod.shift
        qkv, mlp = torch.split(attn.linear1(x_mod), [3 * attn.hidden_size, attn.mlp_hidden_dim], dim=-1)
        for i in range(len(self.lora_weights)):
            qkv = qkv + self.qkv_loras[i](x_mod) * self.lora_weights[i]

        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads)
        q, k = attn.norm(q, k, v)

        # compute attention
        attn_1 = attention(q, k, v, pe=pe)

        # compute activation in mlp stream, cat again and run second linear layer
        output = attn.linear2(torch.cat((attn_1, attn.mlp_act(mlp)), 2))
        for i in range(len(self.lora_weights)):
            output = output + self.proj_loras[i](torch.cat((attn_1, attn.mlp_act(mlp)), 2)) * self.lora_weights[i]
        output = x + mod.gate * output
        return output


class SingleStreamBlockLoraProcessorV2(nn.Module):
    def __init__(self, dim: int, rank: int = 4, network_alpha = None, lora_weight: float = 1):
        super().__init__()
        self.qkv_lora = LoRALinearLayer(dim, dim * 3, rank, network_alpha)
        self.proj_lora = LoRALinearLayer(15360, dim, rank, network_alpha)
        self.lora_weight = lora_weight

    def forward(self, attn: nn.Module, x: Tensor, vec: Tensor, pe: Tensor, lora_weight: float = 1) -> Tensor:
        self.lora_weight = lora_weight
        mod, _ = attn.modulation(vec)
        x_mod = (1 + mod.scale) * attn.pre_norm(x) + mod.shift
        qkv, mlp = torch.split(attn.linear1(x_mod), [3 * attn.hidden_size, attn.mlp_hidden_dim], dim=-1)
        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads)
        self.KQQ = F.scaled_dot_product_attention(k[:,:,:512,:], q[:,:,512:,:], q[:,:,512:,:], dropout_p=0.0, is_causal=False)
        self.Q = q[:,:,512:,:]
        qkv = qkv + self.qkv_lora(x_mod) * self.lora_weight

        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads)
        q, k = attn.norm(q, k, v)

        # compute attention
        attn_1 = attention(q, k, v, pe=pe)

        # compute activation in mlp stream, cat again and run second linear layer
        output = attn.linear2(torch.cat((attn_1, attn.mlp_act(mlp)), 2))
        output = output + self.proj_lora(torch.cat((attn_1, attn.mlp_act(mlp)), 2)) * self.lora_weight
        output = x + mod.gate * output
        return output, self.KQQ, self.Q

class SingleStreamBlockLoraProcessor(nn.Module):
    def __init__(self, dim: int, rank: int = 4, network_alpha = None, lora_weight: float = 1):
        super().__init__()
        self.qkv_lora = LoRALinearLayer(dim, dim * 3, rank, network_alpha)
        self.proj_lora = LoRALinearLayer(15360, dim, rank, network_alpha)
        self.lora_weight = lora_weight

    def forward(self, attn: nn.Module, x: Tensor, vec: Tensor, pe: Tensor) -> Tensor:

        mod, _ = attn.modulation(vec)
        x_mod = (1 + mod.scale) * attn.pre_norm(x) + mod.shift
        qkv, mlp = torch.split(attn.linear1(x_mod), [3 * attn.hidden_size, attn.mlp_hidden_dim], dim=-1)
        qkv = qkv + self.qkv_lora(x_mod) * self.lora_weight

        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads)
        q, k = attn.norm(q, k, v)

        # compute attention
        attn_1 = attention(q, k, v, pe=pe)

        # compute activation in mlp stream, cat again and run second linear layer
        output = attn.linear2(torch.cat((attn_1, attn.mlp_act(mlp)), 2))
        output = output + self.proj_lora(torch.cat((attn_1, attn.mlp_act(mlp)), 2)) * self.lora_weight
        output = x + mod.gate * output
        return output


class SingleStreamBlockProcessor:
    def __call__(self, attn: nn.Module, x: Tensor, vec: Tensor, pe: Tensor, **kwargs) -> Tensor:

        mod, _ = attn.modulation(vec)
        x_mod = (1 + mod.scale) * attn.pre_norm(x) + mod.shift
        qkv, mlp = torch.split(attn.linear1(x_mod), [3 * attn.hidden_size, attn.mlp_hidden_dim], dim=-1)

        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads)
        q, k = attn.norm(q, k, v)

        # compute attention
        attn_1 = attention(q, k, v, pe=pe)

        # compute activation in mlp stream, cat again and run second linear layer
        output = attn.linear2(torch.cat((attn_1, attn.mlp_act(mlp)), 2))
        output = x + mod.gate * output
        return output

class SingleStreamBlock(nn.Module):
    """
    A DiT block with parallel linear layers as described in
    https://arxiv.org/abs/2302.05442 and adapted modulation interface.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qk_scale: float | None = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = qk_scale or self.head_dim**-0.5

        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
        # qkv and mlp_in
        self.linear1 = nn.Linear(hidden_size, hidden_size * 3 + self.mlp_hidden_dim)
        # proj and mlp_out
        self.linear2 = nn.Linear(hidden_size + self.mlp_hidden_dim, hidden_size)

        self.norm = QKNorm(self.head_dim)

        self.hidden_size = hidden_size
        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.mlp_act = nn.GELU(approximate="tanh")
        self.modulation = Modulation(hidden_size, double=False)

        processor = SingleStreamBlockProcessor()
        self.set_processor(processor)


    def set_processor(self, processor) -> None:
        self.processor = processor

    def get_processor(self):
        return self.processor

    def forward(
        self,
        x: Tensor,
        vec: Tensor,
        pe: Tensor,
        image_proj: Tensor | None = None,
        ip_scale: float = 1.0,
        ip_atten_mask: Tensor | None = None,
        txt: Tensor | None = None,
        txt_cloth: Tensor | None = None,
        vec_cloth: Tensor | None = None,
        pe_cloth: Tensor | None = None,
        ip_scale_cloth: float = 1.0,
        lora_weight: float = 1.0,
        lora_weight_list = None,
        **kwargs
    ) -> Tensor:
        if image_proj is None:
            return self.processor(self, x, vec, pe, lora_weight=lora_weight)
        else:
            return self.processor(self, x, vec, pe, image_proj, ip_scale, ip_atten_mask, txt=txt, txt_cloth=txt_cloth, vec_cloth=vec_cloth, pe_cloth=pe_cloth, ip_scale_cloth=ip_scale_cloth, lora_weight_list=lora_weight_list, **kwargs)



class LastLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, x: Tensor, vec: Tensor) -> Tensor:
        shift, scale = self.adaLN_modulation(vec).chunk(2, dim=1)
        x = (1 + scale[:, None, :]) * self.norm_final(x) + shift[:, None, :]
        x = self.linear(x)
        return x

class ImageProjModel(torch.nn.Module):
    """Projection Model
    https://github.com/tencent-ailab/IP-Adapter/blob/main/ip_adapter/ip_adapter.py#L28
    """

    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024, clip_extra_context_tokens=4):
        super().__init__()

        self.generator = None
        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.proj = torch.nn.Linear(clip_embeddings_dim, self.clip_extra_context_tokens * cross_attention_dim)
        self.norm = torch.nn.LayerNorm(cross_attention_dim)

    def forward(self, image_embeds):
        embeds = image_embeds
        clip_extra_context_tokens = self.proj(embeds).reshape(
            -1, self.clip_extra_context_tokens, self.cross_attention_dim
        )
        clip_extra_context_tokens = self.norm(clip_extra_context_tokens)
        return clip_extra_context_tokens

class ImageProjModelLocal(torch.nn.Module):
    """Projection Model
    https://github.com/tencent-ailab/IP-Adapter/blob/main/ip_adapter/ip_adapter.py#L28
    """

    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024, clip_pooing_embeddings_dim=768, faces_embedding_dim=512):
        super().__init__()

        self.generator = None
        self.cross_attention_dim = cross_attention_dim
        self.clip_embeddings_dim = clip_embeddings_dim
        self.faces_embedding_dim = faces_embedding_dim

        self.proj_0 = torch.nn.Linear(clip_embeddings_dim, 2 * cross_attention_dim)
        self.norm_0 = torch.nn.LayerNorm(cross_attention_dim * 2)

        self.proj_1 = torch.nn.Linear(2 * cross_attention_dim, cross_attention_dim)
        self.norm_1 = torch.nn.LayerNorm(cross_attention_dim)

        self.proj = torch.nn.Linear(clip_pooing_embeddings_dim, clip_embeddings_dim)
        self.norm = torch.nn.LayerNorm(clip_embeddings_dim)

    def forward(self, image_embeds, image_embeds_pooling=None, feature_type=None):

        embeds = image_embeds
        # print(image_embeds.shape, image_embeds_pooling.shape)
        if image_embeds_pooling is not None:
            image_embeds_pooling = self.proj(image_embeds_pooling).reshape(1, -1, self.clip_embeddings_dim)
            image_embeds_pooling = self.norm(image_embeds_pooling)

            print(embeds.shape, image_embeds_pooling.shape)
            embeds = torch.cat([embeds, image_embeds_pooling], 1)
        
        embeds = self.proj_0(embeds)
        embeds = self.norm_0(embeds)

        embeds = self.proj_1(embeds)
        embeds = self.norm_1(embeds)

        return embeds

class ImageProjModelLocalB3_face(torch.nn.Module):
    """Projection Model
    https://github.com/tencent-ailab/IP-Adapter/blob/main/ip_adapter/ip_adapter.py#L28
    """

    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024, clip_pooing_embeddings_dim=768, faces_embedding_dim=512, prompt_len=32, face_token_len=64):
        super().__init__()

        self.generator = None
        self.cross_attention_dim = cross_attention_dim
        self.clip_embeddings_dim = clip_embeddings_dim
        self.faces_embedding_dim = faces_embedding_dim

        self.proj_0 = torch.nn.Linear(clip_embeddings_dim, cross_attention_dim)
        self.norm_0 = torch.nn.LayerNorm(cross_attention_dim)
        self.silu_0 = torch.nn.SiLU()

        self.proj_cloth = torch.nn.Linear(clip_embeddings_dim, cross_attention_dim)
        self.norm_cloth = torch.nn.LayerNorm(cross_attention_dim)
        self.silu_cloth = torch.nn.SiLU()

        self.proj = torch.nn.Linear(clip_pooing_embeddings_dim, clip_embeddings_dim)
        self.norm = torch.nn.LayerNorm(clip_embeddings_dim)
        self.silu = torch.nn.SiLU()

        self.proj_face = torch.nn.Linear(faces_embedding_dim, cross_attention_dim * face_token_len)
        self.norm_face = torch.nn.LayerNorm(cross_attention_dim * face_token_len)
        self.silu_face = torch.nn.SiLU()

        self.prompt_len = prompt_len
        self.face_token_len = face_token_len

    def forward(self, image_embeds, image_embeds_pooling=None, face_embeds_pooling=None, feature_type=None):
        
        if feature_type[0] == 0:
            embeds = image_embeds
            # print(image_embeds.shape, image_embeds_pooling.shape)
            if image_embeds_pooling is not None:
                image_embeds_pooling = self.proj(image_embeds_pooling).reshape(1, -1, self.clip_embeddings_dim)
                image_embeds_pooling = self.norm(image_embeds_pooling)
                image_embeds_pooling = self.silu(image_embeds_pooling)
                
                # print(embeds.shape, image_embeds_pooling.shape)
                #
                embeds = torch.cat([image_embeds_pooling, embeds], 1)
                
            # 脸
            # print("XXX===>", feature_type[0])
            # embeds = self.proj_0(embeds)
            # embeds = self.norm_0(embeds)
            # embeds = self.silu_0(embeds)
            # 衣服 = 0
            print("XXX===>", feature_type[0])
            embeds = self.proj_cloth(embeds)
            embeds = self.norm_cloth(embeds)
            embeds = self.silu_face(embeds)

            if feature_type is not None:
                embeds = torch.cat([1 + feature_type[0] + torch.ones(embeds.shape[0], embeds.shape[1], self.prompt_len).to(embeds.device).to(embeds.dtype), embeds], -1)
        else:
            print("XXX===>", feature_type[1])
            face_embeds_pooling = self.proj_face(face_embeds_pooling)
            face_embeds_pooling = self.norm_face(face_embeds_pooling).reshape(-1, self.face_token_len, self.cross_attention_dim)
            face_embeds_pooling = self.silu_face(face_embeds_pooling)
            # print(face_embeds_pooling.shape, embeds.shape)

            if feature_type is not None:
                face_embeds_pooling = torch.cat([torch.zeros(face_embeds_pooling.shape[0], face_embeds_pooling.shape[1], self.prompt_len).to(face_embeds_pooling.device).to(face_embeds_pooling.dtype), face_embeds_pooling], -1)

            # print(face_embeds_pooling.shape, embeds.shape)
            embeds = face_embeds_pooling #torch.cat([face_embeds_pooling], 1)

        # print(embeds[:, :, :128])
        return embeds


class ImageProjModelLocalB2_face(torch.nn.Module):
    """Projection Model
    https://github.com/tencent-ailab/IP-Adapter/blob/main/ip_adapter/ip_adapter.py#L28
    """

    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024, clip_pooing_embeddings_dim=768, faces_embedding_dim=512, prompt_len=32, face_token_len=64):
        super().__init__()

        self.generator = None
        self.cross_attention_dim = cross_attention_dim
        self.clip_embeddings_dim = clip_embeddings_dim
        self.faces_embedding_dim = faces_embedding_dim

        self.proj_local_clip_face = torch.nn.Linear(clip_embeddings_dim, cross_attention_dim)
        self.norm_local_clip_face = torch.nn.LayerNorm(cross_attention_dim)
        self.silu_local_clip_face = torch.nn.SiLU()

        self.proj_local_clip_cloth = torch.nn.Linear(clip_embeddings_dim, cross_attention_dim)
        self.norm_local_clip_cloth = torch.nn.LayerNorm(cross_attention_dim)
        self.silu_local_clip_cloth = torch.nn.SiLU()

        self.proj_pool_clip_face = torch.nn.Linear(clip_pooing_embeddings_dim, clip_embeddings_dim)
        self.norm_pool_clip_face  = torch.nn.LayerNorm(clip_embeddings_dim)
        self.silu_pool_clip_face  = torch.nn.SiLU()

        self.proj_pool_clip_cloth = torch.nn.Linear(clip_pooing_embeddings_dim, clip_embeddings_dim)
        self.norm_pool_clip_cloth  = torch.nn.LayerNorm(clip_embeddings_dim)
        self.silu_pool_clip_cloth  = torch.nn.SiLU()

        # self.proj = torch.nn.Linear(clip_pooing_embeddings_dim, clip_embeddings_dim)
        # self.norm = torch.nn.LayerNorm(clip_embeddings_dim)
        # self.silu = torch.nn.SiLU()

        self.proj_face = torch.nn.Linear(faces_embedding_dim, cross_attention_dim * face_token_len)
        self.norm_face = torch.nn.LayerNorm(cross_attention_dim * face_token_len)
        self.silu_face = torch.nn.SiLU()

        self.prompt_len = prompt_len
        self.face_token_len = face_token_len

    def forward(self, image_embeds, image_embeds_pooling=None, face_embeds_pooling=None, feature_type=None):

        embeds = image_embeds
        # print(image_embeds.shape, image_embeds_pooling.shape)
        if image_embeds_pooling is not None:
            if feature_type[0] == 1:
                image_embeds_pooling = self.proj_pool_clip_face(image_embeds_pooling).reshape(1, -1, self.clip_embeddings_dim)
                image_embeds_pooling = self.norm_pool_clip_face(image_embeds_pooling)
                image_embeds_pooling = self.silu_pool_clip_face(image_embeds_pooling)
            else:
                image_embeds_pooling = self.proj_pool_clip_cloth(image_embeds_pooling).reshape(1, -1, self.clip_embeddings_dim)
                image_embeds_pooling = self.norm_pool_clip_cloth(image_embeds_pooling)
                image_embeds_pooling = self.silu_pool_clip_cloth(image_embeds_pooling)
                
            # print(embeds.shape, image_embeds_pooling.shape)
            #
            embeds = torch.cat([image_embeds_pooling, embeds], 1)
        
        if feature_type[0] == 1:
            # 脸
            print("XXX===>", feature_type[0])
            embeds = self.proj_local_clip_face(embeds)
            embeds = self.norm_local_clip_face(embeds)
            embeds = self.silu_local_clip_face(embeds)
        else:
            # 衣服 = 0
            print("XXX===>", feature_type[0])
            embeds = self.proj_local_clip_cloth(embeds)
            embeds = self.norm_local_clip_cloth(embeds)
            embeds = self.silu_local_clip_cloth(embeds)

        if feature_type is not None:
            embeds = torch.cat([1 + feature_type[0] + torch.ones(embeds.shape[0], embeds.shape[1], self.prompt_len).to(embeds.device).to(embeds.dtype), embeds], -1)

        if face_embeds_pooling is not None:
            face_embeds_pooling = self.proj_face(face_embeds_pooling)
            face_embeds_pooling = self.norm_face(face_embeds_pooling).reshape(-1, self.face_token_len, self.cross_attention_dim)
            face_embeds_pooling = self.silu_face(face_embeds_pooling)
            # print(face_embeds_pooling.shape, embeds.shape)

            if feature_type is not None:
                face_embeds_pooling = torch.cat([torch.zeros(face_embeds_pooling.shape[0], face_embeds_pooling.shape[1], self.prompt_len).to(face_embeds_pooling.device).to(face_embeds_pooling.dtype), face_embeds_pooling], -1)

            # print(face_embeds_pooling.shape, embeds.shape)
            embeds = torch.cat([face_embeds_pooling, embeds], 1)

        # print(embeds[:, :, :128])
        return embeds

class ImageProjModelLocalB0_face(torch.nn.Module):
    """Projection Model
    https://github.com/tencent-ailab/IP-Adapter/blob/main/ip_adapter/ip_adapter.py#L28
    """

    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024, clip_pooing_embeddings_dim=768, faces_embedding_dim=512, prompt_len=32):
        super().__init__()

        self.generator = None
        self.cross_attention_dim = cross_attention_dim
        self.clip_embeddings_dim = clip_embeddings_dim
        self.faces_embedding_dim = faces_embedding_dim

        self.proj_0 = torch.nn.Linear(clip_embeddings_dim, cross_attention_dim)
        self.norm_0 = torch.nn.LayerNorm(cross_attention_dim)

        self.proj = torch.nn.Linear(clip_pooing_embeddings_dim, clip_embeddings_dim)
        self.norm = torch.nn.LayerNorm(clip_embeddings_dim)

        self.proj_face = torch.nn.Linear(faces_embedding_dim, cross_attention_dim * 4)
        self.norm_face = torch.nn.LayerNorm(cross_attention_dim * 4)
        self.prompt_len = prompt_len
        

    def forward(self, image_embeds, image_embeds_pooling=None, face_embeds_pooling=None, feature_type=None):

        embeds = image_embeds
        # print(image_embeds.shape, image_embeds_pooling.shape)
        if image_embeds_pooling is not None:
            image_embeds_pooling = self.proj(image_embeds_pooling).reshape(1, -1, self.clip_embeddings_dim)
            image_embeds_pooling = self.norm(image_embeds_pooling)
            
            # print(embeds.shape, image_embeds_pooling.shape)
            #
            embeds = torch.cat([image_embeds_pooling, embeds], 1)
        
        embeds = self.proj_0(embeds)
        embeds = self.norm_0(embeds)

        if feature_type is not None:
            embeds = torch.cat([feature_type[0] + torch.ones(embeds.shape[0], embeds.shape[1], self.prompt_len).to(embeds.device).to(embeds.dtype), embeds], -1)

        if face_embeds_pooling is not None:
            face_embeds_pooling = self.proj_face(face_embeds_pooling)
            face_embeds_pooling = self.norm_face(face_embeds_pooling).reshape(-1, 4, self.cross_attention_dim)
            # print(face_embeds_pooling.shape, embeds.shape)

            if feature_type is not None:
                face_embeds_pooling = torch.cat([torch.zeros(face_embeds_pooling.shape[0], face_embeds_pooling.shape[1], self.prompt_len).to(face_embeds_pooling.device).to(face_embeds_pooling.dtype), face_embeds_pooling], -1)

            embeds = torch.cat([face_embeds_pooling, embeds], 1)

        # print(embeds[:, :, :128])
        return embeds

class ImageProjModelLocalB0(torch.nn.Module):
    """Projection Model
    https://github.com/tencent-ailab/IP-Adapter/blob/main/ip_adapter/ip_adapter.py#L28
    """

    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024, clip_pooing_embeddings_dim=768, faces_embedding_dim=512, prompt_len=32):
        super().__init__()

        self.generator = None
        self.cross_attention_dim = cross_attention_dim
        self.clip_embeddings_dim = clip_embeddings_dim
        self.faces_embedding_dim = faces_embedding_dim

        self.proj_0 = torch.nn.Linear(clip_embeddings_dim, cross_attention_dim)
        self.norm_0 = torch.nn.LayerNorm(cross_attention_dim)

        self.proj = torch.nn.Linear(clip_pooing_embeddings_dim, clip_embeddings_dim)
        self.norm = torch.nn.LayerNorm(clip_embeddings_dim)

        self.proj_face = torch.nn.Linear(faces_embedding_dim, cross_attention_dim * 4)
        self.norm_face = torch.nn.LayerNorm(cross_attention_dim * 4)
        self.prompt_len = prompt_len
        

    def forward(self, image_embeds, image_embeds_pooling=None, face_embeds_pooling=None, feature_type=None):

        embeds = image_embeds
        # print(image_embeds.shape, image_embeds_pooling.shape)
        if image_embeds_pooling is not None:
            image_embeds_pooling = self.proj(image_embeds_pooling).reshape(1, -1, self.clip_embeddings_dim)
            image_embeds_pooling = self.norm(image_embeds_pooling)
            
            # print(embeds.shape, image_embeds_pooling.shape)
            #
            embeds = torch.cat([embeds, image_embeds_pooling], 1)
        
        embeds = self.proj_0(embeds)
        embeds = self.norm_0(embeds)

        if feature_type is not None:
            embeds = torch.cat([feature_type[0] + torch.ones(embeds.shape[0], embeds.shape[1], self.prompt_len).to(embeds.device).to(embeds.dtype), embeds], -1)

        if face_embeds_pooling is not None:
            face_embeds_pooling = self.proj_face(face_embeds_pooling)
            face_embeds_pooling = self.norm_face(face_embeds_pooling).reshape(-1, 4, self.cross_attention_dim)
            # print(face_embeds_pooling.shape, embeds.shape)

            if feature_type is not None:
                face_embeds_pooling = torch.cat([torch.zeros(face_embeds_pooling.shape[0], face_embeds_pooling.shape[1], self.prompt_len).to(face_embeds_pooling.device).to(face_embeds_pooling.dtype), face_embeds_pooling], -1)

            embeds = torch.cat([face_embeds_pooling, embeds], 1)

        # print(embeds[:, :, :128])
        return embeds

class ImageProjModelLocalB1(torch.nn.Module):
    """Projection Model
    https://github.com/tencent-ailab/IP-Adapter/blob/main/ip_adapter/ip_adapter.py#L28
    """

    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024, clip_pooing_embeddings_dim=768, faces_embedding_dim=512):
        super().__init__()

        self.generator = None
        self.cross_attention_dim = cross_attention_dim
        self.clip_embeddings_dim = clip_embeddings_dim
        self.faces_embedding_dim = faces_embedding_dim

        self.proj_0 = torch.nn.Linear(clip_embeddings_dim, 2 * cross_attention_dim)
        self.norm_0 = torch.nn.LayerNorm(cross_attention_dim * 2)

        self.proj_1 = torch.nn.Linear(2 * cross_attention_dim, 2*cross_attention_dim)
        self.norm_1 = torch.nn.LayerNorm(2*cross_attention_dim)

        self.proj_2 = torch.nn.Linear(2 * cross_attention_dim, 2*cross_attention_dim)
        self.norm_2 = torch.nn.LayerNorm(2*cross_attention_dim)
        
        self.proj_3 = torch.nn.Linear(2 * cross_attention_dim, cross_attention_dim)
        self.norm_3 = torch.nn.LayerNorm(cross_attention_dim)

        self.proj = torch.nn.Linear(clip_pooing_embeddings_dim, clip_embeddings_dim)
        self.norm = torch.nn.LayerNorm(clip_embeddings_dim)

    def forward(self, image_embeds, image_embeds_pooling=None):

        embeds = image_embeds
        # print(image_embeds.shape, image_embeds_pooling.shape)
        if image_embeds_pooling is not None:
            image_embeds_pooling = self.proj(image_embeds_pooling).reshape(image_embeds_pooling.shape[0], -1, self.clip_embeddings_dim)
            image_embeds_pooling = self.norm(image_embeds_pooling)
        
            embeds = torch.cat([embeds, image_embeds_pooling], 1)
        
        embeds = self.proj_0(embeds)
        embeds = self.norm_0(embeds)

        embeds = self.proj_1(embeds)
        embeds = self.norm_1(embeds)

        embeds = self.proj_2(embeds)
        embeds = self.norm_2(embeds)

        embeds = self.proj_3(embeds)
        embeds = self.norm_3(embeds)

        return embeds

class ImageProjModelFaceLocal(torch.nn.Module):
    """Projection Model
    https://github.com/tencent-ailab/IP-Adapter/blob/main/ip_adapter/ip_adapter.py#L28
    """

    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024, clip_pooing_embeddings_dim=768, faces_embedding_dim=512):
        super().__init__()

        self.generator = None
        self.cross_attention_dim = cross_attention_dim
        self.clip_embeddings_dim = clip_embeddings_dim
        self.faces_embedding_dim = faces_embedding_dim

        self.proj_0 = torch.nn.Linear(clip_embeddings_dim + faces_embedding_dim, 2 * cross_attention_dim)
        self.norm_0 = torch.nn.LayerNorm(cross_attention_dim * 2)

        self.proj_1 = torch.nn.Linear(2 * cross_attention_dim, cross_attention_dim)
        self.norm_1 = torch.nn.LayerNorm(cross_attention_dim)

        self.proj = torch.nn.Linear(clip_pooing_embeddings_dim, clip_embeddings_dim)
        self.norm = torch.nn.LayerNorm(clip_embeddings_dim)

        # self.proj_face = torch.nn.Linear(faces_embedding_dim, clip_embeddings_dim)
        # self.norm_face = torch.nn.LayerNorm(clip_embeddings_dim)

    def forward(self, image_embeds, image_embeds_pooling=None, faces_embedding_pooling=None):

        embeds = image_embeds
        

        image_embeds_pooling = self.proj(image_embeds_pooling).reshape(image_embeds_pooling.shape[0], -1, self.clip_embeddings_dim)
        image_embeds_pooling = self.norm(image_embeds_pooling)

        faces_embedding_pooling = faces_embedding_pooling.reshape(faces_embedding_pooling.shape[0], -1, self.faces_embedding_dim)
        
        # print(faces_embedding_pooling.shape)
        
        embeds = torch.cat([embeds, image_embeds_pooling], 1)

        embeds = torch.cat([embeds] +  [torch.cat([faces_embedding_pooling] * embeds.shape[1], 1)], -1)
        
        
        embeds = self.proj_0(embeds)
        embeds = self.norm_0(embeds)

        embeds = self.proj_1(embeds)
        embeds = self.norm_1(embeds)

        return embeds


class InputProjModel(torch.nn.Module):
    """Projection Model
    https://github.com/tencent-ailab/IP-Adapter/blob/main/ip_adapter/ip_adapter.py#L28
    """

    def __init__(self, in_channels=9, out_channels=3):
        super().__init__()

         
        self.proj_0 = torch.nn.Linear(in_channels, out_channels)
        self.norm_0 = torch.nn.LayerNorm(out_channels)

    def forward(self, embeds):

        embeds = self.proj_0(embeds)
        embeds = self.norm_0(embeds)

        return embeds

class SkeletonMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim, bias=True)
        self.silu = nn.SiLU()
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=True)

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(self.out_layer(self.silu(self.in_layer(x))))


class MLPMixPortraitModel(nn.Module):
    def __init__(self, vae_embeddings_dim: int, pose_point_dim: int, pose_token: int, cross_attention_dim: int):
        super().__init__()
        
        self.vae_embeddings_dim = vae_embeddings_dim
        self.pose_point_dim = pose_point_dim
        self.pose_token = pose_token
        self.cross_attention_dim = cross_attention_dim
        
        self.proj_1 = nn.Sequential(
            nn.Linear(vae_embeddings_dim, cross_attention_dim, bias=True),
            nn.SiLU(),
            nn.Linear(cross_attention_dim, cross_attention_dim, bias=True),
            nn.LayerNorm(cross_attention_dim)
        )
        self.proj_2 = nn.Sequential(
            nn.Linear(pose_point_dim, pose_token * cross_attention_dim, bias=True),
            nn.SiLU(),
            nn.Linear(pose_token * cross_attention_dim, pose_token * cross_attention_dim, bias=True),
            nn.LayerNorm(pose_token * cross_attention_dim)
        )

    def forward(self, image_embeds: Tensor, pose_embeds: Tensor) -> Tensor:
        image_embeds = self.proj_1(image_embeds)
        pose_embeds = self.proj_2(pose_embeds).view(pose_embeds.shape[0], self.pose_token, self.cross_attention_dim)
        embeds = torch.cat([pose_embeds, image_embeds], 1)
        # print("pose_embeds",pose_embeds.shape)
        # print("image_embeds",image_embeds.shape)
        # print("embeds",embeds.shape)
        return embeds
    
class LandmarkEncoder(ModelMixin):
    def __init__(
        self,
        conditioning_embedding_channels: int,
        conditioning_channels: int = 3,
        block_out_channels: Tuple[int] = (16, 32, 64, 128),
    ):
        super().__init__()
        self.conv_in = nn.Conv2d(
            conditioning_channels, block_out_channels[0], kernel_size=3, padding=1
        )

        self.blocks = nn.ModuleList([])

        for i in range(len(block_out_channels) - 1):
            channel_in = block_out_channels[i]
            channel_out = block_out_channels[i + 1]
            self.blocks.append(
                nn.Conv2d(channel_in, channel_in, kernel_size=3, padding=1)
            )
            self.blocks.append(
                nn.Conv2d(
                    channel_in, channel_out, kernel_size=3, padding=1, stride=2
                )
            )
        # self.conv_out = zero_module(
        #     nn.Conv2d(
        #         block_out_channels[-1],
        #         conditioning_embedding_channels,
        #         kernel_size=3,
        #         padding=1,
        #     )
        # )
        self.out_layer = nn.Linear(block_out_channels[-1], conditioning_embedding_channels, bias=True)


    def forward(self, conditioning):
        embedding = self.conv_in(conditioning)
        embedding = F.silu(embedding)

        for block in self.blocks:
            embedding = block(embedding)
            embedding = F.silu(embedding)
        embedding = self.out_layer(embedding.permute(0, 2, 3, 1))
        embedding = embedding.permute(0, 3, 1, 2)

        return embedding

class VAEControlNet(nn.Module):
    def __init__(self, controlnet=None, need_proj=True):
        super().__init__()
        
        self.controlnet = controlnet
        self.need_proj = need_proj
        if need_proj:
            self.proj_1 = nn.Linear(1024, 4096, bias=True)
        # self.proj_1 = zero_module(nn.Linear(1024, 4096, bias=True))

    def forward(self, inp, timesteps, guidance, do_vae):
        if self.need_proj:
            txt = self.proj_1(inp["txt"])
        else:
            txt = inp["txt"]
        block_controlnet_hidden_states = self.controlnet(
                        img=inp["img"],
                        img_ids=inp["img_ids"],
                        controlnet_cond=None,
                        txt=txt,
                        txt_ids=inp["txt_ids"],
                        y=inp["vec"],
                        timesteps=timesteps,
                        guidance=guidance,
                        do_vae=do_vae
                    )
        # print("block_controlnet_hidden_states", block_controlnet_hidden_states[-1])
        return block_controlnet_hidden_states[-1:]


class LivePortraitEncoder(nn.Module):
    def __init__(self, ):
        super().__init__()

        self.proj_t = nn.Linear(3, 66, bias=True)
        self.proj_exp = nn.Linear(63, 66, bias=True)
        self.proj_scale = nn.Linear(1, 66, bias=True)
        self.proj_kp = nn.Linear(63, 66, bias=True)
        
        # self.out = nn.Linear(66, 1024, bias=True)

    def forward(self, x_info):
        pitch = x_info['pitch']
        yaw = x_info['yaw']
        roll = x_info['roll']
        t = x_info['t']
        exp = x_info['exp']
        scale = x_info['scale']
        kp = x_info['kp']

        t = self.proj_t(t)
        exp = self.proj_exp(exp)
        scale = self.proj_scale(scale)
        kp = self.proj_kp(kp)

        embeds = torch.cat([pitch, yaw, roll, t, exp, scale, kp], 0)
        # embeds = self.out(embeds)
        embeds = embeds.unsqueeze(0)
        print("embeds", embeds.shape)
        return embeds
    

class VGGFeatureEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj_4 = nn.Linear(128, 64, bias=True)
        self.proj_8 = nn.Linear(256, 64, bias=True)
        self.proj_16 = nn.Linear(512, 64, bias=True)

    def forward(self, x):
        return [x[0], self.proj_4(x[1]), self.proj_8(x[2]), self.proj_16(x[3])]
        

class conv_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))
    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        x = self.up(x)
        return x


class U_Net(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, n1=16):
        super(U_Net, self).__init__()
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        
        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        # self.Up3 = up_conv(filters[2], filters[1])
        # self.Up_conv3 = conv_block(filters[2], filters[1])

        # self.Up2 = up_conv(filters[1], filters[0])
        # self.Up_conv2 = conv_block(filters[1], filters[0])

        # self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)
        self.proj_d5 = nn.Linear(128, 64, bias=True)
        # self.proj_d4 = nn.Linear(64, 64, bias=True)
        # self.proj_d3 = nn.Linear(32, 64, bias=True)
        # self.proj_d2 = nn.Linear(16, 64, bias=True)
        
    def forward(self, x):
        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)  

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)  

        d5 = self.Up5(e5)    
        padding = (e4.shape[-1] - d5.shape[-1])  ### 本行和下行代码是为了解决-输入图像尺寸大小不能被32整除的问题
        d5 = F.pad(d5, (0, padding, 0, padding), mode='constant', value=0)
        d5 = torch.cat((e4, d5), dim=1)

        d5_n = self.Up_conv5(d5)

        d4 = self.Up4(d5_n)
        d4 = torch.cat((e3, d4), dim=1)
        d4_n = self.Up_conv4(d4)

        # d3 = self.Up3(d4_n)
        # d3 = torch.cat((e2, d3), dim=1)
        # d3_n = self.Up_conv3(d3)

        # d2 = self.Up2(d3)
        # d2 = torch.cat((e1, d2_n), dim=1)
        # d2 = self.Up_conv2(d2)

        # out = self.Conv(d2)
        d5_out = self.proj_d5(d5_n.permute(0, 2, 3, 1))
        # d4_out = self.proj_d4(d4_n.permute(0, 2, 3, 1))
        d4_out = d4_n.permute(0, 2, 3, 1)
        # d3_out = self.proj_d3(d3_n.permute(0, 2, 3, 1))
        # d2_out = self.proj_d2(d2_n.permute(0, 2, 3, 1))
        d5_out = rearrange(d5_out, "b h w c -> b (h w) c")
        d4_out = rearrange(d4_out, "b h w c -> b (h w) c")
        # d3_out = rearrange(d3_out, "b h w c -> b (h w) c")
        # d2_out = rearrange(d2_out, "b h w c -> b (h w) c")
        out = torch.cat([d5_out, d4_out], 1)
        print("d5_n", d5_n.shape, "d4_n", d4_n.shape, "out", out.shape)
        return out
