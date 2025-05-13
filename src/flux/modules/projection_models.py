import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
# from utils.training_utils import fuse_multi_object_embeddings
import random

from transformers import Blip2QFormerModel, Blip2QFormerConfig

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, use_residual=True):
        super().__init__()
        if use_residual:
            assert in_dim == out_dim
        self.layernorm = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.use_residual = use_residual
        self.act_fn = nn.GELU()

    def forward(self, x):
        residual = x
        x = self.layernorm(x)
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.fc2(x)
        if self.use_residual:
            x = x + residual
        return x

class ImageProjModel(torch.nn.Module):
    """Projection Model"""

    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024, clip_extra_context_tokens=4):
        super().__init__()

        self.generator = None
        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.proj = torch.nn.Linear(clip_embeddings_dim, self.clip_extra_context_tokens * cross_attention_dim)
        self.norm = torch.nn.LayerNorm(cross_attention_dim)

    def forward(self, image_embeds):
        image_embeds = image_embeds.to(self.proj.parameters().__next__().dtype)
        embeds = image_embeds
        clip_extra_context_tokens = self.proj(embeds).reshape(
            -1, self.clip_extra_context_tokens, self.cross_attention_dim
        )
        clip_extra_context_tokens = self.norm(clip_extra_context_tokens)
        return clip_extra_context_tokens


class MLPProjModel(torch.nn.Module):
    """SD model with image prompt"""
    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024):
        super().__init__()
        
        self.proj = torch.nn.Sequential(
            torch.nn.Linear(clip_embeddings_dim, clip_embeddings_dim),
            torch.nn.GELU(),
            torch.nn.Linear(clip_embeddings_dim, cross_attention_dim),
            torch.nn.LayerNorm(cross_attention_dim)
        )
        
    def forward(self, image_embeds):
        clip_extra_context_tokens = self.proj(image_embeds)
        return clip_extra_context_tokens



class ImageProjModelLocal(torch.nn.Module):
    """Projection Model
    https://github.com/tencent-ailab/IP-Adapter/blob/main/ip_adapter/ip_adapter.py#L28
    """

    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024, clip_pooing_embeddings_dim=768, faces_embedding_dim=512, vae_embeddings_dim=None):
        super().__init__()

        self.generator = None
        self.cross_attention_dim = cross_attention_dim
        self.clip_embeddings_dim = clip_embeddings_dim
        self.faces_embedding_dim = faces_embedding_dim
        self.vae_embeddings_dim = vae_embeddings_dim

        if vae_embeddings_dim is not None:
            self.clip_dim = clip_embeddings_dim*2
            self.proj_0 = torch.nn.Linear(self.clip_dim, 2 * cross_attention_dim)
        else:
            self.clip_dim = clip_embeddings_dim
            self.proj_0 = torch.nn.Linear(self.clip_dim, 2 * cross_attention_dim)

        self.norm_0 = torch.nn.LayerNorm(cross_attention_dim * 2)

        self.proj_1 = torch.nn.Linear(2 * cross_attention_dim, cross_attention_dim)
        self.norm_1 = torch.nn.LayerNorm(cross_attention_dim)
        if vae_embeddings_dim is not None:
            self.proj = torch.nn.Linear(clip_pooing_embeddings_dim, self.clip_dim)
        else:
            self.proj = torch.nn.Linear(clip_pooing_embeddings_dim, self.clip_dim)
        self.norm = torch.nn.LayerNorm(self.clip_dim)

        if vae_embeddings_dim is not None:
            self.proj_vae = torch.nn.Linear(vae_embeddings_dim*4, clip_embeddings_dim)
            self.norm_vae = torch.nn.LayerNorm(clip_embeddings_dim)

    def forward(self, image_embeds, image_embeds_pooling=None, vae_embeds=None):
        bs = image_embeds.shape[0]
        local_embeds = image_embeds
        if vae_embeds is not None:
            vae_embeds = rearrange(vae_embeds, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=4, pw=4)
            vae_embeds = self.proj_vae(vae_embeds).reshape(vae_embeds.shape[0], -1, self.clip_embeddings_dim)
            vae_embeds = self.norm_vae(vae_embeds)
            vae_embeds = torch.cat([vae_embeds, torch.zeros((bs, 1, self.clip_embeddings_dim), device=vae_embeds.device)], 1).to(local_embeds.device) # [bs, 257, 1024]
            local_embeds = torch.cat([local_embeds, vae_embeds], -1) # [bs, 257, 2048]
        
        # print(image_embeds.shape, image_embeds_pooling.shape)
        if image_embeds_pooling is not None:
            image_embeds_pooling = self.proj(image_embeds_pooling).reshape(image_embeds_pooling.shape[0], -1, self.clip_dim)
            image_embeds_pooling = self.norm(image_embeds_pooling)
        
            embeds = torch.cat([local_embeds, image_embeds_pooling], 1).to(self.proj.parameters().__next__().dtype)
        
        embeds = self.proj_0(embeds)
        embeds = self.norm_0(embeds)

        embeds = self.proj_1(embeds)
        embeds = self.norm_1(embeds)

        return embeds


class ImageProjModelLocalGELU(torch.nn.Module):
    """Projection Model
    https://github.com/tencent-ailab/IP-Adapter/blob/main/ip_adapter/ip_adapter.py#L28
    """

    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024, clip_pooing_embeddings_dim=768, faces_embedding_dim=512, vae_embeddings_dim=None):
        super().__init__()

        self.generator = None
        self.cross_attention_dim = cross_attention_dim
        self.clip_embeddings_dim = clip_embeddings_dim
        self.faces_embedding_dim = faces_embedding_dim
        self.vae_embeddings_dim = vae_embeddings_dim

        if vae_embeddings_dim is not None:
            self.clip_dim = clip_embeddings_dim*2
            self.proj_0 = torch.nn.Linear(self.clip_dim, 2 * cross_attention_dim)
        else:
            self.clip_dim = clip_embeddings_dim
            self.proj_0 = torch.nn.Sequential(
                torch.nn.Linear(self.clip_dim, 2 * cross_attention_dim),
                torch.nn.GELU(),
            )

        self.proj_1 = torch.nn.Sequential(
            torch.nn.Linear(2 * cross_attention_dim, cross_attention_dim),
            torch.nn.GELU(),
        )
        self.norm_1 = torch.nn.LayerNorm(cross_attention_dim)
        if vae_embeddings_dim is not None:
            self.proj = torch.nn.Linear(clip_pooing_embeddings_dim, self.clip_dim)
        else:
            self.proj = torch.nn.Sequential(
                torch.nn.Linear(clip_pooing_embeddings_dim, self.clip_dim),
                torch.nn.GELU(),
            )

        if vae_embeddings_dim is not None:
            self.proj_vae = torch.nn.Linear(vae_embeddings_dim*4, clip_embeddings_dim)
            self.norm_vae = torch.nn.LayerNorm(clip_embeddings_dim)

    def forward(self, image_embeds, image_embeds_pooling=None, vae_embeds=None):
        bs = image_embeds.shape[0]
        local_embeds = image_embeds
        if vae_embeds is not None:
            vae_embeds = rearrange(vae_embeds, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=4, pw=4)
            vae_embeds = self.proj_vae(vae_embeds).reshape(vae_embeds.shape[0], -1, self.clip_embeddings_dim)
            vae_embeds = self.norm_vae(vae_embeds)
            vae_embeds = torch.cat([vae_embeds, torch.zeros((bs, 1, self.clip_embeddings_dim), device=vae_embeds.device)], 1).to(local_embeds.device) # [bs, 257, 1024]
            local_embeds = torch.cat([local_embeds, vae_embeds], -1) # [bs, 257, 2048]
        
        # print(image_embeds.shape, image_embeds_pooling.shape)
        if image_embeds_pooling is not None:
            image_embeds_pooling = self.proj(image_embeds_pooling).reshape(image_embeds_pooling.shape[0], -1, self.clip_dim)
        
            embeds = torch.cat([local_embeds, image_embeds_pooling], 1).to(self.proj.parameters().__next__().dtype)
        
        embeds = self.proj_0(embeds)

        embeds = self.proj_1(embeds)
        embeds = self.norm_1(embeds)

        return embeds

class SequenceDropout(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        
    def forward(self, x):
        if not self.training:  # 推理时不做dropout
            return x
            
        # x shape: [batch_size, seq_len, hidden_dim]
        batch_size, seq_len, hidden_dim = x.shape
        
        # 在seq_len维度上生成mask
        mask = torch.rand(batch_size, seq_len, 1, device=x.device) > self.p  # [batch_size, seq_len, 1]
        mask = mask.expand(-1, -1, hidden_dim)  # [batch_size, seq_len, hidden_dim]
        
        # 应用mask并缩放
        return x * mask

class TokenDropout(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        
    def forward(self, x):
        if not self.training:  # 推理时不做dropout
            return x
            
        # x shape: [batch_size, seq_len, hidden_dim]
        batch_size, seq_len, hidden_dim = x.shape
        
        # 生成单个mask用于整个batch
        keep_mask = torch.rand(seq_len, device=x.device) > self.p  # [seq_len]
        
        # 直接用布尔索引选择保留的token
        # print("x", x.shape)
        return x[:, keep_mask, :]  # [batch_size, new_seq_len, hidden_dim]
    

class ImageProjModelLocalID(torch.nn.Module):
    """Projection Model
    https://github.com/tencent-ailab/IP-Adapter/blob/main/ip_adapter/ip_adapter.py#L28
    """

    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1280, clip_pooing_embeddings_dim=1024, face_embeddings_dim=512, id_tokens=4, drop_p=0.1):
        super().__init__()

        self.generator = None
        self.cross_attention_dim = cross_attention_dim
        self.clip_embeddings_dim = clip_embeddings_dim
        self.face_embeddings_dim = face_embeddings_dim
        self.id_tokens = id_tokens
        self.drop_p = drop_p
        self.clip_dim = clip_embeddings_dim
        self.proj_0 = torch.nn.Linear(self.clip_dim, 2 * cross_attention_dim)

        self.norm_0 = torch.nn.LayerNorm(cross_attention_dim * 2)

        self.proj_1 = torch.nn.Linear(2 * cross_attention_dim, cross_attention_dim)
        self.norm_1 = torch.nn.LayerNorm(cross_attention_dim)

        self.proj = torch.nn.Linear(clip_pooing_embeddings_dim, self.clip_dim)
        self.norm = torch.nn.LayerNorm(self.clip_dim)

        self.proj_face = torch.nn.Sequential(
            torch.nn.Linear(face_embeddings_dim, self.clip_dim*2),
            torch.nn.GELU(),
            torch.nn.Linear(self.clip_dim*2, self.clip_dim*id_tokens)
        )
        self.norm_face = torch.nn.LayerNorm(self.clip_dim*id_tokens)

        self.drop_token = SequenceDropout(drop_p)

    def forward(self, image_embeds, image_embeds_pooling=None, face_embeddings=None):
        bs = image_embeds.shape[0]
        local_embeds = image_embeds
        
        # print(image_embeds.shape, image_embeds_pooling.shape)
        if image_embeds_pooling is not None:
            image_embeds_pooling = self.proj(image_embeds_pooling).reshape(image_embeds_pooling.shape[0], -1, self.clip_dim)
            image_embeds_pooling = self.norm(image_embeds_pooling)
        
            embeds = torch.cat([local_embeds, image_embeds_pooling], 1).to(self.proj.parameters().__next__().dtype)
        
        if face_embeddings is not None:
            if len(face_embeddings.shape) == 2:
                face_embeddings = face_embeddings.unsqueeze(1)
            face_embeddings = self.proj_face(face_embeddings)
            face_embeddings = self.norm_face(face_embeddings)
            face_embeddings = face_embeddings.view(bs, -1, self.clip_dim)

        if self.drop_p > 0 and self.drop_p < 1:
            if random.random() < 0.5:
                embeds = self.drop_token(embeds)
            else:
                face_embeddings = self.drop_token(face_embeddings)
        elif self.drop_p == 1:
            embeds = torch.zeros_like(embeds).to(embeds.device, dtype=embeds.dtype)
        embeds = torch.cat([embeds, face_embeddings], 1).to(self.proj.parameters().__next__().dtype)

        embeds = self.proj_0(embeds)
        embeds = self.norm_0(embeds)

        embeds = self.proj_1(embeds)
        embeds = self.norm_1(embeds)

        return embeds
    
class MLPMixIDModel(nn.Module):
    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1280, clip_pooing_embeddings_dim=1024, face_embeddings_dim=512, id_tokens=4, drop_p=0.1, apply_pos_emb=True, max_seq_len=258):
        super().__init__()

        self.generator = None
        self.cross_attention_dim = cross_attention_dim
        self.clip_embeddings_dim = clip_embeddings_dim
        self.face_embeddings_dim = face_embeddings_dim
        self.id_tokens = id_tokens
        self.drop_p = drop_p
        self.clip_dim = clip_embeddings_dim
        self.num_id_token = id_tokens
        self.max_seq_len = max_seq_len
        self.proj_0 = torch.nn.Linear(self.clip_dim, 2 * cross_attention_dim)

        self.norm_0 = torch.nn.LayerNorm(cross_attention_dim * 2)

        self.proj_1 = torch.nn.Linear(2 * cross_attention_dim, cross_attention_dim)
        self.norm_1 = torch.nn.LayerNorm(cross_attention_dim)

        self.proj = torch.nn.Linear(clip_pooing_embeddings_dim, self.clip_dim)
        self.norm = torch.nn.LayerNorm(self.clip_dim)

        self.proj_id = nn.Sequential(
        nn.Linear(1024, self.clip_dim),
        nn.LeakyReLU(),
        nn.LayerNorm(self.clip_dim),
        nn.Linear(self.clip_dim, self.num_id_token * self.clip_dim),
        nn.LayerNorm(self.num_id_token * self.clip_dim),
        )

        self.drop_token = TokenDropout(drop_p)

        self.pos_emb = nn.Embedding(max_seq_len+id_tokens, self.clip_dim) if apply_pos_emb else None

    def forward(self, image_embeds, image_embeds_pooling=None, id_embeddings_ante=None, id_embeddings_circular=None, is_inference=False):
        # id_embeddings_1: (bs, 512)
        # id_embeddings_2: (bs, 512)
        # clip_image_embeddings: (bs, 257, 1280)
        bsz = image_embeds.shape[0]
        # project id embeddings
        # print("id_embeddings", id_embeddings_ante.shape, id_embeddings_circular.shape)
        # if id_embeddings_ante.shape[0] == 0:
        #     id_embeddings_ante = torch.zeros_like(id_embeddings_circular).to(id_embeddings_circular.device).to(id_embeddings_circular.dtype)
        # print(id_embeddings_ante.shape)
        id_embeddings = torch.cat((id_embeddings_ante, id_embeddings_circular), dim=-1)
        if id_embeddings.shape[0] <1024:
            id_embeddings = torch.zeros(1024).to(id_embeddings_circular.device).to(id_embeddings_circular.dtype)
        id_embeddings = self.proj_id(id_embeddings).view(bsz, self.num_id_token, self.clip_dim) # (bs, 1024) -> (bs, 4, 1280)

        if self.pos_emb is not None:
            n, device = image_embeds.shape[1], image_embeds.device
            pos_emb = self.pos_emb(torch.arange(n, device=device))
            image_embeds = image_embeds + pos_emb

        image_embeds_pooling = self.proj(image_embeds_pooling).reshape(bsz, -1, self.clip_dim)
        image_embeds_pooling = self.norm(image_embeds_pooling)
        embeds = torch.cat([image_embeds, image_embeds_pooling, id_embeddings], 1).to(self.proj.parameters().__next__().dtype)

        if self.pos_emb is not None:
            n, device = embeds.shape[1], image_embeds.device
            pos_emb = self.pos_emb(torch.arange(n, device=device))
            embeds = embeds + pos_emb

        clip_embeds = embeds[:, :self.max_seq_len, :]
        id_embeds = embeds[:, self.max_seq_len:, :]
        
        if not is_inference:
            if random.random() < 0.65:
                clip_embeds = self.drop_token(clip_embeds)
            else:
                id_embeds = self.drop_token(id_embeds)

        embeds = torch.cat([clip_embeds, id_embeds], 1)

        embeds = self.proj_0(embeds)
        embeds = self.norm_0(embeds)

        embeds = self.proj_1(embeds)
        embeds = self.norm_1(embeds)

        return embeds
    
# 衣服用
class MLPMixIDModel_v2(nn.Module):
    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1280, clip_pooing_embeddings_dim=1024, face_embeddings_dim=512, id_tokens=4, drop_p=0.1, apply_pos_emb=True, max_seq_len=258, drop_local=False):
        super().__init__()

        self.generator = None
        self.cross_attention_dim = cross_attention_dim
        self.clip_embeddings_dim = clip_embeddings_dim
        self.face_embeddings_dim = face_embeddings_dim
        self.id_tokens = id_tokens
        self.drop_p = drop_p
        self.clip_dim = clip_embeddings_dim
        self.num_id_token = id_tokens
        self.max_seq_len = max_seq_len
        self.proj_0 = torch.nn.Linear(self.clip_dim, 2 * cross_attention_dim)

        self.norm_0 = torch.nn.LayerNorm(cross_attention_dim * 2)

        self.proj_1 = torch.nn.Linear(2 * cross_attention_dim, cross_attention_dim)
        self.norm_1 = torch.nn.LayerNorm(cross_attention_dim)

        self.proj = torch.nn.Linear(clip_pooing_embeddings_dim, self.clip_dim)
        self.norm = torch.nn.LayerNorm(self.clip_dim)

        self.proj_id = nn.Sequential(
        nn.Linear(1024, self.clip_dim),
        nn.LeakyReLU(),
        nn.LayerNorm(self.clip_dim),
        nn.Linear(self.clip_dim, self.num_id_token * self.clip_dim),
        nn.LayerNorm(self.num_id_token * self.clip_dim),
        )

        self.drop_token = TokenDropout(drop_p)

        self.pos_emb = nn.Embedding(max_seq_len*5+id_tokens, self.clip_dim) if apply_pos_emb else None

        self.drop_local = drop_local

    def forward(self, image_embeds, image_embeds_pooling=None, id_embeddings_ante=None, id_embeddings_circular=None, is_inference=False):
        # id_embeddings_1: (bs, 512)
        # id_embeddings_2: (bs, 512)
        # clip_image_embeddings: (bs, 257, 1280)
        bsz = image_embeds.shape[0]
        # project id embeddings
        # print("id_embeddings", id_embeddings_ante.shape, id_embeddings_circular.shape)
        # if id_embeddings_ante.shape[0] == 0:
        #     id_embeddings_ante = torch.zeros_like(id_embeddings_circular).to(id_embeddings_circular.device).to(id_embeddings_circular.dtype)
        # print(id_embeddings_ante.shape)
        if id_embeddings_ante is None:
            id_embeddings = id_embeddings_circular
        else:
            id_embeddings = torch.cat((id_embeddings_ante, id_embeddings_circular), dim=-1)
        if id_embeddings.shape[0] <1024:
            id_embeddings = torch.zeros(1024).to(id_embeddings_circular.device).to(id_embeddings_circular.dtype)
        id_embeddings = self.proj_id(id_embeddings).view(bsz, self.num_id_token, self.clip_dim) # (bs, 1024) -> (bs, 4, 1280)

        if self.pos_emb is not None:
            n, device = image_embeds.shape[1], image_embeds.device
            pos_emb = self.pos_emb(torch.arange(n, device=device))
            image_embeds = image_embeds + pos_emb

        image_embeds_pooling = self.proj(image_embeds_pooling).reshape(bsz, -1, self.clip_dim)
        image_embeds_pooling = self.norm(image_embeds_pooling)
        if self.drop_local:
            print("drop_local")
            embeds = torch.cat([image_embeds_pooling, id_embeddings], 1).to(self.proj.parameters().__next__().dtype)
        else:
            embeds = torch.cat([image_embeds, image_embeds_pooling, id_embeddings], 1).to(self.proj.parameters().__next__().dtype)

        if self.pos_emb is not None:
            n, device = embeds.shape[1], image_embeds.device
            pos_emb = self.pos_emb(torch.arange(n, device=device))
            embeds = embeds + pos_emb

        id_embeds = embeds[:, :self.num_id_token, :]
        clip_embeds = embeds[:, self.num_id_token:, :]
        if not is_inference:
            if random.random() < 0.65:
                clip_embeds = self.drop_token(clip_embeds)
            else:
                id_embeds = self.drop_token(id_embeds)

        embeds = torch.cat([clip_embeds, id_embeds], 1)

        embeds = self.proj_0(embeds)
        embeds = self.norm_0(embeds)

        embeds = self.proj_1(embeds)
        embeds = self.norm_1(embeds)

        return embeds
    
# 衣服用
class MLPMixIDModelQFormer(nn.Module):
    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1280, clip_pooing_embeddings_dim=1024, face_embeddings_dim=512, id_tokens=4, drop_p=0.1, apply_pos_emb=True, max_seq_len=258):
        super().__init__()

        self.generator = None
        self.cross_attention_dim = cross_attention_dim
        self.clip_embeddings_dim = clip_embeddings_dim
        self.face_embeddings_dim = face_embeddings_dim
        self.id_tokens = id_tokens
        self.drop_p = drop_p
        self.clip_dim = clip_embeddings_dim
        self.num_id_token = id_tokens
        self.max_seq_len = max_seq_len
        self.proj_0 = torch.nn.Linear(self.clip_dim, 2 * cross_attention_dim)

        self.norm_0 = torch.nn.LayerNorm(cross_attention_dim * 2)

        self.proj_1 = torch.nn.Linear(2 * cross_attention_dim, cross_attention_dim)
        self.norm_1 = torch.nn.LayerNorm(cross_attention_dim)

        self.proj = torch.nn.Linear(clip_pooing_embeddings_dim, self.clip_dim)
        self.norm = torch.nn.LayerNorm(self.clip_dim)

        self.proj_id = nn.Sequential(
        nn.Linear(1024, self.clip_dim),
        nn.LeakyReLU(),
        nn.LayerNorm(self.clip_dim),
        nn.Linear(self.clip_dim, self.num_id_token * self.clip_dim),
        nn.LayerNorm(self.num_id_token * self.clip_dim),
        )

        self.drop_token = TokenDropout(drop_p)

        self.pos_emb = nn.Embedding(max_seq_len*5+id_tokens, self.clip_dim) if apply_pos_emb else None
        
        qformerconfig = Blip2QFormerConfig(hidden_size=4096, num_attention_heads=32, encoder_hidden_size=4096)
        self.qformer = Blip2QFormerModel(qformerconfig)
        self.learnable_q = nn.Parameter(torch.randn(1, 32, 4096))

    def forward(self, image_embeds, image_embeds_pooling=None, id_embeddings_ante=None, id_embeddings_circular=None, is_inference=False):
        bsz = image_embeds.shape[0]
        
        if id_embeddings_ante is None:
            id_embeddings = id_embeddings_circular
        else:
            id_embeddings = torch.cat((id_embeddings_ante, id_embeddings_circular), dim=-1)
        if id_embeddings.shape[0] <1024:
            id_embeddings = torch.zeros(1024).to(id_embeddings_circular.device).to(id_embeddings_circular.dtype)
        id_embeddings = self.proj_id(id_embeddings).view(bsz, self.num_id_token, self.clip_dim) # (bs, 1024) -> (bs, 4, 1280)

        if self.pos_emb is not None:
            n, device = image_embeds.shape[1], image_embeds.device
            pos_emb = self.pos_emb(torch.arange(n, device=device))
            image_embeds = image_embeds + pos_emb

        image_embeds_pooling = self.proj(image_embeds_pooling).reshape(bsz, -1, self.clip_dim)
        image_embeds_pooling = self.norm(image_embeds_pooling)
        embeds = torch.cat([image_embeds, image_embeds_pooling, id_embeddings], 1).to(self.proj.parameters().__next__().dtype)

        if self.pos_emb is not None:
            n, device = embeds.shape[1], image_embeds.device
            pos_emb = self.pos_emb(torch.arange(n, device=device))
            embeds = embeds + pos_emb

        id_embeds = embeds[:, :self.num_id_token, :]
        clip_embeds = embeds[:, self.num_id_token:, :]
        if not is_inference:
            if random.random() < 0.65:
                print("clip_embeds", clip_embeds.shape)
                clip_embeds = self.drop_token(clip_embeds)
            else:
                id_embeds = self.drop_token(id_embeds)

        embeds = torch.cat([clip_embeds, id_embeds], 1)

        embeds = self.proj_0(embeds)
        embeds = self.norm_0(embeds)

        embeds = self.proj_1(embeds)
        embeds = self.norm_1(embeds)
        
        print("embeds", embeds.shape)
        qformer_result_dic = self.qformer(
            query_embeds=self.learnable_q,
            encoder_hidden_states=embeds,
            output_attentions=True,
            output_hidden_states=True,
            return_dict=True
        )
        embeds = qformer_result_dic.last_hidden_state
        print("embeds after qformer", embeds.shape)

        return embeds

# 衣服用
class MLPMixIDModelTransformer(nn.Module):
    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1280, clip_pooing_embeddings_dim=1024, face_embeddings_dim=512, id_tokens=4, drop_p=0.1, apply_pos_emb=True, max_seq_len=258):
        super().__init__()

        self.generator = None
        self.cross_attention_dim = cross_attention_dim
        self.clip_embeddings_dim = clip_embeddings_dim
        self.face_embeddings_dim = face_embeddings_dim
        self.id_tokens = id_tokens
        self.drop_p = drop_p
        self.clip_dim = clip_embeddings_dim
        self.num_id_token = id_tokens
        self.max_seq_len = max_seq_len
        self.proj_0 = torch.nn.Linear(self.clip_dim, 2 * cross_attention_dim)

        self.norm_0 = torch.nn.LayerNorm(cross_attention_dim * 2)

        self.proj_1 = torch.nn.Linear(2 * cross_attention_dim, cross_attention_dim)
        self.norm_1 = torch.nn.LayerNorm(cross_attention_dim)

        self.proj = torch.nn.Linear(clip_pooing_embeddings_dim, self.clip_dim)
        self.norm = torch.nn.LayerNorm(self.clip_dim)

        self.proj_id = nn.Sequential(
        nn.Linear(1024, self.clip_dim),
        nn.LeakyReLU(),
        nn.LayerNorm(self.clip_dim),
        nn.Linear(self.clip_dim, self.num_id_token * self.clip_dim),
        nn.LayerNorm(self.num_id_token * self.clip_dim),
        )

        self.drop_token = TokenDropout(drop_p)

        self.pos_emb = nn.Embedding(max_seq_len*5+id_tokens, self.clip_dim) if apply_pos_emb else None
        
        self.transformer = torch.nn.TransformerEncoderLayer(d_model=1024, nhead=16, batch_first=True)

    def forward(self, image_embeds, image_embeds_pooling=None, id_embeddings_ante=None, id_embeddings_circular=None, is_inference=False):
        bsz = image_embeds.shape[0]
        
        image_embeds = self.transformer(image_embeds)
        # print("image_embeds", image_embeds.shape, "image_embeds_transformer", image_embeds_transformer.shape)
        
        if id_embeddings_ante is None:
            id_embeddings = id_embeddings_circular
        else:
            id_embeddings = torch.cat((id_embeddings_ante, id_embeddings_circular), dim=-1)
        if id_embeddings.shape[0] <1024:
            id_embeddings = torch.zeros(1024).to(id_embeddings_circular.device).to(id_embeddings_circular.dtype)
        id_embeddings = self.proj_id(id_embeddings).view(bsz, self.num_id_token, self.clip_dim) # (bs, 1024) -> (bs, 4, 1280)

        if self.pos_emb is not None:
            n, device = image_embeds.shape[1], image_embeds.device
            pos_emb = self.pos_emb(torch.arange(n, device=device))
            image_embeds = image_embeds + pos_emb

        image_embeds_pooling = self.proj(image_embeds_pooling).reshape(bsz, -1, self.clip_dim)
        image_embeds_pooling = self.norm(image_embeds_pooling)
        embeds = torch.cat([image_embeds, image_embeds_pooling, id_embeddings], 1).to(self.proj.parameters().__next__().dtype)

        if self.pos_emb is not None:
            n, device = embeds.shape[1], image_embeds.device
            pos_emb = self.pos_emb(torch.arange(n, device=device))
            embeds = embeds + pos_emb

        id_embeds = embeds[:, :self.num_id_token, :]
        clip_embeds = embeds[:, self.num_id_token:, :]
        if not is_inference:
            if random.random() < 0.65:
                clip_embeds = self.drop_token(clip_embeds)
            else:
                id_embeds = self.drop_token(id_embeds)

        embeds = torch.cat([clip_embeds, id_embeds], 1)

        embeds = self.proj_0(embeds)
        embeds = self.norm_0(embeds)

        embeds = self.proj_1(embeds)
        embeds = self.norm_1(embeds)
        
        return embeds
        

class MLPImageProjIDModel(torch.nn.Module):

    def __init__(self, cross_attention_dim=4096, structure_embeddings_dim=1728, id_embeddings_dim=512):
        super().__init__()
        
        self.proj_local = torch.nn.Sequential(
            torch.nn.Linear(structure_embeddings_dim, cross_attention_dim*2),
            torch.nn.GELU(),
            torch.nn.Linear(cross_attention_dim*2, cross_attention_dim),
            torch.nn.LayerNorm(cross_attention_dim)
        )
        self.proj_global = torch.nn.Sequential(
            torch.nn.Linear(id_embeddings_dim, cross_attention_dim),
            torch.nn.GELU(),
            torch.nn.Linear(cross_attention_dim, cross_attention_dim),
        )

        self.proj_out = torch.nn.Linear(cross_attention_dim, cross_attention_dim)
        self.norm_out = torch.nn.LayerNorm(cross_attention_dim)

    def forward(self, structure_embeds, id_embeds):
        local_embeds = self.proj_local(structure_embeds)
        global_embeds = self.proj_global(id_embeds)

        embeds = torch.cat([local_embeds, global_embeds], 1)
        embeds = self.proj_out(embeds)
        embeds = self.norm_out(embeds)

        return embeds

class PureIDModel(torch.nn.Module):

    def __init__(self, cross_attention_dim=4096, id_embeddings_dim=512, num_tokens=16, with_gelu=True, normalize=True):
        super().__init__()
        self.cross_attention_dim = cross_attention_dim
        self.id_embeddings_dim = id_embeddings_dim
        self.num_tokens = num_tokens
        
        self.proj = nn.Sequential(
            nn.Linear(id_embeddings_dim, cross_attention_dim),
            nn.GELU() if with_gelu else nn.Identity(),
            nn.Linear(cross_attention_dim, cross_attention_dim*num_tokens),
        )
        self.norm = nn.LayerNorm(cross_attention_dim*num_tokens)
        self.normalize = normalize
    def forward(self, id_embeds):
        bs = id_embeds.shape[0]
        if self.normalize:
            id_embeds = F.normalize(id_embeds, p=2, dim=-1)
        # print(torch.cosine_similarity(id_embeds[0,:], id_embeds[1,:], dim=-1))
        embeds = self.proj(id_embeds)
        embeds = self.norm(embeds)
        if len(embeds.shape) == 2:
            embeds = embeds.unsqueeze(1)
        embeds = embeds.view(bs, -1, self.cross_attention_dim)
        return embeds

class TextWithImageProjModel(torch.nn.Module):
    def __init__(self, cross_attention_dim=4096, text_embeddings_dim=4096, image_embeddings_dim=4096, face_embeddings_dim=512, simplified=True):
        super().__init__()

        self.generator = None
        self.cross_attention_dim = cross_attention_dim
        self.text_embeddings_dim = text_embeddings_dim
        self.image_embeddings_dim = image_embeddings_dim
        self.face_embeddings_dim = face_embeddings_dim

        self.simplified = simplified

        if not simplified:
            self.proj_text_1 = torch.nn.Linear(text_embeddings_dim*2, cross_attention_dim*2)
            self.norm_text_1 = torch.nn.LayerNorm(cross_attention_dim*2)
            self.proj_text_2 = torch.nn.Linear(cross_attention_dim*2, cross_attention_dim)
            self.norm_text_2 = torch.nn.LayerNorm(cross_attention_dim)

            self.proj_out = torch.nn.Linear(image_embeddings_dim, cross_attention_dim)
            self.norm_out = torch.nn.LayerNorm(cross_attention_dim)
        else:
            self.proj_text_1 = torch.nn.Sequential(
                torch.nn.Linear(text_embeddings_dim*2, cross_attention_dim*2),
                torch.nn.GELU(),
                torch.nn.Linear(cross_attention_dim*2, cross_attention_dim)
            )
            self.mh_attn_to_q = torch.nn.Linear(cross_attention_dim, cross_attention_dim)
            self.mh_attn_to_k = torch.nn.Linear(face_embeddings_dim, cross_attention_dim)
            self.mh_attn_to_v = torch.nn.Linear(face_embeddings_dim, cross_attention_dim)

    def forward(self, text_embeds, image_embeds, face_embeds=None):
        bs = text_embeds.shape[0]
        text_embeds = self.proj_text_1(text_embeds.to(self.proj_text_1.parameters().__next__().dtype).view(bs, 1, -1)) 
        if face_embeds is not None:
            if len(face_embeds.shape) == 2:
                face_embeds = face_embeds.unsqueeze(1)
            face_embeds = face_embeds.to(self.mh_attn_to_q.parameters().__next__().dtype)
            head_num = 64
            head_dim = self.cross_attention_dim // head_num
            
            q = self.mh_attn_to_q(text_embeds)  # [bs, seq_len, dim]
            k = self.mh_attn_to_k(face_embeds)  # [bs, face_len, dim] 
            v = self.mh_attn_to_v(face_embeds)  # [bs, face_len, dim]
            
            q = q.view(bs, -1, head_num, head_dim).transpose(1, 2)  # [bs, head_num, seq_len, head_dim]
            k = k.view(bs, -1, head_num, head_dim).transpose(1, 2)  # [bs, head_num, face_len, head_dim]
            v = v.view(bs, -1, head_num, head_dim).transpose(1, 2)  # [bs, head_num, face_len, head_dim]

            out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False)
            out = out.transpose(1, 2).contiguous().view(bs, -1, self.cross_attention_dim)  # [bs, seq_len, dim]
            text_embeds = text_embeds + out

        if not self.simplified:
            text_embeds = self.norm_text_1(text_embeds) # [bs, 4096 * 2]
            text_embeds = self.proj_text_2(text_embeds)
            text_embeds = self.norm_text_2(text_embeds) # [bs, 4096]

            out_embeds = self.proj_out(image_embeds + text_embeds.expand(bs, image_embeds.shape[1], -1))
            out_embeds = self.norm_out(out_embeds)    
        else:
            out_embeds = image_embeds + text_embeds.expand(bs, image_embeds.shape[1], -1)


        return out_embeds

class TextWithImageProjModelGELU(torch.nn.Module):
    def __init__(self, cross_attention_dim=4096, text_embeddings_dim=4096, image_embeddings_dim=4096):
        super().__init__()

        self.generator = None
        self.cross_attention_dim = cross_attention_dim
        self.text_embeddings_dim = text_embeddings_dim
        self.image_embeddings_dim = image_embeddings_dim

        self.proj_text = torch.nn.Sequential(
            torch.nn.Linear(text_embeddings_dim*2, cross_attention_dim*2),
            torch.nn.GELU(),
            torch.nn.Linear(cross_attention_dim*2, cross_attention_dim)
        )

        self.proj_out = torch.nn.Linear(image_embeddings_dim, cross_attention_dim)
        self.norm_out = torch.nn.LayerNorm(cross_attention_dim)

    def forward(self, text_embeds, image_embeds):
        bs = text_embeds.shape[0]
        text_embeds = self.proj_text(text_embeds.to(self.proj_text.parameters().__next__().dtype).view(bs, 1, -1)) 

        out_embeds = self.proj_out(image_embeds + text_embeds.expand(bs, image_embeds.shape[1], -1))
        out_embeds = self.norm_out(out_embeds)

        return out_embeds

# class PostfuseModule(nn.Module):
#     def __init__(self, embed_dim_text, embed_dim_img):
#         super().__init__()
#         self.mlp1 = MLP(embed_dim_text + embed_dim_img, embed_dim_text , (embed_dim_text + embed_dim_img) // 2, use_residual=False)
#         self.mlp2 = MLP(embed_dim_text, embed_dim_text, embed_dim_text, use_residual=True)
#         self.layer_norm = nn.LayerNorm(embed_dim_text)

#     def fuse_fn(self, text_embeds, object_embeds):
#         text_object_embeds = torch.cat([text_embeds, object_embeds], dim=-1)
#         text_object_embeds = self.mlp1(text_object_embeds) + text_embeds
#         text_object_embeds = self.mlp2(text_object_embeds)
#         text_object_embeds = self.layer_norm(text_object_embeds)
#         return text_object_embeds

#     def forward(
#         self,
#         object_pos_mat,
#         text_embeds,
#         object_embeds,
#     ) -> torch.Tensor:
#         text_object_embeds = fuse_multi_object_embeddings(
#             object_pos_mat, text_embeds, object_embeds, self.fuse_fn
#         )
#         return text_object_embeds
