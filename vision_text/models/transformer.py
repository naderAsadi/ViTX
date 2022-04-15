import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import Tuple, Union, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import VisionTextOutput


class AttentionPool2d(nn.Module):
    def __init__(
        self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None
    ):
        super().__init__()
        self.positional_embedding = nn.Parameter(
            torch.randn(spacial_dim**2 + 1, embed_dim) / embed_dim**0.5
        )
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(
            2, 0, 1
        )  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x,
            key=x,
            value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat(
                [self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]
            ),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False,
        )

        return x[0]


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x.to(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        # NOTE I do not know why this is the default. Slower than nn.GELU or nn.SiLU and use more GPU memory
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        attn_mask: torch.Tensor = None,
        act_layer: Callable = nn.GELU,
    ):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, d_model * 4)),
                    ("gelu", act_layer()),
                    ("c_proj", nn.Linear(d_model * 4, d_model)),
                ]
            )
        )
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = (
            self.attn_mask.to(dtype=x.dtype, device=x.device)
            if self.attn_mask is not None
            else None
        )
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        width: int,
        layers: int,
        heads: int,
        attn_mask: torch.Tensor = None,
        act_layer: Callable = nn.GELU,
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(
            *[
                ResidualAttentionBlock(width, heads, attn_mask, act_layer=act_layer)
                for _ in range(layers)
            ]
        )

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class MultiModalTransformer(nn.Module):
    def __init__(
        self,
        seq_size: int,
        embed_dim: int,
        layers: int,
        heads: int,
        projection_dim: int,
        act_layer: Callable = nn.GELU,
        n_modalities: int = 1,
    ):
        super().__init__()

        self.projection_dim = projection_dim
        self.seq_size = seq_size
        scale = embed_dim**-0.5

        self.vision_class_embedding = nn.Parameter(scale * torch.randn(embed_dim))
        self.text_class_embedding = nn.Parameter(scale * torch.randn(embed_dim))

        self.positional_embedding = nn.Parameter(
            scale * torch.randn((seq_size) * 2 + 2, embed_dim)
        )
        self.ln_pre = LayerNorm(embed_dim)

        self.transformer = Transformer(embed_dim, layers, heads, act_layer=act_layer)

        self.ln_post = LayerNorm(embed_dim)

        self.vision_projection = nn.Parameter(
            scale * torch.randn(embed_dim, projection_dim)
        )
        self.text_projection = nn.Parameter(
            scale * torch.randn(embed_dim, projection_dim)
        )

    def forward(self, vision_embed: torch.Tensor, text_embed: torch.Tensor):
        """_summary_

        Args:
            vision_embed (torch.Tensor): [batch_size, sequence_length, feature_dim]
            text_embed (torch.Tensor): [batch_size, sequence_length, feature_dim]

        Returns:
            _type_: _description_
        """

        x = torch.cat(
            [
                self.vision_class_embedding.to(vision_embed.dtype)
                + torch.zeros(
                    vision_embed.shape[0],
                    1,
                    vision_embed.shape[-1],
                    dtype=vision_embed.dtype,
                    device=vision_embed.device,
                ),
                vision_embed,
                self.text_class_embedding.to(text_embed.dtype)
                + torch.zeros(
                    text_embed.shape[0],
                    1,
                    text_embed.shape[-1],
                    dtype=text_embed.dtype,
                    device=text_embed.device,
                ),
                text_embed,
            ],
            dim=1,
        )  # shape = [*, grid ** 2 + 1, width]

        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        pooled_vision_output = self.ln_post(x[:, 0, :])
        pooled_text_output = self.ln_post(x[:, self.seq_size + 1, :])

        if (self.vision_projection is not None) and (self.text_projection is not None):
            pooled_vision_output = pooled_vision_output @ self.vision_projection
            pooled_text_output = pooled_text_output @ self.text_projection

        return VisionTextOutput(
            vision_pooled_embeds=pooled_vision_output,
            text_pooled_embeds=pooled_text_output,
            vision_model_output=x[:, : self.seq_size + 1, :],
            text_model_output=x[:, self.seq_size + 1 :, :],
        )
