# coding: utf-8

"""
@File   : models6/hm_control.py
@Author : Songkey
@Email  : songkey@pku.edu.cn
@Date   : 8/14/2024
@Desc   : 
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.modeling_utils import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from einops import rearrange
from .hm_blocks import SKCrossAttention


class HMControlNet(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
            self,
            embedding_channels: int,
            input_channels: int = 3,
            scale_factor: int = 8,
            cross_attention_dim: int = 320,
            block_out_channels: Tuple[int] = (128, 320, 640, 1280),
    ):
        super().__init__()
        self.scale_factor = scale_factor
        self.embedding_channels = embedding_channels
        self.cross_attention_dim = cross_attention_dim
        self.conv_in = nn.Conv2d(input_channels, block_out_channels[0], kernel_size=3, padding=1, bias=False)

        self.exp_embedding = Timesteps(cross_attention_dim, True, 0)
        self.exp_proj = TimestepEmbedding(cross_attention_dim, cross_attention_dim)
        self.face_proj = TimestepEmbedding(1024, cross_attention_dim)

        self.blocks_down = nn.ModuleList([])
        for i in range(1, len(block_out_channels)):
            channel_in = block_out_channels[i-1]
            channel_out = block_out_channels[i]
            self.blocks_down.append(
                SKCrossAttention(
                    channel_in=channel_in,
                    channel_out=channel_out,
                    cross_attention_dim=cross_attention_dim,
                    num_positional_embeddings=64,
                    num_positional_embeddings_hidden=64
                )
            )

    def forward(self, condition, drive_coeff, face_parts):
        bs, _, video_length, h, w = condition.shape

        condition = rearrange(condition, "b c f h w -> (b f) c h w")
        conditioning = F.interpolate(condition,
                                   size=(h // self.scale_factor, w // self.scale_factor),
                                   mode='bilinear',
                                   align_corners=False)
        embedding = self.conv_in(conditioning)

        drive_coeff = rearrange(drive_coeff * 500., "b f c -> (b f c)")
        drive_embedding = self.exp_embedding(drive_coeff).to(dtype=embedding.dtype)
        drive_embedding = rearrange(drive_embedding, "(b f c) d -> b f c d",
                                    b=bs, f=video_length, d=self.cross_attention_dim)

        face_parts = rearrange(face_parts, "b f c d -> (b f) c d")
        face_embedding = self.face_proj(face_parts)

        face_embedding = rearrange(face_embedding, "(b f) c d -> b f c d", f=video_length)

        drive_embedding = torch.cat([face_embedding, drive_embedding], dim=2)
        drive_embedding = rearrange(drive_embedding, "b f c d -> (b f) c d")

        drive_embedding = self.exp_proj(drive_embedding)

        ret_dict = {}
        for idx, block in enumerate(self.blocks_down):
            embedding = block(embedding, drive_embedding)
            ret_dict[f'down_{idx}'] = rearrange(embedding,
                                                     "(b f) c h w -> b c f h w",
                                                     f=video_length)
        return ret_dict