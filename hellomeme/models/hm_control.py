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
from .hm_blocks import SKCrossAttention, SmallUnet

class HMControlNetBase(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
            self,
            in_channels: int = 3,
            cross_attention_dim: int = 2048,
    ):
        super().__init__()

        self.emo_embedding = Timesteps(1024, True, 0)
        self.emo_pre_proj = TimestepEmbedding(548, 16)
        self.emo_proj = TimestepEmbedding(1024, cross_attention_dim)

        self.exp_embedding = Timesteps(1024, True, 0)
        self.exp_pre_proj = TimestepEmbedding(51, 16)
        self.face_pre_proj = TimestepEmbedding(3, 16)
        self.exp_proj = TimestepEmbedding(1024, cross_attention_dim)
        self.face_proj = TimestepEmbedding(1024, cross_attention_dim)

        self.ctrl_attn = SmallUnet(in_channels=in_channels, out_channels=320,
                                   mid_channels=[320, 640, 1280],
                                   cross_attention_dim=cross_attention_dim,
                                   temporal_attn=False)

    def forward(self, condition, drive_coeff=None, face_parts=None, emo_embedding=None):
        bs, video_length = condition.size(0), condition.size(2)
        condition = rearrange(condition, "b c f h w -> (b f) c h w")
        condition = F.interpolate(condition, size=(condition.size(2)//8, condition.size(3)//8), mode='bilinear', align_corners=False)
        condition = rearrange(condition, "(b f) c h w -> b c f h w", f=video_length)

        if drive_coeff is None:
            emo_coeff = rearrange(emo_embedding * 20., "b f c -> (b f c)")
            emo_embedding = self.emo_embedding(emo_coeff)
            emo_embedding = rearrange(emo_embedding, "(b f c) d -> (b f) d c", b=bs, f=video_length)
            emo_embedding = self.emo_pre_proj(emo_embedding.to(dtype=condition.dtype))
            emo_embedding = rearrange(emo_embedding, "(b f) d c -> (b f) c d", f=video_length)
            drive_embedding = self.emo_proj(emo_embedding)
        else:
            drive_coeff = rearrange(drive_coeff * 500., "b f c -> (b f c)")
            exp_embedding = self.exp_embedding(drive_coeff)
            exp_embedding = rearrange(exp_embedding, "(b f c) d -> (b f) d c", b=bs, f=video_length)
            exp_embedding = self.exp_pre_proj(exp_embedding.to(dtype=condition.dtype))
            exp_embedding = rearrange(exp_embedding, "(b f) d c -> (b f) c d", f=video_length)

            face_parts = rearrange(face_parts, "b f c d -> (b f) d c")
            face_embedding = self.face_pre_proj(face_parts)
            face_embedding = rearrange(face_embedding, "(b f) d c -> (b f) c d", f=video_length)

            drive_embedding = self.exp_proj(exp_embedding) + self.face_proj(face_embedding)

        drive_embedding = rearrange(drive_embedding, "(b f) c d -> b f c d", f=video_length)
        return self.ctrl_attn(condition, drive_embedding)


class HM4SD15ControlProj(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
            self,
    ):
        super().__init__()
        self.map640_to_640 = nn.Conv2d(640, 640, kernel_size=3, padding=1, bias=False)
        self.map640_to_1280 = nn.Conv2d(640, 1280, kernel_size=3, padding=1, bias=False)
        self.map320_to_320 = nn.Conv2d(320, 320, kernel_size=3, padding=1, bias=False)
        self.map320_to_640 = nn.Conv2d(320, 640, kernel_size=3, padding=1, bias=False)
        self.map320_to_1280 = nn.Conv2d(320, 1280, kernel_size=3, padding=1, bias=False)
        self.map1280_to_1280 = nn.Conv2d(1280, 1280, kernel_size=3, padding=1, bias=False)
        self.map1280_to_1280_2 = nn.Conv2d(1280, 1280, kernel_size=3, padding=1, bias=False)
        self.map1280_to_1280_3 = nn.Conv2d(1280, 1280, kernel_size=3, padding=1, bias=False)

    def forward(self, control_dict):
        video_length = control_dict['feat_0'].size(2)
        ret_dict = dict(
            up3_0=rearrange(
                self.map1280_to_1280(
                    rearrange(control_dict['feat_0'], "b c f h w -> (b f) c h w"),
                ), "(b f) c h w -> b c f h w", f=video_length),
            up3_1=rearrange(
                self.map640_to_1280(
                    rearrange(control_dict['feat_1'], "b c f h w -> (b f) c h w"),
                ), "(b f) c h w -> b c f h w", f=video_length),
            up3_2=rearrange(
                self.map320_to_1280(
                    rearrange(control_dict['feat_2'], "b c f h w -> (b f) c h w"),
                ), "(b f) c h w -> b c f h w", f=video_length),
            up3_3=rearrange(
                self.map320_to_640(
                    rearrange(control_dict['feat_3'], "b c f h w -> (b f) c h w"),
                ), "(b f) c h w -> b c f h w", f=video_length),
            down3_0=rearrange(
                self.map320_to_320(
                    rearrange(control_dict['feat_2'], "b c f h w -> (b f) c h w"),
                ), "(b f) c h w -> b c f h w", f=video_length),
            down3_1=rearrange(
                self.map640_to_640(
                    rearrange(control_dict['feat_1'], "b c f h w -> (b f) c h w"),
                ), "(b f) c h w -> b c f h w", f=video_length),
            down3_2=rearrange(
                self.map1280_to_1280_2(
                    rearrange(control_dict['feat_0'], "b c f h w -> (b f) c h w"),
                ), "(b f) c h w -> b c f h w", f=video_length),
            down3_3=rearrange(
                self.map1280_to_1280_3(
                    rearrange(control_dict['feat_0'], "b c f h w -> (b f) c h w"),
                ), "(b f) c h w -> b c f h w", f=video_length),
        )
        return ret_dict

class HMControlNet(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
            self,
            embedding_channels: int = 1280,
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

class HMControlNet2(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
            self,
            embedding_channels: int = 1280,
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
        self.emo_proj = TimestepEmbedding(cross_attention_dim, cross_attention_dim)

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
                    num_positional_embeddings_hidden=1024
                )
            )

    def forward(self, condition, emo_embedding):
        bs, _, video_length, h, w = condition.shape

        condition = rearrange(condition, "b c f h w -> (b f) c h w")
        condition = F.interpolate(condition,
                                   size=(h // self.scale_factor, w // self.scale_factor),
                                   mode='bilinear',
                                   align_corners=False)
        embedding = self.conv_in(condition)

        drive_coeff = rearrange(emo_embedding * 20., "b f c -> (b f c)")
        drive_embedding = self.exp_embedding(drive_coeff).to(dtype=embedding.dtype)
        drive_embedding = rearrange(drive_embedding, "(b f c) d -> b f c d",
                                    b=bs, f=video_length, d=self.cross_attention_dim)
        drive_embedding = rearrange(drive_embedding, "b f c d -> (b f) c d")
        drive_embedding = self.emo_proj(drive_embedding)

        ret_dict = {}
        for idx, block in enumerate(self.blocks_down):
            embedding = block(embedding, drive_embedding)
            ret_dict[f'down2_{idx}'] = rearrange(embedding,
                                                     "(b f) c h w -> b c f h w",
                                                     f=video_length)
        return ret_dict

class HMV2ControlNet(ModelMixin, ConfigMixin):

    def __init__(
            self,
            embedding_channels: int = 1280,
            input_channels: int = 3,
            scale_factor: int = 4,
            cross_attention_dim: int = 320,
            block_out_channels: Tuple[int] = (128, 640, 1280, 1280, 1280),
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
            channel_in = block_out_channels[i - 1]
            channel_out = block_out_channels[i]
            self.blocks_down.append(
                SKCrossAttention(
                    channel_in=channel_in,
                    channel_out=channel_out,
                    cross_attention_dim=cross_attention_dim,
                    num_positional_embeddings=64,
                    num_positional_embeddings_hidden=1024
                )
            )

    def forward(self, condition, drive_coeff, face_parts):
        bs, _, video_length, h, w = condition.shape

        conditioning = rearrange(condition, "b c f h w -> (b f) c h w")
        conditioning = F.interpolate(conditioning,
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
            ret_dict[f'up_v2_{len(self.blocks_down) - idx - 1}'] = rearrange(embedding,
                                                                          "(b f) c h w -> b c f h w",
                                                                          f=video_length)
        return ret_dict


class HMV3ControlNet(ModelMixin, ConfigMixin):
    def __init__(
            self,
            embedding_channels: int = 1280,
            input_channels: int = 3,
            scale_factor: int = 4,
            cross_attention_dim: int = 320,
            block_out_channels: Tuple[int] = (128, 640, 1280, 1280, 1280),
    ):
        super().__init__()
        self.scale_factor = scale_factor
        self.embedding_channels = embedding_channels
        self.cross_attention_dim = cross_attention_dim
        self.conv_in = nn.Conv2d(input_channels, block_out_channels[0], kernel_size=3, padding=1, bias=False)
        self.conv_up2_down0 = nn.Conv2d(1280, 320, kernel_size=1, padding=0, bias=False)
        self.conv_up1_down1 = nn.Conv2d(1280, 640, kernel_size=1, padding=0, bias=False)

        self.emo_embedding = Timesteps(cross_attention_dim, True, 0)
        self.exp_embedding = Timesteps(cross_attention_dim, True, 0)

        self.exp_proj = TimestepEmbedding(cross_attention_dim, cross_attention_dim)
        self.face_proj = TimestepEmbedding(1024, cross_attention_dim)
        self.emo_proj = TimestepEmbedding(cross_attention_dim, cross_attention_dim)

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
                    num_positional_embeddings_hidden=1024
                )
            )

    def forward(self, condition, drive_coeff=None, face_parts=None, emo_embedding=None):
        bs, _, video_length, h, w = condition.shape

        conditioning = rearrange(condition, "b c f h w -> (b f) c h w")
        conditioning = F.interpolate(conditioning,
                                   size=(h // self.scale_factor, w // self.scale_factor),
                                   mode='bilinear',
                                   align_corners=False)
        embedding = self.conv_in(conditioning)

        if drive_coeff is None or face_parts is None:
            emo_coeff = rearrange(emo_embedding * 20., "b f c -> (b f c)")
            emo_embedding = self.emo_embedding(emo_coeff).to(dtype=embedding.dtype)
            emo_embedding = rearrange(emo_embedding, "(b f c) d -> b f c d",
                                        b=bs, f=video_length, d=self.cross_attention_dim)
            emo_embedding = rearrange(emo_embedding, "b f c d -> (b f) c d")
            drive_embedding = self.emo_proj(emo_embedding)
        else:
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
            up_idx = len(self.blocks_down)-idx-1
            ret_dict[f'up3_{up_idx}'] = rearrange(embedding,
                                                 "(b f) c h w -> b c f h w",
                                                 f=video_length)
            if up_idx == 2:
                ret_dict[f'down3_0'] = rearrange(self.conv_up2_down0(embedding),
                                                "(b f) c h w -> b c f h w",
                                                f=video_length)
            if up_idx == 1:
                ret_dict[f'down3_1'] = rearrange(self.conv_up1_down1(embedding),
                                                "(b f) c h w -> b c f h w",
                                                f=video_length)
            if up_idx == 0:
                ret_dict[f'down3_2'] = ret_dict[f'up3_0']
                ret_dict[f'down3_3'] = ret_dict[f'up3_0']
        return ret_dict


class HMV2ControlNet2(ModelMixin, ConfigMixin):
    def __init__(
            self,
            embedding_channels: int = 1280,
            input_channels: int = 3,
            scale_factor: int = 4,
            cross_attention_dim: int = 320,
            block_out_channels: Tuple[int] = (128, 640, 1280, 1280, 1280),
    ):
        super().__init__()
        self.scale_factor = scale_factor
        self.embedding_channels = embedding_channels
        self.cross_attention_dim = cross_attention_dim
        self.conv_in = nn.Conv2d(input_channels, block_out_channels[0], kernel_size=3, padding=1, bias=False)

        self.exp_embedding = Timesteps(cross_attention_dim, True, 0)
        self.emo_proj = TimestepEmbedding(cross_attention_dim, cross_attention_dim)
        # self.face_proj = TimestepEmbedding(1024, cross_attention_dim)

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
                    num_positional_embeddings_hidden=1024
                )
            )

    def forward(self, condition, emo_embedding):
        bs, _, video_length, h, w = condition.shape

        conditioning = rearrange(condition, "b c f h w -> (b f) c h w")
        conditioning = F.interpolate(conditioning,
                                   size=(h // self.scale_factor, w // self.scale_factor),
                                   mode='bilinear',
                                   align_corners=False)
        embedding = self.conv_in(conditioning)

        drive_coeff = rearrange(emo_embedding * 20., "b f c -> (b f c)")
        drive_embedding = self.exp_embedding(drive_coeff).to(dtype=embedding.dtype)
        drive_embedding = rearrange(drive_embedding, "(b f c) d -> b f c d",
                                    b=bs, f=video_length, d=self.cross_attention_dim)
        drive_embedding = rearrange(drive_embedding, "b f c d -> (b f) c d")
        drive_embedding = self.emo_proj(drive_embedding)

        ret_dict = {}
        for idx, block in enumerate(self.blocks_down):
            embedding = block(embedding, drive_embedding)
            ret_dict[f'up2_v2_{len(self.blocks_down)-idx-1}'] = rearrange(embedding,
                                                     "(b f) c h w -> b c f h w",
                                                     f=video_length)
        return ret_dict