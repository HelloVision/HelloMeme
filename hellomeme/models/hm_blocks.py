# coding: utf-8

"""
@File   : models6/hm_blocks.py
@Author : Songkey
@Email  : songkey@pku.edu.cn
@Date   : 8/14/2024
@Desc   : 
"""

import torch
import torch.nn as nn
import torch.utils.checkpoint
from typing import Tuple

from einops import rearrange
from diffusers.models.attention import Attention, FeedForward
from diffusers.models.unets.unet_motion_model import AnimateDiffTransformer3D
from torchvision.models.resnet import BasicBlock
from diffusers.models.embeddings import (SinusoidalPositionalEmbedding,
                                         TimestepEmbedding)

from diffusers.utils import logging

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class SKReferenceAttention(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_attention_heads: int = 1,
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        num_positional_embeddings: int = 64*2,
    ):
        super().__init__()
        self.pos_embed = SinusoidalPositionalEmbedding(in_channels, max_seq_length=num_positional_embeddings)
        self.attn1 = Attention(
            query_dim=in_channels,
            heads=num_attention_heads,
            dim_head=in_channels // num_attention_heads,
            dropout=0.0,
            bias=False,
            cross_attention_dim=None,
            upcast_attention=False,
            out_bias=True,
        )
        self.attn2 = Attention(
            query_dim=in_channels,
            heads=num_attention_heads,
            dim_head=in_channels // num_attention_heads,
            dropout=0.0,
            bias=False,
            cross_attention_dim=None,
            upcast_attention=False,
            out_bias=True,
        )
        self.norm = nn.LayerNorm(in_channels, elementwise_affine=norm_elementwise_affine, eps=norm_eps)
        self.proj = zero_module(nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1))

    def forward(self, hidden_states, ref_stats, num_frames):
        h, w = hidden_states.shape[-2:]

        input_states = ref_stats.clone()
        if input_states.shape[0] != hidden_states.shape[0]:
            input_states = input_states.repeat_interleave(num_frames, dim=0)
        cat_stats = torch.cat([hidden_states, input_states], dim=-1)

        cat_stats = rearrange(cat_stats.contiguous(), "b c h w -> (b h) w c")
        res1 = self.attn1(self.norm(self.pos_embed(cat_stats)))
        res1 = rearrange(res1[:, :w, :], "(b h) w c -> b c h w", h=h)

        cat_stats2 = torch.cat([res1, input_states], dim=-2)
        cat_stats2 = rearrange(cat_stats2.contiguous(), "b c h w -> (b w) h c")
        res2 = self.attn2(self.norm(self.pos_embed(cat_stats2)))

        res2 = rearrange(res2[:, :h, :], "(b w) h c -> b c h w", w=w)

        return hidden_states + self.proj(res2)

class SKReferenceAttentionV3(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_attention_heads: int = 1,
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        num_positional_embeddings: int = 64*2,
    ):
        super().__init__()
        self.pos_embed = SinusoidalPositionalEmbedding(in_channels, max_seq_length=num_positional_embeddings)
        self.attn1 = Attention(
            query_dim=in_channels,
            heads=num_attention_heads,
            dim_head=in_channels // num_attention_heads,
            dropout=0.0,
            bias=False,
            cross_attention_dim=None,
            upcast_attention=False,
            out_bias=True,
        )
        self.attn2 = Attention(
            query_dim=in_channels,
            heads=num_attention_heads,
            dim_head=in_channels // num_attention_heads,
            dropout=0.0,
            bias=False,
            cross_attention_dim=None,
            upcast_attention=False,
            out_bias=True,
        )
        self.norm = nn.LayerNorm(in_channels, elementwise_affine=norm_elementwise_affine, eps=norm_eps)
        self.proj = zero_module(nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1))

    def forward(self, hidden_states, ref_stats, num_frames):
        h, w = hidden_states.shape[-2:]

        input_states = ref_stats.clone()
        if input_states.shape[0] != hidden_states.shape[0]:
            input_states = input_states.repeat_interleave(num_frames, dim=0)
        cat_stats = torch.cat([hidden_states, input_states], dim=-1).contiguous()

        cat_stats = rearrange(cat_stats, "b c h w -> (b h) w c")
        res1 = self.attn1(self.norm(self.pos_embed(cat_stats)))
        res1 = rearrange(res1, "(b h) w c -> b c h w", h=h, w=w+w)[:, :, :, :w]

        cat_stats2 = torch.cat([res1.contiguous(), input_states], dim=-2).contiguous()
        cat_stats2 = rearrange(cat_stats2, "b c h w -> (b w) h c")
        res2 = self.attn2(self.norm(self.pos_embed(cat_stats2)))

        res2 = rearrange(res2, "(b w) h c -> b c h w", h=h+h, w=w)[:, :, :h, :]

        return hidden_states + self.proj(res2.contiguous())


class SKReferenceAttentionV5(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_attention_heads: int = 1,
        blocks_time_embed_dim: int = 1280,
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.time_emb_proj = nn.Linear(blocks_time_embed_dim, in_channels)
        self.attn1 = Attention(
            query_dim=in_channels,
            heads=num_attention_heads,
            dim_head=in_channels // num_attention_heads,
            dropout=0.0,
            bias=False,
            cross_attention_dim=None,
            upcast_attention=False,
            out_bias=True,
        )
        self.attn2 = Attention(
            query_dim=in_channels,
            heads=num_attention_heads,
            dim_head=in_channels // num_attention_heads,
            dropout=0.0,
            bias=False,
            cross_attention_dim=None,
            upcast_attention=False,
            out_bias=True,
        )
        self.norm = nn.LayerNorm(in_channels, elementwise_affine=norm_elementwise_affine, eps=norm_eps)
        self.proj = zero_module(nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1))

    def forward(self, hidden_states, ref_stats, num_frames):
        h, w = hidden_states.shape[-2:]

        if ref_stats.shape[0] != hidden_states.shape[0]:
            ref_stats = ref_stats.repeat_interleave(num_frames, dim=0)
        cat_stats = torch.cat([hidden_states, ref_stats], dim=-1).contiguous()

        cat_stats = rearrange(cat_stats, "b c h w -> (b h) w c")
        pose_embed1 = get_posembed_linear(cat_stats.shape[1], cat_stats.shape[2],
                                         dtype=cat_stats.dtype, device=cat_stats.device)
        res1 = self.attn1(self.norm(cat_stats+pose_embed1[None,:,:]))
        res1 = rearrange(res1, "(b h) w c -> b c h w", h=h, w=w+w)[:, :, :, :w]

        cat_stats2 = torch.cat([res1.contiguous(), ref_stats], dim=-2).contiguous()
        cat_stats2 = rearrange(cat_stats2, "b c h w -> (b w) h c")
        pose_embed2 = get_posembed_linear(cat_stats.shape[1], cat_stats.shape[2],
                                         dtype=cat_stats.dtype, device=cat_stats.device)
        res2 = self.attn2(self.norm(cat_stats2+pose_embed2[None,:,:]))

        res2 = rearrange(res2, "(b w) h c -> b c h w", h=h+h, w=w)[:, :, :h, :]

        return hidden_states + self.proj(res2.contiguous())


class SmallUnet(nn.Module):
    def __init__(self, in_channels: int = 3,
                 out_channels: int = 16,
                 cross_attention_dim: int = 1024,
                 mid_channels: Tuple[int] = (320, 640, 1280),
                 temporal_attn: bool = False):
        super(SmallUnet, self).__init__()

        down_channels = [in_channels] + mid_channels
        up_channels = mid_channels[::-1] + [out_channels]
        self.down_blocks = nn.ModuleList([
            BasicBlock(
                inplanes=down_channels[i-1],
                planes=down_channels[i],
                stride=2,
                downsample=nn.Sequential(
                    nn.Conv2d(down_channels[i-1], down_channels[i], kernel_size=1, stride=2, bias=False),
                    nn.InstanceNorm2d(down_channels[i]),
                    nn.SiLU(),
                ),
                norm_layer=nn.InstanceNorm2d,
            ) for i in range(1, len(down_channels))
        ])
        self.up_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(up_channels[i-1], up_channels[i], kernel_size=3, padding=1),
                nn.InstanceNorm2d(up_channels[i]),
                nn.SiLU(),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ) for i in range(1, len(up_channels))
        ])
        self.attn_blocks = nn.ModuleList([
            STKCrossAttention(
                channel_in=c,
                channel_mid=c,
                cross_attention_dim=cross_attention_dim,
                num_positional_embeddings=512,
                num_positional_embeddings_hidden=1024,
                temporal_attn=temporal_attn,
            ) for c in mid_channels
        ])

    def forward(self, x, condition):
        f = x.size(2)
        x = rearrange(x, "b c f h w -> (b f) c h w")
        skips = []
        ret_dict = {}
        for down_block, attn_block in zip(self.down_blocks, self.attn_blocks):
            x = rearrange(down_block(x), "(b f) c h w -> b f c h w", f=f)
            x = attn_block(x, condition)
            x = rearrange(x, "b f c h w -> (b f) c h w")
            skips.append(x)
        skips = skips[::-1][1:]
        ret_dict['feat_0'] = rearrange(x, "(b f) c h w -> b c f h w", f=f)
        for i, block in enumerate(self.up_blocks[:-1]):
            x = block(x) + skips[i]
            ret_dict[f'feat_{i+1}'] = rearrange(x, "(b f) c h w -> b c f h w", f=f)
        x = self.up_blocks[-1](x)
        ret_dict[f'feat_{i+2}'] = rearrange(x, "(b f) c h w -> b c f h w", f=f)
        return ret_dict

class SmallUnetV5(nn.Module):
    def __init__(self, in_channels: int = 3,
                 out_channels: int = 16,
                 cross_attention_dim: int = 1024,
                 mid_channels: Tuple[int] = (320, 640, 1280),
                 temporal_attn: bool = False):
        super().__init__()

        down_channels = [in_channels] + mid_channels
        up_channels = mid_channels[::-1] + [out_channels]
        self.down_blocks = nn.ModuleList([
            BasicBlock(
                inplanes=down_channels[i-1],
                planes=down_channels[i],
                stride=2,
                downsample=nn.Sequential(
                    nn.Conv2d(down_channels[i-1], down_channels[i], kernel_size=1, stride=2, bias=False),
                    nn.InstanceNorm2d(down_channels[i]),
                    nn.SiLU(),
                ),
                norm_layer=nn.InstanceNorm2d,
            ) for i in range(1, len(down_channels))
        ])
        self.up_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(up_channels[i-1], up_channels[i], kernel_size=3, padding=1),
                nn.InstanceNorm2d(up_channels[i]),
                nn.SiLU(),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ) for i in range(1, len(up_channels))
        ])
        self.attn_blocks = nn.ModuleList([
            STKCrossAttentionV5(
                channel_in=c,
                channel_mid=c,
                cross_attention_dim=cross_attention_dim,
                temporal_attn=temporal_attn,
            ) for c in mid_channels
        ])

    def forward(self, x, condition):
        f = x.size(2)
        x = rearrange(x, "b c f h w -> (b f) c h w").contiguous()
        skips = []
        ret_dict = {}
        for down_block, attn_block in zip(self.down_blocks, self.attn_blocks):
            x = rearrange(down_block(x), "(b f) c h w -> b f c h w", f=f).contiguous()
            x = attn_block(x, condition)
            x = rearrange(x, "b f c h w -> (b f) c h w").contiguous()
            skips.append(x)
        skips = skips[::-1][1:]
        ret_dict['feat_0'] = rearrange(x, "(b f) c h w -> b c f h w", f=f).contiguous()
        for i, block in enumerate(self.up_blocks[:-1]):
            x = block(x) + skips[i]
            ret_dict[f'feat_{i+1}'] = rearrange(x, "(b f) c h w -> b c f h w", f=f).contiguous()
        x = self.up_blocks[-1](x)
        ret_dict[f'feat_{i+2}'] = rearrange(x, "(b f) c h w -> b c f h w", f=f).contiguous()
        return ret_dict


def get_posembed_linear(length, dim, dtype, device):

    pos = torch.arange(length).float()
    omega = torch.arange(dim // 2).float()
    omega /= dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = torch.outer(pos, omega)  # (M, D/2), outer product

    emb_sin = torch.sin(out)  # (M, D/2)
    emb_cos = torch.cos(out)  # (M, D/2)

    return torch.concat([emb_sin, emb_cos], dim=1).to(device=device, dtype=dtype)  # (M, D)


class STKAttentionV5(nn.Module):
    def __init__(
        self,
        in_channels: int = 16,
        mid_channels: int = 16,
        num_attention_heads: int = 8,
        time_embed_dim: int = 512,
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        temporal_attn: bool = False,
    ):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)

        if not time_embed_dim is None:
            self.emb_proj = TimestepEmbedding(time_embed_dim, mid_channels)

        self.attnx = Attention(
            query_dim=mid_channels,
            heads=num_attention_heads,
            dim_head=mid_channels // num_attention_heads,
            dropout=0.0,
            bias=False,
            cross_attention_dim=None,
            upcast_attention=False,
            out_bias=True,
        )
        self.attny = Attention(
            query_dim=mid_channels,
            heads=num_attention_heads,
            dim_head=mid_channels // num_attention_heads,
            dropout=0.0,
            bias=False,
            cross_attention_dim=None,
            upcast_attention=False,
            out_bias=True,
        )
        if temporal_attn:
            self.attnt = Attention(
                query_dim=mid_channels,
                heads=num_attention_heads,
                dim_head=mid_channels // num_attention_heads,
                dropout=0.0,
                bias=False,
                cross_attention_dim=None,
                upcast_attention=False,
                out_bias=True,
            )
        self.norm = nn.LayerNorm(mid_channels, elementwise_affine=norm_elementwise_affine, eps=norm_eps)
        self.proj = zero_module(nn.Conv2d(mid_channels, in_channels, kernel_size=3, padding=1))

    def forward(self, hidden_states, ref_stats=None, emb=None):
        b, f, h, w, _ = hidden_states.shape

        hidden_states_in = rearrange(hidden_states, "b f h w c -> (b f) c h w")
        hidden_states_in = self.conv_in(hidden_states_in)

        if ref_stats is not None:
            ref_stats_in = rearrange(ref_stats, "b f h w c -> (b f) c h w")
            ref_stats_in = self.conv_in(ref_stats_in)
            if emb is not None and hasattr(self, 'emb_proj'):
                emb = self.emb_proj(emb)[:,:,None,None]
                emb = emb.repeat_interleave(f, dim=0)
                ref_stats_in = ref_stats_in + emb
        hidden_states_in = rearrange(hidden_states_in, "(b f) c h w -> b f h w c", f=f)

        if ref_stats is not None:
            ref_stats_in = rearrange(ref_stats_in, "(b f) c h w -> b f h w c", f=f)
            cat_stats = torch.cat([hidden_states_in, ref_stats_in], dim=3).contiguous()
        else:
            cat_stats = hidden_states_in

        cat_stats = rearrange(cat_stats, "b f h w c -> (b f h) w c")

        pose_embedx = get_posembed_linear(cat_stats.shape[1], cat_stats.shape[2],
                                         dtype=cat_stats.dtype, device=cat_stats.device)
        resx = self.attnx(self.norm(cat_stats+pose_embedx[None,:,:]))
        resx = rearrange(resx[:,:w], "(b f h) w c -> b f h w c", h=h, f=f)

        if ref_stats is not None:
            cat_stats = torch.cat([resx, ref_stats_in], dim=2).contiguous()
        else:
            cat_stats = resx
        cat_stats = rearrange(cat_stats, "b f h w c -> (b f w) h c")

        pose_embedy = get_posembed_linear(cat_stats.shape[1], cat_stats.shape[2],
                                         dtype=cat_stats.dtype, device=cat_stats.device)
        resy = self.attny(self.norm(cat_stats+pose_embedy[None,:,:]))
        resy = rearrange(resy[:,:h], "(b f w) h c -> b f h w c", f=f, w=w)


        if hasattr(self, 'attnt'):
            if ref_stats is not None:
                cat_stats = torch.cat([resy, ref_stats_in], dim=1).contiguous()
            else:
                cat_stats = resy
            cat_stats = rearrange(cat_stats, "b f h w c -> (b h w) f c")
            pose_embedt =  get_posembed_linear(cat_stats.shape[1], cat_stats.shape[2],
                                         dtype=cat_stats.dtype, device=cat_stats.device)
            rest = self.attnt(self.norm(cat_stats+pose_embedt[None,:,:]))
            rest = rearrange(rest[:,:f], "(b h w) f c -> (b f) c h w", h=h, w=w)
        else:
            rest = rearrange(resy, "b f h w c -> (b f) c h w")

        return hidden_states + rearrange(self.proj(rest), "(b f) c h w -> b f h w c", f=f)

class STKCrossAttentionV5(nn.Module):
    def __init__(
            self,
            channel_in: int,
            channel_mid: int,
            heads: int = 8,
            cross_attention_dim: int = 320,
            norm_elementwise_affine: bool = True,
            norm_eps: float = 1e-5,
            temporal_attn: bool = False,
    ):
        super().__init__()

        self.conv_in = nn.Conv2d(channel_in, channel_mid, kernel_size=3, padding=1)

        self.norm = nn.LayerNorm(channel_mid, elementwise_affine=norm_elementwise_affine, eps=norm_eps)
        self.attn1 = Attention(
            query_dim=channel_mid,
            heads=heads,
            dim_head=channel_mid // heads,
            dropout=0.0,
            bias=False,
            cross_attention_dim=cross_attention_dim,
            upcast_attention=False,
            out_bias=True,
        )

        self.attn2 = Attention(
            query_dim=channel_mid,
            heads=heads,
            dim_head=channel_mid // heads,
            dropout=0.0,
            bias=False,
            cross_attention_dim=cross_attention_dim,
            upcast_attention=False,
            out_bias=True,
        )

        if temporal_attn:
            self.attn3 = Attention(
                query_dim=channel_mid,
                heads=heads,
                dim_head=channel_mid // heads,
                dropout=0.0,
                bias=False,
                cross_attention_dim=cross_attention_dim,
                upcast_attention=False,
                out_bias=True,
            )

        self.ff = FeedForward(
            channel_mid,
            dropout=0.0,
            activation_fn="geglu",
            final_dropout=False,
            inner_dim=channel_mid*16,
            bias=True,
        )

        self.proj = nn.Conv2d(channel_mid, channel_in, kernel_size=3, padding=1)

    def forward(self, input, hidden_stats):
        b, f, _, h, w = input.shape
        x = rearrange(input, "b f c h w -> (b f) c h w")
        x = self.conv_in(x)

        hidden_stats = rearrange(hidden_stats, "b f c d -> (b f) c d")

        pose_embed_hidden = get_posembed_linear(hidden_stats.shape[1], hidden_stats.shape[2],
                                               dtype=hidden_stats.dtype, device=hidden_stats.device)[None,:,:]

        x = rearrange(x, "b c h w -> (b h) w c")
        pose_embed1 = get_posembed_linear(x.shape[1], x.shape[2], dtype=x.dtype, device=x.device)[None,:,:]
        x = self.attn1(self.norm(x + pose_embed1), hidden_stats.repeat_interleave(h, dim=0)+pose_embed_hidden)
        x = rearrange(x, "(b h) w c -> (b w) h c", h=h)
        pose_embed2 = get_posembed_linear(x.shape[1], x.shape[2], dtype=x.dtype, device=x.device)[None,:,:]
        x = self.attn2(self.norm(x + pose_embed2), hidden_stats.repeat_interleave(w, dim=0)+pose_embed_hidden)
        if hasattr(self, 'attn3'):
            x = rearrange(x, "(b f w) h c -> (b w h) f c", w=w, f=f)
            pose_embed3 = get_posembed_linear(x.shape[1], x.shape[2], dtype=x.dtype, device=x.device)[None,:,:]
            x = self.attn3(self.norm(x+pose_embed3), hidden_stats.repeat_interleave(w*h, dim=0)+pose_embed_hidden)
            x = rearrange(x, "(b w h) f c -> (b f) (h w) c", w=w, h=h)
        else:
            x = rearrange(x, "(b f w) h c -> (b f) (h w) c", w=w, f=f)
        x = self.norm(self.ff(x))
        x = rearrange(x, "(b f) (h w) c -> (b f) c h w", w=w, f=f)
        x = self.proj(x)
        return rearrange(x, "(b f) c h w -> b f c h w", f=f)

class STKCrossAttention(nn.Module):
    def __init__(
            self,
            channel_in: int,
            channel_mid: int,
            heads: int = 8,
            cross_attention_dim: int = 320,
            norm_elementwise_affine: bool = True,
            norm_eps: float = 1e-5,
            num_positional_embeddings: int = 64,
            num_positional_embeddings_hidden: int = 64,
            temporal_attn: bool = False,
    ):
        super().__init__()

        self.conv_in = nn.Conv2d(channel_in, channel_mid, kernel_size=3, padding=1)
        self.pos_embed = SinusoidalPositionalEmbedding(channel_mid, max_seq_length=num_positional_embeddings)
        self.pos_embed_hidden = SinusoidalPositionalEmbedding(cross_attention_dim, max_seq_length=num_positional_embeddings_hidden)

        self.norm = nn.LayerNorm(channel_mid, elementwise_affine=norm_elementwise_affine, eps=norm_eps)
        self.attn1 = Attention(
            query_dim=channel_mid,
            heads=heads,
            dim_head=channel_mid // heads,
            dropout=0.0,
            bias=False,
            cross_attention_dim=cross_attention_dim,
            upcast_attention=False,
            out_bias=True,
        )

        self.attn2 = Attention(
            query_dim=channel_mid,
            heads=heads,
            dim_head=channel_mid // heads,
            dropout=0.0,
            bias=False,
            cross_attention_dim=cross_attention_dim,
            upcast_attention=False,
            out_bias=True,
        )

        if temporal_attn:
            self.attn3 = Attention(
                query_dim=channel_mid,
                heads=heads,
                dim_head=channel_mid // heads,
                dropout=0.0,
                bias=False,
                cross_attention_dim=cross_attention_dim,
                upcast_attention=False,
                out_bias=True,
            )

        self.ff = FeedForward(
            channel_mid,
            dropout=0.0,
            activation_fn="geglu",
            final_dropout=False,
            inner_dim=channel_mid*16,
            bias=True,
        )

        self.proj = nn.Conv2d(channel_mid, channel_in, kernel_size=3, padding=1)

    def forward(self, input, hidden_stats):
        b, f, _, h, w = input.shape
        x = rearrange(input, "b f c h w -> (b f) c h w")
        x = self.conv_in(x)

        hidden_stats = rearrange(hidden_stats, "b f c d -> (b f) c d")
        x = rearrange(x, "b c h w -> (b h) w c")
        x = self.attn1(self.norm(self.pos_embed(x)),
                       self.pos_embed_hidden(hidden_stats.repeat_interleave(h, dim=0)))
        x = rearrange(x, "(b h) w c -> (b w) h c", h=h)
        x = self.attn2(self.norm(self.pos_embed(x)),
                       self.pos_embed_hidden(hidden_stats.repeat_interleave(w, dim=0)))
        if hasattr(self, 'attn3'):
            x = rearrange(x, "(b f w) h c -> (b w h) f c", w=w, f=f)
            x = self.attn3(self.norm(self.pos_embed(x)),
                           self.pos_embed_hidden(hidden_stats.repeat_interleave(w*h, dim=0)))
            x = rearrange(x, "(b w h) f c -> (b f) (h w) c", w=w, h=h)
        else:
            x = rearrange(x, "(b f w) h c -> (b f) (h w) c", w=w, f=f)
        x = self.norm(self.ff(x))
        x = rearrange(x, "(b f) (h w) c -> (b f) c h w", w=w, f=f)
        x = self.proj(x)
        return rearrange(x, "(b f) c h w -> b f c h w", f=f)

class SKCrossAttention(nn.Module):
    def __init__(
            self,
            channel_in,
            channel_out,
            heads: int=8,
            cross_attention_dim: int = 320,
            norm_elementwise_affine: bool = True,
            norm_eps: float = 1e-5,
            num_positional_embeddings: int = 64,
            num_positional_embeddings_hidden: int = 64,
    ):
        super().__init__()
        self.conv = BasicBlock(
                    inplanes=channel_in,
                    planes=channel_out,
                    stride=2,
                    downsample=nn.Sequential(
                        nn.Conv2d(channel_in, channel_out, kernel_size=1, stride=2, bias=False),
                        nn.InstanceNorm2d(channel_out),
                        nn.SiLU(),
                    ),
                    norm_layer=nn.InstanceNorm2d,
        )

        self.pos_embed = SinusoidalPositionalEmbedding(channel_out, max_seq_length=num_positional_embeddings)
        self.pos_embed_hidden = SinusoidalPositionalEmbedding(cross_attention_dim, max_seq_length=num_positional_embeddings_hidden)

        self.norm1 = nn.LayerNorm(channel_out, elementwise_affine=norm_elementwise_affine, eps=norm_eps)
        self.attn1 = Attention(
            query_dim=channel_out,
            heads=heads,
            dim_head=channel_out // heads,
            dropout=0.0,
            bias=False,
            cross_attention_dim=cross_attention_dim,
            upcast_attention=False,
            out_bias=True,
        )

        self.norm2 = nn.LayerNorm(channel_out, elementwise_affine=norm_elementwise_affine, eps=norm_eps)
        self.attn2 = Attention(
            query_dim=channel_out,
            heads=heads,
            dim_head=channel_out // heads,
            dropout=0.0,
            bias=False,
            cross_attention_dim=cross_attention_dim,
            upcast_attention=False,
            out_bias=True,
        )

        self.ff = FeedForward(
            channel_out,
            dropout=0.0,
            activation_fn="geglu",
            final_dropout=False,
            inner_dim=channel_out*2,
            bias=True,
        )

        self.proj = zero_module(nn.Conv2d(channel_out, channel_out, kernel_size=3, padding=1))

    def forward(self, input, hidden_stats):
        x = self.conv(input)
        h, w = x.shape[-2:]
        x = rearrange(x, "b c h w -> (b h) w c")
        x = self.attn1(self.norm1(self.pos_embed(x)), self.pos_embed_hidden(hidden_stats.repeat_interleave(h, dim=0).contiguous()))
        x = rearrange(x, "(b h) w c -> (b w) h c", h=h)
        x = self.ff(self.attn2(self.norm2(self.pos_embed(x)), self.pos_embed_hidden(hidden_stats.repeat_interleave(w, dim=0).contiguous())))
        x = rearrange(x, "(b w) h c -> b c h w", w=w)
        x = self.proj(x)
        return x

class SKMotionModule(nn.Module):
    def __init__(
            self,
            in_channels: int,
            num_attention_heads: int = 1,
            norm_elementwise_affine: bool = True,
            norm_eps: float = 1e-5,
            blocks_time_embed_dim: int = 1280,
            num_positional_embeddings_t: int = 77*2,
    ):
        super().__init__()
        self.pos_embed_t = SinusoidalPositionalEmbedding(in_channels, max_seq_length=num_positional_embeddings_t)
        self.time_emb_proj = nn.Linear(blocks_time_embed_dim, in_channels)
        self.temp_attn = Attention(
            query_dim=in_channels,
            heads=num_attention_heads,
            dim_head=in_channels // num_attention_heads,
            dropout=0.0,
            bias=False,
            cross_attention_dim=in_channels,
            upcast_attention=False,
            out_bias=True,
        )
        self.norm = nn.LayerNorm(in_channels, elementwise_affine=norm_elementwise_affine, eps=norm_eps)
        self.ff = FeedForward(
            in_channels,
            dropout=0.0,
            activation_fn="geglu",
            final_dropout=False,
            inner_dim=in_channels * 2,
            bias=True,
        )

        self.proj = zero_module(nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1))

    def forward(self, hidden_states, pad_states, temb, num_frames):
        temb = rearrange(self.time_emb_proj(temb), "(b f) c -> b c f 1 1", f=num_frames)[:,:,:1]

        inputt = rearrange(hidden_states, "(b f) c h w -> b c f h w", f=num_frames)
        inputt = torch.cat([pad_states[:,:,:1], inputt, pad_states[:,:,-1:]], dim=2).contiguous()

        h, w = inputt.shape[-2:]
        inputt = rearrange(inputt+temb, "b c f h w -> (b h w) f c")

        res_temp = self.ff(self.temp_attn(self.norm(self.pos_embed_t(inputt))))

        res_temp = rearrange(res_temp, "(b h w) f c -> b c f h w", h=h, w=w)
        res_temp = rearrange(res_temp[:,:,1:-1], "b c f h w -> (b f) c h w")

        return hidden_states + self.proj(res_temp)


class SKMotionModuleV5(nn.Module):
    def __init__(
            self,
            in_channels: int,
            num_attention_heads: int = 8,
    ):
        super().__init__()

        self.attn1 = AnimateDiffTransformer3D(
            in_channels=in_channels,
            num_attention_heads=num_attention_heads,
            attention_head_dim=in_channels // num_attention_heads,
            dropout=0.0,
            double_self_attention=True,
            positional_embeddings='sinusoidal',
            num_positional_embeddings=32,

        )

        self.attn_pad = AnimateDiffTransformer3D(
            in_channels=in_channels,
            num_attention_heads=num_attention_heads,
            attention_head_dim=in_channels // num_attention_heads,
            dropout=0.0,
            double_self_attention=True,
            positional_embeddings='sinusoidal',
            num_positional_embeddings=32,
        )


    def forward(self, hidden_states, pad_states, temb, num_frames):
        c, h, w = hidden_states.shape[-3:]
        res_temp = hidden_states

        if not pad_states is None:
            res_temp = rearrange(res_temp, "(b f) c h w -> b c f h w", f=num_frames)
            res_temp = torch.cat([pad_states[:,:,:1], res_temp, pad_states[:,:,-1:]], dim=2).contiguous()
            res_temp = rearrange(res_temp, "b c f h w -> (b f) c h w")
            res_temp = self.attn_pad(res_temp, num_frames=num_frames+2)
            res_temp = rearrange(res_temp, "(b f) c h w -> b c f h w", f=num_frames+2).contiguous()
            res_temp = rearrange(res_temp[:,:,1:-1], "b c f h w -> (b f) c h w", h=h, w=w) + hidden_states

        res_temp = self.attn1(res_temp, num_frames=num_frames)
        return res_temp

# https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/controlnet.py
def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module