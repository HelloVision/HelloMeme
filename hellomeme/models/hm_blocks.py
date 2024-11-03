# coding: utf-8

"""
@File   : models6/hm_blocks.py
@Author : Songkey
@Email  : songkey@pku.edu.cn
@Date   : 8/14/2024
@Desc   : 
"""

import copy
import torch
import torch.nn as nn
import torch.utils.checkpoint
import torch.nn.functional as F
from typing import Optional, Tuple, Union

from einops import rearrange

from diffusers.models.modeling_utils import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.attention import Attention, FeedForward
from torchvision.models.resnet import BasicBlock
from diffusers.models.embeddings import SinusoidalPositionalEmbedding
from diffusers.utils import is_transformers_available, logging
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel

from diffusers.configuration_utils import FrozenDict
from diffusers.models.attention_processor import (
    AttnProcessor,
    AttnProcessor2_0,
)

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class HMReferenceAdapter(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(self,
                 block_out_channels: Tuple[int] = (320, 640, 1280, 1280),
                 down_block_types: Tuple[str] = (
                         "CrossAttnDownBlock2D",
                         "CrossAttnDownBlock2D",
                         "CrossAttnDownBlock2D",
                         "DownBlock2D",
                 ),
                 up_block_types: Tuple[str] = (
                         "UpBlock2D",
                         "CrossAttnUpBlock2D",
                         "CrossAttnUpBlock2D",
                         "CrossAttnUpBlock2D"),
                 num_attention_heads: Optional[Union[int, Tuple[int]]] = 8,
                 ):
        super().__init__()

        self.reference_modules_down = nn.ModuleList([])

        if isinstance(num_attention_heads, int):
            num_attention_heads = (num_attention_heads,) * len(down_block_types)

        for i, down_block_type in enumerate(down_block_types):
            output_channel = block_out_channels[i]

            self.reference_modules_down.append(
                SKReferenceAttention(
                    in_channels=output_channel,
                    num_attention_heads=num_attention_heads[i],
                )
            )

        self.reference_modules_mid = SKReferenceAttention(
            in_channels=block_out_channels[-1],
            num_attention_heads=num_attention_heads[-1],
        )

        self.reference_modules_up = nn.ModuleList([])

        reversed_block_out_channels = list(reversed(block_out_channels))
        reversed_num_attention_heads = list(reversed(num_attention_heads))

        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]

            if i > 0:
                self.reference_modules_up.append(
                    SKReferenceAttention(
                        in_channels=prev_output_channel,
                        num_attention_heads=reversed_num_attention_heads[i],
                        num_positional_embeddings=64 * 2
                    )
                )

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
        # self.ff = FeedForward(
        #     in_channels,
        #     dropout=0.0,
        #     activation_fn="geglu",
        #     final_dropout=False,
        #     inner_dim=in_channels*2,
        #     bias=True,
        # )
        self.norm = nn.LayerNorm(in_channels, elementwise_affine=norm_elementwise_affine, eps=norm_eps)
        self.proj = zero_module(nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1))

    def forward(self, hidden_states, ref_stats, num_frames):
        h, w = hidden_states.shape[-2:]
        ref_stats = ref_stats.repeat_interleave(num_frames, dim=0)
        cat_stats = torch.cat([hidden_states, ref_stats], dim=-1)

        cat_stats = rearrange(cat_stats.contiguous(), "b c h w -> (b h) w c")
        res1 = self.attn1(self.norm(self.pos_embed(cat_stats)))
        res1 = rearrange(res1[:, :w, :], "(b h) w c -> b c h w", h=h)

        cat_stats2 = torch.cat([res1, ref_stats], dim=-2)
        cat_stats2 = rearrange(cat_stats2.contiguous(), "b c h w -> (b w) h c")
        res2 = self.attn2(self.norm(self.pos_embed(cat_stats2)))

        res2 = rearrange(res2[:, :h, :], "(b w) h c -> b c h w", w=w)
        # res2 = self.ff(res2)

        return hidden_states + self.proj(res2)

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

# https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/controlnet.py
def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module

# https://github.com/huggingface/diffusers/blob/82058a5413ca09561cc5cc236c4abc5eeda7b209/src/diffusers/loaders/ip_adapter.py

if is_transformers_available():
    from diffusers.models.attention_processor import (
        IPAdapterAttnProcessor,
        IPAdapterAttnProcessor2_0,
    )

class CopyWeights(object):
    @classmethod
    def from_unet2d(cls, unet: UNet2DConditionModel):
        # adapted from :https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/unets/unet_motion_model.py

        config = dict(unet.config)

        # Need this for backwards compatibility with UNet2DConditionModel checkpoints
        if not config.get("num_attention_heads"):
            config["num_attention_heads"] = config["attention_head_dim"]

        config = FrozenDict(config)
        model = cls.from_config(config)

        model.conv_in.load_state_dict(unet.conv_in.state_dict())

        model.time_proj.load_state_dict(unet.time_proj.state_dict())
        model.time_embedding.load_state_dict(unet.time_embedding.state_dict())

        if any(
            isinstance(proc, (IPAdapterAttnProcessor, IPAdapterAttnProcessor2_0))
            for proc in unet.attn_processors.values()
        ):
            attn_procs = {}
            for name, processor in unet.attn_processors.items():
                if name.endswith("attn1.processor"):
                    attn_processor_class = (
                        AttnProcessor2_0 if hasattr(F, "scaled_dot_product_attention") else AttnProcessor
                    )
                    attn_procs[name] = attn_processor_class()
                else:
                    attn_processor_class = (
                        IPAdapterAttnProcessor2_0
                        if hasattr(F, "scaled_dot_product_attention")
                        else IPAdapterAttnProcessor
                    )
                    attn_procs[name] = attn_processor_class(
                        hidden_size=processor.hidden_size,
                        cross_attention_dim=processor.cross_attention_dim,
                        scale=processor.scale,
                        num_tokens=processor.num_tokens,
                    )
            for name, processor in model.attn_processors.items():
                if name not in attn_procs:
                    attn_procs[name] = processor.__class__()
            model.set_attn_processor(attn_procs)
            model.config.encoder_hid_dim_type = "ip_image_proj"
            model.encoder_hid_proj = unet.encoder_hid_proj

        for i, down_block in enumerate(unet.down_blocks):
            model.down_blocks[i].resnets.load_state_dict(down_block.resnets.state_dict())
            if hasattr(model.down_blocks[i], "attentions"):
                model.down_blocks[i].attentions.load_state_dict(down_block.attentions.state_dict())
            if model.down_blocks[i].downsamplers:
                model.down_blocks[i].downsamplers.load_state_dict(down_block.downsamplers.state_dict())

        for i, up_block in enumerate(unet.up_blocks):
            model.up_blocks[i].resnets.load_state_dict(up_block.resnets.state_dict())
            if hasattr(model.up_blocks[i], "attentions"):
                model.up_blocks[i].attentions.load_state_dict(up_block.attentions.state_dict())
            if model.up_blocks[i].upsamplers:
                model.up_blocks[i].upsamplers.load_state_dict(up_block.upsamplers.state_dict())

        model.mid_block.resnets.load_state_dict(unet.mid_block.resnets.state_dict())
        model.mid_block.attentions.load_state_dict(unet.mid_block.attentions.state_dict())

        if unet.conv_norm_out is not None:
            model.conv_norm_out.load_state_dict(unet.conv_norm_out.state_dict())
        if unet.conv_act is not None:
            model.conv_act.load_state_dict(unet.conv_act.state_dict())
        model.conv_out.load_state_dict(unet.conv_out.state_dict())

        # ensure that the Motion UNet is the same dtype as the UNet2DConditionModel
        model.to(unet.dtype)

        return model

class InsertReferenceAdapter(object):
    def __init__(self):
        self.reference_modules_down = None
        self.reference_modules_mid = None
        self.reference_modules_up = None

    def insert_reference_adapter(self, adapter: HMReferenceAdapter):
        self.reference_modules_down = copy.deepcopy(adapter.reference_modules_down)
        self.reference_modules_mid = copy.deepcopy(adapter.reference_modules_mid)
        self.reference_modules_up = copy.deepcopy(adapter.reference_modules_up)

