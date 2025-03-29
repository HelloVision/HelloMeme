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
from typing import Optional, Tuple, Union, Dict, List

from einops import rearrange
from diffusers import StableDiffusionImg2ImgPipeline
from diffusers.models.modeling_utils import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.attention import Attention, FeedForward
from torchvision.models.resnet import BasicBlock
from diffusers.models.embeddings import SinusoidalPositionalEmbedding
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel

from diffusers.utils import (logging, is_torch_version,
                             is_peft_available, is_peft_version,
                             is_transformers_available, is_transformers_version,
                             USE_PEFT_BACKEND,
                             scale_lora_layers,
                             unscale_lora_layers,
                             )

from diffusers.configuration_utils import FrozenDict
from diffusers.models.attention_processor import (
    AttnProcessor,
    AttnProcessor2_0,
)

from diffusers.loaders import StableDiffusionLoraLoaderMixin, TextualInversionLoaderMixin
from diffusers.models.lora import adjust_lora_scale_text_encoder

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

_LOW_CPU_MEM_USAGE_DEFAULT_LORA = False
if is_torch_version(">=", "1.9.0"):
    if (
        is_peft_available()
        and is_peft_version(">=", "0.13.1")
        and is_transformers_available()
        and is_transformers_version(">", "4.45.2")
    ):
        _LOW_CPU_MEM_USAGE_DEFAULT_LORA = True

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
                 version='v1'
                 ):
        super().__init__()

        if isinstance(num_attention_heads, int):
            num_attention_heads = (num_attention_heads,) * len(down_block_types)

        if version == 'v1':
            self.reference_modules_down = nn.ModuleList([])
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


class HM3ReferenceAdapter(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(self, block_down_channels: Tuple[int] = (320, 640, 1280, 1280),
                     block_up_channels: Tuple[int] = (1280, 1280, 1280, 640),
                     num_attention_heads: int = 8,
                     use_3d: bool = False):
        super().__init__()

        self.reference_modules_up = nn.ModuleList([])
        for i, in_channels in enumerate(block_up_channels):
            self.reference_modules_up.append(
                SKReferenceAttentionV3(
                    in_channels=in_channels,
                    num_attention_heads=num_attention_heads,
                    num_positional_embeddings=64*2
                )
            )


class HM3MotionAdapter(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(self,  block_down_channels: Tuple[int] = (320, 640, 1280, 1280),
                     block_up_channels: Tuple[int] = (1280, 1280, 1280, 640),
                     num_attention_heads: int = 8,
                     use_3d: bool = True):
        super().__init__()
        blocks_time_embed_dim = 1280
        self.motion_down = nn.ModuleList([])

        for i, in_channels in enumerate(block_down_channels):
            self.motion_down.append(
                SKMotionModule(
                    in_channels=in_channels,
                    num_attention_heads=num_attention_heads,
                    blocks_time_embed_dim=blocks_time_embed_dim,
                )
            )

        self.motion_up = nn.ModuleList([])
        for i, in_channels in enumerate(block_up_channels):
            self.motion_up.append(
                SKMotionModule(
                    in_channels=in_channels,
                    num_attention_heads=num_attention_heads,
                    blocks_time_embed_dim=blocks_time_embed_dim,
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
        self.motion_down = None
        self.motion_up = None

    def insert_reference_adapter(self, adapter):
        if hasattr(adapter, "reference_modules_down"):
            self.reference_modules_down = copy.deepcopy(adapter.reference_modules_down)
        if hasattr(adapter, "reference_modules_mid"):
            self.reference_modules_mid = copy.deepcopy(adapter.reference_modules_mid)
        if hasattr(adapter, "reference_modules_up"):
            self.reference_modules_up = copy.deepcopy(adapter.reference_modules_up)
        if hasattr(adapter, "motion_down"):
            self.motion_down = copy.deepcopy(adapter.motion_down)
        if hasattr(adapter, "motion_up"):
            self.motion_up = copy.deepcopy(adapter.motion_up)


class HMPipeline(StableDiffusionImg2ImgPipeline):
    @torch.no_grad()
    def load_lora_weights_sk(self, unet, pretrained_model_name_or_path_or_dict: Union[str, Dict[str, torch.Tensor]],
                          adapter_name=None, text_encoder=None, lora_scale=1.0, **kwargs):
        if not USE_PEFT_BACKEND:
            raise ValueError("PEFT backend is required for this method.")

        low_cpu_mem_usage = kwargs.pop("low_cpu_mem_usage", _LOW_CPU_MEM_USAGE_DEFAULT_LORA)
        if low_cpu_mem_usage and not is_peft_version(">=", "0.13.1"):
            raise ValueError(
                "`low_cpu_mem_usage=True` is not compatible with this `peft` version. Please update it with `pip install -U peft`."
            )

        # if a dict is passed, copy it instead of modifying it inplace
        if isinstance(pretrained_model_name_or_path_or_dict, dict):
            pretrained_model_name_or_path_or_dict = pretrained_model_name_or_path_or_dict.copy()

        # First, ensure that the checkpoint is a compatible one and can be successfully loaded.
        state_dict, network_alphas = self.lora_state_dict(pretrained_model_name_or_path_or_dict, **kwargs)

        is_correct_format = all("lora" in key for key in state_dict.keys())
        if not is_correct_format:
            raise ValueError("Invalid LoRA checkpoint.")

        self.load_lora_into_unet(
            state_dict,
            network_alphas=network_alphas,
            unet=unet,
            adapter_name=adapter_name,
            _pipeline=self,
            low_cpu_mem_usage=low_cpu_mem_usage,
        )

        if not text_encoder is None:
            self.load_lora_into_text_encoder(
                state_dict,
                network_alphas=network_alphas,
                text_encoder=text_encoder,
                lora_scale=lora_scale,
                adapter_name=adapter_name,
                _pipeline=self,
                low_cpu_mem_usage=low_cpu_mem_usage,
            )

    def encode_prompt_sk(
        self,
        text_encoder,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        lora_scale: Optional[float] = None,
        clip_skip: Optional[int] = None,
    ):

        # set lora scale so that monkey patched LoRA
        # function of text encoder can correctly access it
        if lora_scale is not None and isinstance(self, StableDiffusionLoraLoaderMixin):
            self._lora_scale = lora_scale

            # dynamically adjust the LoRA scale
            if not USE_PEFT_BACKEND:
                adjust_lora_scale_text_encoder(text_encoder , lora_scale)
            else:
                scale_lora_layers(text_encoder , lora_scale)

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            # textual inversion: process multi-vector tokens if necessary
            if isinstance(self, TextualInversionLoaderMixin):
                prompt = self.maybe_convert_prompt(prompt, self.tokenizer)

            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
            ):
                removed_text = self.tokenizer.batch_decode(
                    untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
                )
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                )

            if hasattr(text_encoder .config, "use_attention_mask") and text_encoder .config.use_attention_mask:
                attention_mask = text_inputs.attention_mask.to(device)
            else:
                attention_mask = None

            if clip_skip is None:
                prompt_embeds = text_encoder (text_input_ids.to(device), attention_mask=attention_mask)
                prompt_embeds = prompt_embeds[0]
            else:
                prompt_embeds = text_encoder (
                    text_input_ids.to(device), attention_mask=attention_mask, output_hidden_states=True
                )
                # Access the `hidden_states` first, that contains a tuple of
                # all the hidden states from the encoder layers. Then index into
                # the tuple to access the hidden states from the desired layer.
                prompt_embeds = prompt_embeds[-1][-(clip_skip + 1)]
                # We also need to apply the final LayerNorm here to not mess with the
                # representations. The `last_hidden_states` that we typically use for
                # obtaining the final prompt representations passes through the LayerNorm
                # layer.
                prompt_embeds = text_encoder .text_model.final_layer_norm(prompt_embeds)

        if text_encoder  is not None:
            prompt_embeds_dtype = text_encoder .dtype
        elif self.unet is not None:
            prompt_embeds_dtype = self.unet.dtype
        else:
            prompt_embeds_dtype = prompt_embeds.dtype

        prompt_embeds = prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            # textual inversion: process multi-vector tokens if necessary
            if isinstance(self, TextualInversionLoaderMixin):
                uncond_tokens = self.maybe_convert_prompt(uncond_tokens, self.tokenizer)

            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if hasattr(text_encoder .config, "use_attention_mask") and text_encoder .config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            negative_prompt_embeds = text_encoder (
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embeds[0]

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        if text_encoder  is not None:
            if isinstance(self, StableDiffusionLoraLoaderMixin) and USE_PEFT_BACKEND:
                # Retrieve the original scale by scaling back the LoRA layers
                unscale_lora_layers(text_encoder , lora_scale)

        return prompt_embeds, negative_prompt_embeds
