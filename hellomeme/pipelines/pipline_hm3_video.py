# coding: utf-8

"""
@File   : hm_pipline_image.py
@Author : Songkey
@Email  : songkey@pku.edu.cn
@Date   : 1/3/2025
@Desc   :
adapted from: https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_img2img.py
"""

import copy
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from einops import rearrange

from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.image_processor import PipelineImageInput
from diffusers.utils import deprecate

from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import retrieve_timesteps, retrieve_latents
from diffusers import MotionAdapter, EulerDiscreteScheduler # DPMSolverMultistepScheduler
from diffusers.utils.torch_utils import randn_tensor

from ..models import (HM3Denoising3D, HM3DenoisingMotion, HM3MotionAdapter,
                      HMV3ControlNet, HM3ReferenceAdapter, HMPipeline, HMControlNetBase, HM4SD15ControlProj)

class HM3VideoPipeline(HMPipeline):
    def caryomitosis(self, version='v3', modelscope=False, **kwargs):
        if hasattr(self, "unet_ref"):
            del self.unet_ref
        self.unet_ref = HM3Denoising3D.from_unet2d(self.unet)
        self.unet_ref.cpu()

        if hasattr(self, "unet_pre"):
            del self.unet_pre
        self.unet_pre = HM3Denoising3D.from_unet2d(self.unet)
        self.unet_pre.cpu()

        if modelscope:
            from modelscope import snapshot_download
            if version == 'v3':
                hm_animatediff_dir = snapshot_download('songkey/hm3_animatediff')
            else:
                hm_animatediff_dir = snapshot_download('songkey/hm4_animatediff')
        else:
            if version == 'v3':
                hm_animatediff_dir = 'songkey/hm3_animatediff'
            else:
                hm_animatediff_dir = 'songkey/hm4_animatediff'

        self.num_frames = 12

        adapter = MotionAdapter.from_pretrained(hm_animatediff_dir, torch_dtype=torch.float16)

        unet = HM3DenoisingMotion.from_unet2d(unet=self.unet, motion_adapter=adapter, load_weights=True)
        del self.unet
        self.unet = unet

        self.vae.cpu()
        self.vae_decode = copy.deepcopy(self.vae)
        self.text_encoder.cpu()
        self.text_encoder_ref = copy.deepcopy(self.text_encoder)
        self.safety_checker.cpu()

    def insert_hm_modules(self, version='v3', dtype=torch.float16, modelscope=False):
        self.version = version

        if modelscope:
            from modelscope import snapshot_download
            if version == 'v3':
                hm_reference_dir = snapshot_download('songkey/hm3_reference')
                hm_control_dir = snapshot_download('songkey/hm3_control_mix')
                hm_motion_dir = snapshot_download('songkey/hm3_motion')
            else:
                hm_reference_dir = snapshot_download('songkey/hm4_reference')
                hm_control_dir = snapshot_download('songkey/hm_control_base')
                hm_control_proj_dir = snapshot_download('songkey/hm4_control_proj')
                hm_motion_dir = snapshot_download('songkey/hm4_motion')
        else:
            if version == 'v3':
                hm_reference_dir = 'songkey/hm3_reference'
                hm_control_dir = 'songkey/hm3_control_mix'
                hm_motion_dir = 'songkey/hm3_motion'
            else:
                hm_reference_dir = 'songkey/hm4_reference'
                hm_control_dir = 'songkey/hm_control_base'
                hm_control_proj_dir = 'songkey/hm4_control_proj'
                hm_motion_dir = 'songkey/hm4_motion'

        hm_adapter = HM3ReferenceAdapter.from_pretrained(hm_reference_dir)
        motion_adapter = HM3MotionAdapter.from_pretrained(hm_motion_dir)

        if isinstance(self.unet, HM3DenoisingMotion):
            self.unet.insert_reference_adapter(hm_adapter)
            self.unet.insert_reference_adapter(motion_adapter)
            self.unet.to(device='cpu', dtype=dtype).eval()

        if hasattr(self, "unet_pre"):
            self.unet_pre.insert_reference_adapter(hm_adapter)
            self.unet_pre.insert_reference_adapter(motion_adapter)
            self.unet_pre.to(device='cpu', dtype=dtype).eval()

        if hasattr(self, "unet_ref"):
            self.unet_ref.to(device='cpu', dtype=dtype).eval()

        if hasattr(self, "mp_control"):
            del self.mp_control
        if hasattr(self, "mp_control_proj"):
            del self.mp_control_proj

        if version == 'v3':
            self.mp_control = HMV3ControlNet.from_pretrained(hm_control_dir)
        else:
            self.mp_control = HMControlNetBase.from_pretrained(hm_control_dir)
            self.mp_control_proj = HM4SD15ControlProj.from_pretrained(hm_control_proj_dir)

            self.mp_control_proj.to(device='cpu', dtype=dtype).eval()
        self.mp_control.to(device='cpu', dtype=dtype).eval()

        self.vae.to(device='cpu', dtype=dtype).eval()
        self.vae_decode.to(device='cpu', dtype=dtype).eval()
        self.text_encoder.to(device='cpu', dtype=dtype).eval()

    @torch.no_grad()
    def gen_ref_cache(self, latent, prompt_embeds, added_cond_kwargs, device, add_axis=False):
        self.unet_ref.to(device=device)
        latent = latent.to(device=device, dtype=prompt_embeds.dtype)
        latent_model_input = torch.cat([torch.zeros_like(latent), latent]) if self.do_classifier_free_guidance else latent
        cached_res = self.unet_ref(
            latent_model_input.unsqueeze(2),
            0,
            encoder_hidden_states=prompt_embeds,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False,
        )[1]
        self.unet_ref.cpu()

        if add_axis:
            for k, v in cached_res.items():
                cached_res[k] = rearrange(v, 'b c h w -> b c 1 h w')

        return dicts_to_device([cached_res], device='cpu')[0]

    @torch.no_grad()
    def gen_video(self, indexes, pad_dict, base_noise, depth, control_dict, step, more_params):
        device = more_params['device']
        timesteps = more_params['timesteps']
        timestep_cond = more_params['timestep_cond']
        sigmas = more_params['sigmas']
        prompt_embeds = more_params['prompt_embeds']
        cached_res = more_params['cached_res']
        added_cond_kwargs = more_params['added_cond_kwargs']
        extra_step_kwargs = more_params['extra_step_kwargs']

        num_indexes = len(indexes)
        if num_indexes <= self.num_frames:
            paded_indexes = indexes

            paded_indexes_noise = list(range(len(indexes)))

            paded_indexes = copy.deepcopy(paded_indexes)
            margin_indexes = [indexes[0], indexes[-1]]

            scheduler = EulerDiscreteScheduler(
                num_train_timesteps=1000,
                beta_start=0.00085,
                beta_end=0.012,
                beta_schedule="scaled_linear",
                # algorithm_type="sde-dpmsolver++",
            )

            tmp_timesteps, _ = retrieve_timesteps(scheduler, step, device, timesteps, sigmas)

            margin_latent = base_noise[:,:,[0, 0]] * scheduler.init_noise_sigma
            margin_latent = margin_latent.to(device=device, dtype=prompt_embeds.dtype)
            control_latent = cat_dicts([control_dict[x] for x in margin_indexes], dim=2)

            self.unet_pre.to(device=device)

            with self.progress_bar(total=len(tmp_timesteps)) as progress_bar:
                for i, t in enumerate(tmp_timesteps):
                    progress_bar.set_description(f"D{depth} GEN Margin")
                    progress_bar.update()
                    latent_model_input = torch.cat([margin_latent] * 2, dim=0) if \
                        self.do_classifier_free_guidance else margin_latent
                    latent_model_input = scheduler.scale_model_input(latent_model_input, t)
                    latent_model_input = latent_model_input.to(device=device, dtype=prompt_embeds.dtype)
                    noise_pred = self.unet_pre(
                        latent_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        reference_hidden_states=dicts_to_device([cached_res], device=device)[0],
                        control_hidden_states=dicts_to_device([control_latent], device=device)[0],
                        timestep_cond=timestep_cond,
                        cross_attention_kwargs=self.cross_attention_kwargs,
                        added_cond_kwargs=added_cond_kwargs,
                        return_dict=False,
                    )[0]

                    if self.do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
                    margin_latent = scheduler.step(noise_pred, t, margin_latent,
                                                **extra_step_kwargs, return_dict=False)[0]

            self.unet_pre.cpu()

            if not margin_indexes[0] in pad_dict:
                pad_dict[margin_indexes[0]] = self.gen_ref_cache(margin_latent[:,:,0],
                                                           prompt_embeds, added_cond_kwargs, device, add_axis=True)
            if not margin_indexes[1] in pad_dict:
                pad_dict[margin_indexes[1]] = self.gen_ref_cache(margin_latent[:,:,-1],
                                                           prompt_embeds, added_cond_kwargs, device, add_axis=True)

            drive_idx_chunks = [paded_indexes]
            drive_idx_chunks_noise = [paded_indexes_noise]
        else:
            stride = self.num_frames - 1
            paded_indexes = indexes
            paded_indexes = copy.deepcopy(paded_indexes)

            num_paded_indexes = len(paded_indexes)
            paded_indexes_noise = list(range(num_paded_indexes))

            drive_idx_chunks = [paded_indexes[i:min(i + self.num_frames, num_paded_indexes)] for i in
                                range(0, num_paded_indexes, stride)]
            drive_idx_chunks_noise = [paded_indexes_noise[i:min(i + self.num_frames, num_paded_indexes)] for i in
                                range(0, num_paded_indexes, stride)]

            if len(drive_idx_chunks[-1]) == 1:
                drive_idx_chunks[-1] = drive_idx_chunks[:-1]
                drive_idx_chunks_noise[-1] = drive_idx_chunks_noise[:-1]

            margin_indexes = set()
            for chunk in drive_idx_chunks:
                margin_indexes.add(chunk[0])
                margin_indexes.add(chunk[-1])
            margin_indexes = sorted(list(margin_indexes))

            self.gen_video(margin_indexes, pad_dict, base_noise, depth+1, control_dict, step, more_params)

        ## generate latent
        scheduler = EulerDiscreteScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            # algorithm_type="sde-dpmsolver++",
        )
        timesteps, _ = retrieve_timesteps(scheduler, step, device, timesteps, sigmas)

        N, C, _, H, W = base_noise.shape

        latents = base_noise[:,:,:len(paded_indexes)] * scheduler.init_noise_sigma

        self.unet.to(device=device)
        with self.progress_bar(total=len(timesteps)) as progress_bar:
            for i, t in enumerate(timesteps):
                progress_bar.set_description(f"D{depth} GEN Frames")
                progress_bar.update()

                noise_pred = torch.zeros_like(latents)
                noise_pred = torch.cat([noise_pred, noise_pred], dim=0) if \
                    self.do_classifier_free_guidance else noise_pred
                noise_pred_counter = torch.zeros([1, 1, noise_pred.size(2), 1, 1], dtype=prompt_embeds.dtype).cpu()

                latent_model_input_all = torch.cat([latents] * 2, dim=0) if \
                    self.do_classifier_free_guidance else latents

                latent_model_input_all = scheduler.scale_model_input(latent_model_input_all, t)
                for chunk, chunk_noise in zip(drive_idx_chunks, drive_idx_chunks_noise):
                    if self.interrupt:
                        continue

                    control_latent = cat_dicts([control_dict[ix] for ix in chunk], dim=2)
                    pad_latent = cat_dicts([pad_dict[chunk[0]], pad_dict[chunk[-1]]], dim=2)

                    latent_model_input = latent_model_input_all[:, :, chunk_noise].to(device=device)

                    noise_pred_chunk = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        reference_hidden_states=dicts_to_device([cached_res], device=device)[0],
                        control_hidden_states=dicts_to_device([control_latent], device=device)[0],
                        motion_pad_hidden_states=dicts_to_device([pad_latent], device=device)[0],
                        use_motion=True,
                        timestep_cond=timestep_cond,
                        cross_attention_kwargs=self.cross_attention_kwargs,
                        added_cond_kwargs=added_cond_kwargs,
                        return_dict=False,
                    )[0].cpu()

                    noise_pred[:, :, chunk_noise] = noise_pred[:, :, chunk_noise] + noise_pred_chunk
                    noise_pred_counter[:, :, chunk_noise] = noise_pred_counter[:, :, chunk_noise] + 1

                noise_pred = noise_pred / noise_pred_counter
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
                latents = scheduler.step(noise_pred.cpu(), t, latents,
                                              **extra_step_kwargs, return_dict=False)[0]
        self.unet.cpu()

        if depth == 0:
            return latents[:,:,:num_indexes]
        else:
            for chunk, chunk_noise in zip(drive_idx_chunks, drive_idx_chunks_noise):
                for idx, idx_noise in zip(chunk, chunk_noise):
                    if not idx in pad_dict:
                        pad_dict[idx] = self.gen_ref_cache(latents[:,:,idx_noise], prompt_embeds,
                                                           added_cond_kwargs, device, add_axis=True)

    @torch.no_grad()
    def __call__(
            self,
            prompt: Union[str, List[str]] = None,
            image: PipelineImageInput = None,
            chunk_overlap: int = 4,
            drive_params: Dict[str, Any] = None,
            strength: float = 0.8,
            num_inference_steps: Optional[int] = 50,
            timesteps: List[int] = None,
            sigmas: List[float] = None,
            guidance_scale: Optional[float] = 7.5,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            eta: Optional[float] = 0.0,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            prompt_embeds: Optional[torch.Tensor] = None,
            negative_prompt_embeds: Optional[torch.Tensor] = None,
            ip_adapter_image: Optional[PipelineImageInput] = None,
            ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
            output_type: Optional[str] = "pil",
            device: Optional[str] = "cpu",
            return_dict: bool = True,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            clip_skip: int = None,
            callback_on_step_end: Optional[
                Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
            ] = None,
            callback_on_step_end_tensor_inputs: List[str] = ["latents"],
            **kwargs,
    ):
        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)
        num_images_per_prompt = 1

        if callback is not None:
            deprecate(
                "callback",
                "1.0.0",
                "Passing `callback` as an input argument to `__call__` is deprecated, consider use `callback_on_step_end`",
            )
        if callback_steps is not None:
            deprecate(
                "callback_steps",
                "1.0.0",
                "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider use `callback_on_step_end`",
            )

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            strength,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            ip_adapter_image,
            ip_adapter_image_embeds,
            callback_on_step_end_tensor_inputs,
        )

        self._guidance_scale = guidance_scale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # device = self.device

        # 3. Encode input prompt
        text_encoder_lora_scale = (
            self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
        )
        self.text_encoder_ref.to(device=device)
        prompt_embeds_ref, negative_prompt_embeds_ref = self.encode_prompt_sk(
            self.text_encoder_ref,
            prompt,
            device,
            num_images_per_prompt,
            self.do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
            clip_skip=self.clip_skip,
        )
        self.text_encoder_ref.cpu()

        self.text_encoder.to(device=device)
        prompt_embeds, negative_prompt_embeds = self.encode_prompt_sk(
            self.text_encoder,
            prompt,
            device,
            num_images_per_prompt,
            self.do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
            clip_skip=self.clip_skip,
        )
        self.text_encoder.cpu()

        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
            prompt_embeds_ref = torch.cat([negative_prompt_embeds_ref, prompt_embeds_ref])

        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
                self.do_classifier_free_guidance,
            )

        # 4. Preprocess image
        image = self.image_processor.preprocess(image).to(device=device, dtype=prompt_embeds.dtype)

        raw_video_len = drive_params['condition'].size(2)

        # 6. Prepare latent variables
        self.vae.to(device=device)
        ref_latents = [
            retrieve_latents(self.vae.encode(image[i: i + 1].to(device=device)), generator=generator)
            for i in range(batch_size)
        ]
        self.vae.cpu()

        ref_latents = torch.cat(ref_latents, dim=0)
        ref_latents = self.vae.config.scaling_factor * ref_latents
        c, h, w = ref_latents.shape[1:]

        # 7. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7.1 Add image embeds for IP-Adapter
        added_cond_kwargs = (
            {"image_embeds": image_embeds}
            if ip_adapter_image is not None or ip_adapter_image_embeds is not None
            else None
        )

        # 7.2 Optionally get Guidance Scale Embedding
        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=prompt_embeds.dtype)

        input_noise = randn_tensor([batch_size, c, raw_video_len + self.num_frames + 1, h, w],
                                   generator=generator, dtype=prompt_embeds.dtype)

        cached_res = self.gen_ref_cache(ref_latents, prompt_embeds_ref, added_cond_kwargs, device)

        control_latents = []
        control_latent1_neg, control_latent2_neg = None, None
        self.mp_control.to(device=device)
        if self.version == 'v4':
            self.mp_control_proj.to(device=device)
        with self.progress_bar(total=raw_video_len) as progress_bar:
            for idx in range(raw_video_len):
                progress_bar.set_description(f"GEN stage1")
                progress_bar.update()

                condition = drive_params['condition'][:, :, idx:idx+1].clone().to(device=device)

                control_latent_input = {}
                if 'drive_coeff' in drive_params:
                    drive_coeff = drive_params['drive_coeff'][:, idx:idx+1].clone().to(device=device)
                    face_parts = drive_params['face_parts'][:, idx:idx+1].clone().to(device=device)

                    if control_latent1_neg is None and self.do_classifier_free_guidance:
                        control_latent1_neg = self.mp_control(condition=torch.ones_like(condition)*-1,
                                                    drive_coeff=torch.zeros_like(drive_coeff),
                                                    face_parts=torch.zeros_like(face_parts))
                        if hasattr(self, 'mp_control_proj') and self.version == 'v4':
                            control_latent1_neg = self.mp_control_proj(control_latent1_neg)

                    control_latent1 = self.mp_control(condition=condition,
                                                    drive_coeff=drive_coeff,
                                                    face_parts=face_parts)
                    if hasattr(self, 'mp_control_proj') and self.version == 'v4':
                        control_latent1 = self.mp_control_proj(control_latent1)
                    if self.do_classifier_free_guidance:
                        control_latent1 = cat_dicts([control_latent1_neg, control_latent1], dim=0)

                    control_latent_input.update(control_latent1)
                else:
                    pd_fpg = drive_params['pd_fpg'][:, idx:idx+1].clone().to(device=device)

                    if control_latent2_neg is None and self.do_classifier_free_guidance:
                        control_latent2_neg = self.mp_control(condition=torch.ones_like(condition)*-1,
                                emo_embedding=torch.zeros_like(pd_fpg))
                        if hasattr(self, 'mp_control_proj') and self.version == 'v4':
                            control_latent2_neg = self.mp_control_proj(control_latent2_neg)

                    control_latent2 = self.mp_control(condition=condition, emo_embedding=pd_fpg)
                    if hasattr(self, 'mp_control_proj') and self.version == 'v4':
                        control_latent2 = self.mp_control_proj(control_latent2)
                    if self.do_classifier_free_guidance:
                        control_latent2 = cat_dicts([control_latent2_neg, control_latent2], dim=0)

                    control_latent_input.update(control_latent2)
                control_latents.append(dicts_to_device([control_latent_input], device='cpu')[0])

        self.mp_control.cpu()
        if self.version == 'v4':
            self.mp_control_proj.cpu()

        more_params = dict(
            timesteps=timesteps,
            timestep_cond=timestep_cond,
            sigmas=sigmas,
            prompt_embeds=prompt_embeds,
            cached_res=cached_res,
            device=device,
            added_cond_kwargs=added_cond_kwargs,
            extra_step_kwargs=extra_step_kwargs,
        )

        indexes = list(range(raw_video_len))
        latents = self.gen_video(indexes, dict(), input_noise, 0, control_latents, num_inference_steps, more_params)
        latents_res = latents / self.vae_decode.config.scaling_factor

        self.vae_decode.to(device=device)
        res_frames = []
        with self.progress_bar(total=raw_video_len) as progress_bar:
            for i in range(raw_video_len):
                ret_tensor = self.vae_decode.decode(latents_res[:, :, i, :, :].to(device=device),
                                             return_dict=False, generator=generator)[0]
                ret_image = self.image_processor.postprocess(ret_tensor, output_type=output_type)
                res_frames.append(ret_image)
                progress_bar.set_description(f"VAE DECODE")
                progress_bar.update()
        self.vae_decode.cpu()
        return res_frames, latents_res


def merge_dicts(dictl, dictr, wl=0.5):
    res = {}
    for k in dictl.keys():
        res[k] = dictl[k] * wl + dictr[k] * (1-wl)
    return res


def cat_dicts(dicts, dim=0):
    res = {}
    for k in dicts[0].keys():
        res[k] = torch.cat([d[k].clone() for d in dicts], dim=dim)
    return res


def cat_dicts_ref(dicts):
    res = {}
    for k in dicts[0].keys():
        res[k] = rearrange(torch.cat([d[k].clone().unsqueeze(1) for d in dicts], dim=1), "b f c h w -> (b f) c h w ")
    return res


def dicts_to_device(dicts, device):
    ret = []
    for d in dicts:
        tmpd = {}
        for k in d.keys():
            tmpd[k] = d[k].clone().to(device)
        ret.append(tmpd)
    return ret