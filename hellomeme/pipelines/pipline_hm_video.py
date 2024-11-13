# coding: utf-8

"""
@File   : hm_pipline_image.py
@Author : Songkey
@Email  : songkey@pku.edu.cn
@Date   : 8/29/2024
@Desc   :
adapted from: https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_img2img.py
"""
import copy
from typing import Any, Callable, Dict, List, Optional, Union
import torch

from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.image_processor import PipelineImageInput
from diffusers.utils import deprecate
from diffusers.utils.torch_utils import randn_tensor

from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import retrieve_timesteps, retrieve_latents
from diffusers import StableDiffusionImg2ImgPipeline, MotionAdapter, EulerDiscreteScheduler

from ..models import HMDenoising3D, HMDenoisingMotion, HMControlNet
from ..models import HMReferenceAdapter
from ..utils import dicts_to_device, cat_dicts


class HMVideoPipeline(StableDiffusionImg2ImgPipeline):
    def caryomitosis(self, patch_frames=12, **kwargs):
        if hasattr(self, "unet_ref"):
            del self.unet_ref
        self.unet_ref = HMDenoising3D.from_unet2d(self.unet)

        if patch_frames >= 14:
            self.num_frames = 16
            adapter = MotionAdapter.from_pretrained("songkey/hm_animatediff", torch_dtype=torch.float16)
        else:
            self.num_frames = 12
            adapter = MotionAdapter.from_pretrained("songkey/hm_animatediff_frame12", torch_dtype=torch.float16)
        unet = HMDenoisingMotion.from_unet2d(unet=self.unet, motion_adapter=adapter, load_weights=True)
        # todo: 不够优雅
        del self.unet
        self.unet = unet

    def insert_hm_modules(self, dtype, device):
        hm_adapter = HMReferenceAdapter.from_pretrained('songkey/hm_reference')
        if isinstance(self.unet, HMDenoisingMotion):
            self.unet.insert_reference_adapter(hm_adapter)
            self.unet = self.unet.to(device=device, dtype=dtype).eval()

        if hasattr(self, "unet_ref"):
            self.unet_ref.insert_reference_adapter(hm_adapter)
            self.unet_ref = self.unet_ref.to(device=device, dtype=dtype).eval()

        if hasattr(self, "mp_control"):
            del self.mp_control
        self.mp_control = HMControlNet.from_pretrained('songkey/hm_control')
        self.mp_control = self.mp_control.to(device=device, dtype=dtype).eval()

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

        device = self._execution_device

        # 3. Encode input prompt
        text_encoder_lora_scale = (
            self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
        )
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
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
        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

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

        # 5. set timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps, sigmas
        )
        timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength, device)
        latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)

        raw_video_len = drive_params['condition'].size(2)

        # 6. Prepare latent variables
        ref_latents = [
            retrieve_latents(self.vae.encode(image[i: i + 1].to(device=device)), generator=generator)
            for i in range(batch_size)
        ]
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

        ref_latents = ref_latents.unsqueeze(2)
        cached_res = self.unet_ref(
            torch.cat([torch.zeros_like(ref_latents), ref_latents], dim=0) if
                self.do_classifier_free_guidance else ref_latents,
            0,
            encoder_hidden_states=prompt_embeds,
            return_dict=False,
        )[1]

        # 7.2 Optionally get Guidance Scale Embedding
        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=prompt_embeds.dtype)

        pre_latents = []
        control_latents = []
        control_latent_neg = None
        base_noise = randn_tensor([batch_size, c, 1, h, w], dtype=prompt_embeds.dtype, generator=generator).to(device=device)
        with self.progress_bar(total=raw_video_len) as progress_bar:
            for idx in range(raw_video_len):
                condition = drive_params['condition'][:, :, idx:idx+1].clone().to(device=device)
                drive_coeff = drive_params['drive_coeff'][:, idx:idx+1].clone().to(device=device)
                face_parts = drive_params['face_parts'][:, idx:idx+1].clone().to(device=device)

                if control_latent_neg is None and self.do_classifier_free_guidance:
                    control_latent_neg = self.mp_control(condition=torch.ones_like(condition)*-1,
                                                drive_coeff=torch.zeros_like(drive_coeff),
                                                face_parts=torch.zeros_like(face_parts))

                control_latent = self.mp_control(condition=condition,
                                                drive_coeff=drive_coeff,
                                                face_parts=face_parts)
                if self.do_classifier_free_guidance:
                    control_latent = cat_dicts([control_latent_neg, control_latent], dim=0)

                scheduler = EulerDiscreteScheduler(
                    num_train_timesteps=1000,
                    beta_start=0.00085,
                    beta_end=0.012,
                    beta_schedule="scaled_linear",
                )

                tmp_timesteps, _ = retrieve_timesteps(
                    scheduler, 8, device, None, sigmas
                )
                if idx == 0:
                    pred_latent = scheduler.add_noise(ref_latents, base_noise, tmp_timesteps[:1])
                else:
                    pred_latent = scheduler.add_noise(pred_latent, base_noise, tmp_timesteps[:1])

                for i, t in enumerate(tmp_timesteps):
                    latent_model_input = torch.cat([pred_latent, pred_latent], dim=0) if \
                        self.do_classifier_free_guidance else pred_latent
                    latent_model_input = scheduler.scale_model_input(latent_model_input, t)
                    noise_pred_chunk = self.unet_ref(
                        latent_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        reference_hidden_states=cached_res,
                        control_hidden_states=control_latent,
                        timestep_cond=timestep_cond,
                        cross_attention_kwargs=self.cross_attention_kwargs,
                        added_cond_kwargs=added_cond_kwargs,
                        return_dict=False,
                    )[0]
                    if self.do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred_chunk.chunk(2)
                        noise_pred_chunk = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
                    pred_latent = scheduler.step(noise_pred_chunk, t, pred_latent,
                                                  **extra_step_kwargs, return_dict=False)[0]
                pre_latents.append(pred_latent.cpu())
                control_latents.append(dicts_to_device([control_latent], device='cpu')[0])

                progress_bar.set_description(f"GEN stage1")
                progress_bar.update()

        pre_latents = torch.cat(pre_latents, dim=2)

        chunk_size = self.num_frames
        chunk_overlap = min(max(0, chunk_overlap), chunk_size // 2)
        chunk_stride = chunk_size - chunk_overlap

        drive_idx_list = list(range(raw_video_len))

        steped_video_len = len(drive_idx_list)
        post_pad_num = chunk_stride - (steped_video_len - chunk_overlap) % chunk_stride

        if post_pad_num > 0:
            drive_idx_list = drive_idx_list + [drive_idx_list[-1]] * post_pad_num
        paded_video_len = len(drive_idx_list)

        drive_idx_chunks = [drive_idx_list[i:i + chunk_size] for i in
                            range(0, paded_video_len - chunk_stride + 1, chunk_stride)]

        if post_pad_num > 0:
            pre_latents = torch.cat([pre_latents,
                     (pre_latents[:, :, -2:-1]).clone().repeat_interleave(post_pad_num, 2)], dim=2)
            control_latents = control_latents + [control_latents[-1]] * post_pad_num

        # 5. set timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, None, sigmas
        )
        timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength, device)
        latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)

        # 8. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)

        base_noise = randn_tensor([batch_size, c, paded_video_len, h, w], dtype=prompt_embeds.dtype, generator=generator)
        latents = self.scheduler.add_noise(pre_latents, base_noise, latent_timestep)

        drive_idx_chunks[-1] = list(range(drive_idx_chunks[-1][0], paded_video_len))

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                noise_pred = torch.zeros([batch_size * 2 if self.do_classifier_free_guidance
                                          else batch_size, c, paded_video_len, h, w], dtype=prompt_embeds.dtype).cpu()
                noise_pred_counter = torch.zeros([1, 1, paded_video_len, 1, 1], dtype=prompt_embeds.dtype).cpu()

                latent_model_input_all = torch.cat([latents.clone(), latents.clone()], dim=0) if \
                    self.do_classifier_free_guidance else latents.clone()

                latent_model_input_all = self.scheduler.scale_model_input(latent_model_input_all, t)
                for cidx, chunk in enumerate(drive_idx_chunks):
                    if self.interrupt:
                        continue

                    control_latent = cat_dicts([control_latents[ix] for ix in chunk], dim=2)
                    latent_model_input = latent_model_input_all[:, :, chunk].to(device=device)

                    # predict the noise residual
                    noise_pred_chunk = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        reference_hidden_states=cached_res,
                        control_hidden_states=dicts_to_device([control_latent], device=device)[0],
                        timestep_cond=timestep_cond,
                        cross_attention_kwargs=self.cross_attention_kwargs,
                        added_cond_kwargs=added_cond_kwargs,
                        return_dict=False,
                    )[0].cpu()

                    noise_pred[:, :, chunk] = noise_pred[:, :, chunk] + noise_pred_chunk
                    noise_pred_counter[:, :, chunk] = noise_pred_counter[:, :, chunk] + 1

                noise_pred = noise_pred / noise_pred_counter
                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred.to(device=device), t, latents.to(device=device),
                                              **extra_step_kwargs, return_dict=False)[0].cpu()

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.set_description(f"GEN stage2")
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

        latents = latents / self.vae.config.scaling_factor

        res_frames = [self.vae.decode(latents[:,:,i,:,:].to(device=device), return_dict=False, generator=generator)[0]
                       for i in range(raw_video_len)]
        res_frames = [self.image_processor.postprocess(frame, output_type=output_type) for frame in res_frames]
        return res_frames