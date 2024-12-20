# coding: utf-8

"""
@File   : inference.py
@Author : Songkey
@Email  : songkey@pku.edu.cn
@Date   : 12/12/2024
@Desc   : 
"""

import random
import os
import os.path as osp
import shutil
import torch
import numpy as np
import cv2
import imageio
from PIL import Image
from hellomeme.utils import (get_drive_pose,
                             get_drive_expression,
                             get_drive_expression_pd_fgc,
                             det_landmarks,
                             crop_and_resize,
                             gen_control_heatmaps,
                             ff_cat_video_and_audio,
                             ff_change_fps,
                             load_face_toolkits,
                             generate_random_string,
                             append_pipline_weights)
from hellomeme.pipelines import HMVideoPipeline, HMImagePipeline

DEFAULT_PROMPT = '(best quality), highly detailed, ultra-detailed, headshot, person, well-placed five sense organs, looking at the viewer, centered composition, sharp focus, realistic skin texture'

class Generator(object):
    def __init__(self, gpu_id=0, dtype=torch.float16, modelscope=False):
        self.modelscope = modelscope
        self.gpu_id = gpu_id
        self.dtype = dtype
        self.toolkits = load_face_toolkits(gpu_id=gpu_id, dtype=dtype, modelscope=modelscope)
        self.image_pipeline = None
        self.image_params_token = ''
        self.video_pipeline = None
        self.video_params_token = ''

    @torch.no_grad()
    def pre_download_hf_weights(self, checkpoint_dirs):
        for idx, checkpoint_dir in enumerate(checkpoint_dirs):
            self.load_video_pipeline_hf(hf_path=checkpoint_dir, stylize='x2', version='v1' if idx % 2 == 0 else 'v2')

    @torch.no_grad()
    def load_image_pipeline_hf(self, hf_path="SD1.5", stylize='x1', version='v2'):
        new_token = f"{hf_path}_{stylize}_{version}"
        if new_token == self.image_params_token:
            return
        self.image_params_token = new_token

        if self.image_pipeline is not None:
            del self.image_pipeline

        if self.modelscope:
            from modelscope import snapshot_download
            sd1_5_dir = snapshot_download('songkey/stable-diffusion-v1-5')
        else:
            sd1_5_dir = 'songkey/stable-diffusion-v1-5'

        self.image_pipeline = HMImagePipeline.from_pretrained(sd1_5_dir)
        self.image_pipeline.to(dtype=self.dtype)
        self.image_pipeline.caryomitosis(version=version, modelscope=self.modelscope)
        append_pipline_weights2(self.image_pipeline, hf_path, stylize=stylize)
        self.image_pipeline.insert_hm_modules(dtype=self.dtype, version=version, modelscope=self.modelscope)

    @torch.no_grad()
    def load_image_pipeline(self, checkpoint_path, vae_path, lora_path, stylize='x1', version='v2'):
        new_token = f"{checkpoint_path}_{stylize}_{version}"
        if new_token == self.image_params_token:
            return
        self.image_params_token = new_token

        if self.modelscope:
            from modelscope import snapshot_download
            sd1_5_dir = snapshot_download('songkey/stable-diffusion-v1-5')
        else:
            sd1_5_dir = 'songkey/stable-diffusion-v1-5'

        if self.image_pipeline is not None:
            del self.image_pipeline
        self.image_pipeline = HMImagePipeline.from_pretrained(sd1_5_dir)
        self.image_pipeline.to(dtype=self.dtype)
        self.image_pipeline.caryomitosis(version=version, modelscope=self.modelscope)
        append_pipline_weights(self.image_pipeline, checkpoint_path, lora_path, vae_path, stylize=stylize)
        self.image_pipeline.insert_hm_modules(dtype=self.dtype, version=version, modelscope=self.modelscope)

    @torch.no_grad()
    def load_video_pipeline_hf(self, hf_path="SD1.5", stylize='x1', version='v2'):
        new_token = f"{hf_path}_{stylize}_{version}"
        if new_token == self.video_params_token:
            return
        self.video_params_token = new_token

        if self.modelscope:
            from modelscope import snapshot_download
            sd1_5_dir = snapshot_download('songkey/stable-diffusion-v1-5')
        else:
            sd1_5_dir = 'songkey/stable-diffusion-v1-5'

        if self.video_pipeline is not None:
            del self.video_pipeline
        self.video_pipeline = HMVideoPipeline.from_pretrained(sd1_5_dir)
        self.video_pipeline.to(dtype=self.dtype)
        self.video_pipeline.caryomitosis(version=version, modelscope=self.modelscope)
        append_pipline_weights2(self.video_pipeline, hf_path, stylize=stylize)
        self.video_pipeline.insert_hm_modules(dtype=self.dtype, version=version, modelscope=self.modelscope)

    @torch.no_grad()
    def load_video_pipeline(self, checkpoint_path, vae_path, lora_path, stylize='x1', version='v2'):
        new_token = f"{checkpoint_path}_{stylize}_{version}"
        if new_token == self.video_params_token:
            return
        self.video_params_token = new_token

        if self.modelscope:
            from modelscope import snapshot_download
            sd1_5_dir = snapshot_download('songkey/stable-diffusion-v1-5')
        else:
            sd1_5_dir = 'songkey/stable-diffusion-v1-5'

        if self.video_pipeline is not None:
            del self.video_pipeline
        self.video_pipeline = HMVideoPipeline.from_pretrained(sd1_5_dir)
        self.video_pipeline.to(dtype=self.dtype)
        self.video_pipeline.caryomitosis(version=version, modelscope=self.modelscope)
        append_pipline_weights(self.video_pipeline, checkpoint_path, lora_path, vae_path, stylize=stylize)
        self.video_pipeline.insert_hm_modules(dtype=self.dtype, version=version, modelscope=self.modelscope)

    @torch.no_grad()
    def image_generate(self,
                       ref_image,
                       drive_image,
                       steps,
                       guidance,
                       seed,
                       prompt,
                       negative_prompt,
                       trans_ratio,
                       crop_reference,
                       cntrl_version='cntrl2'
                       ):

        save_size = 512
        dtype = self.toolkits['dtype']
        device = self.toolkits['device']

        input_ref_pil, ref_rot, ref_trans = self.ref_image_preprocess(ref_image, crop_reference, save_size)

        drive_image_np = cv2.cvtColor(np.array(drive_image.convert('RGB')), cv2.COLOR_RGB2BGR)
        resize_scale = save_size / max(drive_image_np.shape[:2])
        drive_image_np = cv2.resize(drive_image_np, (0, 0), fx=resize_scale, fy=resize_scale)

        self.toolkits['face_aligner'].reset_track()
        faces = self.toolkits['face_aligner'].forward(drive_image_np)
        if len(faces) > 0:
            face = sorted(faces, key=lambda x: (x['face_rect'][2] - x['face_rect'][0]) * (
                    x['face_rect'][3] - x['face_rect'][1]))[-1]
            drive_landmark = face['pre_kpt_222']
        else:
            return None


        drive_rot, drive_trans = get_drive_pose(self.toolkits, [drive_image_np], [drive_landmark])

        if cntrl_version == 'cntrl1':
            drive_params = get_drive_expression(self.toolkits, [drive_image_np], [drive_landmark])
        else:
            # for HMControlNet2
            drive_params = get_drive_expression_pd_fgc(self.toolkits, [drive_image_np], [drive_landmark])

        control_heatmaps = gen_control_heatmaps(drive_rot, drive_trans, ref_trans, save_size=save_size,
                                                trans_ratio=trans_ratio)

        drive_params['condition'] = control_heatmaps.unsqueeze(0).to(dtype=dtype, device='cpu')

        generator = torch.Generator().manual_seed(seed if seed >= 0 else random.randint(0, 2**32-1))

        result_img, latents = self.image_pipeline(
            prompt=[prompt],
            strength=1.0,
            image=input_ref_pil,
            drive_params=drive_params,
            num_inference_steps=steps,
            negative_prompt=[negative_prompt],
            guidance_scale=guidance,
            generator=generator,
            output_type='np',
            device=device
        )

        res_image_np = np.clip(result_img[0][0] * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(res_image_np)

    def ref_image_preprocess(self, ref_image_pil, crop_reference, save_size):
        ref_image = ref_image_pil.convert('RGB')
        ref_image_np = cv2.cvtColor(np.array(ref_image), cv2.COLOR_RGB2BGR)
        self.toolkits['face_aligner'].reset_track()
        faces = self.toolkits['face_aligner'].forward(ref_image_np)
        if len(faces) > 0:
            face = sorted(faces, key=lambda x: (x['face_rect'][2] - x['face_rect'][0]) * (
                    x['face_rect'][3] - x['face_rect'][1]))[-1]
            ref_landmark = face['pre_kpt_222']
        else:
            return None
        self.toolkits['face_aligner'].reset_track()

        if crop_reference:
            ref_images, ref_landmarks = crop_and_resize(ref_image_np[np.newaxis, :, :, :],
                                                        ref_landmark[np.newaxis, :, :],
                                                        save_size, crop=True)
            ref_image_np, ref_landmark = ref_images[0], ref_landmarks[0]
        else:
            ref_landmark = (ref_landmark * [save_size / ref_image_np.shape[1], save_size / ref_image_np.shape[0]])
            ref_image_np = cv2.resize(ref_image_np, (save_size, save_size))
        ref_landmark = ref_landmark.astype(np.float32)

        ref_rot, ref_trans = self.toolkits['h3dmm'].forward_params(ref_image_np, ref_landmark)
        return Image.fromarray(cv2.cvtColor(ref_image_np, cv2.COLOR_BGR2RGB)), ref_rot, ref_trans

    @torch.no_grad()
    def video_generate(self,
                       ref_image,
                       drive_video_path,
                       num_steps,
                       guidance,
                       seed,
                       prompt,
                       negative_prompt,
                       trans_ratio,
                       crop_reference,
                       patch_overlap,
                       cntrl_version,
                       fps15):

        dtype = self.toolkits['dtype']
        device = self.toolkits['device']
        save_size = 512
        input_ref_pil, ref_rot, ref_trans = self.ref_image_preprocess(ref_image, crop_reference, save_size)
        random_str = generate_random_string(8)

        drive_video_path_fps15 = osp.splitext(drive_video_path)[0] + f'_{random_str}_proced.mp4'
        save_video_path = osp.splitext(drive_video_path)[0] + f'_{random_str}_save.mp4'

        if osp.exists(drive_video_path_fps15): os.remove(drive_video_path_fps15)
        if fps15:
            ff_change_fps(drive_video_path, drive_video_path_fps15, 15)
            fps = 15
        else:
            shutil.copy(drive_video_path, drive_video_path_fps15)

        cap = cv2.VideoCapture(drive_video_path_fps15)
        if not fps15:
            fps = cap.get(cv2.CAP_PROP_FPS)

        frame_list = []
        ret, frame = cap.read()
        while ret:
            frame_list.append(frame.copy())
            ret, frame = cap.read()

        landmark_list = det_landmarks(self.toolkits['face_aligner'], frame_list)[1]

        drive_rot, drive_trans = get_drive_pose(self.toolkits, frame_list, landmark_list)

        if cntrl_version == 'cntrl1':
            drive_params = get_drive_expression(self.toolkits, frame_list, landmark_list)
        else:
            # for HMControlNet2
            drive_params = get_drive_expression_pd_fgc(self.toolkits, frame_list, landmark_list)

        control_heatmaps = gen_control_heatmaps(drive_rot, drive_trans, ref_trans, save_size=save_size,
                                                trans_ratio=trans_ratio)
        drive_params['condition'] = control_heatmaps.unsqueeze(0).to(dtype=dtype, device='cpu')

        generator = torch.Generator().manual_seed(seed if seed >= 0 else random.randint(0, 2**32-1))
        res_frames, latents = self.video_pipeline(
            prompt=[prompt],
            strength=1.0,
            image=input_ref_pil,
            patch_overlap=patch_overlap,
            drive_params=drive_params,
            num_inference_steps=num_steps,
            negative_prompt=[negative_prompt],
            guidance_scale=guidance,
            generator=generator,
            output_type='np',
            device=device
        )
        res_frames_np = [np.clip(x[0] * 255, 0, 255).astype(np.uint8) for x in res_frames]

        if osp.exists(save_video_path): os.remove(save_video_path)
        imageio.mimsave(save_video_path, res_frames_np, fps=fps)

        save_video_audio_path = osp.splitext(drive_video_path)[0] + f'_{random_str}_audio.mp4'
        if osp.exists(save_video_audio_path): os.remove(save_video_audio_path)
        ff_cat_video_and_audio(save_video_path, drive_video_path_fps15, save_video_audio_path)
        # if osp.exists(drive_video_path_fps15): os.remove(drive_video_path_fps15)

        if not osp.exists(save_video_audio_path):
            save_video_audio_path = save_video_path
        else:
            os.remove(save_video_path)

        return save_video_audio_path

def append_pipline_weights2(pipeline, hf_path=None, stylize='x1'):
    if hf_path and not hf_path.startswith('SD1.5'):
        tmp_pipeline = HMImagePipeline.from_pretrained(hf_path)
        unet_state_dict = tmp_pipeline.unet.state_dict()
        vae_state_dict = tmp_pipeline.vae.state_dict()

        if hasattr(pipeline, 'unet_pre'):
            pipeline.unet_pre.load_state_dict(unet_state_dict, strict=False)
        pipeline.unet.load_state_dict(unet_state_dict, strict=False)
        if stylize == 'x2' and hasattr(pipeline, 'unet_ref'):
            pipeline.unet_ref.load_state_dict(unet_state_dict, strict=False)

        if hasattr(pipeline, 'vae_decode'):
            pipeline.vae_decode.load_state_dict(vae_state_dict, strict=True)
        if stylize == 'x2':
            pipeline.vae.load_state_dict(vae_state_dict, strict=True)
        del tmp_pipeline