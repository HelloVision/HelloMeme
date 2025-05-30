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
import json
import os.path as osp
import shutil
import torch
import numpy as np
import cv2
import imageio
from PIL import Image
from collections import OrderedDict

from hellomeme.utils import (get_drive_pose,
                             get_drive_expression,
                             get_drive_expression_pd_fgc,
                             det_landmarks,
                             gen_control_heatmaps,
                             generate_random_string,
                             ff_cat_video_and_audio,
                             ff_change_fps,
                             load_face_toolkits,
                             append_pipline_weights)
from hellomeme.pipelines import (HMVideoPipeline, HMImagePipeline,
                                 HM3VideoPipeline, HM3ImagePipeline, HM5ImagePipeline)

from hellomeme.tools.sr import RealESRGANer

cur_dir = osp.dirname(osp.abspath(__file__))

config_path = osp.join(cur_dir, 'hellomeme', 'model_config.json')
with open(config_path, 'r') as f:
    MODEL_CONFIG = json.load(f)

DEFAULT_PROMPT = MODEL_CONFIG['prompt']

class Generator(object):
    def __init__(self, gpu_id=0, dtype=torch.float16, pipeline_dict_len=10, sr=True, modelscope=False):
        self.modelscope = modelscope
        self.gpu_id = gpu_id
        self.dtype = dtype
        self.toolkits = load_face_toolkits(gpu_id=gpu_id, dtype=dtype, modelscope=modelscope)
        self.pipeline_dict = OrderedDict()
        self.pipeline_counter = OrderedDict()
        self.pipeline_dict_len = pipeline_dict_len
        if sr:
            self.upsampler = RealESRGANer(scale=2, half=True, gpu_id=gpu_id, modelscope=modelscope)

    @torch.no_grad()
    def load_pipeline(self, type, checkpoint_path, vae_path=None, lora_path=None, lora_scale=1.0, stylize='x1', version='v2'):
        new_token = f"{type}__{osp.basename(checkpoint_path)}__{'none' if lora_path is None else osp.basename(lora_path)}__{lora_scale}__{stylize}__{version}"
        if new_token in self.pipeline_dict:
            self.pipeline_counter[new_token] += 1
            print(f"@@ Pipeline {new_token}({self.pipeline_counter[new_token]}) already exists, reuse it.")
            return new_token

        if self.modelscope:
            from modelscope import snapshot_download
            sd1_5_dir = snapshot_download('songkey/stable-diffusion-v1-5')
        else:
            sd1_5_dir = 'songkey/stable-diffusion-v1-5'

        if version == 'v3' or version == 'v4':
            if type == 'image':
                tmp_pipeline = HM3ImagePipeline.from_pretrained(sd1_5_dir)
            else:
                tmp_pipeline = HM3VideoPipeline.from_pretrained(sd1_5_dir)
        elif version == 'v5' and type == 'image':
            tmp_pipeline = HM5ImagePipeline.from_pretrained(sd1_5_dir)
        else:
            if type == 'image':
                tmp_pipeline = HMImagePipeline.from_pretrained(sd1_5_dir)
            else:
                tmp_pipeline = HMVideoPipeline.from_pretrained(sd1_5_dir)

        tmp_pipeline.to(dtype=self.dtype)
        tmp_pipeline.caryomitosis(version=version, modelscope=self.modelscope)
        append_pipline_weights(tmp_pipeline, checkpoint_path, lora_path, vae_path,
                               stylize=stylize, lora_scale=lora_scale)
        tmp_pipeline.insert_hm_modules(dtype=self.dtype, version=version, modelscope=self.modelscope)

        if len(self.pipeline_dict) >= self.pipeline_dict_len:
            min_key = min(self.pipeline_counter, key=self.pipeline_counter.get)
            print(f"@@ Pipeline {min_key}({self.pipeline_counter[min_key]}) removed.")
            del self.pipeline_dict[min_key]
            del self.pipeline_counter[min_key]
        self.pipeline_dict[new_token] = tmp_pipeline
        self.pipeline_counter[new_token] = 1

        print(f"@@ Pipeline {new_token} created.")
        return new_token

    def image_preprocess(self, images, crop=False):
        _, drive_landmarks = det_landmarks(self.toolkits['face_aligner'], images)
        drive_frames, drive_landmarks, drive_rot, drive_trans = get_drive_pose(self.toolkits,
                                                                               images,
                                                                               drive_landmarks,
                                                                               crop=crop)
        return drive_frames, drive_landmarks, drive_rot, drive_trans

    @torch.no_grad()
    def image_generate(self,
                       pipeline_token,
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

        ref_image_input_np = cv2.cvtColor(np.array(ref_image.convert('RGB')), cv2.COLOR_RGB2BGR)
        ref_frames, ref_landmarks, ref_rot, ref_trans = self.image_preprocess([ref_image_input_np], crop=crop_reference)
        assert len(ref_frames) == 1

        input_ref_pil = Image.fromarray(cv2.cvtColor(ref_frames[0], cv2.COLOR_BGR2RGB))

        drive_image_input_np = cv2.cvtColor(np.array(drive_image.convert('RGB')), cv2.COLOR_RGB2BGR)
        drive_frames, drive_landmarks, drive_rot, drive_trans = self.image_preprocess([drive_image_input_np], crop=True)
        assert len(drive_frames) == 1

        if cntrl_version == 'cntrl1':
            drive_params = get_drive_expression(self.toolkits, drive_frames, drive_landmarks)
        else:
            # for HMControlNet2
            drive_params = get_drive_expression_pd_fgc(self.toolkits, drive_frames, drive_landmarks)

        control_heatmaps = gen_control_heatmaps(drive_rot,
                                                drive_trans,
                                                ref_trans[0],
                                                save_size=save_size,
                                                trans_ratio=trans_ratio)

        drive_params['condition'] = control_heatmaps.unsqueeze(0).to(dtype=dtype, device='cpu')

        generator = torch.Generator().manual_seed(seed if seed >= 0 else random.randint(0, 2**32-1))

        result_img, latents = self.pipeline_dict[pipeline_token](
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
        if hasattr(self, 'upsampler'):
            res_image_np = cv2.cvtColor(res_image_np, cv2.COLOR_RGB2BGR)
            res_image_np, _ = self.upsampler.enhance(res_image_np, outscale=2)
            res_image_np = cv2.cvtColor(res_image_np, cv2.COLOR_RGB2BGR)

        return Image.fromarray(res_image_np)

    @torch.no_grad()
    def video_generate(self,
                       pipeline_token,
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

        rand_token = generate_random_string(8)
        drive_video_path_fps15 = osp.splitext(drive_video_path)[0] + f'_{rand_token}_proced.mp4'
        save_video_path = osp.splitext(drive_video_path)[0] + f'_{rand_token}_save.mp4'

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
        cap.release()

        ref_image_input_np = cv2.cvtColor(np.array(ref_image.convert('RGB')), cv2.COLOR_RGB2BGR)
        ref_frames, ref_landmarks, ref_rot, ref_trans = self.image_preprocess([ref_image_input_np], crop=crop_reference)
        assert len(ref_frames) == 1

        input_ref_pil = Image.fromarray(cv2.cvtColor(ref_frames[0], cv2.COLOR_BGR2RGB))

        drive_frames, drive_landmarks, drive_rot, drive_trans = self.image_preprocess(frame_list, crop=True)

        if cntrl_version == 'cntrl1':
            drive_params = get_drive_expression(self.toolkits, drive_frames, drive_landmarks)
        else:
            # for HMControlNet2
            drive_params = get_drive_expression_pd_fgc(self.toolkits, drive_frames, drive_landmarks)

        control_heatmaps = gen_control_heatmaps(drive_rot, drive_trans, ref_trans[0], save_size=save_size,
                                                trans_ratio=trans_ratio)
        drive_params['condition'] = control_heatmaps.unsqueeze(0).to(dtype=dtype, device='cpu')

        generator = torch.Generator().manual_seed(seed if seed >= 0 else random.randint(0, 2**32-1))
        res_frames, latents = self.pipeline_dict[pipeline_token](
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

        if hasattr(self, 'upsampler'):
            res_frames_np = [cv2.cvtColor(x, cv2.COLOR_RGB2BGR) for x in res_frames_np]
            res_frames_np = [self.upsampler.enhance(x, outscale=2)[0] for x in res_frames_np]
            res_frames_np = [cv2.cvtColor(x, cv2.COLOR_RGB2BGR) for x in res_frames_np]

        if osp.exists(save_video_path): os.remove(save_video_path)
        imageio.mimsave(save_video_path, res_frames_np, fps=fps)

        save_video_audio_path = osp.splitext(drive_video_path)[0] + f'_{rand_token}_audio.mp4'
        if osp.exists(save_video_audio_path): os.remove(save_video_audio_path)
        ff_cat_video_and_audio(save_video_path, drive_video_path_fps15, save_video_audio_path)
        if osp.exists(drive_video_path_fps15): os.remove(drive_video_path_fps15)

        if not osp.exists(save_video_audio_path):
            save_video_audio_path = save_video_path
        else:
            os.remove(save_video_path)

        return save_video_audio_path
