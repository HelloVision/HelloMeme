# coding: utf-8

"""
@File   : inference_image.py
@Author : Songkey
@Email  : songkey@pku.edu.cn
@Date   : 8/29/2024
@Desc   : 
"""

import cv2
import numpy as np
import os
import os.path as osp
import imageio
from PIL import Image
import torch
from hellomeme.utils import (get_drive_pose,
                             get_drive_expression,
                             get_drive_expression_pd_fgc,
                             det_landmarks,
                             gen_control_heatmaps,
                             ff_cat_video_and_audio,
                             ff_change_fps,
                             load_face_toolkits,
                             append_pipline_weights)
from hellomeme.pipelines import HMVideoPipeline


@torch.no_grad()
def inference_video(toolkits, ref_img_path, drive_video_path, save_path, cntrl_version='cntrl2', trans_ratio=0.0):
    dtype = toolkits['dtype']
    device = toolkits['device']
    save_size = 512
    text = "(best quality), highly detailed, ultra-detailed, headshot, person, well-placed five sense organs, looking at the viewer, centered composition, sharp focus, realistic skin texture"

    ref_image_pil = Image.open(ref_img_path).convert('RGB').resize((save_size, save_size))
    ref_image = cv2.cvtColor(np.array(ref_image_pil), cv2.COLOR_RGB2BGR)

    drive_video_path_fps15 = osp.splitext(drive_video_path)[0] + '_fps15.mp4'
    ff_change_fps(drive_video_path, drive_video_path_fps15, 15)

    toolkits['face_aligner'].reset_track()
    faces = toolkits['face_aligner'].forward(ref_image)
    if len(faces) > 0:
        face = sorted(faces, key=lambda x: (x['face_rect'][2] - x['face_rect'][0]) * (
                x['face_rect'][3] - x['face_rect'][1]))[-1]
        ref_landmark = face['pre_kpt_222']
    else:
        print('### no face:', ref_img_path)
        return
    ref_rot, ref_trans = toolkits['h3dmm'].forward_params(ref_image, ref_landmark)

    cap = cv2.VideoCapture(drive_video_path_fps15)
    frame_list = []
    ret, frame = cap.read()
    while ret:
        frame_list.append(frame.copy())
        ret, frame = cap.read()

    landmark_list = det_landmarks(toolkits['face_aligner'], frame_list)[1]

    drive_rot, drive_trans = get_drive_pose(toolkits, frame_list, landmark_list)

    if cntrl_version == 'cntrl1':
        drive_params = get_drive_expression(toolkits, frame_list, landmark_list)
    else:
        # for HMControlNet2
        drive_params = get_drive_expression_pd_fgc(toolkits, frame_list, landmark_list)

    control_heatmaps = gen_control_heatmaps(drive_rot, drive_trans, ref_trans, save_size=512, trans_ratio=trans_ratio)
    drive_params['condition'] = control_heatmaps.unsqueeze(0).to(dtype=dtype, device='cpu')

    res_frames, latents = toolkits['pipeline'](
        prompt=[text],
        strength=1.0,
        image=ref_image_pil,
        drive_params=drive_params,
        num_inference_steps=25,
        negative_prompt=[''],
        guidance_scale=2.0,
        output_type='np',
        device=device
    )
    res_frames_np = [np.clip(x[0] * 255, 0, 255).astype(np.uint8) for x in res_frames]

    os.makedirs(osp.dirname(save_path), exist_ok=True)
    imageio.mimsave(save_path, res_frames_np, fps=15)
    ff_cat_video_and_audio(save_path, drive_video_path_fps15, osp.splitext(save_path)[0] + '_audio.mp4')

if __name__ == '__main__':
    ref_img_path = r"data/reference_images/trump.jpg"
    drive_video_path = r"data/drive_videos/jue.mp4"
    save_dir = r'data/results'

    ref_basname = osp.splitext(osp.basename(ref_img_path))[0]
    drive_basename = osp.splitext(osp.basename(drive_video_path))[0]

    gpu_id = 0
    dtype = torch.float16

    pipeline = HMVideoPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5")
    pipeline.to(dtype=dtype)
    pipeline.caryomitosis(version='v2')

    lora_path = "None"
    checkpoint_path = "None"
    vae_path = "same as checkpoint"

    append_pipline_weights(pipeline, lora_path, checkpoint_path, vae_path, stylize='x1')

    pipeline.insert_hm_modules(dtype=dtype, version='v2')

    toolkits = load_face_toolkits(gpu_id=gpu_id, dtype=dtype)
    toolkits['pipeline'] = pipeline

    save_path = osp.join(save_dir, f'{ref_basname}_{drive_basename}.mp4')
    inference_video(toolkits, ref_img_path, drive_video_path, save_path, cntrl_version='cntrl2', trans_ratio=0.0)
