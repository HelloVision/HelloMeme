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
from hellomeme.utils import (face_params_to_tensor,
                             get_drive_params,
                             ff_cat_video_and_audio,
                             ff_change_fps,
                             load_unet_from_safetensors)
from hellomeme.pipelines import HMVideoPipeline
from hellomeme.tools import Hello3DMMPred, HelloARKitBSPred, HelloFaceAlignment, HelloCameraDemo
from transformers import CLIPVisionModelWithProjection

@torch.no_grad()
def inference_video(engines, ref_img_path, drive_video_path, save_path, trans_ratio=0.0):
    save_size = 512
    text = "(best quality), highly detailed, ultra-detailed, headshot, person, well-placed five sense organs, looking at the viewer, centered composition, sharp focus, realistic skin texture"

    ref_image_pil = Image.open(ref_img_path).convert('RGB').resize((save_size, save_size))
    ref_image = cv2.cvtColor(np.array(ref_image_pil), cv2.COLOR_RGB2BGR)

    drive_video_path_fps15 = osp.splitext(drive_video_path)[0] + '_fps15.mp4'
    ff_change_fps(drive_video_path, drive_video_path_fps15, 15)

    engines['face_aligner'].reset_track()
    faces = engines['face_aligner'].forward(ref_image)
    if len(faces) > 0:
        face = sorted(faces, key=lambda x: (x['face_rect'][2] - x['face_rect'][0]) * (
                x['face_rect'][3] - x['face_rect'][1]))[-1]
        ref_landmark = face['pre_kpt_222']
    else:
        print('### no face:', ref_img_path)
        return
    ref_rot, ref_trans = engines['h3dmm'].forward_params(ref_image, ref_landmark)

    cap = cv2.VideoCapture(drive_video_path_fps15)
    frame_list = []
    ret, frame = cap.read()
    while ret:
        frame_list.append(frame.copy())
        ret, frame = cap.read()

    engines['face_aligner'].reset_track()
    (drive_face_parts, drive_coeff, drive_rot, drive_trans) = get_drive_params(engines['face_aligner'],
                                                             engines['h3dmm'], engines['harkit_bs'],
                                                             frame_list=frame_list,
                                                             save_size=save_size)

    face_parts_embedding, control_heatmaps = face_params_to_tensor(
        engines['clip_image_encoder'], engines['h3dmm'],
        drive_face_parts,
        drive_rot, drive_trans, ref_trans,
        save_size=512, trans_ratio=trans_ratio)

    drive_params = dict(
        face_parts=face_parts_embedding.unsqueeze(0).to(dtype=dtype, device='cpu'),
        drive_coeff=drive_coeff.unsqueeze(0).to(dtype=dtype, device='cpu'),
        condition=control_heatmaps.unsqueeze(0).to(dtype=dtype, device='cpu'),
    )

    res_frames = pipline(
        prompt=[text],
        strength=1.0,
        image=ref_image_pil,
        drive_params=drive_params,
        num_inference_steps=25,
        negative_prompt=[''],
        guidance_scale=2.0,
        output_type='np'
    )
    res_frames = [np.clip(x[0] * 255, 0, 255).astype(np.uint8) for x in res_frames]

    os.makedirs(osp.dirname(save_path), exist_ok=True)
    imageio.mimsave(save_path, res_frames, fps=15)
    ff_cat_video_and_audio(save_path, drive_video_path, osp.splitext(save_path)[0] + '_audio.mp4')

if __name__ == '__main__':
    ref_img_path = r"data/reference_images/trump.jpg"
    drive_video_path = r"data/drive_videos/jue.mp4"
    save_dir = r'data/results'

    ref_basname = osp.splitext(osp.basename(ref_img_path))[0]
    drive_basename = osp.splitext(osp.basename(drive_video_path))[0]

    gpu_id = 0
    dtype = torch.float16
    device = torch.device(f'cuda:{gpu_id}')

    engines = dict(
        face_aligner=HelloCameraDemo(face_alignment_module=HelloFaceAlignment(gpu_id=gpu_id), reset=True),
        harkit_bs=HelloARKitBSPred(gpu_id=gpu_id),
        h3dmm=Hello3DMMPred(gpu_id=gpu_id),
        clip_image_encoder=CLIPVisionModelWithProjection.from_pretrained('h94/IP-Adapter',
                             subfolder='models/image_encoder').to(dtype=dtype, device=device),
    )

    pipline = HMVideoPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5").to(device=device, dtype=dtype)
    pipline.caryomitosis()

    ### load customized checkpoint or lora here:
    ## checkpoints
    # state_dict = load_unet_from_safetensors(r"pretrained_models/disneyPixarCartoon_v10.safetensors", pipline.unet_ref.config)
    # pipline.unet.load_state_dict(state_dict, strict=False)
    ### lora
    # pipline.load_lora_weights("pretrained_models/loras", weight_name="pixel-portrait-v1.safetensors", adapter_name="pixel")

    pipline.insert_hm_modules(dtype=dtype, device=device)
    engines['pipline'] = pipline.to(device=device, dtype=dtype)

    save_path = osp.join(save_dir, f'{ref_basname}_{drive_basename}.mp4')
    inference_video(engines, ref_img_path, drive_video_path, save_path, trans_ratio=0.0)
