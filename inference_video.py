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
                             get_torch_device,
                             load_safetensors)
from hellomeme.pipelines import HMVideoPipeline
from hellomeme.tools import Hello3DMMPred, HelloARKitBSPred, HelloFaceAlignment, HelloCameraDemo, FanEncoder
from transformers import CLIPVisionModelWithProjection

from diffusers.pipelines.stable_diffusion.convert_from_ckpt import (convert_ldm_unet_checkpoint,
                                                                    convert_ldm_vae_checkpoint)


@torch.no_grad()
def inference_video(toolkits, ref_img_path, drive_video_path, save_path, trans_ratio=0.0):
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
    drive_params = get_drive_expression(toolkits, frame_list, landmark_list)
    # for HMControlNet2
    # drive_params = get_drive_expression_pd_fgc(toolkits, frame_list, landmark_list)

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
    device = get_torch_device(gpu_id)

    toolkits = dict(
        device=device,
        dtype=dtype,
        pd_fpg_motion=FanEncoder.from_pretrained("songkey/pd_fgc_motion").to(dtype=dtype),
        face_aligner=HelloCameraDemo(face_alignment_module=HelloFaceAlignment(gpu_id=gpu_id), reset=False),
        harkit_bs=HelloARKitBSPred(gpu_id=gpu_id),
        h3dmm=Hello3DMMPred(gpu_id=gpu_id),
        image_encoder=CLIPVisionModelWithProjection.from_pretrained(
            'h94/IP-Adapter', subfolder='models/image_encoder').to(dtype=dtype)
    )

    pipeline = HMVideoPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5")
    pipeline.to(dtype=dtype)

    ### patch_frames: 12 or 16
    pipeline.caryomitosis(patch_frames=12)

    ### load customized checkpoint or lora here:
    ### checkpoints
    # raw_stats = load_safetensors(r"pretrained_models/disneyPixarCartoon_v10.safetensors")
    # state_dict = convert_ldm_unet_checkpoint(raw_stats, pipeline.unet_ref.config)
    # pipeline.unet.load_state_dict(state_dict, strict=False)
    # if hasattr(pipeline, 'unet_pre'):
    #     pipeline.unet_pre.load_state_dict(state_dict, strict=False)
    # pipeline.unet.load_state_dict(state_dict, strict=False)
    #
    # vae_state_dict = convert_ldm_vae_checkpoint(raw_stats, pipeline.vae_decode.config)
    # if hasattr(pipeline, 'vae_decode'):
    #     pipeline.vae_decode.load_state_dict(vae_state_dict, strict=True)

    ### lora
    # pipeline.load_lora_weights("pretrained_models/loras", weight_name="pixel-portrait-v1.safetensors", adapter_name="pixel")

    pipeline.insert_hm_modules(dtype=dtype)
    toolkits['pipeline'] = pipeline

    save_path = osp.join(save_dir, f'{ref_basname}_{drive_basename}.mp4')
    inference_video(toolkits, ref_img_path, drive_video_path, save_path, trans_ratio=0.0)
