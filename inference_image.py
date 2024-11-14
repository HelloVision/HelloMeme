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
from PIL import Image
import torch
from hellomeme.utils import (get_drive_pose,
                             get_drive_expression,
                             get_drive_expression_pd_fgc,
                             gen_control_heatmaps,
                             load_unet_from_safetensors)
from hellomeme.tools import Hello3DMMPred, HelloARKitBSPred, HelloFaceAlignment, HelloCameraDemo, FanEncoder
from hellomeme.pipelines import HMImagePipeline
from transformers import CLIPVisionModelWithProjection



def inference_image(toolkits, ref_img_path, drive_img_path, seed=0, trans_ratio=0.0):
    save_size = 512
    text = "(best quality), highly detailed, ultra-detailed, headshot, person, well-placed five sense organs, looking at the viewer, centered composition, sharp focus, realistic skin texture"

    ref_image_pil = Image.open(ref_img_path).convert('RGB').resize((save_size, save_size))
    ref_image = cv2.cvtColor(np.array(ref_image_pil), cv2.COLOR_RGB2BGR)

    drive_image = cv2.imread(drive_img_path)
    resize_scale = 512 / max(drive_image.shape[:2])
    drive_image = cv2.resize(drive_image, (0, 0), fx=resize_scale, fy=resize_scale)

    toolkits['face_aligner'].reset_track()
    faces = toolkits['face_aligner'].forward(ref_image)
    if len(faces) > 0:
        face = sorted(faces, key=lambda x: (x['face_rect'][2] - x['face_rect'][0]) * (
                x['face_rect'][3] - x['face_rect'][1]))[-1]
        ref_landmark = face['pre_kpt_222']
    else:
        print('### no face:', ref_img_path)
        return

    toolkits['face_aligner'].reset_track()
    faces = toolkits['face_aligner'].forward(drive_image)
    if len(faces) > 0:
        face = sorted(faces, key=lambda x: (x['face_rect'][2] - x['face_rect'][0]) * (
                x['face_rect'][3] - x['face_rect'][1]))[-1]
        drive_landmark = face['pre_kpt_222']
    else:
        print('### no face:', ref_img_path)
        return

    ref_rot, ref_trans = toolkits['h3dmm'].forward_params(ref_image, ref_landmark)

    drive_rot, drive_trans = get_drive_pose(toolkits, [drive_image], [drive_landmark])
    drive_params = get_drive_expression(toolkits, [drive_image], [drive_landmark])

    # for HMControlNet2
    # drive_params = get_drive_expression_pd_fgc(toolkits, [drive_image], [drive_landmark])

    control_heatmaps = gen_control_heatmaps(drive_rot, drive_trans, ref_trans, save_size=512, trans_ratio=trans_ratio)

    drive_params['condition'] = control_heatmaps.unsqueeze(0).to(dtype=dtype, device='cpu')

    generator = torch.Generator().manual_seed(seed)

    result_img = pipeline(
        prompt=[text],
        strength=1.0,
        image=ref_image_pil,
        drive_params=drive_params,
        num_inference_steps=25,
        negative_prompt=[''],
        guidance_scale=2.0,
        generator=generator,
        output_type='np'
    )

    return cv2.cvtColor(np.clip(result_img[0][0] * 255, 0, 255).astype(np.uint8), cv2.COLOR_BGR2RGB)

if __name__ == '__main__':
    ref_img_path = r"data/reference_images/majicmix2.jpg"
    drive_img_path = r"data/drive_images/yao.jpg"

    gpu_id = 0
    dtype = torch.float16
    device = torch.device(f'cuda:{gpu_id}')

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

    pipeline = HMImagePipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5").to(device=device, dtype=dtype)
    pipeline.caryomitosis()

    ### load customized checkpoint or lora here:
    ## checkpoints
    # state_dict = load_unet_from_safetensors(r"pretrained_models/disneyPixarCartoon_v10.safetensors", pipeline.unet_ref.config)
    # pipeline.unet.load_state_dict(state_dict, strict=False)
    ### lora
    # pipeline.load_lora_weights("pretrained_models/loras", weight_name="pixel-portrait-v1.safetensors", adapter_name="pixel")

    pipeline.insert_hm_modules(dtype=dtype, device=device)
    toolkits['pipeline'] = pipeline.to(device=device, dtype=dtype)

    result_image = inference_image(toolkits, ref_img_path, drive_img_path, seed=1024)
    cv2.imshow('show', result_image)
    cv2.waitKey()
