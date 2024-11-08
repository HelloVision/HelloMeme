import gradio as gr
import os
import re
import torch
import numpy as np
import cv2
import imageio
from PIL import Image
from hellomeme.utils import (face_params_to_tensor, get_drive_params, ff_cat_video_and_audio, load_unet_from_safetensors)
from hellomeme.pipelines import HMVideoPipeline
from hellomeme.tools import Hello3DMMPred, HelloARKitBSPred, HelloFaceAlignment, HelloCameraDemo
from transformers import CLIPVisionModelWithProjection

# Initialize engines and pipeline (preload these for efficiency)
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
pipline.insert_hm_modules()
engines['pipline'] = pipline.to(device=device, dtype=dtype)

def sanitize_filename(filename):
    """Replace spaces and special characters in filename with underscores."""
    return re.sub(r'[^\w\-_\.]', '_', filename)

@torch.no_grad()
def inference_video(ref_img, drive_video, trans_ratio=0.0):
    save_size = 512
    text = "(best quality), highly detailed, ultra-detailed, headshot, person, well-placed five sense organs, looking at the viewer, centered composition, sharp focus, realistic skin texture"

    # Sanitize filenames for safety
    ref_img_path = sanitize_filename("temp_ref_image.jpg")
    
    # Save input files to sanitized paths
    ref_img.save(ref_img_path)
    
    # Load and prepare reference image as NumPy array for OpenCV processing
    ref_image_pil = Image.open(ref_img_path).convert('RGB').resize((save_size, save_size))
    ref_image = cv2.cvtColor(np.array(ref_image_pil), cv2.COLOR_RGB2BGR)  # Keep as NumPy array

    # Get the original FPS of the drive video
    video_capture = cv2.VideoCapture(drive_video)  # Use drive_video directly as the path
    original_fps = video_capture.get(cv2.CAP_PROP_FPS)
    video_capture.release()

    # Run face alignment and get drive parameters without changing FPS
    engines['face_aligner'].reset_track()
    faces = engines['face_aligner'].forward(ref_image)
    if len(faces) > 0:
        face = sorted(faces, key=lambda x: (x['face_rect'][2] - x['face_rect'][0]) * (
                x['face_rect'][3] - x['face_rect'][1]))[-1]
        ref_landmark = face['pre_kpt_222']
    else:
        return "No face detected in the reference image.", None

    # Use ref_image as a NumPy array for OpenCV operations
    ref_rot, ref_trans = engines['h3dmm'].forward_params(ref_image, ref_landmark)

    engines['face_aligner'].reset_track()
    (drive_face_parts, drive_coeff, drive_rot, drive_trans) = get_drive_params(
        engines['face_aligner'], engines['h3dmm'], engines['harkit_bs'],
        video_path=drive_video,
        save_size=save_size
    )

    # Ensure tensors are moved to the correct device
    face_parts_embedding, control_heatmaps = face_params_to_tensor(
        engines['clip_image_encoder'], engines['h3dmm'],
        drive_face_parts,
        drive_rot, drive_trans, ref_trans,
        save_size=512, trans_ratio=trans_ratio)
    
    face_parts_embedding = face_parts_embedding.to(device=device, dtype=dtype)
    control_heatmaps = control_heatmaps.to(device=device, dtype=dtype)
    drive_coeff = drive_coeff.to(device=device, dtype=dtype)

    drive_params = dict(
        face_parts=face_parts_embedding.unsqueeze(0),
        drive_coeff=drive_coeff.unsqueeze(0),
        condition=control_heatmaps.unsqueeze(0),
    )

    # Generate frames in pipeline
    res_frames = pipline(
        prompt=[text],
        strength=1.0,
        image=ref_image_pil,  # ref_image_pil is a PIL image for the pipeline
        drive_params=drive_params,
        num_inference_steps=25,
        negative_prompt=[''],
        guidance_scale=2.0,
        output_type='np'
    )
    res_frames = [np.clip(x[0] * 255, 0, 255).astype(np.uint8) for x in res_frames]

    # Save the output video
    output_path = sanitize_filename("output_video.mp4")
    imageio.mimsave(output_path, res_frames, fps=original_fps)
    final_output_path = sanitize_filename("output_with_audio.mp4")
    ff_cat_video_and_audio(output_path, drive_video, final_output_path)

    return final_output_path

# Gradio interface with layout adjustments
with gr.Blocks() as interface:
    gr.Markdown("# HelloMeme\nIntegrating Spatial Knitting Attentions to Embed High-Level and Fidelity-Rich Conditions in Diffusion Models.")
    
    with gr.Row():
        with gr.Column():
            ref_image_input = gr.Image(type="pil", label="Reference Image")
            drive_video_input = gr.Video(label="Drive Video")
            trans_ratio_slider = gr.Slider(minimum=0.0, maximum=1.0, step=0.1, value=0.0, label="Transition Ratio")
        
        with gr.Column():
            generated_video_output = gr.Video(label="Generated Video")
    
    # Button at the bottom
    generate_button = gr.Button("Generate Animation")

    # Link inputs and outputs
    generate_button.click(
        inference_video,
        inputs=[ref_image_input, drive_video_input, trans_ratio_slider],
        outputs=generated_video_output
    )

interface.launch()
