# coding: utf-8

"""
@File   : new_app.py
@Author : Songkey
@Email  : songkey@pku.edu.cn
@Date   : 12/12/2024
@Desc   : 
"""
import os

import gradio as gr
from generator import Generator, DEFAULT_PROMPT
import sys
import importlib.metadata

installed_packages = [package.name for package in importlib.metadata.distributions()]

REQUIRED = {
    'diffusers':'0.31.0', 'transformers':'4.46.3', 'einops':'0.8.0', 'opencv-python':'4.10.0.84', 'tqdm':'4.67.0',
    'pillow':'10.2.0', 'onnxruntime-gpu':'1.18.1', 'onnx':'1.17.0', 'safetensors':'0.4.5',
    'accelerate':'1.1.1', 'peft':'0.13.2'
}

missing = [name for name in REQUIRED.keys() if name not in installed_packages]
missing_params = ' '.join([f'{k}=={REQUIRED[k]}' for k in missing])
print("missing pkgs", missing_params)

# if missing:
#     os.system(f'{sys.executable} -m pip install {missing_params}')

modelscope = False
if modelscope:
    from modelscope import snapshot_download
    realistic_checkpoint_dir = snapshot_download('songkey/realisticVisionV60B1_v51VAE')
    disney_pixar_checkpoint_dir = snapshot_download('songkey/disney-pixar-cartoon-b')
else:
    realistic_checkpoint_dir = 'songkey/realisticVisionV60B1_v51VAE'
    disney_pixar_checkpoint_dir = 'songkey/disney-pixar-cartoon-b'

with gr.Blocks() as app:
    gr.Markdown('''
        <div style="display: flex; justify-content: center; align-items: center; text-align: center;">
            <div>
                <h1>HelloMeme: Integrating Spatial Knitting Attentions to Embed High-Level and Fidelity-Rich Conditions in Diffusion Models</h1>
                <div style="display: flex; justify-content: center; align-items: center; text-align: center;">
                    <a href='https://songkey.github.io/hellomeme/'><img src='https://img.shields.io/badge/Project-HomePage-Green'></a>  &nbsp;\
                    <a href='https://github.com/HelloVision/HelloMeme'><img src='https://img.shields.io/badge/GitHub-Code-blue'></a>  &nbsp;\
                    <a href='https://arxiv.org/pdf/2410.22901'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>  &nbsp;\
                    <a href='https://github.com/HelloVision/ComfyUI_HelloMeme'><img src='https://img.shields.io/badge/ComfyUI-UI-blue'></a>  &nbsp;\
                    <a href='https://github.com/HelloVision/HelloMeme'><img src='https://img.shields.io/github/stars/HelloVision/HelloMeme'></a>
                </div>
            </div>
        </div>
    ''')

    gen = Generator(gpu_id=0, modelscope=modelscope)
    gen.pre_download_hf_weights([realistic_checkpoint_dir, disney_pixar_checkpoint_dir])
    with gr.Tab("Image Generation"):
        with gr.Row():
            ref_img = gr.Image(type="pil", label="Reference Image")
            drive_img = gr.Image(type="pil", label="Drive Image")
            result_img = gr.Image(type="pil", label="Generated Image")
        exec_btn = gr.Button("Run")
        with gr.Row():
            checkpoint = gr.Dropdown(choices=['SD1.5', realistic_checkpoint_dir,
                                              disney_pixar_checkpoint_dir], value=realistic_checkpoint_dir, label="Checkpoint")
            version = gr.Dropdown(choices=['HelloMemeV1', 'HelloMemeV2'], value="HelloMemeV2", label="Version")
            cntrl_version = gr.Dropdown(choices=['HMControlNet1', 'HMControlNet2'], value="HMControlNet2", label="Control Version")
            stylize = gr.Dropdown(choices=['x1', 'x2'], value="x1", label="Stylize")
        with gr.Accordion("Advanced Options", open=False):
            with gr.Row():
                num_steps = gr.Slider(1, 50, 25, step=1, label="Steps")
                guidance = gr.Slider(1.0, 10.0, 2.0, step=0.1, label="Guidance", interactive=True)
            with gr.Column():
                prompt = gr.Textbox(label="Prompt", value=DEFAULT_PROMPT)
                negative_prompt = gr.Textbox(label="Negative Prompt", value="")
            with gr.Row():
                seed = gr.Number(value=-1, label="Seed (-1 for random)")
                trans_ratio = gr.Slider(0.0, 1.0, 0.0, step=0.01, label="Trans Ratio", interactive=True)
                crop_reference = gr.Checkbox(label="Crop Reference", value=True)

        def img_gen_fnc(ref_img, drive_img, num_steps, guidance, seed, prompt, negative_prompt,
                        trans_ratio, crop_reference, cntrl_version, version, stylize, checkpoint):
            gen.load_image_pipeline_hf(hf_path=checkpoint, stylize=stylize, version='v1' if version == 'HelloMemeV1' else 'v2')
            res = gen.image_generate(ref_img,
                                     drive_img,
                                     num_steps,
                                     guidance,
                                     seed,
                                     prompt,
                                     negative_prompt,
                                     trans_ratio,
                                     crop_reference,
                                     'cntrl1' if cntrl_version == 'HMControlNet1' else 'cntrl2',
                                    )
            return res

        exec_btn.click(fn=img_gen_fnc,
                       inputs=[ref_img, drive_img, num_steps, guidance, seed, prompt, negative_prompt,
                               trans_ratio, crop_reference, cntrl_version, version, stylize, checkpoint],
                       outputs=result_img,
                       api_name="Image Generation")
        gr.Examples(
            examples=[
                ['data/reference_images/civitai1.jpg', 'data/drive_images/ysll.jpg', 25, 2.0, 1024, DEFAULT_PROMPT, '', 0.0,
                 True, 'HMControlNet2', 'HelloMemeV2', 'x1', disney_pixar_checkpoint_dir],
                ['data/reference_images/kjl.jpg', 'data/drive_images/jue.jpg', 25, 2.0, 1024, DEFAULT_PROMPT, '', 0.0,
                 True, 'HMControlNet2', 'HelloMemeV2', 'x1', realistic_checkpoint_dir],
                ['data/reference_images/zzj.jpg', 'data/drive_images/yao.jpg', 25, 2.0, 1024, DEFAULT_PROMPT, '', 0.0,
                 True, 'HMControlNet2', 'HelloMemeV2', 'x1', 'SD1.5'],
            ],
            fn=img_gen_fnc,
            inputs=[ref_img, drive_img, num_steps, guidance, seed, prompt, negative_prompt, trans_ratio,
                    crop_reference, cntrl_version, version, stylize, checkpoint],
            outputs=result_img,
            cache_examples=False,
        )

    with gr.Tab("Video Generation"):
        with gr.Row():
            ref_img = gr.Image(type="pil", label="Reference Image")
            drive_video = gr.Video(label="Drive Video")
            result_video = gr.Video(autoplay=True, loop=True, label="Generated Video")
        exec_btn = gr.Button("Run")
        with gr.Row():
            checkpoint = gr.Dropdown(choices=['SD1.5', realistic_checkpoint_dir,
                                              disney_pixar_checkpoint_dir], value=realistic_checkpoint_dir, label="Checkpoint")
            version = gr.Dropdown(choices=['HelloMemeV1', 'HelloMemeV2'], value="HelloMemeV2", label="Version")
            cntrl_version = gr.Dropdown(choices=['HMControlNet1', 'HMControlNet2'], value="HMControlNet2", label="Control Version")
            stylize = gr.Dropdown(choices=['x1', 'x2'], value="x1", label="Stylize")
        with gr.Accordion("Advanced Options", open=False):
            with gr.Row():
                num_steps = gr.Slider(1, 50, 25, step=1, label="Steps", interactive=True)
                guidance = gr.Slider(1.0, 10.0, 2.0, step=0.1, label="Guidance", interactive=True)
                patch_overlap = gr.Slider(1, 5, 4, step=1, label="Patch Overlap", interactive=True)
            with gr.Column():
                prompt = gr.Textbox(label="Prompt", value=DEFAULT_PROMPT)
                negative_prompt = gr.Textbox(label="Negative Prompt", value="")
            with gr.Row():
                seed = gr.Number(value=-1, label="Seed (-1 for random)")
                trans_ratio = gr.Slider(0.0, 1.0, 0.0, step=0.01, label="Trans Ratio", interactive=True)
                with gr.Column():
                    crop_reference = gr.Checkbox(label="Crop Reference", value=True)
                    fps15 = gr.Checkbox(label="Use fps15", value=True)
        def video_gen_fnc(ref_img, drive_video, num_steps, guidance, seed, prompt, negative_prompt,
                        trans_ratio, crop_reference, cntrl_version, version, stylize, patch_overlap, checkpoint, fps15):
            gen.load_video_pipeline_hf(hf_path=checkpoint, stylize=stylize, version='v1' if version == 'HelloMemeV1' else 'v2')
            res = gen.video_generate(ref_img,
                                     drive_video,
                                     num_steps,
                                     guidance,
                                     seed,
                                     prompt,
                                     negative_prompt,
                                     trans_ratio,
                                     crop_reference,
                                     patch_overlap,
                                     'cntrl1' if cntrl_version == 'HMControlNet1' else 'cntrl2',
                                     fps15
                                    )
            return res
        exec_btn.click(fn=video_gen_fnc,
                       inputs=[ref_img, drive_video, num_steps, guidance, seed, prompt, negative_prompt, trans_ratio,
                               crop_reference, cntrl_version, version, stylize, patch_overlap, checkpoint, fps15],
                       outputs=result_video,
                       api_name="Video Generation")
        gr.Examples(
            examples=[
                ['data/reference_images/zzj.jpg', 'data/drive_videos/tbh.mp4', 25, 2.0, 1024, DEFAULT_PROMPT, '', 0.0,
                 True, 'HMControlNet2', 'HelloMemeV2', 'x1', 4, realistic_checkpoint_dir, True],
                ['data/reference_images/kjl.jpg', 'data/drive_videos/jue.mp4', 25, 2.0, 1024, DEFAULT_PROMPT, '', 0.0,
                 True, 'HMControlNet2', 'HelloMemeV2', 'x1', 4, disney_pixar_checkpoint_dir, True],
            ],
            fn=video_gen_fnc,
            inputs=[ref_img, drive_video, num_steps, guidance, seed, prompt, negative_prompt, trans_ratio,
                               crop_reference, cntrl_version, version, stylize, patch_overlap, checkpoint, fps15],
            outputs=result_video,
            cache_examples=False,
        )

app.launch(inbrowser=True)