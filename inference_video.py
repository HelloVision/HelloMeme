# coding: utf-8

"""
@File   : inference_image.py
@Author : Songkey
@Email  : songkey@pku.edu.cn
@Date   : 8/29/2024
@Desc   : 
"""

import os
from generator import Generator, DEFAULT_PROMPT, MODEL_CONFIG

from PIL import Image

lora_names = [None] + list(MODEL_CONFIG['sd15']['loras'].keys())
checkpoint_names = list(MODEL_CONFIG['sd15']['checkpoints'].keys())

print("Available lora models: ", lora_names)
print("Available checkpoints: ", checkpoint_names)

modelscope = False

if __name__ == '__main__':
    ref_img_path = r"data/reference_images/toon.png"
    drive_video_path = r"data/drive_videos/tiktok.mp4"

    lora = lora_names[2]
    tmp_lora_info = MODEL_CONFIG['sd15']['loras'][lora]
    checkpoint = checkpoint_names[1]

    print("lora: ", lora, "checkpoint: ", checkpoint)
    if modelscope:
        from modelscope import snapshot_download
        checkpoint_path = snapshot_download(MODEL_CONFIG['sd15']['checkpoints'][checkpoint])
        if lora is None:
            lora_path = None
        else:
            lora_path = os.path.join(snapshot_download(tmp_lora_info[0]), tmp_lora_info[1])
    else:
        checkpoint_path = MODEL_CONFIG['sd15']['checkpoints'][checkpoint]
        if lora is None:
            lora_path = None
        else:
            from huggingface_hub import hf_hub_download
            lora_path = hf_hub_download(tmp_lora_info[0], filename=tmp_lora_info[1])
    vae_path = "same as checkpoint"

    gpu_id = 0
    generator = Generator(gpu_id=gpu_id, modelscope=False)
    ref_image = Image.open(ref_img_path)
    generator.load_video_pipeline(checkpoint_path, vae_path, lora_path, stylize='x1', version='v3')

    save_path = generator.video_generate(ref_image=ref_image,
                                      drive_video_path=drive_video_path,
                                      num_steps=25,
                                      guidance=2.2,
                                      seed=-1,
                                      prompt=DEFAULT_PROMPT,
                                      negative_prompt='',
                                      trans_ratio=0.5,
                                      crop_reference=True,
                                      patch_overlap=4,
                                      cntrl_version='cntrl2',
                                      fps15=True)
    print(f"Save path: {save_path}")