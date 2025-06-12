# coding: utf-8

"""
@File   : inference_image.py
@Author : Songkey
@Email  : songkey@pku.edu.cn
@Date   : 8/29/2024
@Desc   : 
"""

import os
from generator import Generator, MODEL_CONFIG

from PIL import Image

lora_names = [None] + list(MODEL_CONFIG['sd15']['loras'].keys())
checkpoint_names = list(MODEL_CONFIG['sd15']['checkpoints'].keys())

print("Available lora models: ", lora_names)
print("Available checkpoints: ", checkpoint_names)

modelscope = False

if __name__ == '__main__':
    ref_img_path = r"data/reference_images/chillout.jpg"
    drive_img_path = r"data/drive_images/yao.jpg"

    lora = lora_names[2]
    checkpoint = checkpoint_names[1]

    tmp_lora_info = MODEL_CONFIG['sd15']['loras'][lora]
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
    generator = Generator(gpu_id=gpu_id, modelscope=modelscope)
    ref_image = Image.open(ref_img_path)
    drive_image = Image.open(drive_img_path)
    token = generator.load_pipeline('image', checkpoint_path, vae_path, lora_path, stylize='x1', version='v5b')
    result = generator.image_generate(token,
                                      ref_image,
                                      drive_image,
                                      25,
                                      1.5,
                                      1,
                                      '',
                                      '',
                                      0.5,
                                      False,
                                      'cntrl2')
    result.show()