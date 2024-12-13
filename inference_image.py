# coding: utf-8

"""
@File   : inference_image.py
@Author : Songkey
@Email  : songkey@pku.edu.cn
@Date   : 8/29/2024
@Desc   : 
"""

from generator import Generator, DEFAULT_PROMPT

from PIL import Image

if __name__ == '__main__':
    ref_img_path = r"data/reference_images/trump.jpg"
    drive_img_path = r"data/drive_images/yao.jpg"

    lora_path = "None"
    checkpoint_path = "None"
    vae_path = "same as checkpoint"

    gpu_id = 0
    generator = Generator(gpu_id=gpu_id)
    ref_image = Image.open(ref_img_path)
    drive_image = Image.open(drive_img_path)
    generator.load_image_pipeline(checkpoint_path, vae_path, lora_path, stylize='x1', version='v2')
    result = generator.image_generate(ref_image,
                                      drive_image,
                                      25,
                                      2.0,
                                      1,
                                      DEFAULT_PROMPT,
                                      '',
                                      0.5,
                                      False,
                                      'cntrl2')
    result.show()