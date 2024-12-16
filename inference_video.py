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
    drive_video_path = r"data/drive_videos/jue.mp4"

    lora_path = "None"
    checkpoint_path = "None"
    vae_path = "same as checkpoint"

    gpu_id = 0
    generator = Generator(gpu_id=gpu_id, modelscope=False)
    ref_image = Image.open(ref_img_path)
    generator.load_video_pipeline(checkpoint_path, vae_path, lora_path, stylize='x1', version='v2')

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