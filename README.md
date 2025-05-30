<h1 align='center'>HelloMeme: Integrating Spatial Knitting Attentions to Embed High-Level and Fidelity-Rich Conditions in Diffusion Models</h1>

<div align='center'>
    <a href='https://github.com/songkey' target='_blank'>Shengkai Zhang</a>, <a href='https://github.com/RhythmJnh' target='_blank'>Nianhong Jiao</a>, <a href='https://github.com/Shelton0215' target='_blank'>Tian Li</a>, <a href='https://github.com/chaojie12131243' target='_blank'>Chaojie Yang</a>, <a href='https://github.com/xchgit' target='_blank'>Chenhui Xue</a><sup>*</sup>, <a href='https://github.com/boya34' target='_blank'>Boya Niu</a><sup>*</sup>, <a href='https://github.com/HelloVision/HelloMeme' target='_blank'>Jun Gao</a> 
</div>

<div align='center'>
    HelloVision | HelloGroup Inc.
</div>

<div align='center'>
    <small><sup>*</sup> Intern</small>
</div>

<br>
<div align='center'>
    <a href='https://github.com/HelloVision/HelloMeme'><img src='https://img.shields.io/github/stars/HelloVision/HelloMeme'></a>
    <a href='https://songkey.github.io/hellomeme/'><img src='https://img.shields.io/badge/Project-HomePage-Green'></a>
    <a href='https://arxiv.org/pdf/2410.22901'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>
    <a href='https://huggingface.co/songkey'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Model-yellow'></a>
    <a href='https://github.com/HelloVision/ComfyUI_HelloMeme'><img src='https://img.shields.io/badge/ComfyUI-UI-blue'></a>
    <a href='https://www.modelscope.cn/studios/songkey/HelloMeme'><img src='https://img.shields.io/badge/modelscpe-Demo-red'></a>
</div>

<p align="center">
  <img src="data/demo.gif" alt="showcase">
</p>

## 🔆 New Features/Updates

- ☐ [`ExperimentsOnSKAttentions`](https://github.com/HelloVision/ExperimentsOnSKAttentions) for ablation experiments.
- ☐ SDXL version.
- ✅ `02/09/2025` **HelloMemeV3** is now available.
[YouTube Demo](https://www.youtube.com/watch?v=DAUA0EYjsZA)

- ✅ `12/17/2024` Added modelscope [Demo](https://www.modelscope.cn/studios/songkey/HelloMeme).
- ✅ `12/13/2024` Rewrite the code for the Gradio app.
- ✅ `12/12/2024` Added HelloMeme V2 (synchronize code from the [`ComfyUI`](https://github.com/HelloVision/ComfyUI_HelloMeme) repo).
- ✅ `11/14/2024` Added the `HMControlNet2` module
- ✅ `11/12/2024` Added a newly fine-tuned version of [`Animatediff`](https://huggingface.co/songkey/hm_animatediff_frame12) with a patch size of 12, which uses less VRAM (Tested on 2080Ti).
- ✅ `11/5/2024`  [`ComfyUI`](https://github.com/HelloVision/ComfyUI_HelloMeme) interface for HelloMeme.
- ✅ `11/1/2024` Release the code for the core functionalities..

## Introduction
This repository contains the official code implementation of the paper [`HelloMeme`](https://arxiv.org/pdf/2410.22901). Any updates related to the code or models from the paper will be posted here. The code for the ablation experiments discussed in the paper will be added to the [`ExperimentsOnSKAttentions`](https://github.com/HelloVision/ExperimentsOnSKAttentions) section. Additionally, we plan to release a `ComfyUI` interface for HelloMeme, with updates posted here as well.

## Getting Started

### 1. Create a Conda Environment

```bash
conda create -n hellomeme python=3.10.11
conda activate hellomeme
```

### 2. Install PyTorch and FFmpeg
To install the latest version of PyTorch, please refer to the official [PyTorch](https://pytorch.org/get-started/locally/) website for detailed installation instructions. Additionally, the code will invoke the system's ffmpeg command for video and audio editing, so the runtime environment must have ffmpeg pre-installed. For installation guidance, please refer to the official [FFmpeg](https://ffmpeg.org/) website.

### 3. Install dependencies

```bash
pip install diffusers transformers einops scipy opencv-python tqdm pillow onnxruntime-gpu onnx safetensors accelerate peft imageio imageio[ffmpeg] torchvision 
```

> [!IMPORTANT]  
> 
> Note the version of diffusers required: frequent updates to diffusers may lead to dependency conflicts. We will periodically check the repo’s compatibility with the latest diffusers version. The currently tested and supported version is **diffusers==0.33.1**.

### 4. Clone the repository

```bash
git clone https://github.com/HelloVision/HelloMeme
cd HelloMeme
```

### 5. Run the code
```bash
python inference_image.py # for image generation
python inference_video.py # for video generation
```

### 6. Install for Gradio App

We recommend setting up the environment with conda.

```bash
pip install gradio
python app.py
```

After run the app, all models will be downloaded.

## Examples

### Image Generation

The input for the image generation script `inference_image.py` consists of a reference image and a drive image, as shown in the figure below:

<table>
    <tr>
        <td><img src="./data/reference_images/harris.jpg" width="256" height="256"> <br> Reference Image</td>
        <td ><img src="./data/drive_images/yao.jpg" width="192" height="256"> <br> Drive Image </td>
    </tr>
</table>

The output of the image generation script is shown below:

<table>
    <tr>
        <td><img src="./data/harris_yao.jpg" width="256" height="256"> <br> Based on SD1.5 </td>
        <td ><img src="./data/harris_yao_toon.jpg" height="256" height="256"> <br> Based on <a href="https://civitai.com/models/75650/disney-pixar-cartoon-type-b">disneyPixarCartoon</a>  </td>
    </tr>
</table>

### Video Generation

The input for the video generation script `inference_video.py` consists of a reference image and a drive video, as shown in the figure below:

<table>
    <tr>
        <td><img src="./data/reference_images/trump.jpg" width="256" height="256"> <br> Reference Image</td>
        <td ><img src="./data/jue.gif" width="256" height="256"> <br> Drive Video  </td>
    </tr>
</table>

The output of the video generation script is shown below:

<table>
    <tr>
        <td><img src="./data/trump_jue.gif" width="256" height="256"> <br> Based on <a href="https://civitai.com/models/25694/epicrealism">epicrealism</a> </td>
        <td ><img src="./data/trump_jue-toon.gif" width="256" height="256"> <br> Based on <a href="https://civitai.com/models/75650/disney-pixar-cartoon-type-b">disneyPixarCartoon</a> </td>
    </tr>
</table>

> [!Note]
> 
> If the face in the driving video has significant movement (such as evident camera motion), it is recommended to set the `trans_ratio` parameter to 0 to prevent distorted outputs.
> 
>`inference_video(engines, ref_img_path, drive_video_path, save_path, trans_ratio=0.0)`

## Pretrained Models

Our models are all hosted on [🤗](https://huggingface.co/songkey), and the startup script will download them automatically. The specific model information is as follows:

| model | size  | url  | Info                                                  |
|-------|-------|------|-------------------------------------------------------|
| songkey/hm_reference  | 312M  | <a href='https://huggingface.co/songkey/hm_reference'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Model-yellow'></a> | The weights of the ReferenceAdapter module            |
| songkey/hm_control  | 149M  | <a href='https://huggingface.co/songkey/hm_control'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Model-yellow'></a> | The weights of the HMControlNet module                |
| songkey/hm_animatediff  | 835M  | <a href='https://huggingface.co/songkey/hm_animatediff'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Model-yellow'></a> | The weights of the Turned Animatediff (patch size 16) |
| songkey/hm_animatediff_frame12 | 835M  | <a href='https://huggingface.co/songkey/hm_animatediff_frame12'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Model-yellow'></a> | The weights of the Turned Animatediff (patch size 12) |
| hello_3dmm.onnx  | 311M  | <a href='https://huggingface.co/songkey/hello_group_facemodel'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Model-yellow'></a> | For face RT Extractor                                 |
| hello_arkit_blendshape.onnx | 9.11M | <a href='https://huggingface.co/songkey/hello_group_facemodel'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Model-yellow'></a> | Extract ARKit blendshape parameters                   |
| hello_face_det.onnx | 317K  | <a href='https://huggingface.co/songkey/hello_group_facemodel'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Model-yellow'></a> | Face Detector                                         |
| hello_face_landmark.onnx | 2.87M | <a href='https://huggingface.co/songkey/hello_group_facemodel'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Model-yellow'></a> | Face Landmarks (222 points)                           |

Our pipeline also supports loading stylized base models (safetensors). For video generation tasks, using some customized models for portrait generation, such as [**Realistic Vision V6.0 B1**](https://civitai.com/models/4201/realistic-vision-v60-b1), can produce better results. You can download checkpoints and loras into the directories `pretrained_models/` and `pretrained_models/loras/`, respectively.

## Acknowledgements

Thanks to 🤗 for providing [diffusers](https://huggingface.co/docs/diffusers), which has greatly enhanced development efficiency in diffusion-related work. We also drew considerable inspiration from [MagicAnimate](https://github.com/magic-research/magic-animate) and [EMO](https://github.com/HumanAIGC/EMO), and [Animatediff](https://github.com/guoyww/AnimateDiff) allowed us to implement the video version at a very low cost. Finally, we thank our colleagues **Shengjie Wu** and **Zemin An**, whose foundational modules played a significant role in this work.

## Citation

```bibtex
@misc{zhang2024hellomemeintegratingspatialknitting,
        title={HelloMeme: Integrating Spatial Knitting Attentions to Embed High-Level and Fidelity-Rich Conditions in Diffusion Models}, 
        author={Shengkai Zhang and Nianhong Jiao and Tian Li and Chaojie Yang and Chenhui Xue and Boya Niu and Jun Gao},
        year={2024},
        eprint={2410.22901},
        archivePrefix={arXiv},
        primaryClass={cs.CV},
        url={https://arxiv.org/abs/2410.22901}, 
  }
```

## Contact
**Shengkai Zhang** (songkey@pku.edu.cn)
