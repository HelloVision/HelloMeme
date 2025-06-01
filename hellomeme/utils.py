# coding: utf-8

"""
@File   : utils.py
@Author : Songkey
@Email  : songkey@pku.edu.cn
@Date   : 8/28/2024
@Desc   : 
"""

import os
import cv2
import os.path as osp
import numpy as np
import random
from tqdm import tqdm
import torch
from PIL import Image
import subprocess
from einops import rearrange
from .tools.utils import transform_points
from .tools.hello_3dmm import get_project_points_rect

from safetensors import safe_open
from diffusers.pipelines.stable_diffusion.convert_from_ckpt import (convert_ldm_unet_checkpoint,
                                                                    convert_ldm_vae_checkpoint)
from .tools import Hello3DMMPred, HelloARKitBSPred, HelloFaceAlignment, HelloCameraDemo, FanEncoder
from .pipelines import HMImagePipeline
from transformers import CLIPVisionModelWithProjection

def generate_random_string(length=8):
    characters = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
    result = ''
    for i in range(0, length):
        result += random.choice(characters)
    return result

def get_torch_device(gpu_id=-1):
    if gpu_id < 0: return torch.device("cpu")
    gpu_id = min(max(-1, gpu_id), torch.cuda.device_count())
    return torch.device(f"cuda:{gpu_id}")

def load_face_toolkits(dtype=torch.float16, gpu_id=-1, modelscope=False):
    if modelscope:
        from modelscope import snapshot_download
        pd_fpg_dir = snapshot_download('songkey/pd_fgc_motion')
        clip_image_dir = snapshot_download('songkey/IP-Adapter')
    else:
        pd_fpg_dir = 'songkey/pd_fgc_motion'
        clip_image_dir = 'songkey/IP-Adapter'

    pd_fpg_motion = FanEncoder.from_pretrained(pd_fpg_dir)
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(clip_image_dir, subfolder='models/image_encoder')

    image_encoder.to(dtype=dtype).cpu()
    pd_fpg_motion.to(dtype=dtype).cpu()
    return dict(
            device=get_torch_device(gpu_id),
            dtype=dtype,
            pd_fpg_motion=pd_fpg_motion,
            face_aligner=HelloCameraDemo(face_alignment_module=HelloFaceAlignment(gpu_id=gpu_id, modelscope=modelscope), reset=False),
            harkit_bs=HelloARKitBSPred(gpu_id=gpu_id, modelscope=modelscope),
            h3dmm=Hello3DMMPred(gpu_id=gpu_id, modelscope=modelscope),
            image_encoder=image_encoder
        )

def append_pipline_weights(pipeline, checkpoint_path=None, lora_path=None, vae_path=None, stylize='x1', lora_scale=1.0):
    ### load customized checkpoint or lora here:

    # print("## checkpoint_path", checkpoint_path)
    # print("## lora_path", lora_path)
    # print("## vae_path", vae_path)

    unet_stats, vae_stats = None, None
    if checkpoint_path and not checkpoint_path.startswith('SD1.5'):
        if osp.isfile(checkpoint_path):
            print("Loading checkpoint from", checkpoint_path)
            if checkpoint_path.endswith('.safetensors'):
                raw_stats = load_safetensors(checkpoint_path)
                unet_stats = convert_ldm_unet_checkpoint(raw_stats, pipeline.unet_ref.config)
                vae_stats = convert_ldm_vae_checkpoint(raw_stats, pipeline.vae.config)
            else:
                raw_stats = torch.load(checkpoint_path)
                unet_stats = convert_ldm_unet_checkpoint(raw_stats, pipeline.unet_ref.config)
                vae_stats = convert_ldm_vae_checkpoint(raw_stats, pipeline.vae.config)
        else:
            tmp_pipeline = HMImagePipeline.from_pretrained(checkpoint_path)
            unet_stats = tmp_pipeline.unet.state_dict()
            vae_stats = tmp_pipeline.vae.state_dict()

        if unet_stats:
            try:
                if hasattr(pipeline, 'unet_pre'):
                    pipeline.unet_pre.load_state_dict(unet_stats, strict=False)
                pipeline.unet.load_state_dict(unet_stats, strict=False)

                if stylize == 'x2' and hasattr(pipeline, 'unet_ref'):
                    pipeline.unet_ref.load_state_dict(unet_stats, strict=False)
            except:
                raise ValueError("Failed to load checkpoint from", checkpoint_path)

    if vae_path and not vae_path.startswith('SD1.5 default vae'):
        if osp.isfile(vae_path):
            print("Loading vae from", vae_path)
            if vae_path.endswith('.safetensors'):
                raw_vae_stats = load_safetensors(vae_path)
            else:
                raw_vae_stats = torch.load(vae_path)
            vae_stats = convert_ldm_vae_checkpoint(raw_vae_stats, pipeline.vae.config)

        if vae_stats:
            try:
                if hasattr(pipeline, 'vae_decode'):
                    pipeline.vae_decode.load_state_dict(vae_stats, strict=True)
                if stylize == 'x2':
                    pipeline.vae.load_state_dict(vae_stats, strict=True)
            except:
                raise ValueError("Failed to load vae from", vae_path)

    ### lora
    if lora_path and not lora_path.startswith('None'):
        if osp.exists(lora_path):
            print("Loading lora from", lora_path)
            try:
                pipeline.load_lora_weights_sk(pipeline.unet, osp.dirname(lora_path),
                                              weight_name=osp.basename(lora_path),
                                              adapter_name="lora", text_encoder=pipeline.text_encoder,
                                              lora_scale=lora_scale)
                if hasattr(pipeline, 'unet_pre'):
                    pipeline.load_lora_weights_sk(pipeline.unet_pre, osp.dirname(lora_path),
                                              weight_name=osp.basename(lora_path),
                                              adapter_name="lora", text_encoder=None,
                                              lora_scale=lora_scale)
                if stylize == 'x2' and hasattr(pipeline, 'unet_ref'):
                    pipeline.load_lora_weights_sk(pipeline.unet_ref, osp.dirname(lora_path),
                                              weight_name=osp.basename(lora_path),
                                              adapter_name="lora", text_encoder=None,
                                              lora_scale=lora_scale)

            except:
                raise ValueError("Failed to load lora from", lora_path)

def load_safetensors(model_path):
    tensors = {}
    with safe_open(model_path, framework="pt", device=0) as f:
        for k in f.keys():
            tensors[k] = f.get_tensor(k) # loads the full tensor given a key
    return tensors


def image_preprocess(np_bgr, size, dtype=torch.float32):
    img_np = cv2.resize(np_bgr, size)
    return np_bgr_to_tensor(img_np, dtype)


def np_bgr_to_tensor(img_np, dtype):
    img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB) / 255. * 2 - 1
    return torch.tensor(img_rgb).permute(2, 0, 1).to(dtype=dtype)


def np_rgb_to_tensor(img_np, dtype):
    img_rgb = img_np / 255. * 2 - 1
    return torch.tensor(img_rgb).permute(2, 0, 1).to(dtype=dtype)


def tensor_to_np_rgb(img_np):
    img_rgb = np.clip((img_np.permute(1, 2, 0).detach().cpu().numpy() + 1) / 2 * 255, 0, 255)
    return img_rgb.astype(np.uint8)


def clip_preprocess_from_bgr(image_bgr, dtype=torch.float32):
    temp_image = cv2.resize(image_bgr, (224, 224), interpolation=cv2.INTER_CUBIC)
    temp_image = cv2.cvtColor(temp_image, cv2.COLOR_BGR2RGB)
    temp_image_np = temp_image * 0.00392156862745098
    temp_image_np = (temp_image_np - [0.48145466, 0.4578275, 0.40821073]) / [0.26862954, 0.26130258, 0.27577711]
    temp_image_np = np.transpose(temp_image_np, (2, 0, 1))
    return torch.tensor(temp_image_np, dtype=dtype)


def clip_preprocess_to_bgr(image_tensor):
    image_tensor_np = image_tensor.detach().cpu().numpy().astype(float)
    image_tensor_np = np.transpose(image_tensor_np, (1, 2, 0))
    image_tensor_np = (image_tensor_np * [0.26862954, 0.26130258, 0.27577711]) + [0.48145466, 0.4578275, 0.40821073]
    image_tensor_np = np.clip(image_tensor_np / 0.00392156862745098, 0, 255).astype(np.uint8)
    return cv2.cvtColor(image_tensor_np, cv2.COLOR_BGR2RGB)


def clip_preprocess_from_pil(image_pil, dtype):
    temp_image = image_pil.resize((224, 224), resample=Image.BICUBIC)
    temp_image_np = np.asarray(temp_image) * 0.00392156862745098
    temp_image_np = (temp_image_np - [0.48145466, 0.4578275, 0.40821073]) / [0.26862954, 0.26130258, 0.27577711]
    temp_image_np = np.transpose(temp_image_np, (2, 0, 1))
    return torch.tensor(temp_image_np, dtype=dtype)


def draw_skl_by_rect(save_size, rect):
    rect = rect.astype(np.int32)
    ret_img = np.zeros((save_size, save_size, 3), dtype=np.uint8)
    # cv2.rectangle(ret_img, rect[0], rect[2], (255, 255, 255), -1)
    cv2.line(ret_img, rect[0], rect[1], (0, 255, 0), 15)
    cv2.line(ret_img, rect[1], rect[2], (255, 0, 0), 15)
    cv2.line(ret_img, rect[2], rect[3], (0, 0, 255), 15)
    cv2.line(ret_img, rect[3], rect[0], (255, 0, 255), 15)
    return ret_img


face_parts_regions = [
    [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94],
    [61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116],
    [118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148],
]
face_region = [33, 74, 96, 133, 149]
face_temp = np.array([[0.50951539, 0.40128625], [0.29259476, 0.40203411], [0.72196212, 0.40857479], [0.63420714, 0.71678643], [0.38437035, 0.71723224]])


def get_drive_pose(toolkits, frames, landmarks, save_size=512, crop=True):
    if isinstance(landmarks, list): landmarks = np.stack(landmarks, axis=0)

    new_frames, new_landmarks = crop_and_resize(np.stack(frames, axis=0), landmarks,
                                                save_size=save_size, crop=crop)

    rot_list = []
    trans_list = []
    with tqdm(total=len(new_landmarks)) as pbar:
        for landmark, frame in zip(new_landmarks, new_frames):
            drive_rot, drive_trans = toolkits['h3dmm'].forward_params(frame, landmark)
            rot_list.append(drive_rot)
            trans_list.append(drive_trans)

            pbar.set_description('HEAD POSE')
            pbar.update()
    return new_frames, new_landmarks, rot_list, trans_list


def get_drive_expression(toolkits, images, landmarks):
    face_parts_embedding = get_face_parts(toolkits, images, landmarks, save_size=256)
    drive_coeff = get_arkit_bs(toolkits, images, landmarks)

    return dict(
        face_parts=face_parts_embedding.unsqueeze(0),
        drive_coeff=drive_coeff.unsqueeze(0)
    )


def get_drive_expression_pd_fgc(toolkits, images, landmarks):
    dtype = toolkits['dtype']
    device = toolkits['device']
    emo_list = []

    motion_model = toolkits['pd_fpg_motion'].to(device=device)
    with tqdm(total=len(images)) as pbar:
        for frame, landmark in zip(images, landmarks):
            emo_image = warp_face_pd_fgc(frame, landmark, save_size=224)
            input_tensor = image_preprocess(emo_image, (224, 224), dtype).to(device=device).unsqueeze(0)
            emo_tensor = motion_model(input_tensor)
            emo_list.append(emo_tensor.cpu())
            pbar.set_description('PD_FPG_MOTION')
            pbar.update()

    neg_tensor = motion_model(torch.ones_like(input_tensor)*-1).cpu()

    ret_tensor = torch.cat(emo_list, dim=0)
    toolkits['pd_fpg_motion'].to(device='cpu')
    return dict(pd_fpg=ret_tensor.unsqueeze(0), neg_pd_fpg=neg_tensor.unsqueeze(0))


def gen_control_heatmaps(drive_rot, drive_trans, ref_trans, save_size=512, trans_ratio=0.0):
    control_list = []
    for rot, trans in zip(drive_rot, drive_trans):
        rect = get_project_points_rect(rot, ref_trans + (trans - drive_trans[0]) * trans_ratio, save_size, save_size)
        control_heatmap = draw_skl_by_rect(save_size, rect)

        control_heatmap = image_preprocess(control_heatmap, (save_size, save_size))
        control_list.append(control_heatmap)
    return torch.stack(control_list, dim=1)


def det_landmarks(face_aligner, frame_list):
    rect_list = []
    new_frame_list = []

    assert len(frame_list) > 0
    face_aligner.reset_track()
    with tqdm(total=len(frame_list)) as pbar:
        for frame in frame_list:
            faces = face_aligner.forward(frame)
            if len(faces) > 0:
                face = sorted(faces, key=lambda x: (x['face_rect'][2] - x['face_rect'][0]) * (
                        x['face_rect'][3] - x['face_rect'][1]))[-1]
                rect_list.append(face['face_rect'])
                new_frame_list.append(frame)
            pbar.set_description('DET stage1')
            pbar.update()

    assert len(new_frame_list) > 0
    face_aligner.reset_track()
    save_frame_list = []
    save_landmark_list = []
    with tqdm(total=len(new_frame_list)) as pbar:
        for frame, rect in zip(new_frame_list, rect_list):
            faces = face_aligner.forward(frame, pre_rect=rect)
            if len(faces) > 0:
                face = sorted(faces, key=lambda x: (x['face_rect'][2] - x['face_rect'][0]) * (
                        x['face_rect'][3] - x['face_rect'][1]))[-1]
                landmarks = face['pre_kpt_222']
                save_frame_list.append(frame)
                save_landmark_list.append(landmarks)
            pbar.set_description('DET stage2')
            pbar.update()

    assert len(save_frame_list) > 0
    save_landmark_list = np.stack(save_landmark_list, axis=0)
    face_aligner.reset_track()
    return save_frame_list, save_landmark_list


def get_face_parts(toolkits, images, landmarks, save_size=256):
    dtype = toolkits['dtype']
    device = toolkits['device']
    H, W = images[0].shape[:2]
    warped_imgs = []
    warped_landmarks = []

    for landmark, frame in zip(landmarks, images):
        src_landmarks = landmark[face_region, :]
        dst_landmarks = face_temp * (W, H)

        M = umeyama(src_landmarks, dst_landmarks, True)[0:2]
        warped_img = cv2.warpAffine(frame, M, (W, H))

        warped_landmark = transform_points(landmark.astype(np.float32), M)[:,:2]

        warped_imgs.append(warped_img)
        warped_landmarks.append(warped_landmark)
    warped_imgs = np.stack(warped_imgs, axis=0)
    warped_landmarks = np.stack(warped_landmarks, axis=0)

    clip_encoder = toolkits['image_encoder'].to(device=device)

    face_parts_embedding = []

    for rdx, region in enumerate(face_parts_regions):
        part_landmarks = warped_landmarks[:, region, :]
        part_tls, part_brs = np.min(np.min(part_landmarks, axis=1), axis=0), np.max(np.max(part_landmarks, axis=1), axis=0)
        center = (part_tls + part_brs) * 0.5
        wh = max(part_brs - part_tls) * 1.2

        scale = save_size / wh
        M = cv2.getRotationMatrix2D(center, 0, scale)
        M[:, 2] += save_size * 0.5 - center

        tmp_parts_embeddings = []
        with tqdm(total=len(warped_imgs)) as pbar:
            for warped_img in warped_imgs:
                part = cv2.warpAffine(warped_img, M, (save_size, save_size))
                part = cv2.GaussianBlur(part, (5, 5), 0)
                part_input = clip_preprocess_from_bgr(part).unsqueeze(0)
                part_embedding = clip_encoder(part_input.to(device=device, dtype=dtype)).image_embeds
                tmp_parts_embeddings.append(part_embedding[0].cpu())

                pbar.set_description(f'FACE part{rdx}')
                pbar.update()
        face_parts_embedding.append(torch.stack(tmp_parts_embeddings, dim=0))

    ret_embedding = torch.stack(face_parts_embedding, dim=0)
    ret_embedding = rearrange(ret_embedding, "p f d -> f p d")
    toolkits['image_encoder'].to(device='cpu')
    return ret_embedding


def get_arkit_bs(toolkits, frames, landmarks):
    harkit_bs = toolkits['harkit_bs']
    arkit_bs_list = []
    with tqdm(total=len(landmarks)) as pbar:
        for landmark, frame in zip(landmarks, frames):
            arkit_bs_list.append(harkit_bs.forward(frame, landmark))
            pbar.set_description('ARKIT BS')
            pbar.update()
    return torch.from_numpy(np.stack(arkit_bs_list, axis=0))


def warp_face_pd_fgc(image, landmarks222, save_size=224):
    pt5_idx = [182, 202, 36, 149, 133]
    dst_pt5 = np.array([[0.3843, 0.27], [0.62, 0.2668], [0.503, 0.4185], [0.406, 0.5273], [0.5977, 0.525]]) * save_size
    src_pt5 = landmarks222[pt5_idx]

    M = umeyama(src_pt5, dst_pt5, True)[0:2]
    warped = cv2.warpAffine(image, M, (save_size, save_size), flags=cv2.INTER_CUBIC)

    return warped


def crop_and_resize(frames, landmarks, save_size=512, crop=True):
    H, W = frames[0].shape[:2]
    if crop:
        all_tl, all_br = np.min(landmarks, axis=1), np.max(landmarks, axis=1)
        mean_wh = np.mean(all_br - all_tl, axis=0)
        tl, br = np.min(all_tl, axis=0), np.max(all_br, axis=0)
        new_size = min(max(mean_wh) * 3.0, min(H, W) - 1)
        fcenter = (tl + br) * 0.5
        fcenter[1] -= new_size * 0.115
        ftl = fcenter - new_size * 0.5
        fbr = fcenter + new_size * 0.5

        if ftl[0] < 0:
            fbr[0] -= ftl[0]
            ftl[0] = 0
        if ftl[1] < 0:
            fbr[1] -= ftl[1]
            ftl[1] = 0

        if fbr[0] >= W:
            ftl[0] -= fbr[0] - W + 1
            fbr[0] = W - 1
        if fbr[1] >= H:
            ftl[1] -= fbr[1] - H + 1
            fbr[1] = H - 1

        opt_center = (ftl + fbr) * 0.5
        opt_size = np.floor(fbr - ftl).min()
        opt_tl = (opt_center - opt_size * 0.5).astype(int)
        opt_br = (opt_center + opt_size * 0.5).astype(int)

        frames = frames[:, opt_tl[1]:opt_br[1], opt_tl[0]:opt_br[0], :].copy()

        landmarks = landmarks - opt_tl
        ratio = save_size / (opt_br - opt_tl)
    else:
        ratio = (save_size / W, save_size / H)
    landmarks = landmarks * ratio
    frames = np.stack([cv2.resize(frame, (save_size, save_size), interpolation=cv2.INTER_CUBIC) for frame in frames], axis=0)
    return frames, landmarks


def umeyama(src, dst, estimate_scale):
    """Estimate N-D similarity transformation with or without scaling.
    Parameters
    ----------
    src : (M, N) array
        Source coordinates.
    dst : (M, N) array
        Destination coordinates.
    estimate_scale : bool
        Whether to estimate scaling factor.
    Returns
    -------
    T : (N + 1, N + 1)
        The homogeneous similarity transformation matrix. The matrix contains
        NaN values only if the problem is not well-conditioned.
    References
    ----------
    .. [1] "Least-squares estimation of transformation parameters between two
            point patterns", Shinji Umeyama, PAMI 1991, DOI: 10.1109/34.88573
    """

    num = src.shape[0]
    dim = src.shape[1]

    # Compute mean of src and dst.
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)

    # Subtract mean from src and dst.
    src_demean = src - src_mean
    dst_demean = dst - dst_mean

    # Eq. (38).
    A = np.dot(dst_demean.T, src_demean) / num

    # Eq. (39).
    d = np.ones((dim,), dtype=np.double)
    if np.linalg.det(A) < 0:
        d[dim - 1] = -1

    T = np.eye(dim + 1, dtype=np.double)

    U, S, V = np.linalg.svd(A)

    # Eq. (40) and (43).
    rank = np.linalg.matrix_rank(A)
    if rank == 0:
        return np.nan * T
    elif rank == dim - 1:
        if np.linalg.det(U) * np.linalg.det(V) > 0:
            T[:dim, :dim] = np.dot(U, V)
        else:
            s = d[dim - 1]
            d[dim - 1] = -1
            T[:dim, :dim] = np.dot(U, np.dot(np.diag(d), V))
            d[dim - 1] = s
    else:
        T[:dim, :dim] = np.dot(U, np.dot(np.diag(d), V.T))

    if estimate_scale:
        # Eq. (41) and (42).
        scale = 1.0 / src_demean.var(axis=0).sum() * np.dot(S, d)
    else:
        scale = 1.0

    T[:dim, dim] = dst_mean - scale * np.dot(T[:dim, :dim], src_mean.T)
    T[:dim, :dim] *= scale

    return T


def ff_cat_video_and_audio(video_path, audio_path, output_path):
    cmd = [
        "ffmpeg",
        "-y",
        "-v", "quiet", "-stats",
        "-i", video_path,
        "-i", audio_path,
        "-c:v", "copy",
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-shortest",
        output_path
    ]
    subprocess.run(cmd)


def ff_change_fps(input_video, output_video, fps=15):
    cmd = [
        "ffmpeg",
        "-y",
        "-v", "quiet", "-stats",
        # "-hwaccel", "cuda",
        "-i", input_video,
        "-c:v", "libx264",
        "-r", f"{fps}",
        "-c:a", "copy",
        "-crf", "18",
        output_video
    ]

    subprocess.run(cmd)


def load_data_list(data_dir, post_fix='.pickle;.txt'):
    post_fixs = post_fix.split(';')
    ret_list = []
    for root, dirnames, filenames in os.walk(data_dir):
            for name in filenames:
                if os.path.splitext(name)[1].lower() in post_fixs and not name.startswith('.'):
                    data_path = os.path.join(root, name)
                    ret_list.append(data_path)
    return ret_list