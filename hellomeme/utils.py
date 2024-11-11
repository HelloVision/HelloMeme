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
import numpy as np
from tqdm import tqdm
import torch
from PIL import Image
import subprocess
from einops import rearrange
from .tools.utils import transform_points
from diffusers.pipelines.stable_diffusion.convert_from_ckpt import convert_ldm_unet_checkpoint

from safetensors import safe_open

def merge_dicts(dictl, dictr, wl=0.5):
    res = {}
    for k in dictl.keys():
        res[k] = dictl[k] * wl + dictr[k] * (1-wl)
    return res

def cat_dicts(dicts, dim=0):
    res = {}
    for k in dicts[0].keys():
        res[k] = torch.cat([d[k].clone() for d in dicts], dim=dim)
    return res

def dicts_to_device(dicts, device):
    ret = []
    for d in dicts:
        tmpd = {}
        for k in d.keys():
            tmpd[k] = d[k].clone().to(device)
        ret.append(tmpd)
    return ret

def load_safetensors(model_path):
    tensors = {}
    with safe_open(model_path, framework="pt", device=0) as f:
        for k in f.keys():
            tensors[k] = f.get_tensor(k) # loads the full tensor given a key
    return tensors

def load_unet_from_safetensors(safetensors_path, unet_config):
    original_stats = load_safetensors(safetensors_path)
    return convert_ldm_unet_checkpoint(original_stats, unet_config)

def image_preprocess(np_bgr, size, dtype=torch.float32):
    img_np = cv2.resize(np_bgr, size)
    return np_bgr_to_tensor(img_np, dtype)

def np_bgr_to_tensor(img_np, dtype):
    img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB) / 255. * 2 - 1
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

def get_face_part_1frame(image, parts, save_size=256):
    ret_list = []
    for part_landmarks in parts:
        part_tls, part_brs = np.min(part_landmarks, axis=0), np.max(part_landmarks, axis=0)
        center = (part_tls + part_brs) * 0.5
        wh = max(part_brs - part_tls) * 1.2

        scale = save_size / wh

        M = cv2.getRotationMatrix2D(center, 0, scale)
        M[:, 2] += save_size * 0.5 - center

        part = cv2.warpAffine(image, M, (save_size, save_size))
        ret_list.append(part)
    return np.stack(ret_list, axis=0)

face_parts_regions = [
    [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94],
    [61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116],
    [118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148],
]
face_region = [33, 74, 96, 133, 149]
face_temp = np.array([[0.50951539, 0.40128625], [0.29259476, 0.40203411], [0.72196212, 0.40857479], [0.63420714, 0.71678643], [0.38437035, 0.71723224]])

def get_face_parts(images, landmarks, save_size=256):
    H, W = images[0].shape[:2]
    warped_imgs = []
    warped_landmarks = []

    for landmark, frame in zip(landmarks, images):
        src_landmarks = landmark[face_region, :]
        dst_landmarks = face_temp * (W, H)

        M = umeyama(src_landmarks, dst_landmarks, True)[0:2]
        warped_img = cv2.warpAffine(frame, M, (W, H))

        warped_landmark = transform_points(landmark.astype(np.float32), M)[:,:2]

        # for x, y in warped_landmark:
        #     warped_img = cv2.circle(warped_img, (int(x), int(y)), 1, (255, 255, 0), -1)
        # for x, y in dst_landmarks:
        #     warped_img = cv2.circle(warped_img, (int(x), int(y)), 1, (0, 255, 0), -1)
        # cv2.imshow('warped_img', warped_img)
        # cv2.waitKey()

        warped_imgs.append(warped_img)
        warped_landmarks.append(warped_landmark)
    warped_imgs = np.stack(warped_imgs, axis=0)
    warped_landmarks = np.stack(warped_landmarks, axis=0)

    ret_list = []
    for region in face_parts_regions:
        part_landmarks = warped_landmarks[:, region, :]
        part_tls, part_brs = np.min(np.min(part_landmarks, axis=1), axis=0), np.max(np.max(part_landmarks, axis=1), axis=0)
        center = (part_tls + part_brs) * 0.5
        wh = max(part_brs - part_tls) * 1.2

        scale = save_size / wh

        M = cv2.getRotationMatrix2D(center, 0, scale)
        M[:, 2] += save_size * 0.5 - center
        tmp_part_list = []
        for warped_img in warped_imgs:
            part = cv2.warpAffine(warped_img, M, (save_size, save_size))
            # cv2.imshow('warped_img', warped_img)
            # cv2.imshow('part', part)
            # cv2.waitKey()
            tmp_part_list.append(part)
        ret_list.append(np.stack(tmp_part_list, axis=0))
    return np.stack(ret_list, axis=0).transpose(1, 0, 2, 3, 4)

def get_face_params(h3dmm, harkit_bs, frames, landmarks, save_size=(512, 512), align=True):
    face_parts = []
    for tmp_face_parts_list in get_face_parts(frames, landmarks, save_size=256):
        tmp_face_parts_list = [cv2.GaussianBlur(x, (9, 9), 0) for x in tmp_face_parts_list]
        face_parts.append(tmp_face_parts_list)

    if align:
        new_frames, new_landmarks = crop_and_resize(np.stack(frames, axis=0), np.stack(landmarks, axis=0),
                                                    save_size=save_size, crop=False)
    else:
        new_frames, new_landmarks = frames, landmarks

    rot_list = []
    trans_list = []
    arkit_bs_list = []

    with tqdm(total=len(new_landmarks)) as pbar:
        for landmark, frame in zip(new_landmarks, new_frames):
            drive_rot, drive_trans = h3dmm.forward_params(frame, landmark)
            rot_list.append(drive_rot)
            trans_list.append(drive_trans)
            arkit_bs_list.append(harkit_bs.forward(frame, landmark))

            pbar.set_description('RT & ARKIT')
            pbar.update()
    drive_coeff = torch.from_numpy(np.stack(arkit_bs_list, axis=0))
    return face_parts, drive_coeff, rot_list, trans_list

def face_params_to_tensor(clip_encoder, h3dmm, face_parts, drive_rot, drive_trans, ref_trans, save_size=512, trans_ratio=0.0):
    face_parts_list = []
    for tmp_face_parts_list in face_parts:
        tmp_face_parts_list = [clip_preprocess_from_bgr(x).unsqueeze(1) for x in tmp_face_parts_list]
        face_parts_list.append(torch.cat(tmp_face_parts_list, dim=1))
    face_parts = torch.stack(face_parts_list, dim=2)

    face_parts_tensor = rearrange(face_parts, "c f p h w -> (f p) c h w")
    face_parts_embedding_list = []
    with tqdm(total=face_parts_tensor.size(0)) as pbar:
        for i in range(0, face_parts_tensor.size(0)):
            face_parts_embedding = clip_encoder(face_parts_tensor[i:i+1].to(device=clip_encoder.device, dtype=clip_encoder.dtype)).image_embeds
            face_parts_embedding_list.append(face_parts_embedding.cpu())
            pbar.set_description('CLIP IMAGE ENCODER')
            pbar.update()

    face_parts_embedding = torch.cat(face_parts_embedding_list, dim=0)
    face_parts_embedding = rearrange(face_parts_embedding,
                                     "(f p) c -> f p c",
                                     f=face_parts.size(2))

    control_list = []
    control_show_list = []
    for rot, trans in zip(drive_rot, drive_trans):
        rect = h3dmm.get_project_points_rect(rot, ref_trans + (trans - drive_trans[0]) * trans_ratio, save_size, save_size)
        control_heatmap = draw_skl_by_rect(save_size, rect)

        control_show_list.append(control_heatmap.copy())
        control_heatmap = image_preprocess(control_heatmap, (save_size, save_size))
        control_list.append(control_heatmap)
    control_heatmaps = torch.stack(control_list, dim=1)

    return face_parts_embedding, control_heatmaps

def det_landmarks(face_aligner, frame_list, save_size=(512, 512), reset=False):
    rect_list = []
    new_frame_list = []
    with tqdm(total=len(frame_list)) as pbar:
        for frame in frame_list:
            frame = cv2.resize(frame, save_size)
            faces = face_aligner.forward(frame, reset=reset)
            if len(faces) > 0:
                face = sorted(faces, key=lambda x: (x['face_rect'][2] - x['face_rect'][0]) * (
                        x['face_rect'][3] - x['face_rect'][1]))[-1]
                rect_list.append(face['face_rect'])
                new_frame_list.append(frame)
            pbar.set_description('DET stage1')
            pbar.update()

    face_aligner.reset_track()
    save_frame_list = []
    save_landmark_list = []
    with tqdm(total=len(new_frame_list)) as pbar:
        for frame, rect in zip(new_frame_list, rect_list):
            faces = face_aligner.forward(frame, pre_rect=rect, reset=reset)
            if len(faces) > 0:
                face = sorted(faces, key=lambda x: (x['face_rect'][2] - x['face_rect'][0]) * (
                        x['face_rect'][3] - x['face_rect'][1]))[-1]
                landmarks = face['pre_kpt_222']
                save_frame_list.append(frame)
                save_landmark_list.append(landmarks)
            pbar.set_description('DET stage2')
            pbar.update()
    save_landmark_list = np.stack(save_landmark_list, axis=0).astype(np.float16)

    face_aligner.reset_track()
    return save_frame_list, save_landmark_list

def get_drive_params(face_aligner, h3dmm, harkit_bs, frame_list, save_size, align=True):
    frame_num = len(frame_list)
    frame_list, landmark_list = det_landmarks(face_aligner, frame_list, save_size=(512, 512), reset=False)
    assert len(frame_list) == frame_num
    return get_face_params(h3dmm, harkit_bs, frame_list, landmark_list, save_size=save_size, align=align)

def crop_and_resize(frames, landmarks, save_size=512, crop=True):
    H, W = frames[0].shape[:2]
    if crop:
        all_tl, all_br = np.min(landmarks, axis=1), np.max(landmarks, axis=1)
        mean_wh = np.mean(all_br - all_tl, axis=0)
        tl, br = np.min(all_tl, axis=0), np.max(all_br, axis=0)
        new_size = min(max(mean_wh) * 2.2, min(H, W) - 1)
        fcenter = (tl + br) * 0.5
        ftl = fcenter - new_size * 0.5
        ftl[1] -= mean_wh[1] * 0.2
        fbr = fcenter + new_size * 0.5
        fbr[1] -= mean_wh[1] * 0.2

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
        ftl = ftl.astype(int)
        fbr = (fbr - ftl)[0].astype(int) + ftl

        frames = frames[:, ftl[1]:fbr[1], ftl[0]:fbr[0], :].copy()
        landmarks = landmarks - ftl
        ratio = save_size / (fbr - ftl)
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
                if os.path.splitext(name)[1] in post_fixs and not name.startswith('.'):
                    data_path = os.path.join(root, name)
                    ret_list.append(data_path)
    return ret_list