# coding: utf-8

"""
@File   : test.py
@Author : Songkey
@Email  : songkey@pku.edu.cn
@Date   : 11/1/2024
@Desc   : Created by Shengjie Wu (wu.shengjie@immomo.com)
这可能是一个很强大的模型
"""

import numpy as np
import cv2

from huggingface_hub import hf_hub_download
from .utils import get_warp_mat_bbox_by_gt_pts_float, create_onnx_session

def crop_transl_to_full_transl(crop_trans, crop_center, scale, full_center, focal_length):
    """
    :param crop_trans: (3), float
    :param crop_center: (2), float
    :param scale: (1), float
    :param full_center: (2), float
    :param focal_length: (1), float
    :return:
    """
    crop_c_x, crop_c_y = crop_center
    full_c_x, full_c_y = full_center
    bs = 2 * focal_length / scale / crop_trans[2]
    full_x = crop_trans[0] - 2 * (crop_c_x - full_c_x) / bs
    full_y = crop_trans[1] + 2 * (crop_c_y - full_c_y) / bs
    full_z = crop_trans[2] * scale

    full_trans = np.array([full_x, full_y, full_z], dtype=np.float32)

    return full_trans

class Hello3DMMPred(object):
    def __init__(self, gpu_id=None):
        self.deep3d_pred_net = create_onnx_session(hf_hub_download('songkey/hello_group_facemodel', filename='hello_3dmm.onnx'), gpu_id=gpu_id)
        self.deep3d_pred_net_input_name = self.deep3d_pred_net.get_inputs()[0].name
        self.deep3d_pred_net_output_name = [output.name for output in self.deep3d_pred_net.get_outputs()]

        self.image_size = 224
        self.camera_init_z = -0.4
        self.camera_init_focal_len = 386.2879122887948
        self.used_focal_len = -5.0 / self.camera_init_z * self.camera_init_focal_len
        self.id_dims = 526
        self.exp_dims = 203
        self.tex_dims = 439

    def forward_params(self, src_image, src_pt):
        align_mat_info = get_warp_mat_bbox_by_gt_pts_float(src_pt, base_angle=0, dst_size=self.image_size, expand_ratio=0.35, return_info=True)
        align_mat = align_mat_info["M"]

        align_image_rgb_uint8 = cv2.cvtColor(cv2.warpAffine(src_image, align_mat, (self.image_size, self.image_size)), cv2.COLOR_BGR2RGB)

        # cv2.imshow('align_image_rgb_uint8', align_image_rgb_uint8)

        align_image_rgb_fp32 = align_image_rgb_uint8.astype(np.float32) / 255.0
        align_image_rgb_fp32_onnx_input = align_image_rgb_fp32.copy().transpose((2, 0, 1))[np.newaxis, ...]
        pred_coeffs = self.deep3d_pred_net.run(self.deep3d_pred_net_output_name,
                                               {self.deep3d_pred_net_input_name: align_image_rgb_fp32_onnx_input})[0]

        angles = pred_coeffs[:, self.id_dims + self.exp_dims + self.tex_dims:self.id_dims + self.exp_dims + self.tex_dims + 3]
        translations = pred_coeffs[:, self.id_dims + self.exp_dims + self.tex_dims + 3 + 27:]

        crop_global_transl = crop_transl_to_full_transl(translations[0],
                                                        crop_center=[align_mat_info["center_x"],
                                                                     align_mat_info["center_y"]],
                                                        scale=align_mat_info["scale"],
                                                        full_center=[src_image.shape[1] * 0.5, src_image.shape[0] * 0.5],
                                                        focal_length=self.used_focal_len)
        return angles, crop_global_transl[np.newaxis, :]

    def compute_rotation_matrix(self, angles):
        n_b = angles.shape[0]
        sinx = np.sin(angles[:, 0])
        siny = np.sin(angles[:, 1])
        sinz = np.sin(angles[:, 2])
        cosx = np.cos(angles[:, 0])
        cosy = np.cos(angles[:, 1])
        cosz = np.cos(angles[:, 2])
        rotXYZ = np.eye(3).reshape(1, 3, 3).repeat(n_b*3, 0).reshape(3, n_b, 3, 3)
        rotXYZ[0, :, 1, 1] = cosx
        rotXYZ[0, :, 1, 2] = -sinx
        rotXYZ[0, :, 2, 1] = sinx
        rotXYZ[0, :, 2, 2] = cosx
        rotXYZ[1, :, 0, 0] = cosy
        rotXYZ[1, :, 0, 2] = siny
        rotXYZ[1, :, 2, 0] = -siny
        rotXYZ[1, :, 2, 2] = cosy
        rotXYZ[2, :, 0, 0] = cosz
        rotXYZ[2, :, 0, 1] = -sinz
        rotXYZ[2, :, 1, 0] = sinz
        rotXYZ[2, :, 1, 1] = cosz
        rotation = np.matmul(np.matmul(rotXYZ[2], rotXYZ[1]), rotXYZ[0])
        return rotation.transpose(0, 2, 1)

    def rigid_transform(self, vs, rot, trans):
        vs_r = np.matmul(vs, rot)
        vs_t = vs_r + trans.reshape(-1, 1, 3)
        return vs_t

    def perspective_projection_points(self, points, image_w, image_h, used_focal_len):
        batch_size = points.shape[0]
        K = np.zeros([batch_size, 3, 3])
        K[:, 0, 0] = used_focal_len
        K[:, 1, 1] = used_focal_len
        K[:, 2, 2] = 1.
        K[:, 0, 2] = image_w * 0.5
        K[:, 1, 2] = image_h * 0.5

        reverse_z = np.array([[1, 0, 0], [0, 1, 0], [0, 0, -1]])[np.newaxis, :, :].repeat(batch_size, 0)

        # Transform points
        aug_projection = np.matmul(points, reverse_z)
        aug_projection = np.matmul(aug_projection, K.transpose((0, 2, 1)))

        # Apply perspective distortion
        projected_points = aug_projection[:, :, :2] / aug_projection[:, :, 2:]
        return projected_points

    def get_project_points_rect(self, angle, trans, image_w, image_h):
        vs = np.array(
            [[-1, -1, 0], [-1, 1, 0], [1, 1, 0], [1, -1, 0]],
        ) * 0.05
        vs = vs[np.newaxis, :, :]

        rotation = self.compute_rotation_matrix(angle)
        translation = trans.copy()
        translation[0, 2] *= 0.05

        vs_t = self.rigid_transform(vs, rotation, translation)

        project_points = self.perspective_projection_points(vs_t, image_w, image_h, self.used_focal_len*0.05)
        project_points = np.stack([project_points[:, :, 0], image_h - project_points[:, :, 1]], axis=2)

        return project_points[0]

