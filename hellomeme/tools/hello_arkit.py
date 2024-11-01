"""
@File   : test.py
@Author : Songkey
@Email  : songkey@pku.edu.cn
@Date   : 11/1/2024
@Desc   : Created by Shengjie Wu (wu.shengjie@immomo.com)
"""

import numpy as np
import cv2
from huggingface_hub import hf_hub_download
from .utils import create_onnx_session, get_warp_mat_bbox_by_gt_pts_float

class HelloARKitBSPred(object):
    def __init__(self, gpu_id=0):
        self.face_rig_net = create_onnx_session(hf_hub_download('songkey/hello_group_facemodel', filename='hello_arkit_blendshape.onnx'), gpu_id=gpu_id)
        self.onnx_input_name = self.face_rig_net.get_inputs()[0].name
        self.onnx_output_name = [output.name for output in self.face_rig_net.get_outputs()]
        self.image_size = 224
        self.expand_ratio = 0.15

    def forward(self, src_image, src_pt):
        left_eye_corner = src_pt[74]
        right_eye_corner = src_pt[96]
        radian = np.arctan2(right_eye_corner[1] - left_eye_corner[1], right_eye_corner[0] - left_eye_corner[0] + 0.00000001)
        rotate_angle = np.rad2deg(radian)
        align_warp_mat = get_warp_mat_bbox_by_gt_pts_float(src_pt, base_angle=rotate_angle, dst_size=self.image_size,
                                                           expand_ratio=self.expand_ratio)
        face_rig_input = cv2.warpAffine(src_image, align_warp_mat, (self.image_size, self.image_size))

        face_rig_onnx_input = face_rig_input.transpose((2, 0, 1)).astype(np.float32)[np.newaxis, :, :, :] / 255.0
        face_rig_params = self.face_rig_net.run(self.onnx_output_name,
                                                {self.onnx_input_name: face_rig_onnx_input})
        face_rig_params = face_rig_params[0][0]
        return face_rig_params
