# Copyright (c) Phigent Robotics. All rights reserved.
import torch
import torch.nn.functional as F
from mmcv.runner import force_fp32

from mmdet3d.ops.bev_pool_v2.bev_pool import TRTBEVPoolv2
from mmdet.models import DETECTORS
from .. import builder
from .centerpoint import CenterPoint
from .bevdet import BEVDet

@DETECTORS.register_module()
class BEVDet_intri(BEVDet):
    r"""BEVDet paradigm for multi-camera 3D object detection.

    Please refer to the `paper <https://arxiv.org/abs/2112.11790>`_

    Args:
        img_view_transformer (dict): Configuration dict of view transformer.
        img_bev_encoder_backbone (dict): Configuration dict of the BEV encoder
            backbone.
        img_bev_encoder_neck (dict): Configuration dict of the BEV encoder neck.
    """
   
    def extract_img_feat(self, img, img_metas, **kwargs):
        """Extract features of images."""
        x = self.image_encoder(img[0])
        x, depth = self.img_view_transformer([x] + img[1:8]) # 多输入一个
        x = self.bev_encoder(x)
        return [x], depth



