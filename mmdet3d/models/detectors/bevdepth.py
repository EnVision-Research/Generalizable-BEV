# Copyright (c) Phigent Robotics. All rights reserved.
import torch
import torch.nn.functional as F
from mmcv.runner import force_fp32

from mmdet.models import DETECTORS
from .. import builder
from .bevdet import BEVDet

import pdb

@DETECTORS.register_module()
class BEVDepth(BEVDet):
    def extract_img_feat(self, img, img_metas, **kwargs):
        """Extract features of images."""
        imgs, rots, trans, intrins, post_rots, post_trans, bda, _ = img
        mlp_input = self.img_view_transformer.get_mlp_input(
                rots, trans, intrins, post_rots, post_trans, bda)
        x = self.image_encoder(imgs)
        x, depth = self.img_view_transformer([x, rots, trans, intrins, post_rots, post_trans, bda, mlp_input])
        x = self.bev_encoder(x)
        
        return [x], depth
    
    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img_inputs=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      **kwargs):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """
        img_feats, pts_feats, depth = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas, **kwargs)
        gt_depth = kwargs['gt_depth']
        loss_depth = self.img_view_transformer.get_depth_loss(gt_depth, depth)
        losses = dict(loss_depth=loss_depth)
        losses_pts = self.forward_pts_train(img_feats, gt_bboxes_3d,
                                            gt_labels_3d, img_metas,
                                            gt_bboxes_ignore)
        losses.update(losses_pts)
        return losses


@DETECTORS.register_module()
class BEVDepth_DG(BEVDet):

    def extract_feat(self, points, img, img_metas, **kwargs):
        """Extract features from images and points."""
        img_feats, depth_real, depth_vitual = self.extract_img_feat(img, img_metas, **kwargs)
        pts_feats = None
        return (img_feats, pts_feats, depth_real, depth_vitual)

    def extract_img_feat(self, img, img_metas, **kwargs):
        """Extract features of images."""
        imgs, rots, trans, intrins, post_rots, post_trans, bda, intri_actually = img
        mlp_input = self.img_view_transformer.get_mlp_input(
            rots, trans, intrins, post_rots, post_trans, bda, intri_actually)
        x = self.image_encoder(imgs)
        x, depth_real, depth_vitual = self.img_view_transformer(
            [x, rots, trans, intrins, post_rots, post_trans, bda, mlp_input, intri_actually])
        x = self.bev_encoder(x)
        return [x], depth_real, depth_vitual

    def simple_test(self,
                    points,
                    img_metas,
                    img=None,
                    rescale=False,
                    **kwargs):
        """Test function without augmentaiton."""
        img_feats, _, _, _ = self.extract_feat(
            points, img=img, img_metas=img_metas, **kwargs)
        bbox_list = [dict() for _ in range(len(img_metas))]
        bbox_pts = self.simple_test_pts(img_feats, img_metas, rescale=rescale)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
            result_dict['img_metas'] = img_metas
        return bbox_list  # img_metas #

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img_inputs=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      **kwargs):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """
        img_feats, pts_feats, depth_real, depth_vitual = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas, **kwargs)
        gt_depth = kwargs['gt_depth']
        loss_depth = self.img_view_transformer.get_depth_loss(gt_depth, depth_vitual)
        losses = dict(loss_depth=loss_depth)
        losses_pts = self.forward_pts_train(img_feats, gt_bboxes_3d,
                                            gt_labels_3d, img_metas,
                                            gt_bboxes_ignore)
        losses.update(losses_pts)
        return losses










