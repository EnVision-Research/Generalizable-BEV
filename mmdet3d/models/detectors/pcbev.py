# Copyright (c) Phigent Robotics. All rights reserved.
import torch
import torch.nn.functional as F
from mmcv.runner import force_fp32
import copy
import cv2
import numpy as np

from mmdet3d.ops.bev_pool_v2.bev_pool import TRTBEVPoolv2
from mmdet.models import DETECTORS
from .. import builder
from .centerpoint import CenterPoint
from .bevdet import BEVDet



@DETECTORS.register_module()
class PCBEV_DG(BEVDet):
    def __init__(self, img_view_transformer, img_bev_encoder_backbone,
                 img_bev_encoder_neck, img_aug, bev_img_aux, detach_flag=1, **kwargs):
        super(PCBEV_DG, self).__init__(img_view_transformer, img_bev_encoder_backbone,
                 img_bev_encoder_neck, **kwargs)
        self.detach_flag = detach_flag
        self.img_aug_cfg = img_aug
        self.bev_img_aux_cfg = bev_img_aux
        self.img_aux = builder.build_neck(self.img_aug_cfg)
        self.bev_img_aux = builder.build_neck(self.bev_img_aux_cfg)
        self.bev_index = [0]

    def forward_pts_train(self,
                          pts_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          img_metas,
                          bev_heatmap=None,
                          pseudo_flag=False,
                          gt_bboxes_ignore=None):
        """Forward function for point cloud branch.

        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.

        Returns:
            dict: Losses of each branch.
        """
        outs = self.pts_bbox_head(pts_feats)
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
        losses = self.pts_bbox_head.loss(*loss_inputs)
        loss_bev = self.pts_bbox_head.bev_loss(bev_heatmap)
        losses.update({'loss_bev_heatmap_source': loss_bev})


        return losses

    def extract_feat(self, points, img, img_metas, **kwargs):
        """Extract features from images and points."""
        img_feats, depth_real, depth_vitual, loss_dict = self.extract_img_feat(img, img_metas, **kwargs)
        pts_feats = None
        return (img_feats, pts_feats, depth_real, depth_vitual, loss_dict)


    def extract_img_feat(self, points, img, img_metas, gt_bboxes_3d, gt_labels_3d, **kwargs):

        """Extract features of images."""
        imgs, rots, trans, intrins, post_rots, post_trans, bda, intri_actually = img
        mlp_input = self.img_view_transformer.get_mlp_input(
            rots, trans, intrins, post_rots, post_trans, bda, intri_actually)

        rot_augs = kwargs['bev_aug']['rot_augs']
        tran_augs = kwargs['bev_aug']['tran_augs']
        mlp_input_aug = self.img_view_transformer.get_mlp_input(
            rot_augs, tran_augs, intrins, post_rots, post_trans, bda, intri_actually)

        img_feature = self.image_encoder(imgs)

        loss_dict = dict()
        x, depth_real, depth_vitual = self.img_view_transformer(
            [img_feature, rots, trans, intrins, post_rots, post_trans, bda, mlp_input, intri_actually])
        bev_feats_source = self.bev_encoder(x)
        # source 2d data
        loss_heatmaps_source, loss_regess_source, heatmaps_source = self.img_aux(img_feature, **kwargs)
        # source bev data
        BEV_features_source, unused_loss_bev_source = self.img_view_transformer.get_BEV_feats_from_voxel(bev_feats_source)
        PV_features, unused_loss = self.img_view_transformer.get_PV_feats(bev_feats_source)
        PV_features_aug, unused_loss_aug = self.img_view_transformer.get_PV_feats_aug(bev_feats_source, [intrins, post_rots, post_trans, bda, intri_actually], **kwargs)
        loss_pv_heatmaps, loss_pv_regess, pv_heatmaps= self.bev_img_aux(PV_features, mlp_input,'img', depth_flag='real', **kwargs)
        loss_pv_heatmaps_aug, loss_pv_regess_aug, pv_heatmaps_aug = self.bev_img_aux(PV_features_aug, mlp_input_aug,'img_aug', depth_flag='real', **kwargs)
        bev_heatmaps, bev_unused_loss = self.bev_img_aux.get_heatmaps(BEV_features_source)
        # consistency
        loss_dict.update({'loss_heatmaps_source': loss_heatmaps_source})
        loss_dict.update({'loss_regess_source': loss_regess_source})
        loss_dict.update({'loss_pv_heatmaps_source': loss_pv_heatmaps})
        loss_dict.update({'loss_pv_regess_source': loss_pv_regess})
        loss_dict.update({'loss_pv_heatmaps_aug_source': loss_pv_heatmaps_aug})
        loss_dict.update({'loss_pv_regess_aug_source': loss_pv_regess_aug})

        loss_dict.update({'loss_unused_xxx': 0.0*(unused_loss_bev_source+unused_loss+unused_loss_aug+bev_unused_loss
                                                )})
        pts_feats = None
        return [bev_feats_source], bev_heatmaps,  pts_feats, depth_real, depth_vitual, loss_dict


    def simple_test(self,
                    points,
                    img_metas,
                    img=None,
                    rescale=False,
                    **kwargs):
        """Test function without augmentaiton."""
        img_feats, _, _, _, _= self.extract_feat(
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


        img_feats, bev_heatmaps, pts_feats, depth_real, depth_vitual, loss_dict = self.extract_img_feat(
            points, img=img_inputs, img_metas=img_metas, gt_bboxes_3d=gt_bboxes_3d, gt_labels_3d=gt_labels_3d, **kwargs)

        gt_depth = kwargs['ann_maps_2d']
        loss_depth = self.img_view_transformer.get_depth_loss(gt_depth, depth_vitual)
        losses = dict(loss_depth_source=loss_depth)
        losses.update(loss_dict)
        losses_pts = self.forward_pts_train(img_feats, gt_bboxes_3d,
                                            gt_labels_3d, img_metas, bev_heatmap=bev_heatmaps,
                                            pseudo_flag=False, gt_bboxes_ignore=None)
        losses.update(losses_pts)

        return losses



@DETECTORS.register_module()
class PCBEV_UDA(BEVDet):
    def __init__(self, loss_flag, thr, img_view_transformer, img_bev_encoder_backbone,
                 img_bev_encoder_neck, img_aug, bev_img_aux,detach_flag=1, **kwargs):
        super(PCBEV_UDA, self).__init__(img_view_transformer, img_bev_encoder_backbone,
                 img_bev_encoder_neck, **kwargs)
        self.eps = 1e-12
        self.alpha = 2.0
        self.gamma = 4.0
        self.thr = thr
        self.thr_low = 0.5
        self.loss_flag = loss_flag
        self.detach_flag = detach_flag
        self.img_aug_cfg = img_aug
        self.bev_img_aux_cfg = bev_img_aux
        self.img_aux = builder.build_neck(self.img_aug_cfg)
        self.bev_img_aux = builder.build_neck(self.bev_img_aux_cfg)
        self.bev_index = [0]

    def forward_pts_train(self,
                          pts_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          img_metas,
                          bev_heatmap=None,
                          pseudo_flag=False,
                          gt_bboxes_ignore=None):
        """Forward function for point cloud branch.

        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.

        Returns:
            dict: Losses of each branch.
        """
        outs = self.pts_bbox_head(pts_feats)
        if pseudo_flag == False:
            loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
            losses = self.pts_bbox_head.loss(*loss_inputs)
            # loss_bev = self.pts_bbox_head.bev_loss(bev_heatmap)
            # losses.update({'loss_bev_heatmap_source': loss_bev})
        else:
            loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
            losses = self.pts_bbox_head.loss_pseudo(*loss_inputs)
            loss_pseudo, loss_consis = self.pts_bbox_head.bev_loss_pseudo(bev_heatmap, outs)
            losses.update({'loss_bev_pseudo_target': loss_pseudo})
            losses.update({'loss_bev_consis_target': loss_consis})
        return losses

    def extract_feat(self, points, img, img_metas, **kwargs):
        """Extract features from images and points."""
        img_feats, depth_real, depth_vitual, loss_dict = self.extract_img_feat(img, img_metas, **kwargs)
        pts_feats = None
        return (img_feats, pts_feats, depth_real, depth_vitual, loss_dict)





    def get_psudo_label(self, pre1, pre2):


        gts = pre1.clone().detach()

        pos_weights = gts.gt(self.thr)

        pos_temp = pos_weights.reshape(-1)
        pos_num = len(pos_temp[pos_temp > 0.9])
        neg_num = len(pos_temp) - pos_num


        neg_weights = (1 - gts).pow(self.gamma)
        neg_weights = neg_weights.gt(self.thr_low)
        pos_loss = -(pre2 + self.eps).log() * (1 - pre2).pow(self.alpha) * pos_weights
        neg_loss = -(1 - pre2 + self.eps).log() * pre2.pow(self.alpha) * neg_weights

        pos_loss = pos_loss.sum() / (pos_num+0.1)
        neg_loss = neg_loss.sum() / (neg_num+0.1)

        return pos_loss+neg_loss

    def neg_person(self, heatmap1, heatmap2):
        if self.detach_flag==1:
            heatmap1 = heatmap1.clone().detach()
        elif self.detach_flag==2:
            heatmap1 = heatmap2.clone().detach()
        B, C, H, W = heatmap1.shape
        loss_consis = 0
        for index in self.bev_index:
            pred1 = heatmap1[:, index, ...]
            pred2 = heatmap2[:, index, ...]
            if self.loss_flag=='sim':
                cossim = torch.cosine_similarity(pred1.reshape(B, -1), pred2.reshape(B, -1), dim=1)
            else:
                cossim = self.get_psudo_label(pred1, pred2)
            loss_consis = loss_consis + 1-cossim
        return loss_consis




    def extract_img_feat_split(self, points, img, img_metas, gt_bboxes_3d, gt_labels_3d, **kwargs):
        img_inputs_source, img_metas_source, depth_source, \
        kwargs_source, bboxes_3d_source, labels_3d_source, index = self.Data_split(img, img_metas, kwargs['gt_depth'],
                                                                                   gt_bboxes_3d, gt_labels_3d, kwargs,
                                                                                   kwargs['Target_Domain'],
                                                                                   domain='source')

        img_inputs_target, img_metas_target, depth_target, \
        kwargs_target, bboxes_3d_target, labels_3d_target, index_target = self.Data_split(img, img_metas,
                                                                                          kwargs['gt_depth'],
                                                                                          gt_bboxes_3d, gt_labels_3d,
                                                                                          kwargs,
                                                                                          kwargs['Target_Domain'],
                                                                                          domain='target')
        """Extract features of images."""
        imgs, rots, trans, intrins, post_rots, post_trans, bda, intri_actually = img
        mlp_input = self.img_view_transformer.get_mlp_input(
            rots, trans, intrins, post_rots, post_trans, bda, intri_actually)

        rot_augs = kwargs['bev_aug']['rot_augs']
        tran_augs = kwargs['bev_aug']['tran_augs']
        mlp_input_aug = self.img_view_transformer.get_mlp_input(
            rot_augs, tran_augs, intrins, post_rots, post_trans, bda, intri_actually)

        img_feature = self.image_encoder(imgs)

        loss_dict = dict()

        x, depth_real, depth_vitual = self.img_view_transformer(
            [img_feature, rots, trans, intrins, post_rots, post_trans, bda, mlp_input, intri_actually])
        # print('x', x.shape)  # ([4, 80, 128, 128]
        bev_feats = self.bev_encoder(x)
        # print('bev_feats', bev_feats.shape) # ([4, 256, 128, 128]
        bev_feats_source = bev_feats[index]
        bev_feats_target = bev_feats[index_target]
        # source 2d data
        loss_heatmaps_source, loss_regess_source, heatmaps_source = self.img_aux(img_feature[index], **kwargs_source)
        # source bev data
        BEV_features_source, unused_loss_bev_source = self.img_view_transformer.get_BEV_feats_from_voxel(bev_feats_source)
        PV_features, unused_loss = self.img_view_transformer.get_PV_feats(bev_feats_source, index=index)
        PV_features_aug, unused_loss_aug = self.img_view_transformer.get_PV_feats_aug(bev_feats_source, [intrins[index], post_rots[index], post_trans[index], bda[index], intri_actually[index]], **kwargs_source)
        loss_pv_heatmaps, loss_pv_regess, pv_heatmaps= self.bev_img_aux(PV_features, mlp_input[index], 'img', depth_flag='real', **kwargs_source)
        loss_pv_heatmaps_aug, loss_pv_regess_aug, pv_heatmaps_aug = self.bev_img_aux(PV_features_aug, mlp_input_aug[index], 'img_aug', depth_flag='real', **kwargs_source)
        bev_heatmaps, bev_unused_loss = self.bev_img_aux.get_heatmaps(BEV_features_source)

        #  heatmaps_source pv_heatmaps
        loss_dict.update({'loss_heatmaps_source': loss_heatmaps_source})
        loss_dict.update({'loss_regess_source': loss_regess_source})
        loss_dict.update({'loss_pv_heatmaps_source': loss_pv_heatmaps})
        loss_dict.update({'loss_pv_regess_source': loss_pv_regess})
        loss_dict.update({'loss_pv_heatmaps_aug_source': loss_pv_heatmaps_aug})
        loss_dict.update({'loss_pv_regess_aug_source': loss_pv_regess_aug})
        # target 2d data
        loss_heatmaps_pseudo_target, loss_unused_target, heatmaps_pseudo_target = self.img_aux.pseudo(img_feature[index_target], **kwargs_target)
        # target bev data
        BEV_features_target, unused_loss_bev_target = self.img_view_transformer.get_BEV_feats_from_voxel(
            bev_feats_target)
        bev_heatmaps_target, bev_unused_loss_target = self.bev_img_aux.get_heatmaps(BEV_features_target)
        PV_features_target, unused_loss_target = self.img_view_transformer.get_PV_feats(bev_feats_target, index=index_target)

        loss_pv_heatmaps_target, pv_unused_target, pv_heatmaps_target = self.bev_img_aux.pseudo(PV_features_target)

        loss_2Dconsis_target = self.neg_person(heatmaps_pseudo_target, pv_heatmaps_target)

        loss_dict.update({'loss_2Dconsis_target': loss_2Dconsis_target})
        loss_dict.update({'loss_unused_xxx': 0.0*(unused_loss_bev_source+unused_loss_bev_target+loss_unused_target+\
                                                  unused_loss_target+unused_loss+unused_loss_aug+bev_unused_loss\
                                                  +bev_unused_loss_target+pv_unused_target)})
        pts_feats = None
        return [bev_feats_source], [bev_feats_target], bev_heatmaps, bev_heatmaps_target, pts_feats, depth_real, depth_vitual, loss_dict


    def simple_test(self,
                    points,
                    img_metas,
                    img=None,
                    rescale=False,
                    **kwargs):
        """Test function without augmentaiton."""
        img_feats, _, _, _, _= self.extract_feat(
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



        img_inputs_source, img_metas_source, depth_source, \
        kwargs_source, bboxes_3d_source, labels_3d_source, index = self.Data_split(img_inputs, img_metas, kwargs['gt_depth'],
                                                              gt_bboxes_3d, gt_labels_3d, kwargs, kwargs['Target_Domain'],
                                                              domain='source')
        # 提取
        img_feats_source, img_feats_target, bev_heatmaps, bev_heatmaps_target, pts_feats, depth_real, depth_vitual, loss_dict = self.extract_img_feat_split(
            points, img=img_inputs, img_metas=img_metas, gt_bboxes_3d=gt_bboxes_3d, gt_labels_3d=gt_labels_3d, **kwargs)

        BN, C, H, W = depth_vitual.shape
        depth_vitual = depth_vitual.reshape(-1, 6, C, H, W)
        gt_depth_source = kwargs_source['ann_maps_2d']
        loss_depth = self.img_view_transformer.get_depth_loss(gt_depth_source, depth_vitual[index].reshape(-1, C, H, W))
        losses = dict(loss_depth_source=loss_depth)
        losses.update(loss_dict)
        losses_pts_source = self.forward_pts_train(img_feats_source, bboxes_3d_source,
                                            labels_3d_source, img_metas_source, bev_heatmap=bev_heatmaps,
                                            pseudo_flag=False, gt_bboxes_ignore=None)
        losses.update(losses_pts_source)
        # losses_pts_target = self.forward_pts_train(img_feats_target, bboxes_3d_source,
        #                                     labels_3d_source, img_metas_source, bev_heatmap=bev_heatmaps_target,
        #                                     pseudo_flag=True, gt_bboxes_ignore=None)
        # losses.update(losses_pts_target)

        return losses


    def Img_split(self, img_inputs, index):
        imgs, rots, trans, intrins, post_rots, post_trans, bda, intri_actually = img_inputs
        return imgs[index], rots[index], trans[index], intrins[index], post_rots[index], post_trans[index], bda[index], intri_actually[index]

    def List_split(self, img_metas, index):
        return [img_metas[i] for i in index]

    def List_split2(self, img_metas, index):
        return [img_metas[i] for i in index]

    def Tensor_split(self, tensor, index):
        return tensor[index]

    def Get_Domain_index(self, flag, domain='target'):
        if domain=='target':
            index = torch.nonzero(flag)
        else:
            index = torch.nonzero(~flag)
        index = index.cpu().numpy()
        index = [item[0] for item in index]
        return index

    def Data_split(self, img_inputs, img_metas, depth, bboxes_3d, labels_3d, kwargs, flag, domain='target'):
        if domain=='target':
            index = torch.nonzero(flag)
        else:
            index = torch.nonzero(~flag)
        index = index.cpu().numpy()
        index = [item[0] for item in index]
        kwargs_split = dict()
        kwargs_split.update(dict(bev_aug=dict()))
        kwargs_split.update({'heatmaps_2d': self.List_split(kwargs['heatmaps_2d'], index)})
        kwargs_split.update({'ann_maps_2d': self.List_split(kwargs['ann_maps_2d'], index)})
        kwargs_split.update({'heatmap_masks_2d': self.List_split(kwargs['heatmap_masks_2d'], index)})
        kwargs_split.update({'heatmaps_2d_aug': self.List_split(kwargs['heatmaps_2d_aug'], index)})
        kwargs_split.update({'ann_maps_2d_aug': self.List_split(kwargs['ann_maps_2d_aug'], index)})
        kwargs_split.update({'heatmap_masks_2d_aug': self.List_split(kwargs['heatmap_masks_2d_aug'], index)})
        kwargs_split['bev_aug'].update({'tran_augs': self.Tensor_split(kwargs['bev_aug']['tran_augs'], index)})
        kwargs_split['bev_aug'].update({'rot_augs': self.Tensor_split(kwargs['bev_aug']['rot_augs'], index)})
        img_inputs_split = self.Img_split(img_inputs, index)
        img_metas_split = self.List_split(img_metas, index)
        depth_split = self.Tensor_split(depth, index)
        bboxes_3d_split = self.List_split(bboxes_3d, index)
        labels_3d_split = self.List_split(labels_3d, index)
        return img_inputs_split, img_metas_split, depth_split, kwargs_split, bboxes_3d_split, labels_3d_split, index



