# Copyright (c) Phigent Robotics. All rights reserved.

import torch.utils.checkpoint as checkpoint
from torch import nn
import torch
import copy

from mmcv.cnn import ConvModule
from ..builder import NECKS
from mmdet.models import BACKBONES
from mmdet.models.backbones.resnet import BasicBlock, Bottleneck
from ..builder import HEADS, build_loss
from mmdet.core import build_bbox_coder, multi_apply, reduce_mean
from mmdet3d.models.utils import clip_sigmoid

class Mlp(nn.Module):

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.ReLU,
                 drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class SELayer(nn.Module):

    def __init__(self, channels, act_layer=nn.ReLU, gate_layer=nn.Sigmoid):
        super().__init__()
        self.conv_reduce = nn.Conv2d(channels, channels, 1, bias=True)
        self.act1 = act_layer()
        self.conv_expand = nn.Conv2d(channels, channels, 1, bias=True)
        self.gate = gate_layer()

    def forward(self, x, x_se):
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate(x_se)

def gaussian_focal_loss(pred, gaussian_target, alpha=2.0, gamma=4.0):
    """`Focal Loss <https://arxiv.org/abs/1708.02002>`_ for targets in gaussian
    distribution.

    Args:
        pred (torch.Tensor): The prediction.
        gaussian_target (torch.Tensor): The learning target of the prediction
            in gaussian distribution.
        alpha (float, optional): A balanced form for Focal Loss.
            Defaults to 2.0.
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 4.0.
    """
    eps = 1e-12
    # print('gaussian_target.shape', gaussian_target.shape)
    # print('pred.shape', pred.shape)
    pos_weights = gaussian_target.eq(1)
    pos_temp = pos_weights.reshape(-1)
    pos_num = len(pos_temp[pos_temp>0.9])
    neg_num = len(pos_temp) - pos_num
    neg_weights = (1 - gaussian_target).pow(gamma)
    pos_loss = -(pred + eps).log() * (1 - pred).pow(alpha) * pos_weights
    neg_loss = -(1 - pred + eps).log() * pred.pow(alpha) * neg_weights
    # print(pos_loss.size())
    # print(neg_loss.size())
    pos_loss = pos_loss.sum() / (pos_num+0.1)
    neg_loss = neg_loss.sum() / (neg_num+0.1)
    # print('pos_num', pos_num)
    # print('neg_num', neg_num)
    # print('pos_loss, neg_loss', pos_loss, neg_loss)
    # print('pos_loss', pos_loss.max().max().max())
    # print('pos_lossmin', pos_loss.min().min().min())
    # print('neg_loss', neg_loss.max().max().max())
    # print('neg_lossmin', neg_loss.min().min().min())
    # print('neg_loss', neg_loss.shape)
    # print('pos_loss', pos_loss.)
    # print('neg_loss', neg_loss)
    return pos_loss + neg_loss


class GaussianFocalLoss(nn.Module):

    def __init__(self,
                 alpha=2.0,
                 gamma=4.0,
                 reduction='mean',
                 loss_weight=1.0):
        super(GaussianFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction
                in gaussian distribution.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_reg = self.loss_weight * gaussian_focal_loss(pred, target)
        # print('loss_reg.shape', loss_reg.shape)
        return loss_reg






@NECKS.register_module(force=True)
class Img_Aux(nn.Module):

    def __init__(
            self,
            numC_input,
            class_name,
            num_layer=[2, 2, 2],
            num_channels=None,
            stride=[1, 1, 1],
            upsample=[1, 1, 1],
            backbone_output_ids=None,
            norm_cfg=dict(type='BN'),
            with_cp=False,
            block_type='Basic',
    ):
        super(Img_Aux, self).__init__()
        # build backbone
        assert len(num_layer) == len(stride)
        self.class_num = len(class_name)
        self.eps = 1e-12
        self.alpha = 2.0
        self.gamma = 4.0
        self.thr = 0.5
        self.thr_low = 0.5

        self.pseudo_index = [0]
        self.class_num_pseudo = len(self.pseudo_index)
        num_channels = [numC_input*2**(i+1) for i in range(len(num_layer))] \
            if num_channels is None else num_channels
        self.backbone_output_ids = range(len(num_layer)) \
            if backbone_output_ids is None else backbone_output_ids
        layers = []
        if block_type == 'BottleNeck':
            curr_numC = numC_input
            for i in range(len(num_layer)):
                layer = [
                    Bottleneck(
                        curr_numC,
                        num_channels[i] // 4,
                        stride=stride[i],
                        downsample=nn.Conv2d(curr_numC, num_channels[i], 3,
                                             stride[i], 1),
                        norm_cfg=norm_cfg)
                ]
                curr_numC = num_channels[i]
                layer.extend([
                    nn.ConvTranspose2d(curr_numC, curr_numC,
                                       kernel_size=upsample[i], stride=upsample[i], padding=0),
                    nn.BatchNorm2d(curr_numC),
                    nn.ReLU(inplace=True),
                ])
                layer.extend([
                    Bottleneck(curr_numC, curr_numC // 4, norm_cfg=norm_cfg)
                    for _ in range(num_layer[i] - 1)
                ])
                layers.append(nn.Sequential(*layer))
        elif block_type == 'Basic':
            curr_numC = numC_input
            for i in range(len(num_layer)):
                layer = [
                    BasicBlock(
                        curr_numC,
                        num_channels[i],
                        stride=stride[i],
                        downsample=nn.Conv2d(curr_numC, num_channels[i], 3,
                                             stride[i], 1),
                        norm_cfg=norm_cfg)
                ]
                curr_numC = num_channels[i]
                layer.extend([
                    nn.ConvTranspose2d(curr_numC, curr_numC,
                                       kernel_size=upsample[i], stride=upsample[i], padding=0),
                    nn.BatchNorm2d(curr_numC),
                    nn.ReLU(inplace=True),
                ])
                layer.extend([
                    BasicBlock(curr_numC, curr_numC, norm_cfg=norm_cfg)
                    for _ in range(num_layer[i] - 1)
                ])
                layers.append(nn.Sequential(*layer))
        else:
            assert False
        self.layers = nn.Sequential(*layers)
        self.with_cp = with_cp

        # loss_cls = dict(type='GaussianFocalLoss', reduction='mean')
        # loss_bbox = dict(type='L1Loss', reduction='mean', loss_weight=0.25)
        # bbox_coder = dict(
        #     type='CenterPointBBoxCoder',
        #     pc_range=point_cloud_range[:2],
        #     post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
        #     max_num=500,
        #     score_threshold=0.1,
        #     out_size_factor=8,
        #     voxel_size=voxel_size[:2],
        #     code_size=9)
        self.loss_cls = GaussianFocalLoss()
        # self.loss_bbox = build_loss(loss_bbox)
        self.task_loss_weight = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        #self.bbox_coder = build_bbox_coder(bbox_coder)

    def get_target(self, flag, **kwargs):

        if flag=='img':
            heatmaps_2d = kwargs['heatmaps_2d']
            ann_maps_2d = kwargs['ann_maps_2d']
            heatmap_masks_2d = kwargs['heatmap_masks_2d']
        elif flag=='img_aug':
            heatmaps_2d = kwargs['heatmaps_2d_aug']
            ann_maps_2d = kwargs['ann_maps_2d_aug']
            heatmap_masks_2d = kwargs['heatmap_masks_2d_aug']
        if type(heatmaps_2d) in [list, tuple]:
            heatmaps_2d = torch.stack(heatmaps_2d)  # [6, 6, 96, 176, 6]
            ann_maps_2d = torch.stack(ann_maps_2d)  # [6, 6, 96, 176, 7, 6]
            heatmap_masks_2d = torch.stack(heatmap_masks_2d)  # ([6, 6, 96, 176, 6])



        # print('flag')
        # print(heatmaps_2d.shape)
        # # heatmaps_2d = heatmaps_2d.permute(0, 1, 4, 2, 3)
        # # print('B_h, Cam_Num, C_h, W_h, H_h = heatmaps_2d.shape', heatmaps_2d.shape)
        # heatmaps_2d = torch.stack(heatmaps_2d)  # [6, 6, 96, 176, 6]
        # ann_maps_2d = torch.stack(ann_maps_2d)  # [6, 6, 96, 176, 7, 6]
        # heatmap_masks_2d = torch.stack(heatmap_masks_2d) # ([6, 6, 96, 176, 6])
        # print('heatmaps_2d', heatmaps_2d.shape)
        # print('ann_maps_2d', ann_maps_2d.shape) # [6, 6, 96, 176, 7, 6]
        # print('heatmap_masks_2d', heatmap_masks_2d.shape)
        return heatmaps_2d, ann_maps_2d, heatmap_masks_2d



    def tensor_split(self, feats, index_source, index_target, Cam_Num=6):
        BN, C, H, W = feats.shape
        feats = feats.reshape(-1, Cam_Num, C, H, W)
        feats_source = feats[index_source, ...].reshape(-1, C, H, W)
        index_target = feats[index_target, ...].reshape(-1, C, H, W)
        return feats_source, index_target


    def get_cls_loss(self, pre, gts):
        if gts.dim()==4:
            gts = gts.unsqueeze(0)
        if gts.dim() == 6:
            gts = gts.squeeze(0)
        B, N, H, W, C = gts.shape
        gts = gts.reshape(-1, H, W, C)
        loss_heatmaps = 0
        for task_id in range(self.class_num):
            # print('pre[:, task_id, :, :]', pre[:, task_id, :, :].shape)
            # print('gts[:, :, :, task_id]', gts[:, :, :, task_id].shape)
            loss_heatmap = self.loss_cls(pre[:, task_id, :, :], gts[:, :, :, task_id])
            loss_heatmaps = loss_heatmaps + self.task_loss_weight[task_id]*loss_heatmap
        return loss_heatmaps


    # def get_cls_loss(self, pre, gts):
    #     eps = 1e-12
    #     # print('gaussian_target.shape', gaussian_target.shape)
    #     # print('pred.shape', pred.shape)
    #     B, N, H, W, C = gts.shape
    #     gts = gts.reshape(-1, H, W, C)
    #     loss_heatmaps = pre.mean().mean().mean().mean()
    #     # loss_heatmaps = 0
    #     # for task_id in range(self.class_num):
    #     #     #pos_weights = gts[:, :, :, task_id].eq(1)
    #     #     #neg_weights = (1 - gts[:, :, :, task_id]).pow(4.0)
    #     #     # pos_loss = -(pre[:, task_id, :, :] + eps).log() * (1 - pre[:, task_id, :, :]).pow(2.0) * pos_weights
    #     #     pos_loss = pre[:, task_id, :, :].mean().mean().mean()
    #     #     #neg_loss = -(1 - pre[:, task_id, :, :] + eps).log() * pre[:, task_id, :, :].pow(2.0) * neg_weights
    #     #     loss_heatmaps = loss_heatmaps + self.task_loss_weight[task_id] * (pos_loss)
    #     return loss_heatmaps

    def get_regess_loss(self, pre, gts, masks, depth_flag='vitual'):
        # print('?????')
        # print('pre', pre.shape)
        # print('gts', gts.shape)
        # print('masks', masks.shape)
        # print('------')
        if gts.dim()==5:
            gts = gts.unsqueeze(0)
            masks = masks.unsqueeze(0)
        if gts.dim()==7:
            gts = gts.squeeze(0)
            masks = masks.squeeze(0)
        B, N, H, W, N_ann, C = gts.shape
        gts = gts.reshape(-1, H, W, N_ann, C)
        masks = masks.reshape(-1, H, W, 1, C).repeat(1, 1, 1, 3, 1)
        pre = pre.reshape(B*N, 6, C, H, W).permute(0, 3, 4, 1, 2)
        losses = 0
        # print('pre', pre.shape)
        # print('gts', gts.shape)
        # print('masks', masks.shape)
        for task_id in range(self.class_num):
            # print('pre[..., 0:2, task_id]', pre[..., 0:2, task_id].shape)
            # print('gts[..., 0:2, task_id]', gts[..., 0:2, task_id].shape)
            # print('masks[..., task_id]', masks[..., task_id].shape)
            loss_uv = torch.mul(masks[..., 0:2, task_id], torch.abs(pre[..., 0:2, task_id] - gts[..., 0:2, task_id]))
            loss_dim = torch.mul(masks[..., 0:3, task_id], torch.abs(pre[..., 3:6, task_id] - gts[..., 3:6, task_id]))
            if depth_flag=='vitual':
                loss_d = torch.mul(masks[..., 0, task_id], torch.abs(pre[..., 2, task_id] - gts[..., 2, task_id]))
            else:
                loss_d = torch.mul(masks[..., 0, task_id], torch.abs(pre[..., 2, task_id] - gts[..., 6, task_id]))
            # loss_rot = torch.mul(masks[..., 0:2, task_id], torch.abs(pre[..., 5:7, task_id] - gts[..., 5:7, task_id]))
            # print('loss_uv', loss_uv.shape)
            # print('loss_d', loss_d.shape)
            # print('loss_dim', loss_dim.shape)
            loss = loss_uv.mean(dim=-1) + loss_d + loss_dim.mean(dim=-1) # + loss_rot.mean(dim=-1)
            losses = losses + self.task_loss_weight[task_id] * loss/3
        return losses


    def get_psudo_label(self, cls_pre, ann_pre):
        all = 0
        for index in self.pseudo_index:
            pred = cls_pre[:, index, ...]
            gts = pred.clone().detach()
            pos_weights = gts.gt(self.thr)

            pos_temp = pos_weights.reshape(-1)
            pos_num = len(pos_temp[pos_temp > 0.9])
            neg_num = len(pos_temp) - pos_num


            neg_weights = (1 - gts).pow(self.gamma)
            neg_weights = neg_weights.gt(self.thr_low)
            pos_loss = -(pred + self.eps).log() * (1 - pred).pow(self.alpha) * pos_weights
            neg_loss = -(1 - pred + self.eps).log() * pred.pow(self.alpha) * neg_weights

            pos_loss = pos_loss.sum() / (pos_num+0.1)
            neg_loss = neg_loss.sum() / (neg_num+0.1)

            all = all+pos_loss+neg_loss
        return all

    def pseudo(self, x, **kwargs):
        if x.dim()==5:
            B, Cam_Num, C, H, W = x.shape
            x = x.reshape(-1, C, H, W)
        feats = []
        x_tmp = x
        for lid, layer in enumerate(self.layers):
            if self.with_cp:
                x_tmp = checkpoint.checkpoint(layer, x_tmp)
            else:
                x_tmp = layer(x_tmp)
            if lid in self.backbone_output_ids:
                feats.append(x_tmp)
        if type(feats) in [list, tuple]:
            feats = feats[-1]
        unused_loss = 0.0 * feats.mean().mean().mean().mean()
        cls_pre = torch.sigmoid(feats[:, 0:self.class_num, ...])
        ann_pre = feats[:, self.class_num:, ...]
        pseudo_loss = self.get_psudo_label(cls_pre, ann_pre)
        return pseudo_loss, unused_loss, cls_pre

    def get_heatmaps(self, x, flag='img', depth_flag='vitual', **kwargs):
        if x.dim() == 5:
            B, Cam_Num, C, H, W = x.shape
            x = x.reshape(-1, C, H, W)
        feats = []
        x_tmp = x
        for lid, layer in enumerate(self.layers):
            if self.with_cp:
                x_tmp = checkpoint.checkpoint(layer, x_tmp)
            else:
                x_tmp = layer(x_tmp)
            if lid in self.backbone_output_ids:
                feats.append(x_tmp)
        if type(feats) in [list, tuple]:
            feats = feats[-1]
        heatmaps = torch.sigmoid(feats[:, 0:self.class_num, ...])
        return heatmaps, feats.mean().mean().mean().mean()

    def forward(self, x, flag='img', depth_flag='vitual', **kwargs):
        if x.dim()==5:
            B, Cam_Num, C, H, W = x.shape
            x = x.reshape(-1, C, H, W)
        feats = []
        x_tmp = x
        for lid, layer in enumerate(self.layers):
            if self.with_cp:
                x_tmp = checkpoint.checkpoint(layer, x_tmp)
            else:
                x_tmp = layer(x_tmp)
            if lid in self.backbone_output_ids:
                feats.append(x_tmp)
        if type(feats) in [list, tuple]:
            feats = feats[-1]

        # feats = self.layers(x)
        # feats = x
        # for lid, layer in enumerate(self.layers):
        #     print(layer)
        #     feats = layer(feats)
        # for name, param in self.layers.named_parameters():
        #     if param.grad is None:
        #         print(name)
        # print('x', x.shape)
        # x_tmp = x
        # # BEV features:
        # for lid, layer in enumerate(self.layers):
        #     if self.with_cp:
        #         x_tmp = checkpoint.checkpoint(layer, x_tmp)
        #     else:
        #         x_tmp = layer(x_tmp)
        # feats = x_tmp

        #     # if self.with_cp:
        #     #     x_tmp = checkpoint.checkpoint(layer, x_tmp)
        #     # else:
        #
        #     # if lid in self.backbone_output_ids:
        #     #     feats.append(x_tmp)
        # feats = x_tmp

        heatmaps_2d, ann_maps_2d, heatmap_masks_2d = self.get_target(flag, **kwargs)

        # 计算source domain
        # loss_heatmaps_source =feats[:, 0:self.class_num, ...].mean().mean().mean().mean()
        # loss_regess_source = feats[:, self.class_num:, ...].mean().mean().mean().mean()
        # print('feats[:, 0:self.class_num, ...]', feats[:, 0:self.class_num, ...].shape)
        # print('heatmaps_2d', heatmaps_2d.shape)
        # print('feats[:, self.class_num:, ...]', feats[:, self.class_num:, ...].shape)
        # print('ann_maps_2d', ann_maps_2d.shape)
        heatmaps_pre = torch.sigmoid(feats[:, 0:self.class_num, ...])
        loss_heatmaps_source = self.get_cls_loss(heatmaps_pre, heatmaps_2d)
        loss_regess_source = self.get_regess_loss(feats[:, self.class_num:, ...],
                                               ann_maps_2d, heatmap_masks_2d, depth_flag)
        # kwargs.update({'heatmaps_source': loss_heatmaps_source})
        # kwargs.update({'regess_source': loss_regess_source})
        # # print('feats.shape', feats.shape)
        # # print('heatmaps_2d.shape', heatmaps_2d.shape)
        #
        # # num_pos = heatmaps[task_id].eq(1).float().sum().item()
        # # cls_avg_factor = torch.clamp(reduce_mean(heatmaps_2d[task_id].new_tensor(num_pos)), min=1)
        # loss_heatmaps = 0
        # for task_id in range(self.class_num):
        #     # num_pos = (heatmaps_2d[:, task_id, :, :].float().sum()/(H*W)).item()
        #     # print('num_pos', num_pos)
        #     # cls_avg_factor = torch.clamp(reduce_mean(heatmaps_2d[task_id].new_tensor(num_pos)), min=1)
        #     # print('cls_avg_factor', cls_avg_factor)
        #     # print('heatmaps_2d[:, task_id, :, :].max().max().max()', heatmaps_2d[:, task_id, :, :].max().max().max())
        #     # print('torch.where(heatmaps_2d[:, task_id, :, :] > 0.99)', torch.where(heatmaps_2d[:, task_id, :, :] > 0.99))
        #     # print('torch.where(heatmaps_2d[:, task_id, :, :] > 0.95)', torch.where(heatmaps_2d[:, task_id, :, :] > 0.95))
        #     # print('torch.where(heatmaps_2d[:, task_id, :, :] > 0.9)', torch.where(heatmaps_2d[:, task_id, :, :] > 0.9))
        #     # print('torch.where(heatmaps_2d[:, task_id, :, :] > 0.8)', torch.where(heatmaps_2d[:, task_id, :, :] > 0.8))
        #     loss_heatmap = self.loss_cls(feats[:, task_id, :, :], heatmaps_2d[:, task_id, :, :]) # , avg_factor=cls_avg_factor
        #     # print('loss_heatmap', loss_heatmap.shape)
        #     loss_heatmaps = loss_heatmaps + self.task_loss_weight[task_id]*loss_heatmap

        return loss_heatmaps_source, loss_regess_source, heatmaps_pre






@NECKS.register_module(force=True)
class Img_Aux_Dy(nn.Module):

    def __init__(
            self,
            numC_input,
            class_name,
            num_layer=[2, 2, 2],
            num_channels=None,
            stride=[1, 1, 1],
            upsample=[1, 1, 1],
            backbone_output_ids=None,
            norm_cfg=dict(type='BN'),
            with_cp=False,
            block_type='Basic',
    ):
        super(Img_Aux_Dy, self).__init__()
        # build backbone
        assert len(num_layer) == len(stride)
        self.class_num = len(class_name)
        self.eps = 1e-12
        self.alpha = 2.0
        self.gamma = 4.0
        self.thr = 0.5
        self.thr_low = 0.5

        self.bn = nn.BatchNorm1d(31)
        self.Pos_mlp = Mlp(31, numC_input, numC_input)
        self.Pos_se = SELayer(numC_input)  # NOTE: add camera-aware

        self.pseudo_index = [0]
        self.class_num_pseudo = len(self.pseudo_index)
        num_channels = [numC_input*2**(i+1) for i in range(len(num_layer))] \
            if num_channels is None else num_channels
        self.backbone_output_ids = range(len(num_layer)) \
            if backbone_output_ids is None else backbone_output_ids
        layers = []
        if block_type == 'BottleNeck':
            curr_numC = numC_input
            for i in range(len(num_layer)):
                layer = [
                    Bottleneck(
                        curr_numC,
                        num_channels[i] // 4,
                        stride=stride[i],
                        downsample=nn.Conv2d(curr_numC, num_channels[i], 3,
                                             stride[i], 1),
                        norm_cfg=norm_cfg)
                ]
                curr_numC = num_channels[i]
                layer.extend([
                    nn.ConvTranspose2d(curr_numC, curr_numC,
                                       kernel_size=upsample[i], stride=upsample[i], padding=0),
                    nn.BatchNorm2d(curr_numC),
                    nn.ReLU(inplace=True),
                ])
                layer.extend([
                    Bottleneck(curr_numC, curr_numC // 4, norm_cfg=norm_cfg)
                    for _ in range(num_layer[i] - 1)
                ])
                layers.append(nn.Sequential(*layer))
        elif block_type == 'Basic':
            curr_numC = numC_input
            for i in range(len(num_layer)):
                layer = [
                    BasicBlock(
                        curr_numC,
                        num_channels[i],
                        stride=stride[i],
                        downsample=nn.Conv2d(curr_numC, num_channels[i], 3,
                                             stride[i], 1),
                        norm_cfg=norm_cfg)
                ]
                curr_numC = num_channels[i]
                layer.extend([
                    nn.ConvTranspose2d(curr_numC, curr_numC,
                                       kernel_size=upsample[i], stride=upsample[i], padding=0),
                    nn.BatchNorm2d(curr_numC),
                    nn.ReLU(inplace=True),
                ])
                layer.extend([
                    BasicBlock(curr_numC, curr_numC, norm_cfg=norm_cfg)
                    for _ in range(num_layer[i] - 1)
                ])
                layers.append(nn.Sequential(*layer))
        else:
            assert False
        self.layers = nn.Sequential(*layers)
        self.with_cp = with_cp


        self.loss_cls = GaussianFocalLoss()
        # self.loss_bbox = build_loss(loss_bbox)
        self.task_loss_weight = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        #self.bbox_coder = build_bbox_coder(bbox_coder)

    def get_target(self, flag, **kwargs):

        if flag=='img':
            heatmaps_2d = kwargs['heatmaps_2d']
            ann_maps_2d = kwargs['ann_maps_2d']
            heatmap_masks_2d = kwargs['heatmap_masks_2d']
        elif flag=='img_aug':
            heatmaps_2d = kwargs['heatmaps_2d_aug']
            ann_maps_2d = kwargs['ann_maps_2d_aug']
            heatmap_masks_2d = kwargs['heatmap_masks_2d_aug']
        if type(heatmaps_2d) in [list, tuple]:
            heatmaps_2d = torch.stack(heatmaps_2d)  # [6, 6, 96, 176, 6]
            ann_maps_2d = torch.stack(ann_maps_2d)  # [6, 6, 96, 176, 7, 6]
            heatmap_masks_2d = torch.stack(heatmap_masks_2d)  # ([6, 6, 96, 176, 6])

        return heatmaps_2d, ann_maps_2d, heatmap_masks_2d



    def tensor_split(self, feats, index_source, index_target, Cam_Num=6):
        BN, C, H, W = feats.shape
        feats = feats.reshape(-1, Cam_Num, C, H, W)
        feats_source = feats[index_source, ...].reshape(-1, C, H, W)
        index_target = feats[index_target, ...].reshape(-1, C, H, W)
        return feats_source, index_target


    def get_cls_loss(self, pre, gts):
        if gts.dim()==4:
            gts = gts.unsqueeze(0)
        if gts.dim() == 6:
            gts = gts.squeeze(0)
        B, N, H, W, C = gts.shape
        gts = gts.reshape(-1, H, W, C)
        loss_heatmaps = 0
        for task_id in range(self.class_num):
            # print('pre[:, task_id, :, :]', pre[:, task_id, :, :].shape)
            # print('gts[:, :, :, task_id]', gts[:, :, :, task_id].shape)
            loss_heatmap = self.loss_cls(pre[:, task_id, :, :], gts[:, :, :, task_id])
            loss_heatmaps = loss_heatmaps + self.task_loss_weight[task_id]*loss_heatmap
        return loss_heatmaps


    def get_regess_loss(self, pre, gts, masks, depth_flag='vitual'):
        # print('?????')
        # print('pre', pre.shape)
        # print('gts', gts.shape)
        # print('masks', masks.shape)
        # print('------')
        if gts.dim()==5:
            gts = gts.unsqueeze(0)
            masks = masks.unsqueeze(0)
        if gts.dim()==7:
            gts = gts.squeeze(0)
            masks = masks.squeeze(0)
        B, N, H, W, N_ann, C = gts.shape
        gts = gts.reshape(-1, H, W, N_ann, C)
        masks = masks.reshape(-1, H, W, 1, C).repeat(1, 1, 1, 3, 1)
        pre = pre.reshape(B*N, 6, C, H, W).permute(0, 3, 4, 1, 2)
        losses = 0
        # print('pre', pre.shape)
        # print('gts', gts.shape)
        # print('masks', masks.shape)
        for task_id in range(self.class_num):
            # print('pre[..., 0:2, task_id]', pre[..., 0:2, task_id].shape)
            # print('gts[..., 0:2, task_id]', gts[..., 0:2, task_id].shape)
            # print('masks[..., task_id]', masks[..., task_id].shape)
            loss_uv = torch.mul(masks[..., 0:2, task_id], torch.abs(pre[..., 0:2, task_id] - gts[..., 0:2, task_id]))
            loss_dim = torch.mul(masks[..., 0:3, task_id], torch.abs(pre[..., 3:6, task_id] - gts[..., 3:6, task_id]))
            if depth_flag=='vitual':
                loss_d = torch.mul(masks[..., 0, task_id], torch.abs(pre[..., 2, task_id] - gts[..., 2, task_id]))
            else:
                loss_d = torch.mul(masks[..., 0, task_id], torch.abs(pre[..., 2, task_id] - gts[..., 6, task_id]))
            # loss_rot = torch.mul(masks[..., 0:2, task_id], torch.abs(pre[..., 5:7, task_id] - gts[..., 5:7, task_id]))
            # print('loss_uv', loss_uv.shape)
            # print('loss_d', loss_d.shape)
            # print('loss_dim', loss_dim.shape)
            loss = loss_uv.mean(dim=-1) + loss_d + loss_dim.mean(dim=-1) # + loss_rot.mean(dim=-1)
            losses = losses + self.task_loss_weight[task_id] * loss/3
        return losses


    def get_psudo_label(self, cls_pre, ann_pre):
        all = 0
        for index in self.pseudo_index:
            pred = cls_pre[:, index, ...]
            gts = pred.clone().detach()
            pos_weights = gts.gt(self.thr)

            pos_temp = pos_weights.reshape(-1)
            pos_num = len(pos_temp[pos_temp > 0.9])
            neg_num = len(pos_temp) - pos_num


            neg_weights = (1 - gts).pow(self.gamma)
            neg_weights = neg_weights.gt(self.thr_low)
            pos_loss = -(pred + self.eps).log() * (1 - pred).pow(self.alpha) * pos_weights
            neg_loss = -(1 - pred + self.eps).log() * pred.pow(self.alpha) * neg_weights

            pos_loss = pos_loss.sum() / (pos_num+0.1)
            neg_loss = neg_loss.sum() / (neg_num+0.1)

            all = all+pos_loss+neg_loss
        return all

    def pseudo(self, x, **kwargs):
        if x.dim()==5:
            B, Cam_Num, C, H, W = x.shape
            x = x.reshape(-1, C, H, W)
        feats = []
        x_tmp = x
        for lid, layer in enumerate(self.layers):
            if self.with_cp:
                x_tmp = checkpoint.checkpoint(layer, x_tmp)
            else:
                x_tmp = layer(x_tmp)
            if lid in self.backbone_output_ids:
                feats.append(x_tmp)
        if type(feats) in [list, tuple]:
            feats = feats[-1]
        unused_loss = 0.0 * feats.mean().mean().mean().mean()
        cls_pre = torch.sigmoid(feats[:, 0:self.class_num, ...])
        ann_pre = feats[:, self.class_num:, ...]
        pseudo_loss = self.get_psudo_label(cls_pre, ann_pre)
        return pseudo_loss, unused_loss, cls_pre

    def get_heatmaps(self, x, flag='img', depth_flag='vitual', **kwargs):
        if x.dim() == 5:
            B, Cam_Num, C, H, W = x.shape
            x = x.reshape(-1, C, H, W)
        feats = []
        x_tmp = x
        for lid, layer in enumerate(self.layers):
            if self.with_cp:
                x_tmp = checkpoint.checkpoint(layer, x_tmp)
            else:
                x_tmp = layer(x_tmp)
            if lid in self.backbone_output_ids:
                feats.append(x_tmp)
        if type(feats) in [list, tuple]:
            feats = feats[-1]
        heatmaps = torch.sigmoid(feats[:, 0:self.class_num, ...])
        return heatmaps, feats.mean().mean().mean().mean()

    def forward(self, x, mlp_input, flag='img', depth_flag='vitual', **kwargs):
        if x.dim()==5:
            B, Cam_Num, C, H, W = x.shape
            x = x.reshape(-1, C, H, W)
        feats = []


        mlp_input = mlp_input.squeeze(-1)
        mlp_input = self.bn(mlp_input.reshape(-1, mlp_input.shape[-1]))
        context_mpl = self.Pos_mlp(mlp_input)[..., None, None]
        x_tmp = self.Pos_se(x, context_mpl)

        for lid, layer in enumerate(self.layers):
            if self.with_cp:
                x_tmp = checkpoint.checkpoint(layer, x_tmp)
            else:
                x_tmp = layer(x_tmp)
            if lid in self.backbone_output_ids:
                feats.append(x_tmp)
        if type(feats) in [list, tuple]:
            feats = feats[-1]



        heatmaps_2d, ann_maps_2d, heatmap_masks_2d = self.get_target(flag, **kwargs)


        heatmaps_pre = torch.sigmoid(feats[:, 0:self.class_num, ...])
        loss_heatmaps_source = self.get_cls_loss(heatmaps_pre, heatmaps_2d)
        loss_regess_source = self.get_regess_loss(feats[:, self.class_num:, ...],
                                               ann_maps_2d, heatmap_masks_2d, depth_flag)


        return loss_heatmaps_source, loss_regess_source, heatmaps_pre


@NECKS.register_module(force=True)
class BEV_Aux(nn.Module):
    def __init__(
            self,
            numC_input,
            class_name,
            num_layer=[2, 2, 2],
            num_channels=None,
            stride=[1, 1, 1],
            height_num=1,
            backbone_output_ids=None,
            norm_cfg=dict(type='BN'),
            with_cp=False,
            block_type='Basic',
    ):
        super(BEV_Aux, self).__init__()
        # build backbone
        assert len(num_layer) == len(stride)
        self.class_num = len(class_name)
        self.height_num = height_num
        num_channels = [numC_input*2**(i+1) for i in range(len(num_layer))] \
            if num_channels is None else num_channels
        # self.num_hight_channels = [int(num_layer[i]/4) for i in range(len(num_channels))] + [height_num]
        # print('self.num_hight_channels', self.num_hight_channels)
        self.backbone_output_ids = range(len(num_layer)) \
            if backbone_output_ids is None else backbone_output_ids
        # BEV feature:
        features = []
        if block_type == 'BottleNeck':
            curr_numC = numC_input
            for i in range(len(num_layer)):
                layer = [
                    Bottleneck(
                        curr_numC,
                        num_channels[i] // 4,
                        stride=stride[i],
                        downsample=nn.Conv2d(curr_numC, num_channels[i], 3,
                                             stride[i], 1),
                        norm_cfg=norm_cfg)
                ]
                curr_numC = num_channels[i]

                layer.extend([
                    Bottleneck(curr_numC, curr_numC // 4, norm_cfg=norm_cfg)
                    for _ in range(num_layer[i] - 1)
                ])
                features.append(nn.Sequential(*layer))
        elif block_type == 'Basic':
            curr_numC = numC_input
            for i in range(len(num_layer)):
                layer = [
                    BasicBlock(
                        curr_numC,
                        num_channels[i],
                        stride=stride[i],
                        downsample=nn.Conv2d(curr_numC, num_channels[i], 3,
                                             stride[i], 1),
                        norm_cfg=norm_cfg)
                ]
                curr_numC = num_channels[i]
                layer.extend([
                    BasicBlock(curr_numC, curr_numC, norm_cfg=norm_cfg)
                    for _ in range(num_layer[i] - 1)
                ])
                features.append(nn.Sequential(*layer))
        else:
            assert False

        self.BEV_features = nn.Sequential(*features)
        #self.BEV_height = nn.Sequential(*BEV_heights)
        self.with_cp = with_cp
        self.loss_cls = GaussianFocalLoss()
        self.task_loss_weight = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]



    def forward(self, x, **kwargs):
        B_CamNum, C, H, W = x.shape
        feats = []
        x_tmp = x
        for lid, layer in enumerate(self.BEV_features):
            if self.with_cp:
                x_tmp = checkpoint.checkpoint(layer, x_tmp)
            else:
                x_tmp = layer(x_tmp)
            if lid in self.backbone_output_ids:
                feats.append(x_tmp)
        if type(feats) in [list, tuple]:
            feats = feats[-1]


        #print('feats', feats.size())
        bev_height = feats[:, 0:self.height_num, ...]
        bev_feature = feats[:, self.height_num:, ...]
        bev_voxel = bev_height.unsqueeze(1) * bev_feature.unsqueeze(2)
        unused_loss = 0.0*bev_voxel.mean().mean().mean().mean().mean()
        #print('bev_voxel', bev_voxel.shape)
        return bev_voxel, unused_loss
