# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_conv_layer
from mmcv.runner import BaseModule, force_fp32
from torch.cuda.amp.autocast_mode import autocast
from torch.utils.checkpoint import checkpoint
from copy import deepcopy

from mmdet3d.ops.bev_pool_v2.bev_pool import bev_pool_v2
from mmdet.models.backbones.resnet import BasicBlock
from .. import builder
from ..builder import NECKS
from .view_transformer import LSSViewTransformer
from scipy import io



class _ASPPModule(nn.Module):

    def __init__(self, inplanes, planes, kernel_size, padding, dilation,
                 BatchNorm):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(
            inplanes,
            planes,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            dilation=dilation,
            bias=False)
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ASPP(nn.Module):

    def __init__(self, inplanes, mid_channels=256, BatchNorm=nn.BatchNorm2d):
        super(ASPP, self).__init__()

        dilations = [1, 6, 12, 18]

        self.aspp1 = _ASPPModule(
            inplanes,
            mid_channels,
            1,
            padding=0,
            dilation=dilations[0],
            BatchNorm=BatchNorm)
        self.aspp2 = _ASPPModule(
            inplanes,
            mid_channels,
            3,
            padding=dilations[1],
            dilation=dilations[1],
            BatchNorm=BatchNorm)
        self.aspp3 = _ASPPModule(
            inplanes,
            mid_channels,
            3,
            padding=dilations[2],
            dilation=dilations[2],
            BatchNorm=BatchNorm)
        self.aspp4 = _ASPPModule(
            inplanes,
            mid_channels,
            3,
            padding=dilations[3],
            dilation=dilations[3],
            BatchNorm=BatchNorm)

        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(inplanes, mid_channels, 1, stride=1, bias=False),
            BatchNorm(mid_channels),
            nn.ReLU(),
        )
        self.conv1 = nn.Conv2d(
            int(mid_channels * 5), mid_channels, 1, bias=False)
        self.bn1 = BatchNorm(mid_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(
            x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


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





class DepthNet(nn.Module):

    def __init__(self,
                 in_channels,
                 mid_channels,
                 context_channels,
                 depth_channels,
                 mlp_channels=27,
                 use_dcn=True,
                 use_aspp=True):
        super(DepthNet, self).__init__()
        self.reduce_conv = nn.Sequential(
            nn.Conv2d(
                in_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.mlp_input = mlp_channels
        self.context_conv = nn.Conv2d(
            mid_channels, context_channels, kernel_size=1, stride=1, padding=0)
        self.bn = nn.BatchNorm1d(mlp_channels)
        self.depth_mlp = Mlp(mlp_channels, mid_channels, mid_channels)
        self.depth_se = SELayer(mid_channels)  # NOTE: add camera-aware
        self.context_mlp = Mlp(mlp_channels, mid_channels, mid_channels)
        self.context_se = SELayer(mid_channels)  # NOTE: add camera-aware
        depth_conv_list = [
            BasicBlock(mid_channels, mid_channels),
            BasicBlock(mid_channels, mid_channels),
            BasicBlock(mid_channels, mid_channels),
        ]
        if use_aspp:
            depth_conv_list.append(ASPP(mid_channels, mid_channels))
        if use_dcn:
            depth_conv_list.append(
                build_conv_layer(
                    cfg=dict(
                        type='DCN',
                        in_channels=mid_channels,
                        out_channels=mid_channels,
                        kernel_size=3,
                        padding=1,
                        groups=4,
                        im2col_step=128,
                    )))
        depth_conv_list.append(
            nn.Conv2d(
                mid_channels,
                depth_channels,
                kernel_size=1,
                stride=1,
                padding=0))
        self.depth_conv = nn.Sequential(*depth_conv_list)

    def forward(self, x, mlp_input):
        mlp_input = mlp_input.squeeze(-1)
        mlp_input = self.bn(mlp_input.reshape(-1, mlp_input.shape[-1]))
        x = self.reduce_conv(x)
        context_se = self.context_mlp(mlp_input)[..., None, None]
        context = self.context_se(x, context_se)
        context = self.context_conv(context)
        depth_se = self.depth_mlp(mlp_input)[..., None, None]
        depth = self.depth_se(x, depth_se)
        depth = self.depth_conv(depth)
        return torch.cat([depth, context], dim=1)







@NECKS.register_module(force=True)
class LSSViewTransformer_pc(LSSViewTransformer):
    def __init__(self, BEV_Aux, downsample_from_ann=None, loss_depth_weight=3.0, depthnet_cfg=dict(), **kwargs):
        super(LSSViewTransformer_pc, self).__init__(**kwargs)
        self.downsample_from_ann = downsample_from_ann
        if BEV_Aux==None:
            self.BEV_aux_cfg = BEV_Aux
        else:
            self.BEV_aux_cfg = BEV_Aux
            self.BEV_aux = builder.build_neck(self.BEV_aux_cfg)
            self.revised_interval = int(self.grid_interval[2] / self.BEV_aux_cfg.height_num)
        self.loss_depth_weight = loss_depth_weight
        self.depth_net = DepthNet(self.in_channels, self.in_channels,
                                  self.out_channels, self.D, 31, **depthnet_cfg)
        self.c = torch.sqrt(100.0 / torch.square(torch.tensor(450)) + 100.0 / torch.square(torch.tensor(450)))

    def get_lidar_coor(self, rots, trans, cam2imgs, post_rots, post_trans,
                       bda, mlp_input, intri_actrully):

        B, N, _ = trans.shape
        # post-transformation 增广
        # B x N x D x H x W x 3
        points = self.frustum.to(rots) - post_trans.view(B, N, 1, 1, 1, 3)
        points = torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3)\
            .matmul(points.unsqueeze(-1))
        # depth: vitural2real 虚拟深度转换为真实深度
        B, N, D, H, W, _, _ = points.shape
        k = torch.sqrt(100.0 / torch.square(intri_actrully[:, :, 0, 0]) + 100.0 / torch.square(
            intri_actrully[:, :, 1, 1])) / self.c
        k = 1 / k.view(B, N, 1, 1, 1).repeat(1, 1, D, H, W)
        points[..., 2, 0] = torch.mul(k, points[..., 2, 0])
        # cam_to_ego
        points = torch.cat(
            (points[..., :2, :] * points[..., 2:3, :], points[..., 2:3, :]), 5)
        combine = rots.matmul(torch.inverse(cam2imgs))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += trans.view(B, N, 1, 1, 1, 3)
        points = bda.view(B, 1, 1, 1, 1, 3,
                          3).matmul(points.unsqueeze(-1)).squeeze(-1)
        return points


    def view_transform_core(self, input, depth_real, depth_vitural, tran_feat):
        B, N, C, H, W = input[0].shape

        # Lift-Splat
        if self.accelerate:
            feat = tran_feat.view(B, N, self.out_channels, H, W)
            feat = feat.permute(0, 1, 3, 4, 2)
            depth = depth_vitural.view(B, N, self.D, H, W)
            bev_feat_shape = (depth.shape[0], int(self.grid_size[2]),
                              int(self.grid_size[1]), int(self.grid_size[0]),
                              feat.shape[-1])  # (B, Z, Y, X, C)
            bev_feat = bev_pool_v2(depth, feat, self.ranks_depth,
                                   self.ranks_feat, self.ranks_bev,
                                   bev_feat_shape, self.interval_starts,
                                   self.interval_lengths)

            bev_feat = bev_feat.squeeze(2)
        else:
            self.coor = self.get_lidar_coor(*input[1:9])
            bev_feat = self.voxel_pooling_v2(
                self.coor, depth_vitural.view(B, N, self.D, H, W),
                tran_feat.view(B, N, self.out_channels, H, W))
        return bev_feat, depth_real, depth_vitural


    def get_PV_feats(self, bev_feat, index=None):
        coor_revise = self.get_bev_coor(self.coor)
        bev_voxel, unused_loss = self.BEV_aux(bev_feat)
        Frutums_feature = self.get_feature_Voxel2Frutums(bev_voxel, coor_revise, index)
        PV_feats = self.nerf(Frutums_feature)
        return PV_feats, unused_loss

    def get_lidar_coor_aug(self, rots, trans, cam2imgs, post_rots, post_trans,
                       bda, intri_actrully):
        B, N, _ = trans.shape
        points = self.frustum.to(rots) - post_trans.view(B, N, 1, 1, 1, 3)
        points = torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3)\
            .matmul(points.unsqueeze(-1))
        # depth: vitural2real 虚拟深度转换为真实深度
        B, N, D, H, W, _, _ = points.shape
        k = torch.sqrt(100.0 / torch.square(intri_actrully[:, :, 0, 0]) + 100.0 / torch.square(
            intri_actrully[:, :, 1, 1])) / self.c
        k = 1 / k.view(B, N, 1, 1, 1).repeat(1, 1, D, H, W)
        points[..., 2, 0] = torch.mul(k, points[..., 2, 0])
        # cam_to_ego
        points = torch.cat(
            (points[..., :2, :] * points[..., 2:3, :], points[..., 2:3, :]), 5)
        combine = rots.matmul(torch.inverse(cam2imgs))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += trans.view(B, N, 1, 1, 1, 3)
        points = bda.view(B, 1, 1, 1, 1, 3,
                          3).matmul(points.unsqueeze(-1)).squeeze(-1)
        return points

    def get_PV_feats_aug(self, bev_feat, input, **kwargs):
        intrins, post_rots, post_trans, bda, intri_actually = input
        rot_augs = kwargs['bev_aug']['rot_augs']
        tran_augs = kwargs['bev_aug']['tran_augs']
        coor = self.get_lidar_coor_aug(rot_augs, tran_augs, intrins, post_rots, post_trans, bda, intri_actually)
        coor_revise = self.get_bev_coor(coor)
        bev_voxel, unused_loss = self.BEV_aux(bev_feat)
        # print('bev_voxel', bev_voxel.shape)
        Frutums_feature = self.get_feature_Voxel2Frutums(bev_voxel, coor_revise, index=None)
        # print('Frutums_feature', Frutums_feature.shape)
        PV_feats = self.nerf(Frutums_feature)
        return PV_feats, unused_loss

    def get_BEV_feats_from_voxel(self, bev_feat):
        bev_voxel, unused_loss = self.BEV_aux(bev_feat)
        # B, C, Height, X, Y = bev_voxel.shape
        bev_feat = bev_voxel.sum(dim=2)
        return bev_feat, unused_loss

    def nerf(self, PV_feature, depth=None, depth_range=52):
        rays = PV_feature[:, :, range(1, depth_range), ...].sum(dim=2)
        return rays

    def get_bev_coor(self, coor):
        coor_revise = ((coor - self.grid_lower_bound.to(coor)) /
                       torch.Tensor([0.8, 0.8, self.revised_interval]).to(coor))
        coor_revise = coor_revise.long()
        return coor_revise

    def get_feature_Voxel2Frutums(self, bev, coor_revise, index=None):
        if index==None:
            coor_revise = coor_revise
        else:
            coor_revise = coor_revise[index]
        B, C, Height, X, Y = bev.shape
        B, N, D, H, W, XYZ = coor_revise.shape
        temp_feature = bev.new_zeros(B, C, N, D, H, W)
        temp_feature = temp_feature.reshape(B, C, -1)
        # print('coor_revise.shape', coor_revise.shape) # [6, 6, 99, 24, 44, 3]
        # print('bev_feat.shape', bev.shape) # 6, 80, Height, 128, 128]
        # print('temp_feature.shape', temp_feature.shape) # [6, 80, 6, 99, 24, 44]
        # print('self.grid_size[0]', self.grid_size[0])
        for b_index in range(B):
            # print('coor_revise[b_index, :, :, :, :, 0].shape', coor_revise[b_index, :, :, :, :, 0].shape)
            X_index = coor_revise[b_index, :, :, :, :, 0].reshape(-1)  #  N, D, H, W
            Y_index = coor_revise[b_index, :, :, :, :, 1].reshape(-1)
            Z_index = coor_revise[b_index, :, :, :, :, 2].reshape(-1)
            mask = (X_index >= 0) & (X_index < self.grid_size[0]) & (Y_index >= 0) & (Y_index < self.grid_size[1]) & (Z_index >= 0) & (Z_index < self.BEV_aux_cfg.height_num)
            temp_feature[b_index, :, mask] = bev[b_index, :, Z_index[mask], X_index[mask], Y_index[mask]]
        temp_feature = temp_feature.reshape(B, C, N, D, H, W)
        temp_feature = temp_feature.transpose(1, 2).reshape(B*N, C, D, H, W)

        return temp_feature


    def get_mlp_input(self, rot, tran, intrin, post_rot, post_tran, bda, intri_actually):
        B, N, _, _ = rot.shape
        bda = bda.view(B, 1, 3, 3).repeat(1, N, 1, 1)
        mlp_input = torch.stack([
            intrin[:, :, 0, 0],
            intrin[:, :, 1, 1],
            intrin[:, :, 0, 2],
            intrin[:, :, 1, 2],
            intri_actually[:, :, 0, 0],
            intri_actually[:, :, 1, 1],
            intri_actually[:, :, 0, 2],
            intri_actually[:, :, 1, 2],
            post_rot[:, :, 0, 0],
            post_rot[:, :, 0, 1],
            post_tran[:, :, 0],
            post_rot[:, :, 1, 0],
            post_rot[:, :, 1, 1],
            post_tran[:, :, 1],
            bda[:, :, 0, 0],
            bda[:, :, 0, 1],
            bda[:, :, 1, 0],
            bda[:, :, 1, 1],
            bda[:, :, 2, 2],
        ],
            dim=-1)
        sensor2ego = torch.cat([rot, tran.reshape(B, N, 3, 1)],
                               dim=-1).reshape(B, N, -1)
        mlp_input = torch.cat([mlp_input, sensor2ego], dim=-1)
        return mlp_input

    def view_transform(self, input, depth_real, depth_vitural, tran_feat):
        if self.accelerate:
            self.pre_compute(input)
        return self.view_transform_core(input, depth_real, depth_vitural, tran_feat)

    @force_fp32()
    def get_depth_loss(self, depth_labels, depth_preds):
        if type(depth_labels) in [list, tuple]:
            depth_labels = torch.stack(depth_labels)
        # print('depth_labels', depth_labels.shape)
        depth_labels = depth_labels[:, :, :, :, 2, :].max(dim=-1).values # [6, 6, 96, 176, 7, 6]
        depth_labels = self.get_downsampled_gt_depth(depth_labels)
        depth_preds = depth_preds.permute(0, 2, 3,
                                          1).contiguous().view(-1, self.D)
        # print('depth_labels[ffffff > 0]', depth_labels[depth_labels > 0])
        fg_mask = torch.max(depth_labels, dim=1).values > 0.0
        depth_labels = depth_labels[fg_mask]
        depth_preds = depth_preds[fg_mask]
        with autocast(enabled=False):
            depth_loss = F.binary_cross_entropy(
                depth_preds,
                depth_labels,
                reduction='none',
            ).sum() / max(1.0, fg_mask.sum())
        return self.loss_depth_weight * depth_loss


    def get_downsampled_gt_depth(self, gt_depths):
        """
        Input:
            gt_depths: [B, N, H, W]
        Output:
            gt_depths: [B*N*h*w, d]
        """
        if self.downsample_from_ann==None:
            downsample = self.downsample
        else:
            downsample = self.downsample_from_ann
        B, N, H, W = gt_depths.shape

        gt_depths = gt_depths.view(B * N, H // downsample,
                                   downsample, W // downsample,
                                   downsample, 1)
        gt_depths = gt_depths.permute(0, 1, 3, 5, 2, 4).contiguous()
        gt_depths = gt_depths.view(-1, downsample * downsample)
        gt_depths_tmp = torch.where(gt_depths == 0.0,
                                    1e5 * torch.ones_like(gt_depths),
                                    gt_depths)
        gt_depths = torch.min(gt_depths_tmp, dim=-1).values
        gt_depths = gt_depths.view(B * N, H // downsample,
                                   W // downsample)
        gt_depths = (
            gt_depths -
            (self.grid_config['depth'][0] -
             self.grid_config['depth'][2])) / self.grid_config['depth'][2]
        gt_depths = torch.where((gt_depths < self.D + 1) & (gt_depths >= 0.0),
                                gt_depths, torch.zeros_like(gt_depths))

        gt_depths = F.one_hot(
            gt_depths.long(), num_classes=self.D + 1).view(-1, self.D + 1)[:,
                                                                           1:]
        return gt_depths.float()


    def forward(self, input):
        (x, rots, trans, intrins, post_rots, post_trans, bda,
         mlp_input, intri_actrully) = input[:9]
        B, N, C, H, W = x.shape
        x = x.view(B * N, C, H, W)
        x = self.depth_net(x, mlp_input)
        depth_digit = x[:, :self.D, ...]
        tran_feat = x[:, self.D:self.D + self.out_channels, ...]
        depth_real = 0
        depth_vitural = depth_digit.softmax(dim=1)
        return self.view_transform(input, depth_real, depth_vitural, tran_feat)




@NECKS.register_module(force=True)
class LSSViewTransformerBEVDepth_DG(LSSViewTransformer):
    def __init__(self, loss_depth_weight=3.0, depthnet_cfg=dict(), **kwargs):
        super(LSSViewTransformerBEVDepth_DG, self).__init__(**kwargs)
        self.loss_depth_weight = loss_depth_weight
        self.depth_net = DepthNet(self.in_channels, self.in_channels,
                                  self.out_channels, self.D, 31, **depthnet_cfg)
        self.c = torch.sqrt(100.0 / torch.square(torch.tensor(450)) + 100.0 / torch.square(torch.tensor(450)))

    def get_lidar_coor(self, rots, trans, cam2imgs, post_rots, post_trans,
                       bda, mlp_input, intri_actrully):
        B, N, _ = trans.shape
        # post-transformation 增广
        # B x N x D x H x W x 3
        points = self.frustum.to(rots) - post_trans.view(B, N, 1, 1, 1, 3)
        points = torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3)\
            .matmul(points.unsqueeze(-1))
        # depth: vitural2real 虚拟深度转换为真实深度
        B, N, D, H, W, _, _ = points.shape
        k = torch.sqrt(100.0 / torch.square(intri_actrully[:, :, 0, 0]) + 100.0 / torch.square(
            intri_actrully[:, :, 1, 1])) / self.c
        k = 1 / k.view(B, N, 1, 1, 1).repeat(1, 1, D, H, W)
        points[..., 2, 0] = torch.mul(k, points[..., 2, 0])

        # cam_to_ego
        points = torch.cat(
            (points[..., :2, :] * points[..., 2:3, :], points[..., 2:3, :]), 5)
        combine = rots.matmul(torch.inverse(cam2imgs))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += trans.view(B, N, 1, 1, 1, 3)
        points = bda.view(B, 1, 1, 1, 1, 3,
                          3).matmul(points.unsqueeze(-1)).squeeze(-1)
        return points


    def view_transform_core(self, input, depth_real, depth_vitural, tran_feat):
        B, N, C, H, W = input[0].shape

        # Lift-Splat
        if self.accelerate:
            feat = tran_feat.view(B, N, self.out_channels, H, W)
            feat = feat.permute(0, 1, 3, 4, 2)
            depth = depth_vitural.view(B, N, self.D, H, W)
            bev_feat_shape = (depth.shape[0], int(self.grid_size[2]),
                              int(self.grid_size[1]), int(self.grid_size[0]),
                              feat.shape[-1])  # (B, Z, Y, X, C)
            bev_feat = bev_pool_v2(depth, feat, self.ranks_depth,
                                   self.ranks_feat, self.ranks_bev,
                                   bev_feat_shape, self.interval_starts,
                                   self.interval_lengths)

            bev_feat = bev_feat.squeeze(2)
        else:
            coor = self.get_lidar_coor(*input[1:9])
            bev_feat = self.voxel_pooling_v2(
                coor, depth_vitural.view(B, N, self.D, H, W),
                tran_feat.view(B, N, self.out_channels, H, W))
        return bev_feat, depth_real, depth_vitural





    def get_mlp_input(self, rot, tran, intrin, post_rot, post_tran, bda, intri_actually):
        B, N, _, _ = rot.shape
        bda = bda.view(B, 1, 3, 3).repeat(1, N, 1, 1)
        mlp_input = torch.stack([
            intrin[:, :, 0, 0],
            intrin[:, :, 1, 1],
            intrin[:, :, 0, 2],
            intrin[:, :, 1, 2],
            intri_actually[:, :, 0, 0],
            intri_actually[:, :, 1, 1],
            intri_actually[:, :, 0, 2],
            intri_actually[:, :, 1, 2],
            post_rot[:, :, 0, 0],
            post_rot[:, :, 0, 1],
            post_tran[:, :, 0],
            post_rot[:, :, 1, 0],
            post_rot[:, :, 1, 1],
            post_tran[:, :, 1],
            bda[:, :, 0, 0],
            bda[:, :, 0, 1],
            bda[:, :, 1, 0],
            bda[:, :, 1, 1],
            bda[:, :, 2, 2],
        ],
            dim=-1)
        sensor2ego = torch.cat([rot, tran.reshape(B, N, 3, 1)],
                               dim=-1).reshape(B, N, -1)
        mlp_input = torch.cat([mlp_input, sensor2ego], dim=-1)
        return mlp_input

    def view_transform(self, input, depth_real, depth_vitural, tran_feat):
        if self.accelerate:
            self.pre_compute(input)
        return self.view_transform_core(input, depth_real, depth_vitural, tran_feat)

    @force_fp32()
    def get_depth_loss(self, depth_labels, depth_preds):

        depth_labels = self.get_downsampled_gt_depth(depth_labels)
        depth_preds = depth_preds.permute(0, 2, 3,
                                          1).contiguous().view(-1, self.D)
        fg_mask = torch.max(depth_labels, dim=1).values > 0.0
        depth_labels = depth_labels[fg_mask]
        depth_preds = depth_preds[fg_mask]
        with autocast(enabled=False):
            depth_loss = F.binary_cross_entropy(
                depth_preds,
                depth_labels,
                reduction='none',
            ).sum() / max(1.0, fg_mask.sum())
        return self.loss_depth_weight * depth_loss


    def get_downsampled_gt_depth(self, gt_depths):
        """
        Input:
            gt_depths: [B, N, H, W]
        Output:
            gt_depths: [B*N*h*w, d]
        """
        B, N, H, W = gt_depths.shape
        gt_depths = gt_depths.view(B * N, H // self.downsample,
                                   self.downsample, W // self.downsample,
                                   self.downsample, 1)
        gt_depths = gt_depths.permute(0, 1, 3, 5, 2, 4).contiguous()
        gt_depths = gt_depths.view(-1, self.downsample * self.downsample)
        gt_depths_tmp = torch.where(gt_depths == 0.0,
                                    1e5 * torch.ones_like(gt_depths),
                                    gt_depths)
        gt_depths = torch.max(gt_depths_tmp, dim=-1).values
        gt_depths = gt_depths.view(B * N, H // self.downsample,
                                   W // self.downsample)

        gt_depths = (
            gt_depths -
            (self.grid_config['depth'][0] -
             self.grid_config['depth'][2])) / self.grid_config['depth'][2]
        gt_depths = torch.where((gt_depths < self.D + 1) & (gt_depths >= 0.0),
                                gt_depths, torch.zeros_like(gt_depths))
        gt_depths = F.one_hot(
            gt_depths.long(), num_classes=self.D + 1).view(-1, self.D + 1)[:,
                                                                           1:]
        return gt_depths.float()


    def forward(self, input):
        (x, rots, trans, intrins, post_rots, post_trans, bda,
         mlp_input, intri_actrully) = input[:9]
        B, N, C, H, W = x.shape
        x = x.view(B * N, C, H, W)
        x = self.depth_net(x, mlp_input)
        depth_digit = x[:, :self.D, ...]
        tran_feat = x[:, self.D:self.D + self.out_channels, ...]
        depth_real = 0
        depth_vitural = depth_digit.softmax(dim=1)
        return self.view_transform(input, depth_real, depth_vitural, tran_feat)







