"""
ContextCluster implementation
# --------------------------------------------------------
# Context Cluster -- Image as Set of Points, ICLR'23 Oral
# Licensed under The MIT License [see LICENSE for details]
# Written by Xu Ma (ma.xu1@northeastern.com)
# --------------------------------------------------------
"""
import os
import copy
import torch
import torch.nn as nn
import torchvision
import numpy as np
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model
from timm.models.layers.helpers import to_2tuple
from einops import rearrange
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch import nn

from mmdet.models import BACKBONES
from mmdet.models.backbones.resnet import BasicBlock, Bottleneck
from mmdet.utils import get_root_logger
from mmcv.runner import _load_checkpoint




def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224),
        'crop_pct': .95, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    'model_small': _cfg(crop_pct=0.9),
    'model_medium': _cfg(crop_pct=0.95),
}


class PointRecuder(nn.Module):
    """
    Point Reducer is implemented by a layer of conv since it is mathmatically equal.
    Input: tensor in shape [B, in_chans, H, W]
    Output: tensor in shape [B, embed_dim, H/stride, W/stride]
    """
    def __init__(self, patch_size=16, stride=16, padding=0,
                 in_chans=3, embed_dim=768, norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        stride = to_2tuple(stride)
        padding = to_2tuple(padding)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size,
                              stride=stride, padding=padding)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x


class GroupNorm(nn.GroupNorm):
    """
    Group Normalization with 1 group.
    Input: tensor in shape [B, C, H, W]
    """
    def __init__(self, num_channels, **kwargs):
        super().__init__(1, num_channels, **kwargs)


def pairwise_cos_sim(x1: torch.Tensor, x2: torch.Tensor):
    """
    return pair-wise similarity matrix between two tensors
    :param x1: [B,...,M,D]
    :param x2: [B,...,N,D]
    :return: similarity matrix [B,...,M,N]
    """
    x1 = F.normalize(x1, dim=-1)
    x2 = F.normalize(x2, dim=-1)
    sim = torch.matmul(x1, x2.transpose(-2, -1))
    return sim




class Mlp(nn.Module):
    """
    Implementation of MLP with nn.Linear (would be slightly faster in both training and inference).
    Input: tensor with shape [B, C, H, W]
    """
    def __init__(self, in_features, hidden_features=None,
                 out_features=None, mlp_group=1, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    def forward(self, x):
        x = self.fc1(x.permute(0, 2, 3, 1))
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x).permute(0, 3, 1, 2)
        x = self.drop(x)
        return x


class Cluster(nn.Module):
    def __init__(self, dim, out_dim, proposal_w=2, proposal_h=2, fold_w=2, fold_h=2, heads=4, head_dim=24, pro_group=1,
                 return_center=False):
        """
        :param dim:  channel nubmer
        :param out_dim: channel nubmer
        :param proposal_w: the sqrt(proposals) value, we can also set a different value
        :param proposal_h: the sqrt(proposals) value, we can also set a different value
        :param fold_w: the sqrt(number of regions) value, we can also set a different value
        :param fold_h: the sqrt(number of regions) value, we can also set a different value
        :param heads:  heads number in context cluster
        :param head_dim: dimension of each head in context cluster
        :param return_center: if just return centers instead of dispatching back (deprecated).
        """
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        self.f = nn.Conv2d(dim, heads * head_dim, kernel_size=1)  # for similarity
        self.proj = nn.Conv2d(heads * head_dim, out_dim, kernel_size=1, groups=pro_group)  # for projecting channel number
        self.v = nn.Conv2d(dim, heads * head_dim, kernel_size=1)  # for value
        self.sim_alpha = nn.Parameter(torch.ones(1))
        self.sim_beta = nn.Parameter(torch.zeros(1))
        self.centers_proposal = nn.AdaptiveMaxPool2d((proposal_w, proposal_h))
        self.fold_w = fold_w
        self.fold_h = fold_h
        self.return_center = return_center
    def forward(self, x):  # [b,c,w,h]
        value = self.v(x)
        x = self.f(x)
        x = rearrange(x, "b (e c) w h -> (b e) c w h", e=self.heads)
        value = rearrange(value, "b (e c) w h -> (b e) c w h", e=self.heads)
        if self.fold_w > 1 and self.fold_h > 1:
            # split the big feature maps to small local regions to reduce computations.
            b0, c0, w0, h0 = x.shape
            assert w0 % self.fold_w == 0 and h0 % self.fold_h == 0, \
                f"Ensure the feature map size ({w0}*{h0}) can be divided by fold {self.fold_w}*{self.fold_h}"
            x = rearrange(x, "b c (f1 w) (f2 h) -> (b f1 f2) c w h", f1=self.fold_w,
                          f2=self.fold_h)  # [bs*blocks,c,ks[0],ks[1]]
            value = rearrange(value, "b c (f1 w) (f2 h) -> (b f1 f2) c w h", f1=self.fold_w, f2=self.fold_h)
        b, c, w, h = x.shape
        centers = self.centers_proposal(x)  # [b,c,C_W,C_H], we set M = C_W*C_H and N = w*h
        value_centers = rearrange(self.centers_proposal(value), 'b c w h -> b (w h) c')  # [b,C_W,C_H,c]
        b, c, ww, hh = centers.shape
        sim = torch.sigmoid(
            self.sim_beta +
            self.sim_alpha * pairwise_cos_sim(
                centers.reshape(b, c, -1).permute(0, 2, 1),
                x.reshape(b, c, -1).permute(0, 2, 1)
            )
        )  # [B,M,N]
        # we use mask to sololy assign each point to one center
        sim_max, sim_max_idx = sim.max(dim=1, keepdim=True)
        mask = torch.zeros_like(sim)  # binary #[B,M,N]
        mask.scatter_(1, sim_max_idx, 1.)
        sim = sim * mask
        value2 = rearrange(value, 'b c w h -> b (w h) c')  # [B,N,D]
        # aggregate step, out shape [B,M,D]
        ###
        # Update Comment: Mar/26/2022
        # a small bug: mask.sum should be sim.sum according to Eq. (1), mask can be considered as a hard version of sim in out implementation.
        # We will update all checkpoints and the bug once all models are re-trained.
        ###
        out = ((value2.unsqueeze(dim=1) * sim.unsqueeze(dim=-1)).sum(dim=2) + value_centers) / (
                    mask.sum(dim=-1, keepdim=True) + 1.0)  # [B,M,D]
        if self.return_center:
            out = rearrange(out, "b (w h) c -> b c w h", w=ww)
        else:
            # dispatch step, return to each point in a cluster
            out = (out.unsqueeze(dim=2) * sim.unsqueeze(dim=-1)).sum(dim=1)  # [B,N,D]
            out = rearrange(out, "b (w h) c -> b c w h", w=w)
        if self.fold_w > 1 and self.fold_h > 1:
            # recover the splited regions back to big feature maps if use the region partition.
            out = rearrange(out, "(b f1 f2) c w h -> b c (f1 w) (f2 h)", f1=self.fold_w, f2=self.fold_h)
        out = rearrange(out, "(b e) c w h -> b (e c) w h", e=self.heads)
        out = self.proj(out)
        return out



class ClusterBlock(nn.Module):
    """
    Implementation of one block.
    --dim: embedding dim
    --mlp_ratio: mlp expansion ratio
    --act_layer: activation
    --norm_layer: normalization
    --drop: dropout rate
    --drop path: Stochastic Depth,
        refer to https://arxiv.org/abs/1603.09382
    --use_layer_scale, --layer_scale_init_value: LayerScale,
        refer to https://arxiv.org/abs/2103.17239
    """
    def __init__(self, dim, mlp_ratio=4.,
                 act_layer=nn.GELU, norm_layer=GroupNorm,
                 drop=0., drop_path=0.,
                 use_layer_scale=True, layer_scale_init_value=1e-5,
                 # for context-cluster
                 proposal_w=2, proposal_h=2, fold_w=2, fold_h=2, heads=4, head_dim=24, mlp_group=1, pro_group=1, return_center=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        # dim, out_dim, proposal_w=2,proposal_h=2, fold_w=2, fold_h=2, heads=4, head_dim=24, return_center=False
        self.token_mixer = Cluster(dim=dim, out_dim=dim, proposal_w=proposal_w, proposal_h=proposal_h,
                                   fold_w=fold_w, fold_h=fold_h, heads=heads, head_dim=head_dim, pro_group=pro_group, return_center=False)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,  mlp_group=mlp_group,
                       act_layer=act_layer, drop=drop)
        # The following two techniques are useful to train deep ContextClusters.
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
    def forward(self, x):
        if self.use_layer_scale:
            x = x + self.drop_path(
                self.layer_scale_1.unsqueeze(-1).unsqueeze(-1)
                * self.token_mixer(self.norm1(x)))
            x = x + self.drop_path(
                self.layer_scale_2.unsqueeze(-1).unsqueeze(-1)
                * self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.token_mixer(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


def basic_blocks(dim, index, layers,
                 mlp_ratio=4.,
                 act_layer=nn.GELU, norm_layer=GroupNorm,
                 drop_rate=.0, drop_path_rate=0.,
                 use_layer_scale=True, layer_scale_init_value=1e-5,
                 # for context-cluster
                 proposal_w=2, proposal_h=2, fold_w=2, fold_h=2, heads=4, head_dim=24, mlp_group=1, pro_group=1, return_center=False):
    blocks = []
    for block_idx in range(layers[index]):
        block_dpr = drop_path_rate * (block_idx + sum(layers[:index])) / (sum(layers) - 1)
        blocks.append(ClusterBlock(
            dim, mlp_ratio=mlp_ratio,
            act_layer=act_layer, norm_layer=norm_layer,
            drop=drop_rate, drop_path=block_dpr,
            use_layer_scale=use_layer_scale,
            layer_scale_init_value=layer_scale_init_value,
            proposal_w=proposal_w, proposal_h=proposal_h, fold_w=fold_w, fold_h=fold_h,
            heads=heads, head_dim=head_dim, mlp_group=mlp_group, pro_group=pro_group, return_center=return_center
        ))
    blocks = nn.Sequential(*blocks)
    return blocks


class ContextCluster(nn.Module):
    """
    ContextCluster, the main class of our model
    --layers: [x,x,x,x], number of blocks for the 4 stages
    --embed_dims, --mlp_ratios, the embedding dims, mlp ratios
    --downsamples: flags to apply downsampling or not
    --norm_layer, --act_layer: define the types of normalization and activation
    --num_classes: number of classes for the image classification
    --in_patch_size, --in_stride, --in_pad: specify the patch embedding
        for the input image
    --down_patch_size --down_stride --down_pad:
        specify the downsample (patch embed.)
    --fork_feat: whether output features of the 4 stages, for dense prediction
    --init_cfg, --pretrained:
        for mmdetection and mmsegmentation to load pretrained weights
    """
    def __init__(self, layers, embed_dims=None,
                 mlp_ratios=None, downsamples=None,
                 norm_layer=nn.BatchNorm2d, act_layer=nn.GELU,
                 out_indices=[2, 3],
                 num_classes=1000,
                 in_patch_size=4, in_stride=4, in_pad=0,
                 down_patch_size=2, down_stride=2, down_pad=0,
                 drop_rate=0., drop_path_rate=0.,
                 use_layer_scale=True, layer_scale_init_value=1e-5,
                 fork_feat=True,
                 init_cfg=None,
                 pretrained=None,
                 # the parameters for context-cluster
                 proposal_w=[2, 2, 2, 2], proposal_h=[2, 2, 2, 2], fold_w=[8, 4, 2, 1], fold_h=[8, 4, 2, 1],
                 heads=[2, 4, 6, 8], head_dim=[16, 16, 32, 32], mlp_groups=[1, 1, 1, 1], pro_groups=[1, 1, 1, 1],
                 **kwargs):
        super().__init__()
        if not fork_feat:
            self.num_classes = num_classes
        self.out_indices = out_indices
        self.fork_feat = fork_feat
        self.patch_embed = PointRecuder(
            patch_size=in_patch_size, stride=in_stride, padding=in_pad,
            in_chans=5, embed_dim=embed_dims[0])
        # set the main block in network
        network = []
        for i in range(len(layers)):
            stage = basic_blocks(embed_dims[i], i, layers,
                                 mlp_ratio=mlp_ratios[i],
                                 act_layer=act_layer, norm_layer=norm_layer,
                                 drop_rate=drop_rate,
                                 drop_path_rate=drop_path_rate,
                                 use_layer_scale=use_layer_scale,
                                 layer_scale_init_value=layer_scale_init_value,
                                 proposal_w=proposal_w[i], proposal_h=proposal_h[i],
                                 fold_w=fold_w[i], fold_h=fold_h[i], heads=heads[i], head_dim=head_dim[i],
                                 mlp_group=mlp_groups[i], pro_group=pro_groups[i],
                                 return_center=False
                                 )
            network.append(stage)
            if i >= len(layers) - 1:
                break
            if downsamples[i] or embed_dims[i] != embed_dims[i + 1]:
                # downsampling between two stages
                network.append(
                    PointRecuder(
                        patch_size=down_patch_size, stride=down_stride,
                        padding=down_pad,
                        in_chans=embed_dims[i], embed_dim=embed_dims[i + 1]
                    )
                )
        self.network = nn.ModuleList(network)
        if self.fork_feat:
            # add a norm layer for each output
            self.end_indices = [0, 2, 4, 6]
            for i_emb, i_layer in enumerate(self.end_indices):
                if i_emb == 0 and os.environ.get('FORK_LAST3', None):
                    # TODO: more elegant way
                    """For RetinaNet, `start_level=1`. The first norm layer will not used.
                    cmd: `FORK_LAST3=1 python -m torch.distributed.launch ...`
                    """
                    layer = nn.Identity()
                else:
                    layer = norm_layer(embed_dims[i_emb])
                layer_name = f'norm{i_layer}'
                self.add_module(layer_name, layer)
        else:
            # Classifier head
            self.norm = norm_layer(embed_dims[-1])
            self.head = nn.Linear(
                embed_dims[-1], num_classes) if num_classes > 0 \
                else nn.Identity()
        self.apply(self.cls_init_weights)
        self.init_cfg = copy.deepcopy(init_cfg)
        # load pre-trained model
        if self.fork_feat and (
                self.init_cfg is not None or pretrained is not None):
            self.init_weights()
    # init for classification
    def cls_init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
    # init for mmdetection or mmsegmentation by loading
    # imagenet pre-trained weights
    def init_weights(self, pretrained=None):
        logger = get_root_logger()
        if self.init_cfg is None and pretrained is None:
            logger.warn(f'No pre-trained weights for '
                        f'{self.__class__.__name__}, '
                        f'training start from scratch')
            pass
        else:
            assert 'checkpoint' in self.init_cfg, f'Only support ' \
                                                  f'specify `Pretrained` in ' \
                                                  f'`init_cfg` in ' \
                                                  f'{self.__class__.__name__} '
            if self.init_cfg is not None:
                ckpt_path = self.init_cfg['checkpoint']
            elif pretrained is not None:
                ckpt_path = pretrained
            ckpt = _load_checkpoint(
                ckpt_path, logger=logger, map_location='cpu')
            if 'state_dict' in ckpt:
                _state_dict = ckpt['state_dict']
            elif 'model' in ckpt:
                _state_dict = ckpt['model']
            else:
                _state_dict = ckpt
            state_dict = _state_dict
            missing_keys, unexpected_keys = \
                self.load_state_dict(state_dict, False)
            # show for debug
            # print('missing_keys: ', missing_keys)
            # print('unexpected_keys: ', unexpected_keys)
    def get_classifier(self):
        return self.head
    def reset_classifier(self, num_classes):
        self.num_classes = num_classes
        self.head = nn.Linear(
            self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
    def forward_embeddings(self, x):
        _, c, img_w, img_h = x.shape
        # print(f"det img size is {img_w} * {img_h}")
        # register positional information buffer.
        range_w = torch.arange(0, img_w, step=1) / (img_w - 1.0)
        range_h = torch.arange(0, img_h, step=1) / (img_h - 1.0)
        fea_pos = torch.stack(torch.meshgrid(range_w, range_h, indexing='ij'), dim=-1).float()
        fea_pos = fea_pos.to(x.device)
        fea_pos = fea_pos - 0.5
        pos = fea_pos.permute(2, 0, 1).unsqueeze(dim=0).expand(x.shape[0], -1, -1, -1)
        x = self.patch_embed(torch.cat([x, pos], dim=1))
        return x
    def forward_tokens(self, x):
        outs = []
        for idx, block in enumerate(self.network):
            x = block(x)
            if self.fork_feat and idx in self.end_indices:
                norm_layer = getattr(self, f'norm{idx}')
                x_out = norm_layer(x)
                outs.append(x_out)
        if self.fork_feat:
            # output the features of four stages for dense prediction
            return outs
        # output only the features of last layer for image classification
        return x
    def forward(self, x):
        # input embedding
        x = self.forward_embeddings(x)
        # through backbone
        x = self.forward_tokens(x)
        # if self.fork_feat:
        #     # otuput features of four stages for dense prediction
        #     return x
        # x = self.norm(x)
        # cls_out = self.head(x.mean([-2, -1]))
        # for image classification
        return [x[2], x[3]]



@BACKBONES.register_module(force=True)
def coc_tiny_9_max(pretrained=False, **kwargs):
    layers = [1, 2, 4, 2]
    norm_layer = GroupNorm
    embed_dims = [48, 128, 512, 1024]
    mlp_ratios = [4, 4, 2, 2]
    downsamples = [True, True, True, True]
    proposal_w = [4, 4, 2, 2]
    proposal_h = [4, 4, 2, 2]
    fold_w = [16, 8, 4, 2]
    fold_h = [16, 8, 4, 2]
    heads = [6, 6, 12, 12]
    head_dim = [24, 24, 24, 24]
    down_patch_size = 3
    down_pad = 1
    model = ContextCluster(
        layers, embed_dims=embed_dims, norm_layer=norm_layer,
        mlp_ratios=mlp_ratios, downsamples=downsamples,
        down_patch_size=down_patch_size, down_pad=down_pad,
        proposal_w=proposal_w, proposal_h=proposal_h, fold_w=fold_w, fold_h=fold_h,
        heads=heads, head_dim=head_dim,
        **kwargs)
    model.default_cfg = default_cfgs['model_small']
    return model



@BACKBONES.register_module(force=True)
def coc_tiny_10_max(pretrained=False, **kwargs):
    layers = [1, 2, 4, 2]
    norm_layer = GroupNorm
    embed_dims = [48, 128, 512, 1024]
    mlp_ratios = [4, 4, 2, 2]
    mlp_groups = [1, 4, 16, 32]
    downsamples = [True, True, True, True]
    proposal_w = [4, 4, 2, 2]
    proposal_h = [4, 4, 2, 2]
    fold_w = [16, 8, 4, 2]
    fold_h = [16, 8, 4, 2]
    heads = [8, 8, 16, 16]
    head_dim = [24, 24, 24, 24]
    pro_groups = [4, 16, 32, 64]   # heads*head_dim → embed_dims
    down_patch_size = 3
    down_pad = 1
    model = ContextCluster(
        layers, embed_dims=embed_dims, norm_layer=norm_layer,
        mlp_ratios=mlp_ratios, downsamples=downsamples,
        down_patch_size=down_patch_size, down_pad=down_pad,
        proposal_w=proposal_w, proposal_h=proposal_h, fold_w=fold_w, fold_h=fold_h,
        heads=heads, head_dim=head_dim, mlp_groups=mlp_groups, pro_groups=pro_groups,
        **kwargs)
    model.default_cfg = default_cfgs['model_small']
    return model


@BACKBONES.register_module(force=True)
def coc_tiny_11_max(pretrained=False, **kwargs):
    layers = [1, 2, 4, 2]
    norm_layer = GroupNorm
    embed_dims = [48, 128, 512, 1024]
    mlp_ratios = [4, 2, 1, 0.5]
    mlp_groups = [1, 4, 16, 32]
    downsamples = [True, True, True, True]
    proposal_w = [4, 4, 2, 2]
    proposal_h = [4, 4, 2, 2]
    fold_w = [16, 8, 4, 2]
    fold_h = [16, 8, 4, 2]
    heads = [8, 8, 16, 16]
    head_dim = [24, 24, 24, 24]
    pro_groups = [4, 16, 32, 64]   # heads*head_dim → embed_dims
    down_patch_size = 3
    down_pad = 1
    model = ContextCluster(
        layers, embed_dims=embed_dims, norm_layer=norm_layer,
        mlp_ratios=mlp_ratios, downsamples=downsamples,
        down_patch_size=down_patch_size, down_pad=down_pad,
        proposal_w=proposal_w, proposal_h=proposal_h, fold_w=fold_w, fold_h=fold_h,
        heads=heads, head_dim=head_dim, mlp_groups=mlp_groups, pro_groups=pro_groups,
        **kwargs)
    model.default_cfg = default_cfgs['model_small']
    return model




@BACKBONES.register_module(force=True)
def coc_tiny_12_max(pretrained=False, **kwargs):
    layers = [1, 2, 6, 2]
    norm_layer = GroupNorm
    embed_dims = [48, 128, 512, 1024]
    mlp_ratios = [4, 4, 2, 2]
    mlp_groups = [1, 4, 8, 16]
    downsamples = [True, True, True, True]
    proposal_w = [4, 4, 4, 4]
    proposal_h = [4, 4, 4, 4]
    fold_h = [8, 4, 2, 1]
    fold_w = [8, 4, 2, 1]
    heads = [8, 8, 16, 16]
    head_dim = [24, 24, 24, 24]
    pro_groups = [8, 32, 64, 128]   # heads*head_dim → embed_dims
    down_patch_size = 3
    down_pad = 1
    in_patch_size = 7
    in_stride = 4
    in_pad = 3,
    model = ContextCluster(
        layers, in_patch_size=in_patch_size, in_stride=in_stride, in_pad=in_pad,
        embed_dims=embed_dims, norm_layer=norm_layer,
        mlp_ratios=mlp_ratios, downsamples=downsamples,
        down_patch_size=down_patch_size, down_pad=down_pad,
        proposal_w=proposal_w, proposal_h=proposal_h, fold_w=fold_w, fold_h=fold_h,
        heads=heads, head_dim=head_dim, mlp_groups=mlp_groups, pro_groups=pro_groups,
        **kwargs)
    model.default_cfg = default_cfgs['model_small']
    return model


if __name__ == '__main__':
    input = torch.rand(2, 3, 384, 704).cuda()
    model = coc_tiny_12_max().cuda()
    out = model(input)
    print(out[0].shape)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    Calculate_gpu_memory(model, input)
    # del input
    # torch.cuda.empty_cache()
    # for z, p in zip(model.state_dict(), model.parameters()):
    #     if p.requires_grad:
    #         print(z, p.numel())
    # print("number of params: {:.2f}M".format(n_parameters/1024**2))

# model = torchvision.models.resnet50(pretrained=False).cuda()
# n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
# print("number of params: {:.2f}M".format(n_parameters/1024**2))

# model1 = torchvision.models.resnet50(pretrained=False).cuda()

# del model
# del input
# del out
# del n_parameters
# torch.cuda.empty_cache()
# torch.cuda.empty_cache()
# torch.cuda.empty_cache()
# torch.cuda.memory_allocated('cuda:0')
# memory_allocated = torch.cuda.memory_allocated('gpu:0') / 1024 / 1024
    # coc_tiny
    # number of params: 5.01M
    # 2817MiB
    # coc_tiny_1
    # 2815
    # params: 5.01
    # coc_tiny_2
    # 3105 MiB
    # params: 6.00M
    # coc_tiny_3
    # 3081 MiB
    # coc_tiny_4
    # 2897MiB
    # 5.96M
    # coc_tiny_5
    # 2905MiB
    # 8.65M
    # coc_tiny_6
    # 3057MiB
    # 8.65M
    # coc_tiny_7
    # 2991MiB
    # 6.09M
    # 注意以上3-7是用head [8, 8, 16, 16] 现在改成了 [6, 6, 12, 12]
    # coc_tiny_9_max
    #  2739MiB
    # 20.94M
    # coc_tiny_10_max
    # 20.59M
    # 2823MiB
    # coc_tiny_11_max
    # 12.46 M
    # 2711MiB

