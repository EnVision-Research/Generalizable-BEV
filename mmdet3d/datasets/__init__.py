# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.datasets.builder import build_dataloader
from .builder import DATASETS, PIPELINES, build_dataset
from .custom_3d import Custom3DDataset
from .custom_3d_seg import Custom3DSegDataset
from .kitti_dataset import KittiDataset
from .kitti_mono_dataset import KittiMonoDataset
from .lyft_dataset import LyftDataset
from .shift_dataset import ShiftDataset_my
from .CARLA_dataset import CarlaDataset
from .DeepAccident_dataset import DeepAccidentDataset
from .lyft_dataset_DA import LyftDataset_DA
from .nuscenes_dataset import NuScenesDataset
from .nuscenes_dataset_DA import NuScenesDataset_DA
from .nuscenes_mono_dataset import NuScenesMonoDataset
# yapf: disable
from .pipelines import (AffineResize, BackgroundPointsFilter, GlobalAlignment,
                        GlobalRotScaleTrans, IndoorPatchPointSample,
                        IndoorPointSample, LoadAnnotations3D,
                        LoadPointsFromDict, LoadPointsFromFile,
                        LoadPointsFromMultiSweeps, MultiViewWrapper,
                        NormalizePointsColor, ObjectNameFilter, ObjectNoise,
                        ObjectRangeFilter, ObjectSample, PointSample,
                        PointShuffle, PointsRangeFilter, RandomDropPointsColor,
                        RandomFlip3D, RandomJitterPoints, RandomRotate,
                        RandomShiftScale, RangeLimitedRandomCrop,
                        VoxelBasedPointSampler)
# yapf: enable
from .s3dis_dataset import S3DISDataset, S3DISSegDataset
from .scannet_dataset import (ScanNetDataset, ScanNetInstanceSegDataset,
                              ScanNetSegDataset)
from .semantickitti_dataset import SemanticKITTIDataset
from .sunrgbd_dataset import SUNRGBDDataset
from .utils import get_loading_pipeline
from .waymo_dataset import WaymoDataset
from .UDA_DA_LYFT_dataset import UDA_DA_Lyft

__all__ = [
    'KittiDataset', 'KittiMonoDataset', 'build_dataloader', 'DATASETS',
    'build_dataset', 'NuScenesDataset', 'NuScenesMonoDataset', 'LyftDataset',
    'LyftDataset_my', 'ShiftDataset_my', 'LyftDataset_my1', 'LyftDataset_DA',
    'CarlaDataset', 'DeepAccidentDataset',
    'ObjectSample', 'RandomFlip3D', 'ObjectNoise', 'GlobalRotScaleTrans',
    'PointShuffle', 'ObjectRangeFilter', 'PointsRangeFilter',
    'LoadPointsFromFile', 'S3DISSegDataset', 'S3DISDataset',
    'NormalizePointsColor', 'IndoorPatchPointSample', 'IndoorPointSample',
    'PointSample', 'LoadAnnotations3D', 'GlobalAlignment', 'SUNRGBDDataset',
    'ScanNetDataset', 'ScanNetSegDataset', 'ScanNetInstanceSegDataset',
    'SemanticKITTIDataset', 'Custom3DDataset', 'Custom3DSegDataset',
    'LoadPointsFromMultiSweeps', 'WaymoDataset', 'BackgroundPointsFilter',
    'VoxelBasedPointSampler', 'get_loading_pipeline', 'RandomDropPointsColor',
    'RandomJitterPoints', 'ObjectNameFilter', 'AffineResize',
    'RandomShiftScale', 'LoadPointsFromDict', 'PIPELINES',
    'RangeLimitedRandomCrop', 'RandomRotate', 'MultiViewWrapper', 'NuScenesDataset_DA',
    'UDA_DA_Lyft'
]
