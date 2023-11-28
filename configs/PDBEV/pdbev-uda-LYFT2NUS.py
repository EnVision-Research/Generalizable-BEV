# Copyright (c) Phigent Robotics. All rights reserved.

_base_ = ['../_base_/datasets/nus-3d.py', '../_base_/default_runtime.py']
# Global
# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
# For nuScenes we usually do 10-class detection
class_names_train = [
    'car', 'truck', 'bus', 'emergency_vehicle', 'other_vehicle',
     'motorcycle', 'bicycle', 'pedestrian', 'animal'
]
class_names = class_names_train
class_names_test = [
    'car', 'truck', 'van',
    'motorcycle', 'cyclist', 'pedestrian'
]
loss_flag='Focal'
thr=0.9


Source_classes=['car', 'truck', 'bus', 'emergency_vehicle', 'other_vehicle',
     'motorcycle', 'bicycle', 'pedestrian', 'animal']
Target_classes=['car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
                'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']
Test_classes = ['car', 'truck', 'bicycle', 'pedestrian']

data_config = {
    'cams': ['Camera_FrontLeft', 'Camera_Front', 'Camera_FrontRight',
               'Camera_BackLeft', 'Camera_Back',  'Camera_BackRight'],
    'Ncams': 6,
    'input_size': (384, 704),  # 2.2727
    'src_size': (900, 1600),
    # Augmentation
    'resize': (-0.06, 0.11),
    # 'resize': (-0.00, 0.00),
    'rot': (-25.4, 25.4),
    'flip': True,
    'crop_h': (0.0, 0.1),
    'resize_test': 0.00,
    're_ratio': False,
    'pitch_aug': (-0.04, 0.04),
    'yaw_aug': (-0.4, 0.4),
    'roll_aug': (-0.04, 0.04),
    'extri_x_aug': (-2.0, 2.0),
    'extri_y_aug': (-2.0, 2.0),
    'extri_z_aug': (-2.0, 2.0),
    # Aligned Intri
    # 'Target_Focal': 559,
    # # Intri Augmentation
    # 'fc_w_ratio': 0.2,   # 改变x光心
    # 'fc_h_ratio': 0.2,   # 改变y光心
    # 'resize_intri': (0.5, 1.5),  # 改变视角
}

loss_weight = [1.0, # Depth
               1.0, 1.0, # 2D
               1.0, 1.0, 1.0, 1.0, # BEV 2D
               0.0, 0.0, 0.0,  # pseudo
               0.00, # Domian
               0.0,  # final pseudo
               0.0, 0.0, 0.0,  # bevheatmap_consist pseudo target_consist
               0.0, 0.0   # 2D consis_source , consis_targe
               ]
loss_flag='sim'
thr=0.7
# Model
grid_config = {
    'x': [-51.2, 51.2, 0.8],
    'y': [-51.2, 51.2, 0.8],
    'z': [-5, 3, 8],
    'depth': [1.0, 100.0, 1.0],
}

voxel_size = [0.1, 0.1, 0.2]

numC_Trans = 80
model = dict(
    type='PCBEV_UDA',
    loss_flag=loss_flag,
    thr=thr,
    img_backbone=dict(
        pretrained='/mnt/cfs/algorithm/yunpeng.zhang/pretrained/resnet50-0676ba61.pth',
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        with_cp=True,
        style='pytorch'),
    img_aug=dict(
        type='Img_Aux',
        numC_input=512,
        class_name=class_names_train,
        upsample=[1, 2, 1],
        num_channels=[256, 128, len(class_names_train) + len(class_names_train) * 6]),
    bev_img_aux=dict(
        type='Img_Aux_Dy',
        numC_input=80,
        class_name=class_names_train,
        num_layer=[2, 1, 1],
        upsample=[2, 1, 1],
        num_channels=[256, 128, len(class_names_train) + len(class_names_train) * 6]),
    img_neck=dict(
        type='CustomFPN',
        in_channels=[1024, 2048],
        out_channels=512,
        num_outs=1,
        start_level=0,
        out_ids=[0]),
    img_view_transformer=dict(
        type='LSSViewTransformer_pc',
        BEV_Aux=dict(
            type='BEV_Aux',
            class_name=class_names_train,
            numC_input=256,
            num_layer=[2, 2, 2],
            height_num=4,
            num_channels=[numC_Trans, numC_Trans, numC_Trans + 4]),
        downsample_from_ann=2,
        grid_config=grid_config,
        input_size=data_config['input_size'],
        in_channels=512,
        depthnet_cfg=dict(use_dcn=False),
        out_channels=numC_Trans,
        downsample=16,
    ),
    img_bev_encoder_backbone=dict(
        type='CustomResNet',
        numC_input=numC_Trans,
        num_channels=[numC_Trans * 2, numC_Trans * 4, numC_Trans * 8]),
    img_bev_encoder_neck=dict(
        type='FPN_LSS',
        in_channels=numC_Trans * 8 + numC_Trans * 2,
        out_channels=256),
    pts_bbox_head=dict(
        type='CenterHead',
        in_channels=256,
        tasks=[
            dict(num_class=1, class_names=['car']),
            dict(num_class=2, class_names=['truck', 'van']),
            dict(num_class=1, class_names=['bus']),
            dict(num_class=2, class_names=['motorcycle', 'cyclist']),
            dict(num_class=1, class_names=['pedestrian']),
        ],
        common_heads=dict(
            reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)),
        share_conv_channel=64,
        bbox_coder=dict(
            type='CenterPointBBoxCoder',
            pc_range=point_cloud_range[:2],
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_num=500,
            score_threshold=0.1,
            out_size_factor=8,
            voxel_size=voxel_size[:2],
            code_size=9),
        separate_head=dict(
            type='SeparateHead', init_bias=-2.19, final_kernel=3),
        loss_cls=dict(type='GaussianFocalLoss', reduction='mean'),
        loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=0.25),
        norm_bbox=True),
    # model training and testing settings
    train_cfg=dict(
        pts=dict(
            point_cloud_range=point_cloud_range,
            grid_size=[1024, 1024, 40],
            voxel_size=voxel_size,
            out_size_factor=8,
            dense_reg=1,
            gaussian_overlap=0.1,
            max_objs=500,
            min_radius=2,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2])),
    test_cfg=dict(
        pts=dict(
            pc_range=point_cloud_range[:2],
            post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_per_img=500,
            max_pool_nms=False,
            min_radius=[4, 12, 10, 1, 0.85, 0.175],
            score_threshold=0.1,
            out_size_factor=8,
            voxel_size=voxel_size[:2],
            pre_max_size=1000,
            post_max_size=83,
            # Scale-NMS
            nms_type=[
                'rotate', 'rotate', 'rotate', 'circle', 'rotate', 'rotate'
            ],
            nms_thr=[0.2, 0.2, 0.2, 0.2, 0.2, 0.5],
            nms_rescale_factor=[
                1.0, [0.7, 0.7], [0.4, 0.55], 1.1, [1.0, 1.0], [4.5, 9.0]
            ]))
)

# Data
dataset_type = 'UDA_DA_Lyft'
data_root = './data/lyft/'
file_client_args = dict(backend='disk')

bda_aug_conf = dict(
    rot_lim=(-22.5, 22.5),
    scale_lim=(0.95, 1.05),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5)

train_pipeline = [
    dict(
        type='PrepareImageInputs_UDA',
        is_train=True,
        data_config=data_config),
    dict(
        type='LoadAnnotationsBEVDepth',
        bda_aug_conf=bda_aug_conf,
        classes=class_names),
    dict(
        type='LoadPointsFromFile_UDA',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=3,
        file_client_args=file_client_args),
    dict(type='PointToMultiViewDepth_UDA', downsample=1, grid_config=grid_config),
    dict(type='Load3DBoxesHeatmap', classes=class_names_train, downsample_feature=8), # CustomCenterNet_Single提高则downsample_feature降低
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(
        type='Collect3D', keys=['img_inputs', 'gt_bboxes_3d', 'gt_labels_3d', 'gt_depth', 'gt_depth_real',
                                'heatmaps_2d', 'ann_maps_2d', 'heatmap_masks_2d', 'heatmaps_2d_aug', 'ann_maps_2d_aug', 'heatmap_masks_2d_aug',
                                'bev_aug', 'Target_Domain'])
]



test_pipeline = [
    dict(type='PrepareImageInputs_UDA', data_config=data_config),
    dict(
        type='LoadAnnotationsBEVDepth',
        bda_aug_conf=bda_aug_conf,
        classes=class_names_test,
        is_train=False),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1600, 900),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names_test,
                with_label=False),
            dict(type='Collect3D', keys=['img_inputs', 'gt_bboxes_3d', 'gt_labels_3d'])
        ])
]

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)

share_data_config = dict(
    type=dataset_type,
    classes=class_names_train,
    modality=input_modality,
    Source_CLASSES=Source_classes,
    Target_CLASSES=Target_classes,
    Test_CLASSES=Test_classes,
    # img_info_prototype='bevdet',
)

test_data_config = dict(
    pipeline=test_pipeline,
    ann_file_source='./data/lyft/' + 'lyft_infos_ori_ann_train.pkl',
    ann_file_target= './data/nuscenes/' + 'bevdetv2-nuscenes_infos_train.pkl',
    ann_file=data_root + 'lyft_infos_ori_ann_val.pkl')

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        ann_file_source='./data/lyft/' + 'lyft_infos_ori_ann_train.pkl',
        ann_file_target='./data/nuscenes/' + 'bevdetv2-nuscenes_infos_train.pkl',
        pipeline=train_pipeline,
        classes=class_names_train,
        test_mode=False,
        # use_valid_flag=True,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d='LiDAR'),
        val=test_data_config,
        test=test_data_config)

for key in ['val', 'test']:
    data[key].update(share_data_config)
data['train'].update(share_data_config)

# Optimizer
optimizer = dict(type='AdamW', lr=2e-4, weight_decay=1e-2)
optimizer_config = dict(grad_clip=dict(max_norm=5, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=200,
    warmup_ratio=0.001,
    step=[12])
runner = dict(type='EpochBasedRunner', max_epochs=3)

custom_hooks = [
    dict(
        type='MEGVIIEMAHook',
        init_updates=10560,
        priority='NORMAL',
    ),
]


