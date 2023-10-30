# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import numpy as np
import torch
from PIL import Image
import copy
from pyquaternion import Quaternion
import math

from mmdet3d.core.points import BasePoints, get_points_type
from mmdet.datasets.pipelines import LoadAnnotations, LoadImageFromFile
from ...core.bbox import LiDARInstance3DBoxes
from ..builder import PIPELINES
from .loading import PrepareImageInputs, PrepareImageInputs_DeepAccident


@PIPELINES.register_module()
class LoadPointsFromFile_DeepAccident_intri(object):
    """Load Points From File.
    Load points from file.
    Args:
        coord_type (str): The type of coordinates of points cloud.
            Available options includes:
            - 'LIDAR': Points in LiDAR coordinates.
            - 'DEPTH': Points in depth coordinates, usually for indoor dataset.
            - 'CAMERA': Points in camera coordinates.
        load_dim (int, optional): The dimension of the loaded points.
            Defaults to 6.
        use_dim (list[int], optional): Which dimensions of the points to use.
            Defaults to [0, 1, 2]. For KITTI dataset, set use_dim=4
            or use_dim=[0, 1, 2, 3] to use the intensity dimension.
        shift_height (bool, optional): Whether to use shifted height.
            Defaults to False.
        use_color (bool, optional): Whether to use color features.
            Defaults to False.
        file_client_args (dict, optional): Config dict of file clients,
            refer to
            https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py
            for more details. Defaults to dict(backend='disk').
    """
    def __init__(self,
                 coord_type,
                 load_dim=6,
                 use_dim=[0, 1, 2],
                 shift_height=False,
                 use_color=False,
                 file_client_args=dict(backend='disk')):
        self.shift_height = shift_height
        self.use_color = use_color
        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))
        assert max(use_dim) < load_dim, \
            f'Expect all used dimensions < {load_dim}, got {use_dim}'
        assert coord_type in ['CAMERA', 'LIDAR', 'DEPTH']

        self.coord_type = coord_type
        self.load_dim = load_dim
        self.use_dim = use_dim
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def _load_points(self, pts_filename):
        points = np.load(pts_filename)
        points = points['data']

        return points

    def __call__(self, results):
        """Call function to load points data from file.

        Args:
            results (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the point clouds data.
                Added key and value are described below.

                - points (:obj:`BasePoints`): Point clouds data.
        """
        pts_filename = results['pts_filename']
        points = self._load_points(pts_filename)
        points = points.reshape(-1, 4)
        points = points[:, self.use_dim]
        attribute_dims = None

        if self.shift_height:
            floor_height = np.percentile(points[:, 2], 0.99)
            height = points[:, 2] - floor_height
            points = np.concatenate(
                [points[:, :3],
                 np.expand_dims(height, 1), points[:, 3:]], 1)
            attribute_dims = dict(height=3)

        if self.use_color:
            assert len(self.use_dim) >= 6
            if attribute_dims is None:
                attribute_dims = dict()
            attribute_dims.update(
                dict(color=[
                    points.shape[1] - 3,
                    points.shape[1] - 2,
                    points.shape[1] - 1,
                ]))

        points_class = get_points_type(self.coord_type)
        points = points_class(
            points, points_dim=points.shape[-1], attribute_dims=attribute_dims)
        results['points'] = points

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__ + '('
        repr_str += f'shift_height={self.shift_height}, '
        repr_str += f'use_color={self.use_color}, '
        repr_str += f'file_client_args={self.file_client_args}, '
        repr_str += f'load_dim={self.load_dim}, '
        repr_str += f'use_dim={self.use_dim})'
        return repr_str





@PIPELINES.register_module(force=True)
class PrepareImageInputs_DeepAccident_intri(PrepareImageInputs_DeepAccident):

    def img_transform_core(self, img, resize_dims, crop, flip, rotate):
        # adjust image
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)
        return img

    def img_transform(self, img, post_rot, post_tran, resize_w, resize_h, resize_dims,
                      crop, flip, rotate):
        # adjust image
        img = self.img_transform_core(img, resize_dims, crop, flip, rotate)

        # post-homography transformation
        post_rot[0, :] = resize_w * post_rot[0, :]
        post_rot[1, :] = resize_h * post_rot[1, :]
        post_tran -= torch.Tensor(crop[:2])
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            post_rot = A.matmul(post_rot)
            post_tran = A.matmul(post_tran) + b
        A = self.get_rot(rotate / 180 * np.pi)
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        post_rot = A.matmul(post_rot)
        post_tran = A.matmul(post_tran) + b

        return img, post_rot, post_tran

    # def intri_aug(self, intrin, img):
    #     import random
    #     import copy
    #     fc_w_ratio = self.data_config['fc_w_ratio']
    #     fc_h_ratio = self.data_config['fc_h_ratio']
    #     # 输入一个图像
    #     # 进行crop改变光心
    #     # resize改变视场角
    #     height, width = img.height, img.width
    #     crop_x_intri, crop_y_intri = int(random.randint(0, width*fc_w_ratio)), int(random.randint(0, height*fc_h_ratio))
    #     fx_resize, fy_resize = np.random.uniform(*self.data_config['resize_intri']), np.random.uniform(*self.data_config['resize_intri'])
    #
    #     intrin_aug = copy.deepcopy(intrin)
    #     intrin_aug[0, 2] = intrin_aug[0, 2] - crop_x_intri
    #     intrin_aug[1, 2] = intrin_aug[1, 2] - crop_y_intri
    #     resize_matrix = np.array([fx_resize, fy_resize, 1])
    #     intrin_aug = resize_matrix@intrin_aug
    #     return intrin_aug, fx_resize, fy_resize, crop_x_intri, crop_y_intri

    def sample_augmentation(self, H, W, flip=None, scale=None):
        fH, fW = self.data_config['input_size']
        if self.is_train:
            resize = float(fW) / float(W)
            resize_w = resize + np.random.uniform(*self.data_config['resize'])
            resize_h = copy.deepcopy(resize_w)
            if 're_ratio' in self.data_config.keys():
                if self.data_config['re_ratio']==True:
                    resize_h = resize + np.random.uniform(*self.data_config['resize'])
            resize_dims = (int(W * resize_w), int(H * resize_h))
            newW, newH = resize_dims
            crop_h = int((1 - np.random.uniform(*self.data_config['crop_h'])) * newH) - fH
            #crop_h = int((1 - np.random.uniform(*self.data_config['crop_h']))*(newH-fH)/2)
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = self.data_config['flip'] and np.random.choice([0, 1])
            rotate = np.random.uniform(*self.data_config['rot'])
        else:
            resize = float(fW) / float(W)
            resize += self.data_config.get('resize_test', 0.0)
            resize_w = resize
            resize_h = resize
            if scale is not None:
                resize = scale
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.data_config['crop_h'])) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False if flip is None else flip
            rotate = 0
        return resize_w, resize_h, resize_dims, crop, flip, rotate

    def get_inputs(self, results, flip=None, scale=None):
        imgs = []
        rots = []
        trans = []
        intrins = []
        post_rots = []
        post_trans = []
        intrins_actuals = []
        cam_names = self.choose_cams()
        results['cam_names'] = cam_names
        canvas = []
        sensor2sensors = []
        for cam_name in cam_names:
            # print('results.keys()')
            # print(results.keys())
            # DA
            # dict_keys(['sample_idx', 'pts_filename', 'timestamp', 'ann_infos', 'curr', 'img_fields', 'bbox3d_fields',
            #            'pts_mask_fields', 'pts_seg_fields', 'bbox_fields', 'ma$
            #            k_fields', 'seg_fields', 'box_type_3d', 'box_mode_3d', 'cam_names'])
            # lyft
            # ['sample_idx', 'pts_filename', 'sweeps', 'timestamp', 'curr', 'img_filename', 'lidar2img', 'ann_infos',
            #  'img_fields', 'bbox3d_fields', 'pts_mask_fields', 'pts_seg_fields', 'bbox_fields', 'mask_fields',
            #  'seg_fields', 'box_type_3d', 'box_mode_3d', 'cam_names']
            # Nus
            # dict_keys(['sample_idx', 'pts_filename', 'sweeps', 'timestamp', 'ann_infos', 'img_filename', 'lidar2img',
            #            'img_fields', 'bbox3d_fields', 'pts_mask_fields', 'pts_seg_fields', 'bbox_fields', 'mask_fields',
            #            'seg_fields', 'box_type_3d', 'box_mode_3d', 'cam_names'])
            cam_data = results['curr']['cams'][cam_name]
            # cam_data.keys
            # DA
            # dict_keys(
            #     ['image_path', 'lidar_to_camera_matrix', 'camera_intrinsic_matrix', 'timestamp', 'sensor2ego_rotation',
            #      'sensor2ego_translation'])
            # lyft
            # dict_keys(['data_path', 'type', 'sample_data_token', 'sensor2ego_translation', 'sensor2ego_rotation', 'ego2global_translation',
            #  'ego2global_rotation', 'timestamp', 'sensor2lidar_rotation', 'sensor2lidar_translation', 'cam_intrinsic']
            # lyft
            # dict_keys(['data_path', 'type', 'sample_data_token', 'sensor2ego_translation', 'sensor2ego_rotation', 'ego2global_translation',
            # 'ego2global_rotation', 'timestamp', 'sensor2lidar_rotation', 'sensor2lidar_translation', 'cam_intrinsic'])
            if 'image_path' in cam_data.keys():
                filename = cam_data['image_path']
            else:
                filename = cam_data['data_path']
            img = Image.open(filename)
            post_rot = torch.eye(2)
            post_tran = torch.zeros(2)
            if cam_name == 'Camera_Back':
                intrin = torch.Tensor([[560.16603057, 0.0, 800.0],
                                       [0.0, 560.16603057, 450.0],
                                       [0.0, 0.0, 1.0]])
            else:
                intrin = torch.Tensor([[1142.5184053936916, 0.0, 800.0],
                                       [0.0, 1142.5184053936916, 450.0],
                                       [0.0, 0.0, 1.0]])
            if 'cam_intrinsic' in cam_data.keys():
                intrin = torch.Tensor(cam_data['cam_intrinsic'])
            sensor2keyego, sensor2sensor = \
                self.get_sensor2ego_transformation(results['curr'],
                                                   results['curr'],
                                                   cam_name,
                                                   self.ego_cam)
            rot = sensor2keyego[:3, :3]
            tran = sensor2keyego[:3, 3]
            # image view augmentation (resize, crop, horizontal flip, rotate)
            img_augs = self.sample_augmentation(
                H=img.height, W=img.width, flip=flip, scale=scale)
            resize_w, resize_h, resize_dims, crop, flip, rotate = img_augs
            # get the intri of input
            # print('***************************************')
            intrins_actual = copy.deepcopy(intrin)
            # print('intrin oooooo', intrins_actual)
            intrins_actual[0, :] = (resize_w) * intrins_actual[0, :]
            intrins_actual[1, :] = (resize_h) * intrins_actual[1, :]
            intrins_actual[0, 2] = intrins_actual[0, 2] - crop[0]
            intrins_actual[1, 2] = intrins_actual[1, 2] - crop[1]
            # print(intrins_actual)
            # print(resize_w, resize_h, crop)
            # print('intrins_actual', intrins_actual)
            # aug
            img, post_rot2, post_tran2 = \
                self.img_transform(img, post_rot,
                                   post_tran,
                                   resize_w=resize_w,
                                   resize_h=resize_h,
                                   resize_dims=resize_dims,
                                   crop=crop,
                                   flip=flip,
                                   rotate=rotate)

            # for convenience, make augmentation matrices 3x3
            post_tran = torch.zeros(3)
            post_rot = torch.eye(3)
            post_tran[:2] = post_tran2
            post_rot[:2, :2] = post_rot2

            canvas.append(np.array(img))
            imgs.append(self.normalize_img(img))

            if self.sequential:
                assert 'adjacent' in results
                for adj_info in results['adjacent']:
                    filename_adj = adj_info['cams'][cam_name]['data_path']
                    img_adjacent = Image.open(filename_adj)
                    img_adjacent = self.img_transform_core(
                        img_adjacent,
                        resize_dims=resize_dims,
                        crop=crop,
                        flip=flip,
                        rotate=rotate)
                    imgs.append(self.normalize_img(img_adjacent))
            intrins.append(intrin)
            intrins_actuals.append(intrins_actual)
            rots.append(rot)
            trans.append(tran)
            post_rots.append(post_rot)
            post_trans.append(post_tran)
            sensor2sensors.append(sensor2sensor)

        if self.sequential:
            for adj_info in results['adjacent']:
                post_trans.extend(post_trans[:len(cam_names)])
                post_rots.extend(post_rots[:len(cam_names)])
                intrins.extend(intrins[:len(cam_names)])

                # align
                trans_adj = []
                rots_adj = []
                sensor2sensors_adj = []
                for cam_name in cam_names:
                    adjsensor2keyego, sensor2sensor = \
                        self.get_sensor2ego_transformation(adj_info,
                                                           results['curr'],
                                                           cam_name,
                                                           self.ego_cam)
                    rot = adjsensor2keyego[:3, :3]
                    tran = adjsensor2keyego[:3, 3]
                    rots_adj.append(rot)
                    trans_adj.append(tran)
                    sensor2sensors_adj.append(sensor2sensor)
                rots.extend(rots_adj)
                trans.extend(trans_adj)
                sensor2sensors.extend(sensor2sensors_adj)
        imgs = torch.stack(imgs)
        rots = torch.stack(rots)
        trans = torch.stack(trans)
        intrins = torch.stack(intrins)
        intrins_actuals = torch.stack(intrins_actuals)
        post_rots = torch.stack(post_rots)
        post_trans = torch.stack(post_trans)
        sensor2sensors = torch.stack(sensor2sensors)
        results['canvas'] = canvas
        results['sensor2sensors'] = sensor2sensors
        return (imgs, rots, trans, intrins, post_rots, post_trans, intrins_actuals)



@PIPELINES.register_module()
class PointToMultiViewDepth_Dual(object):

    def __init__(self, grid_config, downsample=1):
        self.downsample = downsample
        self.grid_config = grid_config
        self.c = torch.sqrt(100.0/torch.square(torch.tensor(450)) + 100.0/torch.square(torch.tensor(450)))

    def points2depthmap_vitural(self, points, height, width, act_intrins):
        height, width = height // self.downsample, width // self.downsample
        depth_map = torch.zeros((height, width), dtype=torch.float32)
        coor = torch.round(points[:, :2] / self.downsample)
        depth = points[:, 2]
        # 转换为相对深度
        depth = (depth * torch.sqrt(
            100.0 / torch.square(act_intrins[0][0]) + 100.0 / torch.square(act_intrins[1][1]))) / self.c
        kept1 = (coor[:, 0] >= 0) & (coor[:, 0] < width) & (
            coor[:, 1] >= 0) & (coor[:, 1] < height) & (
                depth < 100) & (
                    depth >= -100)
        coor, depth = coor[kept1], depth[kept1]
        ranks = coor[:, 0] + coor[:, 1] * width
        sort = (ranks + depth / 100.).argsort()
        coor, depth, ranks = coor[sort], depth[sort], ranks[sort]
        kept2 = torch.ones(coor.shape[0], device=coor.device, dtype=torch.bool)
        kept2[1:] = (ranks[1:] != ranks[:-1])
        coor, depth = coor[kept2], depth[kept2]
        coor = coor.to(torch.long)
        depth_map[coor[:, 1], coor[:, 0]] = depth
        return depth_map

    def points2depthmap_real(self, points, height, width, act_intrins):
        height, width = height // self.downsample, width // self.downsample
        depth_map = torch.zeros((height, width), dtype=torch.float32)
        coor = torch.round(points[:, :2] / self.downsample)
        depth = points[:, 2]
        kept1 = (coor[:, 0] >= 0) & (coor[:, 0] < width) & (
            coor[:, 1] >= 0) & (coor[:, 1] < height) & (
                depth < 100) & (
                    depth >= -100)
        coor, depth = coor[kept1], depth[kept1]
        ranks = coor[:, 0] + coor[:, 1] * width
        sort = (ranks + depth / 100.).argsort()
        coor, depth, ranks = coor[sort], depth[sort], ranks[sort]
        kept2 = torch.ones(coor.shape[0], device=coor.device, dtype=torch.bool)
        kept2[1:] = (ranks[1:] != ranks[:-1])
        coor, depth = coor[kept2], depth[kept2]
        coor = coor.to(torch.long)
        depth_map[coor[:, 1], coor[:, 0]] = depth
        return depth_map

    def __call__(self, results):
        points_lidar = results['points']
        imgs, rots, trans, intrins = results['img_inputs'][:4]
        post_rots, post_trans, bda, act_intrins = results['img_inputs'][4:8]
        depth_map_list = []
        depth_map_list_real = []
        for cid in range(len(results['cam_names'])):
            cam_name = results['cam_names'][cid]
            lidar2lidarego = np.eye(4, dtype=np.float32)
            if 'lidar2ego_rotation' in results['curr'].keys():
                lidar2lidarego[:3, :3] = Quaternion(
                    results['curr']['lidar2ego_rotation']).rotation_matrix
                lidar2lidarego[:3, 3] = results['curr']['lidar2ego_translation']
                lidar2lidarego = torch.from_numpy(lidar2lidarego)
            else:
                lidar2lidarego = results['curr']['lidar_to_ego_matrix']
                lidar2lidarego[1, :] = -lidar2lidarego[1, :]
                lidar2lidarego = torch.from_numpy(lidar2lidarego).float()


            cam2camego = np.eye(4, dtype=np.float32)
            cam2camego[:3, :3] = Quaternion(
                results['curr']['cams'][cam_name]
                ['sensor2ego_rotation']).rotation_matrix
            cam2camego[:3, 3] = results['curr']['cams'][cam_name][
                'sensor2ego_translation']
            cam2camego = torch.from_numpy(cam2camego)

            cam2img = np.eye(4, dtype=np.float32)
            cam2img = torch.from_numpy(cam2img)
            cam2img[:3, :3] = intrins[cid]

            act_intrin = act_intrins[cid, :, :]
            # print('****************')
            # print('act_intrin', act_intrin)

            # lidar2cam = torch.inverse(camego2global.matmul(cam2camego)).matmul(
            #     lidarego2global.matmul(lidar2lidarego))
            lidar2cam = torch.inverse(cam2camego).matmul(lidar2lidarego)
            lidar2img = cam2img.matmul(lidar2cam)
            points_img = points_lidar.tensor[:, :3].matmul(
                lidar2img[:3, :3].T) + lidar2img[:3, 3].unsqueeze(0)
            points_img = torch.cat(
                [points_img[:, :2] / points_img[:, 2:3], points_img[:, 2:3]],
                1)
            points_img = points_img.matmul(
                post_rots[cid].T) + post_trans[cid:cid + 1, :]
            depth_map_real = self.points2depthmap_real(points_img, imgs.shape[2],
                                             imgs.shape[3], act_intrin)
            depth_map_vitural = self.points2depthmap_vitural(points_img, imgs.shape[2],
                                             imgs.shape[3], act_intrin)

            # print('**************')
            # cam_data = results['curr']['cams'][cam_name]
            # if 'image_path' in cam_data.keys():
            #     filename = cam_data['image_path']
            # else:
            #     filename = cam_data['data_path']
            # filename = filename[-20:-4]
            # depth_mapnumpy = depth_map.numpy()
            # depth_mapnumpy = np.clip(depth_mapnumpy, 2, 50)
            # depth_mapnumpy = ((depth_mapnumpy) / 75) * 255
            # depth_mapnumpy = depth_mapnumpy.astype(np.uint8)
            # depthmap_color = cv2.applyColorMap(depth_mapnumpy, cv2.COLORMAP_JET)
            #
            # depth_map_gray = cv2.cvtColor(depthmap_color, cv2.COLOR_BGR2GRAY)
            # depth_map_edges = cv2.Canny(depth_map_gray, 2, 255)
            # depth_map_edges_color = cv2.cvtColor(depth_map_edges, cv2.COLOR_GRAY2BGR)
            # depth_map_combined = cv2.addWeighted(depthmap_color, 0.5, depth_map_edges_color, 0.5, 0)
            # print('depthmap_color')
            # print('depthmap_color', depthmap_color.shape)
            # img_cv2 = np.array(imgs[cid, :, :, :]).transpose((1, 2, 0))
            # img_cv2 = img_cv2[:, :, [2, 1, 0]]*220
            # fusion = depthmap_color*0.8 + img_cv2*0.4
            # depth_map_edges_color = depth_map_edges_color*0.8 + img_cv2*0.4
            # # cv2.imwrite(os.path.join('/mnt/cfs/algorithm/hao.lu/Code/BEVDepth_pg/scripts/RealDepth',
            # #                          'DA' + filename + '_img' + cam_name + '.jpg'), img_cv2)
            # cv2.imwrite(os.path.join('/mnt/cfs/algorithm/hao.lu/Code/BEVDepth_pg/scripts/RealDepth1',
            #                          'DA' + filename + '_Ege' + cam_name + '.jpg'), depth_map_edges_color)
            # cv2.imwrite(os.path.join('/mnt/cfs/algorithm/hao.lu/Code/BEVDepth_pg/scripts/RealDepth1',
            #                          'DA' + filename + '_FusionDepth' + cam_name + '.jpg'), fusion)

            depth_map_list.append(depth_map_vitural)
            depth_map_list_real.append(depth_map_real)
        depth_map = torch.stack(depth_map_list)
        depth_map_real = torch.stack(depth_map_list_real)
        results['gt_depth'] = depth_map
        results['gt_depth_real'] = depth_map_real
        return results





@PIPELINES.register_module(force=True)
class PrepareImageInputs_Aug(PrepareImageInputs_DeepAccident):

    def img_transform(self, img, post_rot, post_tran, resize_w, resize_h, resize_dims,
                      crop, flip, rotate):
        # adjust image
        img = self.img_transform_core(img, resize_dims, crop, flip, rotate)

        # post-homography transformation
        post_rot[0, :] = resize_w * post_rot[0, :]
        post_rot[1, :] = resize_h * post_rot[1, :]
        post_tran -= torch.Tensor(crop[:2])
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            post_rot = A.matmul(post_rot)
            post_tran = A.matmul(post_tran) + b
        A = self.get_rot(rotate / 180 * np.pi)
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        post_rot = A.matmul(post_rot)
        post_tran = A.matmul(post_tran) + b

        return img, post_rot, post_tran


    def sample_augmentation(self, H, W, flip=None, scale=None, resize_increment=0):
        fH, fW = self.data_config['input_size']
        if self.is_train:
            resize = float(fW) / float(W)
            resize_w = resize + resize_increment + np.random.uniform(*self.data_config['resize'])
            resize_h = copy.deepcopy(resize_w)
            if 're_ratio' in self.data_config.keys():
                if self.data_config['re_ratio']==True:
                    resize_h = resize + np.random.uniform(*self.data_config['resize'])
            resize_dims = (int(W * resize_w), int(H * resize_h))
            newW, newH = resize_dims
            crop_h = int((1 - np.random.uniform(*self.data_config['crop_h'])) * newH) - fH
            # crop_h = int((1 - np.random.uniform(*self.data_config['crop_h']))*(newH-fH)/2)
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = self.data_config['flip'] and np.random.choice([0, 1])
            rotate = np.random.uniform(*self.data_config['rot'])
        else:
            resize = float(fW) / float(W)
            resize += self.data_config.get('resize_test', 0.0)
            resize_w = resize
            resize_h = resize
            if scale is not None:
                resize = scale
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.data_config['crop_h'])) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False if flip is None else flip
            rotate = 0
        return resize_w, resize_h, resize_dims, crop, flip, rotate

    def get_inputs(self, results, flip=None, scale=None):
        imgs = []
        rots = []
        trans = []
        intrins = []
        post_rots = []
        post_trans = []
        intrins_actuals = []
        cam_names = self.choose_cams()
        results['cam_names'] = cam_names
        canvas = []
        sensor2sensors = []
        for cam_name in cam_names:
            # print('results.keys()')
            # print(results.keys())
            # DA
            # dict_keys(['sample_idx', 'pts_filename', 'timestamp', 'ann_infos', 'curr', 'img_fields', 'bbox3d_fields',
            #            'pts_mask_fields', 'pts_seg_fields', 'bbox_fields', 'ma$
            #            k_fields', 'seg_fields', 'box_type_3d', 'box_mode_3d', 'cam_names'])
            # lyft
            # ['sample_idx', 'pts_filename', 'sweeps', 'timestamp', 'curr', 'img_filename', 'lidar2img', 'ann_infos',
            #  'img_fields', 'bbox3d_fields', 'pts_mask_fields', 'pts_seg_fields', 'bbox_fields', 'mask_fields',
            #  'seg_fields', 'box_type_3d', 'box_mode_3d', 'cam_names']
            # Nus
            # dict_keys(['sample_idx', 'pts_filename', 'sweeps', 'timestamp', 'ann_infos', 'img_filename', 'lidar2img',
            #            'img_fields', 'bbox3d_fields', 'pts_mask_fields', 'pts_seg_fields', 'bbox_fields', 'mask_fields',
            #            'seg_fields', 'box_type_3d', 'box_mode_3d', 'cam_names'])
            cam_data = results['curr']['cams'][cam_name]
            # cam_data.keys
            # DA
            # dict_keys(
            #     ['image_path', 'lidar_to_camera_matrix', 'camera_intrinsic_matrix', 'timestamp', 'sensor2ego_rotation',
            #      'sensor2ego_translation'])
            # lyft
            # dict_keys(['data_path', 'type', 'sample_data_token', 'sensor2ego_translation', 'sensor2ego_rotation', 'ego2global_translation',
            #  'ego2global_rotation', 'timestamp', 'sensor2lidar_rotation', 'sensor2lidar_translation', 'cam_intrinsic']
            # lyft
            # dict_keys(['data_path', 'type', 'sample_data_token', 'sensor2ego_translation', 'sensor2ego_rotation', 'ego2global_translation',
            # 'ego2global_rotation', 'timestamp', 'sensor2lidar_rotation', 'sensor2lidar_translation', 'cam_intrinsic'])
            if 'image_path' in cam_data.keys():
                filename = cam_data['image_path']
            else:
                filename = cam_data['data_path']
            img = Image.open(filename)
            post_rot = torch.eye(2)
            post_tran = torch.zeros(2)
            if cam_name == 'Camera_Back':
                intrin = torch.Tensor([[560.16603057, 0.0, 800.0],
                                       [0.0, 560.16603057, 450.0],
                                       [0.0, 0.0, 1.0]])
            else:
                intrin = torch.Tensor([[1142.5184053936916, 0.0, 800.0],
                                       [0.0, 1142.5184053936916, 450.0],
                                       [0.0, 0.0, 1.0]])
            if 'cam_intrinsic' in cam_data.keys():
                intrin = torch.Tensor(cam_data['cam_intrinsic'])
            sensor2keyego, sensor2sensor = \
                self.get_sensor2ego_transformation(results['curr'],
                                                   results['curr'],
                                                   cam_name,
                                                   self.ego_cam)
            rot = sensor2keyego[:3, :3]
            tran = sensor2keyego[:3, 3]
            # image view augmentation (resize, crop, horizontal flip, rotate)
            if cam_name == 'Camera_Back' and ('cam_intrinsic' not in cam_data.keys()):
                img_augs = self.sample_augmentation(
                    H=img.height, W=img.width, flip=flip, scale=scale, resize_increment=0.2)
            else:
                img_augs = self.sample_augmentation(
                    H=img.height, W=img.width, flip=flip, scale=scale, resize_increment=0)

            resize_w, resize_h, resize_dims, crop, flip, rotate = img_augs
            # get the intri of input
            # print('***************************************')
            intrins_actual = copy.deepcopy(intrin)
            # print('intrin oooooo', intrins_actual)
            intrins_actual[0, :] = (resize_w) * intrins_actual[0, :]
            intrins_actual[1, :] = (resize_h) * intrins_actual[1, :]
            intrins_actual[0, 2] = intrins_actual[0, 2] - crop[0]
            intrins_actual[1, 2] = intrins_actual[1, 2] - crop[1]
            # print(intrins_actual)
            # print(resize_w, resize_h, crop)
            # print('intrins_actual', intrins_actual)
            # aug
            img, post_rot2, post_tran2 = \
                self.img_transform(img, post_rot,
                                   post_tran,
                                   resize_w=resize_w,
                                   resize_h=resize_h,
                                   resize_dims=resize_dims,
                                   crop=crop,
                                   flip=flip,
                                   rotate=rotate)

            # for convenience, make augmentation matrices 3x3
            post_tran = torch.zeros(3)
            post_rot = torch.eye(3)
            post_tran[:2] = post_tran2
            post_rot[:2, :2] = post_rot2

            canvas.append(np.array(img))
            imgs.append(self.normalize_img(img))

            if self.sequential:
                assert 'adjacent' in results
                for adj_info in results['adjacent']:
                    filename_adj = adj_info['cams'][cam_name]['data_path']
                    img_adjacent = Image.open(filename_adj)
                    img_adjacent = self.img_transform_core(
                        img_adjacent,
                        resize_dims=resize_dims,
                        crop=crop,
                        flip=flip,
                        rotate=rotate)
                    imgs.append(self.normalize_img(img_adjacent))
            intrins.append(intrin)
            intrins_actuals.append(intrins_actual)
            rots.append(rot)
            trans.append(tran)
            post_rots.append(post_rot)
            post_trans.append(post_tran)
            sensor2sensors.append(sensor2sensor)

        if self.sequential:
            for adj_info in results['adjacent']:
                post_trans.extend(post_trans[:len(cam_names)])
                post_rots.extend(post_rots[:len(cam_names)])
                intrins.extend(intrins[:len(cam_names)])

                # align
                trans_adj = []
                rots_adj = []
                sensor2sensors_adj = []
                for cam_name in cam_names:
                    adjsensor2keyego, sensor2sensor = \
                        self.get_sensor2ego_transformation(adj_info,
                                                           results['curr'],
                                                           cam_name,
                                                           self.ego_cam)
                    rot = adjsensor2keyego[:3, :3]
                    tran = adjsensor2keyego[:3, 3]
                    rots_adj.append(rot)
                    trans_adj.append(tran)
                    sensor2sensors_adj.append(sensor2sensor)
                rots.extend(rots_adj)
                trans.extend(trans_adj)
                sensor2sensors.extend(sensor2sensors_adj)
        imgs = torch.stack(imgs)
        rots = torch.stack(rots)
        trans = torch.stack(trans)
        intrins = torch.stack(intrins)
        intrins_actuals = torch.stack(intrins_actuals)
        post_rots = torch.stack(post_rots)
        post_trans = torch.stack(post_trans)
        sensor2sensors = torch.stack(sensor2sensors)
        results['canvas'] = canvas
        results['sensor2sensors'] = sensor2sensors
        return (imgs, rots, trans, intrins, post_rots, post_trans, intrins_actuals)







