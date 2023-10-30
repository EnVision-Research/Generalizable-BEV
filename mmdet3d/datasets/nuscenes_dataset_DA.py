# Copyright (c) OpenMMLab. All rights reserved.
import os
import tempfile
from os import path as osp
import math
from typing import Callable
from scipy.spatial.transform import Rotation as R
from terminaltables import AsciiTable
from pyquaternion import Quaternion
from pyquaternion import Quaternion
import numpy as np
import pandas as pd
import numpy as np
import pyquaternion
from pyquaternion import Quaternion
import torch

import mmcv
from mmcv.utils import print_log
from mmdet3d.core.evaluation.lyft_eval import lyft_eval
from mmdet3d.core.evaluation.lyft_eval import lyft_eval
from mmcv.utils import print_log
from nuscenes.utils.data_classes import Box as NuScenesBox
from nuscenes import NuScenes
from nuscenes.eval.detection.evaluate import NuScenesEval
from nuscenes.eval.common.config import config_factory
from nuscenes.eval.common.data_classes import EvalBox
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import view_points
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.common.utils import center_distance, scale_iou, velocity_l2, attr_acc, cummean
from nuscenes.eval.detection.data_classes import DetectionMetricData
from nuscenes import NuScenes
from nuscenes.eval.common.loaders import load_prediction, load_gt, add_center_dist, filter_eval_boxes
from nuscenes.eval.detection.algo import calc_ap, calc_tp
from nuscenes.eval.detection.data_classes import DetectionConfig, DetectionMetrics, DetectionBox, \
    DetectionMetricDataList
from nuscenes.eval.detection.render import summary_plot, class_pr_curve, class_tp_curve, dist_pr_curve, visualize_sample
from lyft_dataset_sdk.lyftdataset import LyftDataset as Lyft
from lyft_dataset_sdk.utils.data_classes import Box as LyftBox
from lyft_dataset_sdk.eval.detection.mAP_evaluation import (Box3D, get_ap,
                                                            get_class_names,
                                                            get_ious,
                                                            group_by_key,
                                                            wrap_in_box)


from ..core import show_result
from ..core.bbox import Box3DMode, Coord3DMode, LiDARInstance3DBoxes
from .builder import DATASETS
from .custom_3d import Custom3DDataset
from .pipelines import Compose
from .lyft_dataset import LyftDataset, LyftDataset_my
from .DeepAccident_dataset import accumulate


def quaternion_yaw(q: Quaternion) -> float:
    """
    Calculate the yaw angle from a quaternion.
    Note that this only works for a quaternion that represents a box in lidar or global coordinate frame.
    It does not work for a box in the camera frame.
    :param q: Quaternion of interest.
    :return: Yaw angle in radians.
    """

    # Project into xy plane.
    v = np.dot(q.rotation_matrix, np.array([1, 0, 0]))

    # Measure yaw using arctan.
    yaw = np.arctan2(v[1], v[0])

    return yaw

def yaw_diff(gt_box: EvalBox, eval_box: EvalBox, period: float = 2*np.pi) -> float:
    """
    Returns the yaw angle difference between the orientation of two boxes.
    :param gt_box: Ground truth box.
    :param eval_box: Predicted box.
    :param period: Periodicity in radians for assessing angle difference.
    :return: Yaw angle difference in radians in [0, pi].
    """
    yaw_gt = quaternion_yaw(Quaternion(gt_box.rotation))
    yaw_est = quaternion_yaw(Quaternion(eval_box.rotation))

    return abs(angle_diff(yaw_gt, yaw_est, period))


def angle_diff(x: float, y: float, period: float) -> float:
    """
    Get the smallest angle difference between 2 angles: the angle from y to x.
    :param x: To angle.
    :param y: From angle.
    :param period: Periodicity in radians for assessing angle difference.
    :return: <float>. Signed smallest between-angle difference in range (-pi, pi).
    """

    # calculate angle difference, modulo to [0, 2*pi]
    diff = (x - y + period / 2) % period - period / 2
    if diff > np.pi:
        diff = diff - (2 * np.pi)  # shift (pi, 2*pi] to (-pi, 0]

    return diff

@DATASETS.register_module()
class NuScenesDataset_DA(Custom3DDataset):
    r"""NuScenes Dataset.

    This class serves as the API for experiments on the NuScenes Dataset.

    Please refer to `NuScenes Dataset <https://www.nuscenes.org/download>`_
    for data downloading.

    Args:
        ann_file (str): Path of annotation file.
        pipeline (list[dict], optional): Pipeline used for data processing.
            Defaults to None.
        data_root (str): Path of dataset root.
        classes (tuple[str], optional): Classes used in the dataset.
            Defaults to None.
        load_interval (int, optional): Interval of loading the dataset. It is
            used to uniformly sample the dataset. Defaults to 1.
        with_velocity (bool, optional): Whether include velocity prediction
            into the experiments. Defaults to True.
        modality (dict, optional): Modality to specify the sensor data used
            as input. Defaults to None.
        box_type_3d (str, optional): Type of 3D box of this dataset.
            Based on the `box_type_3d`, the dataset will encapsulate the box
            to its original format then converted them to `box_type_3d`.
            Defaults to 'LiDAR' in this dataset. Available options includes.
            - 'LiDAR': Box in LiDAR coordinates.
            - 'Depth': Box in depth coordinates, usually for indoor dataset.
            - 'Camera': Box in camera coordinates.
        filter_empty_gt (bool, optional): Whether to filter empty GT.
            Defaults to True.
        test_mode (bool, optional): Whether the dataset is in test mode.
            Defaults to False.
        eval_version (bool, optional): Configuration version of evaluation.
            Defaults to  'detection_cvpr_2019'.
        use_valid_flag (bool, optional): Whether to use `use_valid_flag` key
            in the info file as mask to filter gt_boxes and gt_names.
            Defaults to False.
        img_info_prototype (str, optional): Type of img information.
            Based on 'img_info_prototype', the dataset will prepare the image
            data info in the type of 'mmcv' for official image infos,
            'bevdet' for BEVDet, and 'bevdet4d' for BEVDet4D.
            Defaults to 'mmcv'.
        multi_adj_frame_id_cfg (tuple[int]): Define the selected index of
            reference adjcacent frames.
        ego_cam (str): Specify the ego coordinate relative to a specified
            camera by its name defined in NuScenes.
            Defaults to None, which use the mean of all cameras.
    """
    NameMapping = {
        'movable_object.barrier': 'barrier',
        'vehicle.bicycle': 'bicycle',
        'vehicle.bus.bendy': 'bus',
        'vehicle.bus.rigid': 'bus',
        'vehicle.car': 'car',
        'vehicle.construction': 'construction_vehicle',
        'vehicle.motorcycle': 'motorcycle',
        'human.pedestrian.adult': 'pedestrian',
        'human.pedestrian.child': 'pedestrian',
        'human.pedestrian.construction_worker': 'pedestrian',
        'human.pedestrian.police_officer': 'pedestrian',
        'movable_object.trafficcone': 'traffic_cone',
        'vehicle.trailer': 'trailer',
        'vehicle.truck': 'truck'
    }
    DefaultAttribute = {
        'car': 'vehicle.parked',
        'pedestrian': 'pedestrian.moving',
        'trailer': 'vehicle.parked',
        'truck': 'vehicle.parked',
        'bus': 'vehicle.moving',
        'motorcycle': 'cycle.without_rider',
        'construction_vehicle': 'vehicle.parked',
        'bicycle': 'cycle.without_rider',
        'barrier': '',
        'traffic_cone': '',
    }
    AttrMapping = {
        'cycle.with_rider': 0,
        'cycle.without_rider': 1,
        'pedestrian.moving': 2,
        'pedestrian.standing': 3,
        'pedestrian.sitting_lying_down': 4,
        'vehicle.moving': 5,
        'vehicle.parked': 6,
        'vehicle.stopped': 7,
    }
    AttrMapping_rev = [
        'cycle.with_rider',
        'cycle.without_rider',
        'pedestrian.moving',
        'pedestrian.standing',
        'pedestrian.sitting_lying_down',
        'vehicle.moving',
        'vehicle.parked',
        'vehicle.stopped',
    ]
    # https://github.com/nutonomy/nuscenes-devkit/blob/57889ff20678577025326cfc24e57424a829be0a/python-sdk/nuscenes/eval/detection/evaluate.py#L222 # noqa
    ErrNameMapping = {
        'trans_err': 'mATE',
        'scale_err': 'mASE',
        'orient_err': 'mAOE',
        'vel_err': 'mAVE',
        'attr_err': 'mAAE'
    }
    CLASSES = ('car', 'truck', 'trailer', 'bus', 'construction_vehicle',
               'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
               'barrier')

    def __init__(self,
                 ann_file,
                 pipeline=None,
                 data_root=None,
                 classes=None,
                 load_interval=1,
                 with_velocity=True,
                 modality=None,
                 box_type_3d='LiDAR',
                 filter_empty_gt=True,
                 test_mode=False,
                 eval_version='detection_cvpr_2019',
                 use_valid_flag=False,
                 img_info_prototype='mmcv',
                 multi_adj_frame_id_cfg=None,
                 ego_cam='CAM_FRONT',
                 Target_CLASSES=None,
                 Source_CLASSES=None,
                 Test_CLASSES=None,
                 ):
        self.load_interval = load_interval
        self.use_valid_flag = use_valid_flag
        super(NuScenesDataset_DA, self).__init__(
            data_root=data_root,
            ann_file=ann_file,
            pipeline=pipeline,
            classes=classes,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode)

        if Target_CLASSES==None:
            self.Target_CLASSES = [ 'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
                                    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']
        else:
            self.Target_CLASSES = Target_CLASSES
        if Source_CLASSES==None:
            self.Source_CLASSES = ['car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
                                    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']
        else:
            self.Source_CLASSES = Source_CLASSES
        if Test_CLASSES==None:
            self.Test_CLASSES = ['car', 'truck', 'bicycle', 'pedestrian']
        else:
            self.Test_CLASSES = Test_CLASSES
        print('Source_CLASSES', self.Source_CLASSES)
        print('Target_CLASSES', self.Target_CLASSES)
        print('Test_CLASSES', self.Test_CLASSES)

        self.with_velocity = with_velocity
        self.eval_version = eval_version
        from nuscenes.eval.detection.config import config_factory
        self.eval_detection_configs = config_factory(self.eval_version)
        if self.modality is None:
            self.modality = dict(
                use_camera=False,
                use_lidar=True,
                use_radar=False,
                use_map=False,
                use_external=False,
            )

        self.img_info_prototype = img_info_prototype
        self.multi_adj_frame_id_cfg = multi_adj_frame_id_cfg
        self.ego_cam = ego_cam
        # # load annotations
        # if hasattr(self.file_client, 'get_local_path'):
        #     with self.file_client.get_local_path(self.ann_file) as local_path:
        #         self.data_infos = self.load_annotations(open(local_path, 'rb'))
        # else:
        #     warnings.warn(
        #         'The used MMCV version does not have get_local_path. '
        #         f'We treat the {self.ann_file} as local paths and it '
        #         'might cause errors if the path is not a local path. '
        #         'Please use MMCV>= 1.3.16 if you meet errors.')
        #     self.data_infos = self.load_annotations(self.ann_file)

    #     self.data_infos = self.del_empty_sample(self.data_infos)
    #
    # def del_empty_sample(self, data_list):
    #     new_list = []
    #     for item in data_list:
    #         if len(item['ann_infos'][0])>=1:
    #             new_list.append(item)
    #     return new_list

    def accumulate(self,
                   gt_boxes: EvalBoxes,
                   pred_boxes: EvalBoxes,
                   class_name: str,
                   dist_fcn: Callable,
                   dist_th: float,
                   verbose: bool = False) -> DetectionMetricData:
        """
        Average Precision over predefined different recall thresholds for a single distance threshold.
        The recall/conf thresholds and other raw metrics will be used in secondary metrics.
        :param gt_boxes: Maps every sample_token to a list of its sample_annotations.
        :param pred_boxes: Maps every sample_token to a list of its sample_results.
        :param class_name: Class to compute AP on.
        :param dist_fcn: Distance function used to match detections and ground truths.
        :param dist_th: Distance threshold for a match.
        :param verbose: If true, print debug messages.
        :return: (average_prec, metrics). The average precision value and raw data for a number of metrics.
        """
        # ---------------------------------------------
        # Organize input and initialize accumulators.
        # ---------------------------------------------
        # Count the positives.
        # gt_boxes=gts
        # pred_boxes=pre
        # dist_fcn=center_distance
        # dist_th=1
        # class_name='car'
        gt_boxes_name = group_by_key(gt_boxes, 'name')
        npos = len(gt_boxes_name[class_name])
        # For missing classes in the GT, return a data structure corresponding to no predictions.
        if npos == 0:
            return DetectionMetricData.no_predictions()

        image_gts = group_by_key(gt_boxes, 'sample_token')
        image_gts = wrap_in_box(image_gts)
        # For missing classes in the GT, return a data structure corresponding to no predictions.
        if npos == 0:
            return DetectionMetricData.no_predictions()
        # Organize the predictions in a single list.
        pred_boxes = group_by_key(pred_boxes, 'name')
        pred_boxes = pred_boxes[class_name]
        pred_boxes_list = pred_boxes
        pred_confs = [box['score'] for box in pred_boxes_list]
        # Sort by confidence.
        sortind = [i for (v, i) in sorted((v, i) for (i, v) in enumerate(pred_confs))][::-1]
        # Do the actual matching.
        tp = []  # Accumulator of true positives.
        fp = []  # Accumulator of false positives.
        conf = []  # Accumulator of confidences.
        # match_data holds the extra metrics we calculate for each match.
        match_data = {'trans_err': [],  # 'vel_err': [],
                      'scale_err': [],
                      'orient_err': [],
                      'attr_err': [],
                      'conf': []}
        match_data_cumsum = {'trans_err': [],  # 'vel_err': [],
                             'scale_err': [],
                             'orient_err': [],
                             'attr_err': [],
                             'conf': []}
        # ---------------------------------------------
        # Match and accumulate match data.
        # ---------------------------------------------
        taken = set()  # Initially no gt bounding box is matched.
        gt_rot=[]
        pre_rot=[]
        diff_rot = []
        for ind in sortind:
            pred_box = pred_boxes_list[ind]
            pred_box = Box3D(**pred_box)
            min_dist = np.inf
            match_gt_idx = None
            if not (pred_box.sample_token in image_gts.keys()):
                # print('Sample not found:', pred_box.sample_token)
                # index = self.find_index(self.data_infos, pred_box.sample_token)
                # print(self.data_infos[index]['ann_infos'][1])
                continue
            for gt_idx, gt_box in enumerate(image_gts[pred_box.sample_token]):
                # Find closest match among ground truth boxes
                if gt_box.name == class_name and not (pred_box.sample_token, gt_idx) in taken:
                    this_distance = dist_fcn(gt_box, pred_box)
                    if this_distance < min_dist:
                        min_dist = this_distance
                        match_gt_idx = gt_idx
            # If the closest match is close enough according to threshold we have a match!
            is_match = min_dist < dist_th
            if is_match:
                taken.add((pred_box.sample_token, match_gt_idx))
                #  Update tp, fp and confs.
                tp.append(1)
                fp.append(0)
                conf.append(pred_box.score)
                # Since it is a match, update match data also.
                gt_box_match = image_gts[pred_box.sample_token][match_gt_idx]
                gt_box_match.attribute_name = gt_box_match.name
                pred_box.attribute_name = pred_box.name
                match_data['trans_err'].append(center_distance(gt_box_match, pred_box))
                # match_data['vel_err'].append(velocity_l2(gt_box_match, pred_box))
                match_data['scale_err'].append(1 - scale_iou(gt_box_match, pred_box))
                # Barrier orientation is only determined up to 180 degree. (For cones orientation is discarded later)
                period = np.pi if class_name == 'barrier' else 2 * np.pi
                # print('gt_box_match', gt_box_match.rotation)
                # print('pred_box', pred_box.rotation)
                # if class_name=='car':
                #     print('????????')
                #     gt_rot.append(quaternion_yaw(Quaternion(gt_box_match.rotation)))
                #     pre_rot.append(quaternion_yaw(Quaternion(pred_box.rotation)))
                #     diff_rot.append(yaw_diff(gt_box_match, pred_box, period=period))
                #     print('quaternion_yaw(Quaternion(gt_box_match.rotation))', quaternion_yaw(Quaternion(gt_box_match.rotation)))
                #     print('quaternion_yaw(Quaternion(pred_box.rotation))', quaternion_yaw(Quaternion(pred_box.rotation)))
                #     print('yaw_diff(gt_box_match, pred_box, period=period)',
                #           yaw_diff(gt_box_match, pred_box, period=period))
                #     print('-----------')
                # print('yaw_diff(gt_box_match, pred_box, period=period)', yaw_diff(gt_box_match, pred_box, period=period))
                match_data['orient_err'].append(yaw_diff(gt_box_match, pred_box, period=period))
                match_data['attr_err'].append(1 - attr_acc(gt_box_match, pred_box))
                match_data['conf'].append(pred_box.score)
            else:
                # No match. Mark this as a false positive.
                tp.append(0)
                fp.append(1)
                conf.append(pred_box.score)
        # print('match_data[orient_err]', match_data['orient_err'])
        # mmcv.dump(match_data['orient_err'], '/mnt/cfs/algorithm/hao.lu/Code/BEVDet/scripts/Eval/zzzz.pkl')
        # if class_name == 'car':
        #     mmcv.dump(gt_rot, '/mnt/cfs/algorithm/hao.lu/Code/BEVDet/scripts/Eval/gt_rot.pkl')
        #     mmcv.dump(pre_rot, '/mnt/cfs/algorithm/hao.lu/Code/BEVDet/scripts/Eval/pre_rot.pkl')
        #     mmcv.dump(diff_rot, '/mnt/cfs/algorithm/hao.lu/Code/BEVDet/scripts/Eval/diff_rot.pkl')
        # Check if we have any matches. If not, just return a "no predictions" array.
        if len(match_data['trans_err']) == 0:
            return DetectionMetricData.no_predictions()
        # ---------------------------------------------
        # Calculate and interpolate precision and recall
        # ---------------------------------------------
        # Accumulate.
        tp = np.cumsum(tp).astype(float)
        fp = np.cumsum(fp).astype(float)
        conf = np.array(conf)
        # Calculate precision and recall.
        prec = tp / (fp + tp)
        rec = tp / float(npos)
        rec_interp = np.linspace(0, 1, DetectionMetricData.nelem)  # 101 steps, from 0% to 100% recall.
        prec = np.interp(rec_interp, rec, prec, right=0)
        conf = np.interp(rec_interp, rec, conf, right=0)
        rec = rec_interp
        # ---------------------------------------------
        # Re-sample the match-data to match, prec, recall and conf.
        # ---------------------------------------------
        for key in match_data.keys():
            if key == "conf":
                continue  # Confidence is used as reference to align with fp and tp. So skip in this step.
            else:
                # For each match_data, we first calculate the accumulated mean.
                tmp = cummean(np.array(match_data[key]))
                # Then interpolate based on the confidences. (Note reversing since np.interp needs increasing arrays)
                match_data_cumsum[key] = np.interp(conf[::-1], match_data['conf'][::-1], tmp[::-1])[::-1]
                # ---------------------------------------------
                # Done. Instantiate MetricData and return
                # ---------------------------------------------
        return DetectionMetricData(recall=rec,
                                   precision=prec,
                                   confidence=conf,
                                   trans_err=match_data_cumsum['trans_err'],
                                   vel_err=match_data_cumsum['trans_err'],
                                   scale_err=match_data_cumsum['scale_err'],
                                   orient_err=match_data_cumsum['orient_err'],
                                   attr_err=match_data_cumsum['attr_err'])

    def get_cat_ids(self, idx):
        """Get category distribution of single scene.

        Args:
            idx (int): Index of the data_info.

        Returns:
            dict[list]: for each category, if the current scene
                contains such boxes, store a list containing idx,
                otherwise, store empty list.
        """
        info = self.data_infos[idx]
        if self.use_valid_flag:
            mask = info['valid_flag']
            gt_names = set(info['gt_names'][mask])
        else:
            gt_names = set(info['gt_names'])

        cat_ids = []
        for name in gt_names:
            if name in self.CLASSES:
                cat_ids.append(self.cat2id[name])
        return cat_ids

    def load_annotations(self, ann_file):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations sorted by timestamps.
        """
        data = mmcv.load(ann_file, file_format='pkl')
        data_infos = list(sorted(data['infos'], key=lambda e: e['timestamp']))
        data_infos = data_infos[::self.load_interval]
        self.metadata = data['metadata']
        self.version = self.metadata['version']
        return data_infos

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - sweeps (list[dict]): Infos of sweeps.
                - timestamp (float): Sample timestamp.
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        info = self.data_infos[index]
        # standard protocol modified from SECOND.Pytorch
        input_dict = dict(
            sample_idx=info['token'],
            pts_filename=info['lidar_path'],
            sweeps=info['sweeps'],
            timestamp=info['timestamp'] / 1e6,
        )
        if 'ann_infos' in info:
            input_dict['ann_infos'] = info['ann_infos']
        if self.modality['use_camera']:
            if self.img_info_prototype == 'mmcv':
                image_paths = []
                lidar2img_rts = []
                for cam_type, cam_info in info['cams'].items():
                    image_paths.append(cam_info['data_path'])
                    # obtain lidar to image transformation matrix
                    lidar2cam_r = np.linalg.inv(
                        cam_info['sensor2lidar_rotation'])
                    lidar2cam_t = cam_info[
                        'sensor2lidar_translation'] @ lidar2cam_r.T
                    lidar2cam_rt = np.eye(4)
                    lidar2cam_rt[:3, :3] = lidar2cam_r.T
                    lidar2cam_rt[3, :3] = -lidar2cam_t
                    intrinsic = cam_info['cam_intrinsic']
                    viewpad = np.eye(4)
                    viewpad[:intrinsic.shape[0], :intrinsic.
                            shape[1]] = intrinsic
                    lidar2img_rt = (viewpad @ lidar2cam_rt.T)
                    lidar2img_rts.append(lidar2img_rt)

                input_dict.update(
                    dict(
                        img_filename=image_paths,
                        lidar2img=lidar2img_rts,
                    ))

                if not self.test_mode:
                    annos = self.get_ann_info(index)
                    input_dict['ann_info'] = annos
            else:
                assert 'bevdet' in self.img_info_prototype
                input_dict.update(dict(curr=info))
                if '4d' in self.img_info_prototype:
                    info_adj_list = self.get_adj_info(info, index)
                    input_dict.update(dict(adjacent=info_adj_list))
        return input_dict

    def get_adj_info(self, info, index):
        info_adj_list = []
        for select_id in range(*self.multi_adj_frame_id_cfg):
            select_id = max(index - select_id, 0)
            if not self.data_infos[select_id]['scene_token'] == info[
                    'scene_token']:
                info_adj_list.append(info)
            else:
                info_adj_list.append(self.data_infos[select_id])
        return info_adj_list

    def get_ann_info(self, index):
        """Get annotation info according to the given index.

        Args:
            index (int): Index of the annotation data to get.

        Returns:
            dict: Annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`):
                    3D ground truth bboxes
                - gt_labels_3d (np.ndarray): Labels of ground truths.
                - gt_names (list[str]): Class names of ground truths.
        """
        info = self.data_infos[index]
        # filter out bbox containing no points
        if self.use_valid_flag:
            mask = info['valid_flag']
        else:
            mask = info['num_lidar_pts'] > 0
        gt_bboxes_3d = info['gt_boxes'][mask]
        gt_names_3d = info['gt_names'][mask]
        gt_labels_3d = []
        for cat in gt_names_3d:
            if cat in self.CLASSES:
                gt_labels_3d.append(self.CLASSES.index(cat))
            else:
                gt_labels_3d.append(-1)
        gt_labels_3d = np.array(gt_labels_3d)

        if self.with_velocity:
            gt_velocity = info['gt_velocity'][mask]
            nan_mask = np.isnan(gt_velocity[:, 0])
            gt_velocity[nan_mask] = [0.0, 0.0]
            gt_bboxes_3d = np.concatenate([gt_bboxes_3d, gt_velocity], axis=-1)

        # the nuscenes box center is [0.5, 0.5, 0.5], we change it to be
        # the same as KITTI (0.5, 0.5, 0)
        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d,
            box_dim=gt_bboxes_3d.shape[-1],
            origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_names=gt_names_3d)
        return anns_results

    def _format_bbox(self, results, metas_pre, jsonfile_prefix=None):
        """Convert the results to the standard format.

        Args:
            results (list[dict]): Testing results of the dataset.
            jsonfile_prefix (str): The prefix of the output jsonfile.
                You can specify the output directory/filename by
                modifying the jsonfile_prefix. Default: None.

        Returns:
            str: Path of the output json file.
        """
        lyft_annos = {}
        mapped_class_names = self.Source_CLASSES

        print('Start to convert detection format...')
        for sample_id, det in enumerate(mmcv.track_iter_progress(results)):
            annos = []
            boxes = output_to_lyft_box(det)
            sample_token = metas_pre[sample_id][0]['sample_idx']
            data_index_from_pre = self.find_index(self.data_infos, sample_token)
            boxes = lidar_lyft_box_to_global(self.data_infos[data_index_from_pre], boxes)
            for i, box in enumerate(boxes):
                name = mapped_class_names[box.label]
                if name in self.Test_CLASSES:
                    lyft_anno = dict(
                        sample_token=sample_token,
                        translation=box.center.tolist(),
                        size=box.wlh.tolist(),
                        rotation=box.orientation.elements.tolist(),
                        name=name,
                        score=box.score)
                    annos.append(lyft_anno)
                    lyft_annos[sample_token] = annos
        lyft_submissions = {
            'meta': self.modality,
            'results': lyft_annos}

        mmcv.mkdir_or_exist(jsonfile_prefix)
        res_path = osp.join(jsonfile_prefix, 'results_nus.json')
        print('Results writes to', res_path)
        mmcv.dump(lyft_submissions, '/mnt/cfs/algorithm/hao.lu/Code/BEVDet/scripts/Eval/results_nus.json')
        mmcv.dump(lyft_submissions, res_path)
        return res_path

    def load_predictions(self, res_path):
        """Load Lyft predictions from json file.
        Args:
            res_path (str): Path of result json file recording detections.

        Returns:
            list[dict]: List of prediction dictionaries.
        """
        predictions = mmcv.load(res_path)
        predictions = predictions['results']
        all_preds = []
        for sample_token in predictions.keys():
            all_preds.extend(predictions[sample_token])
        return all_preds

    def eval_core(self, res_path, save_path, output_dir, logger=None, NDS=True):
        """Evaluation API for Lyft dataset.

        Args:
            lyft (:obj:`LyftDataset`): Lyft class in the sdk.
            data_root (str): Root of data for reading splits.
            res_path (str): Path of result json file recording detections.
            eval_set (str): Name of the split for evaluation.
            output_dir (str): Output directory for output json files.
            logger (logging.Logger | str, optional): Logger used for printing
                    related information during evaluation. Default: None.

        Returns:
            dict[str, float]: The evaluation results.
        """
        # evaluate by lyft metrics
        gts = self.load_gts()
        predictions = self.load_predictions(res_path)
        print('saving gts and predictions to ' + save_path)
        print(len(gts))
        print(len(predictions))
        mmcv.dump(gts, osp.join(save_path, 'tested_on_nus_gts.pkl'))
        mmcv.dump(predictions, osp.join(save_path, 'tested_on_nus_pre.pkl'))

        # gts = group_by_key(gts, 'name')
        # pre = group_by_key(predictions, 'name')
        dist_th_list = [0.5, 1, 2, 4]


        # -----------------------------------
        # Step 1: Accumulate metric data for all classes and distance thresholds.
        # -----------------------------------
        metric_data_list = DetectionMetricDataList()

        for class_name in self.Test_CLASSES:
            for dist_th in dist_th_list:
                md = self.accumulate(gts, predictions, class_name, center_distance, dist_th)
                metric_data_list.set(class_name, dist_th, md)

        # -----------------------------------
        # Step 2: Calculate metrics from the data.
        # -----------------------------------
        TP_METRICS = ['trans_err', 'scale_err', 'orient_err']
        eval_version = 'detection_cvpr_2019'
        eval_detection_configs = config_factory(eval_version)
        metrics = DetectionMetrics(eval_detection_configs)
        for class_name in self.Test_CLASSES:
            # Compute APs.
            for dist_th in dist_th_list:
                metric_data = metric_data_list[(class_name, dist_th)]
                ap = calc_ap(metric_data, eval_detection_configs.min_recall, eval_detection_configs.min_precision)
                metrics.add_label_ap(class_name, dist_th, ap)
            # Compute TP metrics.
            for metric_name in TP_METRICS:
                metric_data = metric_data_list[(class_name, eval_detection_configs.dist_th_tp)]
                if class_name in ['traffic_cone'] and metric_name in ['attr_err', 'vel_err', 'orient_err']:
                    tp = np.nan
                elif class_name in ['barrier'] and metric_name in ['attr_err', 'vel_err']:
                    tp = np.nan
                else:
                    tp = calc_tp(metric_data, eval_detection_configs.min_recall, metric_name)
                metrics.add_label_tp(class_name, metric_name, tp)
        metrics_summary = metrics.serialize()
        # with open(os.path.join('/mnt/cfs/algorithm/hao.lu/Code/BEVDet/scripts/Eval', 'metrics_summary.json'), 'w') as f:
        #     json.dump(metrics_summary, f, indent=2)
        # with open(os.path.join('/mnt/cfs/algorithm/hao.lu/Code/BEVDet/scripts/Eval', 'metrics_details.json'), 'w') as f:
        #     json.dump(metric_data_list.serialize(), f, indent=2)

        # Print high-level metrics.
        print('mAP: %.4f' % (metrics_summary['mean_ap']))
        err_name_mapping = {
            'trans_err': 'mATE',
            'scale_err': 'mASE',
            'orient_err': 'mAOE',
            'vel_err': 'mAVE',
            'attr_err': 'mAAE'
        }
        for tp_name, tp_val in metrics_summary['tp_errors'].items():
            print('%s: %.4f' % (err_name_mapping[tp_name], tp_val))
        print('NDS: %.4f' % (metrics_summary['nd_score']))
        # Print per-class metrics.
        print('Per-class results:')
        print('Object Class\tAP\tATE\tASE\tAOE')
        class_aps = metrics_summary['mean_dist_aps']
        class_tps = metrics_summary['label_tp_errors']
        for class_name in class_aps.keys():
            print('%s\t%.3f\t%.3f\t%.3f\t%.3f'
                  % (class_name, class_aps[class_name],
                     class_tps[class_name]['trans_err'],
                     class_tps[class_name]['scale_err'],
                     class_tps[class_name]['orient_err']))
        metrics_summary['class_names'] = self.Test_CLASSES
        return metrics_summary

    def load_gts(self):
        # 仅仅是为了测评的加在
        all_annotations = []
        # Load annotations and filter predictions and annotations.
        for data_info in mmcv.track_iter_progress(self.data_infos):
            sample_token = data_info['token']
            sample_annotations = data_info['ann_infos']
            for index in range(len(sample_annotations[0])):
                class_name = self.Target_CLASSES[sample_annotations[1][index]]   # 目标数据集的标注顺序解码类别
                # if class_name in self.Test_CLASSES:   # 在测试协议内
                quat = Quaternion(axis=[0, 0, 1], radians=sample_annotations[0][index][6])
                size = sample_annotations[0][index][3:6][[1, 0, 2]]  #  output_to_lyft_box 把预测的转了
                annotation = {
                    'sample_token': sample_token,
                    'translation': sample_annotations[0][index][0:3],
                    'size': [abs(size[0]), abs(size[1]), abs(size[2])],
                    'rotation': quat.elements.tolist(),
                    'name': class_name,
                }
                all_annotations.append(annotation)
        return all_annotations

    def _evaluate_single(self,
                         result_path,
                         save_path,
                         logger=None,
                         metric='bbox',
                         result_name='pts_bbox'):
        """Evaluation for a single model in Lyft protocol.

        Args:
            result_path (str): Path of the result file.
            logger (logging.Logger | str, optional): Logger used for printing
                related information during evaluation. Default: None.
            metric (str, optional): Metric name used for evaluation.
                Default: 'bbox'.
            result_name (str, optional): Result name in the metric prefix.
                Default: 'pts_bbox'.
        Returns:
            dict: Dictionary of evaluation details.
        """
        output_dir = osp.join(*osp.split(result_path)[:-1])
        metrics = self.eval_core(result_path, save_path, output_dir, logger)
        # record metrics
        detail = dict()
        metric_prefix = f'{result_name}_DeepAccident'
        err_name_mapping = {
            'trans_err': 'mATE',
            'scale_err': 'mASE',
            'orient_err': 'mAOE',
            'vel_err': 'mAVE',
            'attr_err': 'mAAE'
        }
        class_aps = metrics['mean_dist_aps']
        class_tps = metrics['label_tp_errors']
        for class_name in class_aps.keys():
            detail[f'{metric_prefix}/{class_name}_AP'] = class_aps[class_name]
            detail[f'{metric_prefix}/{class_name}_ATE'] = class_tps[class_name]['trans_err']
            detail[f'{metric_prefix}/{class_name}_ASE'] = class_tps[class_name]['scale_err']
            detail[f'{metric_prefix}/{class_name}_AOE'] = class_tps[class_name]['orient_err']
        for tp_name, tp_val in metrics['tp_errors'].items():
            detail[f'{metric_prefix}/' + err_name_mapping[tp_name]] = tp_val
        detail[f'{metric_prefix}/mAP'] = metrics['mean_ap']
        detail[f'{metric_prefix}/NDS'] = metrics['nd_score']
        return detail


    def format_results(self, results, metas_pre, jsonfile_prefix=None, csv_savepath=None):
        """Format the results to json (standard format for COCO evaluation).

        Args:
            results (list[dict]): Testing results of the dataset.
            jsonfile_prefix (str): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            csv_savepath (str): The path for saving csv files.
                It includes the file path and the csv filename,
                e.g., "a/b/filename.csv". If not specified,
                the result will not be converted to csv file.

        Returns:
            tuple: Returns (result_files, tmp_dir), where `result_files` is a
                dict containing the json filepaths, `tmp_dir` is the temporal
                directory created for saving json files when
                `jsonfile_prefix` is not specified.
        """
        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: {} != {}'.
            format(len(results), len(self)))

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None
        result_files = self._format_bbox(results, metas_pre, jsonfile_prefix)
        if csv_savepath is not None:
            self.json2csv(result_files['pts_bbox'], csv_savepath)
        return result_files, tmp_dir

    def find_index(self, data_infos, sample_token, token_name='token'):
        # 在原始pkl文件中寻找gt的index
        # data_infos：pkl读入的数据
        # sample_token：测试数据集的token
        # token_name：token的名字，每个数据集不太一样
        for index, sample in enumerate(data_infos):
            if sample[token_name] == sample_token:
                return index

    def form_meta(self, results_all):
        results = list()
        meta = list()
        for result in results_all:
            results.append(result['pts_bbox'])
            meta.append(result['img_metas'])
        return results, meta

    def evaluate(self,
                 results,
                 save_path,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 csv_savepath=None,
                 result_names=['pts_bbox'],
                 show=False,
                 out_dir=None,
                 pipeline=None):
        """Evaluation in Lyft protocol.

        Args:
            results (list[dict]): Testing results of the dataset.
            metric (str | list[str], optional): Metrics to be evaluated.
                Default: 'bbox'.
            logger (logging.Logger | str, optional): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str, optional): The prefix of json files including
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            csv_savepath (str, optional): The path for saving csv files.
                It includes the file path and the csv filename,
                e.g., "a/b/filename.csv". If not specified,
                the result will not be converted to csv file.
            result_names (list[str], optional): Result names in the
                metric prefix. Default: ['pts_bbox'].
            show (bool, optional): Whether to visualize.
                Default: False.
            out_dir (str, optional): Path to save the visualization results.
                Default: None.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.

        Returns:
            dict[str, float]: Evaluation results.
        """
        if 'img_metas' in results[0].keys():
            results_pre, metas_pre = self.form_meta(results)
        else:
            results_pre = results

        result_files, tmp_dir = self.format_results(results_pre, metas_pre, jsonfile_prefix,
                                                    csv_savepath)

        results_dict = self._evaluate_single(result_files, save_path)

        if tmp_dir is not None:
            tmp_dir.cleanup()

        if show or out_dir:
            self.show(results, out_dir, show=show, pipeline=pipeline)
        return results_dict

    def _build_default_pipeline(self):
        """Build the default pipeline for this dataset."""
        pipeline = [
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=5,
                use_dim=5,
                file_client_args=dict(backend='disk')),
            dict(
                type='LoadPointsFromMultiSweeps',
                sweeps_num=10,
                file_client_args=dict(backend='disk')),
            dict(
                type='DefaultFormatBundle3D',
                class_names=self.CLASSES,
                with_label=False),
            dict(type='Collect3D', keys=['points'])
        ]
        return Compose(pipeline)


    def show(self, results, out_dir, show=False, pipeline=None):
        """Results visualization.

        Args:
            results (list[dict]): List of bounding boxes results.
            out_dir (str): Output directory of visualization result.
            show (bool): Whether to visualize the results online.
                Default: False.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.
        """
        assert out_dir is not None, 'Expect out_dir, got none.'
        pipeline = self._get_pipeline(pipeline)
        for i, result in enumerate(results):
            if 'pts_bbox' in result.keys():
                result = result['pts_bbox']
            data_info = self.data_infos[i]
            pts_path = data_info['lidar_path']
            file_name = osp.split(pts_path)[-1].split('.')[0]
            points = self._extract_data(i, pipeline, 'points').numpy()
            # for now we convert points into depth mode
            points = Coord3DMode.convert_point(points, Coord3DMode.LIDAR,
                                               Coord3DMode.DEPTH)
            inds = result['scores_3d'] > 0.1
            gt_bboxes = self.get_ann_info(i)['gt_bboxes_3d'].tensor.numpy()
            show_gt_bboxes = Box3DMode.convert(gt_bboxes, Box3DMode.LIDAR,
                                               Box3DMode.DEPTH)
            pred_bboxes = result['boxes_3d'][inds].tensor.numpy()
            show_pred_bboxes = Box3DMode.convert(pred_bboxes, Box3DMode.LIDAR,
                                                 Box3DMode.DEPTH)
            show_result(points, show_gt_bboxes, show_pred_bboxes, out_dir,
                        file_name, show)

from ..core.bbox import LiDARInstance3DBoxes

@DATASETS.register_module()
class NuScenesDataset_DA1(NuScenesDataset_DA):
    def load_gts(self):
        bev_range = np.array([-50, -50, 50, 50], dtype=np.float32)
        # 仅仅是为了测评的加在
        all_annotations = []
        # Load annotations and filter predictions and annotations.
        for data_info in mmcv.track_iter_progress(self.data_infos):
            sample_token = data_info['token']
            sample_annotations = data_info['ann_infos']
            for index in range(len(sample_annotations[0])):
                gt_boxes = torch.Tensor(sample_annotations[0][index])
                gt_labels = torch.Tensor(sample_annotations[1])
                if len(gt_boxes) == 0:
                    continue
                    gt_boxes = torch.zeros(0, 9)
                if gt_boxes.dim() < 2:
                    gt_boxes = gt_boxes.unsqueeze(0)
                if gt_labels.numel() == 0:
                    continue
                if gt_labels.dim() < 2:
                    gt_labels = gt_labels.unsqueeze(0)
                gt_bboxes_3d = LiDARInstance3DBoxes(gt_boxes, box_dim=9, origin=(0.5, 0.5, 0.5))
                mask = gt_bboxes_3d.in_range_bev(bev_range)
                # print('mask', mask)
                if mask[0]==True:
                    detection = {}
                    detection['boxes_3d'] = gt_bboxes_3d
                    detection['labels_3d'] = gt_labels[0] # 无用，只是为了更好的调用
                    detection['scores_3d'] = gt_labels[0] # 无用，只是为了更好的调用
                    box = output_to_lyft_box(detection)
                    if sample_annotations[1][index]<len(self.Target_CLASSES):
                        class_name = self.Target_CLASSES[sample_annotations[1][index]]   # 目标数据集的标注顺序解码类别
                        if class_name in self.Test_CLASSES:
                            annotation = dict(
                                sample_token=sample_token,
                                translation=box[0].center.tolist(),
                                size=box[0].wlh.tolist(),
                                rotation=box[0].orientation.elements.tolist(),
                                name=class_name)
                            all_annotations.append(annotation)
                # if class_name in self.Test_CLASSES:   # 在测试协议内
                # quat = Quaternion(axis=[0, 0, 1], radians=sample_annotations[0][index][6])
                # size = sample_annotations[0][index][3:6][[1, 0, 2]]  #  output_to_lyft_box 把预测的转了
                # annotation = {
                #     'sample_token': sample_token,
                #     'translation': sample_annotations[0][index][0:3],
                #     'size': [abs(size[0]), abs(size[1]), abs(size[2])],
                #     'rotation': quat.elements.tolist(),
                #     'name': class_name,
                # }
        return all_annotations


def output_to_lyft_box(detection):
    """Convert the output to the box class in the Lyft.

    Args:
        detection (dict): Detection results.

    Returns:
        list[:obj:`LyftBox`]: List of standard LyftBoxes.
    """
    box3d = detection['boxes_3d']
    scores = detection['scores_3d'].numpy()
    labels = detection['labels_3d'].numpy()

    box_gravity_center = box3d.gravity_center.numpy()
    box_dims = box3d.dims.numpy()
    box_yaw = box3d.yaw.numpy()

    # our LiDAR coordinate system -> Lyft box coordinate system
    lyft_box_dims = box_dims[:, [1, 0, 2]]

    box_list = []
    for i in range(len(box3d)):
        quat = Quaternion(axis=[0, 0, 1], radians=box_yaw[i])
        box = LyftBox(
            box_gravity_center[i],
            lyft_box_dims[i],
            quat,
            label=labels[i],
            score=scores[i])
        box_list.append(box)
    return box_list



def lidar_lyft_box_to_global(info, boxes):
    """Convert the box from ego to global coordinate.

    Args:
        info (dict): Info for a specific sample data, including the
            calibration information.
        boxes (list[:obj:`LyftBox`]): List of predicted LyftBoxes.

    Returns:
        list: List of standard LyftBoxes in the global
            coordinate.
    """
    box_list = []
    for box in boxes:
        # # Move box to ego vehicle coord system
        # box.rotate(Quaternion(info['lidar2ego_rotation']))
        # box.translate(np.array(info['lidar2ego_translation']))
        # Move box to global coord system
        # box.rotate(Quaternion(info['ego2global_rotation']))
        # box.translate(np.array(info['ego2global_translation']))
        box_list.append(box)
    return box_list
