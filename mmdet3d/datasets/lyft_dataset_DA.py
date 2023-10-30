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
from nuscenes.eval.common.utils import center_distance, scale_iou, yaw_diff, velocity_l2, attr_acc, cummean
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



@DATASETS.register_module(force=True)
class LyftDataset_DA(LyftDataset_my):
    def __init__(self,
                 ann_file,
                 pipeline=None,
                 data_root=None,
                 classes=None,
                 load_interval=1,
                 modality=None,
                 box_type_3d='LiDAR',
                 filter_empty_gt=True,
                 test_mode=False,
                 Target_CLASSES=None,
                 Source_CLASSES=None,
                 Test_CLASSES=None,
                 **kwargs):
        self.load_interval = load_interval
        self.Target_CLASSES = Target_CLASSES
        self.Source_CLASSES = Source_CLASSES
        self.Test_CLASSES = Test_CLASSES
        super(LyftDataset_DA, self).__init__(
            data_root=data_root,
            ann_file=ann_file,
            pipeline=pipeline,
            classes=classes,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode,
            # Target_CLASSES=Target_CLASSES,
            # Source_CLASSES=Source_CLASSES,
            # Test_CLASSES=Test_CLASSES,
            **kwargs)

        if self.modality is None:
            self.modality = dict(
                use_camera=False,
                use_lidar=True,
                use_radar=False,
                use_map=False,
                use_external=False,
            )
        # Target_CLASSES = None
        # Source_CLASSES = None
        # Test_CLASSES = None
        if Target_CLASSES==None:
            self.Target_CLASSES = ['car', 'truck', 'bus', 'emergency_vehicle', 'other_vehicle',
                              'motorcycle', 'bicycle', 'pedestrian', 'animal']
        else:
            self.Target_CLASSES = Target_CLASSES
        if Source_CLASSES==None:
            self.Source_CLASSES = ['car', 'truck', 'van', 'motorcycle', 'bicycle', 'pedestrian']
        else:
            self.Source_CLASSES = Source_CLASSES
        if Test_CLASSES==None:
            self.Test_CLASSES = ['car', 'truck', 'bicycle', 'pedestrian']
        else:
            self.Test_CLASSES = Test_CLASSES
        print('Source_CLASSES', self.Source_CLASSES)
        print('Target_CLASSES', self.Target_CLASSES)
        print('Test_CLASSES', self.Test_CLASSES)


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
            'results': lyft_annos,
        }

        mmcv.mkdir_or_exist(jsonfile_prefix)
        res_path = osp.join(jsonfile_prefix, 'results_lyft.json')
        print('Results writes to', res_path)
        mmcv.dump(lyft_submissions, '/mnt/cfs/algorithm/hao.lu/Code/BEVDet/scripts/Eval/results_lyft.json')
        mmcv.dump(lyft_submissions, res_path)
        return res_path

    def find_index(self, data_infos, sample_token, token_name='token'):
        # 在原始pkl文件中寻找gt的index
        # data_infos：pkl读入的数据
        # sample_token：测试数据集的token
        # token_name：token的名字，每个数据集不太一样
        for index, sample in enumerate(data_infos):
            if sample[token_name] == sample_token:
                return index

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

        results_dict = self._evaluate_single(result_files,save_path)

        if tmp_dir is not None:
            tmp_dir.cleanup()

        if show or out_dir:
            self.show(results, out_dir, show=show, pipeline=pipeline)
        return results_dict

    def load_gts(self):
        bev_range = np.array([-51.2, -51.2, 51.2, 51.2], dtype=np.float32)
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
                    if sample_annotations[1][index] < len(self.Target_CLASSES):
                        class_name = self.Target_CLASSES[sample_annotations[1][index]]   # 目标数据集的标注顺序解码类别
                        if class_name in self.Test_CLASSES:
                            annotation = dict(
                                sample_token=sample_token,
                                translation=box[0].center.tolist(),
                                size=box[0].wlh.tolist(),
                                rotation=box[0].orientation.elements.tolist(),
                                name=class_name)
                            all_annotations.append(annotation)
        return all_annotations

    # def load_gts(self):
    #     # 仅仅是为了测评的加在
    #     all_annotations = []
    #     # Load annotations and filter predictions and annotations.
    #     for data_info in mmcv.track_iter_progress(self.data_infos):
    #         sample_token = data_info['token']
    #         sample_annotations = data_info['ann_infos']
    #         for index in range(len(sample_annotations[0])):
    #             class_name = self.Target_CLASSES[sample_annotations[1][index]]   # 目标数据集的标注顺序解码类别
    #             if class_name in self.Test_CLASSES:   # 在测试协议内
    #                 quat = Quaternion(axis=[0, 0, 1], radians=sample_annotations[0][index][6])
    #                 size = sample_annotations[0][index][3:6][[1, 0, 2]]  #  output_to_lyft_box 把预测的转了
    #                 annotation = {
    #                     'sample_token': sample_token,
    #                     'translation': sample_annotations[0][index][0:3],
    #                     'size': [abs(size[0]), abs(size[1]), abs(size[2])],
    #                     'rotation': quat.elements.tolist(),
    #                     'name': class_name,
    #                 }
    #                 all_annotations.append(annotation)
    #
    #     return all_annotations

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
        print(osp.join(save_path, 'tested_on_lyft_gts.pkl'))
        print(osp.join(save_path, 'tested_on_lyft_pre.pkl'))
        mmcv.dump(gts, osp.join(save_path, 'tested_on_lyft_gts.pkl'))
        mmcv.dump(predictions, osp.join(save_path, 'tested_on_lyft_pre.pkl'))

        # gts = group_by_key(gts, 'name')
        # pre = group_by_key(predictions, 'name')
        dist_th_list = [0.5, 1, 2, 4]


        # -----------------------------------
        # Step 1: Accumulate metric data for all classes and distance thresholds.
        # -----------------------------------
        metric_data_list = DetectionMetricDataList()

        for class_name in self.Test_CLASSES:
            for dist_th in dist_th_list:
                md = accumulate(gts, predictions, class_name, center_distance, dist_th)
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
        print()
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
