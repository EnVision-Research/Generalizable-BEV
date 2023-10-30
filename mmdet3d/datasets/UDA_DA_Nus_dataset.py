# Copyright (c) OpenMMLab. All rights reserved.
import os
import tempfile
from os import path as osp
import warnings
import math
from typing import Callable
from scipy.spatial.transform import Rotation as R


import mmcv
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from lyft_dataset_sdk.lyftdataset import LyftDataset as Lyft
from lyft_dataset_sdk.utils.data_classes import Box as LyftBox
from pyquaternion import Quaternion
from mmcv.utils import print_log
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion
from nuscenes import NuScenes
from mmcv.utils import print_log
from terminaltables import AsciiTable
import mmcv
import numpy as np
import pyquaternion
from nuscenes.utils.data_classes import Box as NuScenesBox
from nuscenes import NuScenes
from nuscenes.eval.detection.evaluate import NuScenesEval
from torch.utils.data import Dataset

from nuscenes.eval.common.config import config_factory
from nuscenes.eval.common.data_classes import EvalBox
from nuscenes.utils.data_classes import Box
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
from mmdet3d.core.evaluation.lyft_eval import lyft_eval
from ..core import show_result
from ..core.bbox import Box3DMode, Coord3DMode, LiDARInstance3DBoxes, get_box_type
from .builder import DATASETS
from .custom_3d import Custom3DDataset
from .pipelines import Compose
from .lyft_dataset import LyftDataset, LyftDataset_my
from .utils import extract_result_dict, get_loading_pipeline

@DATASETS.register_module(force=True)
class UDA_DA_Nus(Dataset):
    def __init__(self,
                 ann_file_source,
                 ann_file_target,
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
                 file_client_args=dict(backend='disk'),
                 ann_file='./Noused.pkl',
                 **kwargs):
        self.ann_file_source = ann_file_source
        self.ann_file_target = ann_file_target
        self.load_interval = load_interval
        self.Target_CLASSES = Target_CLASSES
        self.Source_CLASSES = Source_CLASSES
        self.Test_CLASSES = Test_CLASSES
        self.data_root = data_root
        self.ann_file = ann_file
        self.test_mode = test_mode
        self.modality = modality
        self.filter_empty_gt = filter_empty_gt
        self.box_type_3d, self.box_mode_3d = get_box_type(box_type_3d)

        self.CLASSES = self.get_classes(classes)
        self.file_client = mmcv.FileClient(**file_client_args)
        self.cat2id = {name: i for i, name in enumerate(self.CLASSES)}

        # load source annotations
        if hasattr(self.file_client, 'get_local_path'):
            with self.file_client.get_local_path(self.ann_file_source) as local_path:
                self.data_infos_source = self.load_annotations(open(local_path, 'rb'))
        else:
            warnings.warn(
                'The used MMCV version does not have get_local_path. '
                f'We treat the {self.ann_file_source} as local paths and it '
                'might cause errors if the path is not a local path. '
                'Please use MMCV>= 1.3.16 if you meet errors.')
            self.data_infos_source = self.load_annotations(self.ann_file_source)
        # load target annotations
        if hasattr(self.file_client, 'get_local_path'):
            with self.file_client.get_local_path(self.ann_file_target) as local_path:
                self.data_infos_target = self.load_annotations(open(local_path, 'rb'))
        else:
            warnings.warn(
                'The used MMCV version does not have get_local_path. '
                f'We treat the {self.ann_file_target} as local paths and it '
                'might cause errors if the path is not a local path. '
                'Please use MMCV>= 1.3.16 if you meet errors.')
            self.data_infos_target = self.load_annotations(self.ann_file_target)
        self.data_infos_target = self.mark_target(self.data_infos_target)
        self.data_infos = self.mix_source_target(self.data_infos_source, self.data_infos_target)
        # process pipeline
        if pipeline is not None:
            self.pipeline = Compose(pipeline)

        # set group flag for the samplers
        if not self.test_mode:
            self._set_group_flag()

        if self.modality is None:
            self.modality = dict(
                use_camera=False,
                use_lidar=True,
                use_radar=False,
                use_map=False,
                use_external=False,
            )

        if self.Target_CLASSES  == None:
            self.Target_CLASSES = ['car', 'truck', 'bus', 'emergency_vehicle', 'other_vehicle',
                                   'motorcycle', 'bicycle', 'pedestrian', 'animal']
        if self.Source_CLASSES == None:
            self.Source_CLASSES = ['car', 'truck', 'van', 'motorcycle', 'bicycle', 'pedestrian']

        if self.Test_CLASSES == None:
            self.Test_CLASSES = ['car', 'truck', 'bicycle', 'pedestrian']

        print('Source_CLASSES', self.Source_CLASSES)
        print('Target_CLASSES', self.Target_CLASSES)
        print('Test_CLASSES', self.Test_CLASSES)

        DefaultAttribute = {
            'car': 'is_stationary',
            'truck': 'is_stationary',
            'bus': 'is_stationary',
            'motorcycle': 'is_stationary',
            'cyclist': 'is_stationary',
            'pedestrian': 'is_stationary'
        }
        CLASSES = ('car', 'truck', 'van',
                   'motorcycle', 'cyclist', 'pedestrian')

    @classmethod
    def get_classes(cls, classes=None):
        """Get class names of current dataset.

        Args:
            classes (Sequence[str] | str): If classes is None, use
                default CLASSES defined by builtin dataset. If classes is a
                string, take it as a file name. The file contains the name of
                classes where each line contains one class name. If classes is
                a tuple or list, override the CLASSES defined by the dataset.

        Return:
            list[str]: A list of class names.
        """
        if classes is None:
            return cls.CLASSES

        if isinstance(classes, str):
            # take it as a file path
            class_names = mmcv.list_from_file(classes)
        elif isinstance(classes, (tuple, list)):
            class_names = classes
        else:
            raise ValueError(f'Unsupported type {type(classes)} of classes.')

        return class_names

    def mark_target(self, data_infos_target):
        return data_infos_target

    def mix_source_target(self, data_infos_source, data_infos_target):
        data_infos = data_infos_source+data_infos_target
        return data_infos

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): sample index
                - pts_filename (str): filename of point clouds
                - sweeps (list[dict]): infos of sweeps
                - timestamp (float): sample timestamp
                - img_filename (str, optional): image filename
                - lidar2img (list[np.ndarray], optional): transformations
                    from lidar to different cameras
                - ann_info (dict): annotation info
        """
        info = self.data_infos[index]

        # standard protocol modified from SECOND.Pytorch
        input_dict = dict(
            sample_idx=info['lidar_path'],
            pts_filename=info['lidar_path'],
            timestamp=info['timestamp'],
        )
        if 'ann_infos' in info:
            input_dict['ann_infos'] = info['ann_infos']
        input_dict.update(dict(curr=info))

        return input_dict

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
        gt_names = set(info['gt_names'])
        # print(gt_names)
        # print('self.cat2id', self.cat2id)
        # print('self.CLASSES', self.CLASSES)
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
        # loading data from a file-like object needs file format
        data = mmcv.load(ann_file, file_format='pkl')
        data_infos = list(sorted(data['infos'], key=lambda e: e['timestamp']))
        data_infos = data_infos[::self.load_interval]
        self.metadata = dict(image_size=[1280, 800],
                             categories=['car', 'truck', 'bus', 'motorcycle', 'cyclist', 'pedestrian'])
        self.version = 'v1'
        return data_infos

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
            # sample_token = self.data_infos[sample_id]['token']
            # boxes = lidar_lyft_box_to_global(self.data_infos[sample_id], boxes)
            sample_token = metas_pre[sample_id][0]['pts_filename']
            data_index_from_pre = self.find_index(self.data_infos, sample_token)
            if data_index_from_pre == -1:
                print("here don't match")
                print(metas_pre[sample_id])
            boxes = lidar_lyft_box_to_global(self.data_infos[data_index_from_pre], boxes)
            for i, box in enumerate(boxes):
                name = mapped_class_names[box.label]
                lyft_anno = dict(
                    sample_token=sample_token,
                    translation=box.center.tolist(),
                    size=box.wlh[[1, 0, 2]].tolist(),
                    rotation=box.orientation.elements.tolist(),
                    name=name,
                    score=box.score)
                annos.append(lyft_anno)
            lyft_annos[sample_token] = annos
        lyft_submissions = {
            'meta': self.modality,
            'results': lyft_annos,
        }
        # mmcv.dump(lyft_submissions, '/mnt/cfs/algorithm/hao.lu/Code/BEVDet/scripts/lyft_submissions.pkl')
        mmcv.mkdir_or_exist(jsonfile_prefix)
        res_path = osp.join(jsonfile_prefix, 'results_lyft.json')
        print('Results writes to', res_path)
        mmcv.dump(lyft_submissions, res_path)
        return res_path

    def find_index(self, data_infos, sample_token):
        for index, sample in enumerate(data_infos):
            # print(sample_token)
            # print(sample.keys())
            if sample['lidar_path'] == sample_token:
                return index
        return -1

    def _evaluate_single(self,
                         result_path,
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
        # lyft = Lyft(
        #     data_path=osp.join(self.data_root, self.version),
        #     json_path=osp.join(self.data_root, self.version, self.version),
        #     verbose=True)
        # print('self.version')
        # print(self.version)
        # eval_set_map = {
        #     'v1.01-train': 'val',
        # }
        metrics = self.eval_core(result_path, output_dir, logger)
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

    def load_gts(self):
        # 仅仅是为了测评的加在
        all_annotations = []
        # Load annotations and filter predictions and annotations.
        for data_info in mmcv.track_iter_progress(self.data_infos):
            sample_token = data_info['token']
            sample_annotations = data_info['ann_infos']
            for index in range(len(sample_annotations[0])):
                class_name = self.Target_CLASSES[sample_annotations[1][index]]  # 目标数据集的标注顺序解码类别
                if class_name in self.Test_CLASSES:  # 在测试协议内
                    quat = Quaternion(axis=[0, 0, 1], radians=sample_annotations[0][index][6])
                    size = sample_annotations[0][index][3:6][[1, 0, 2]]  # output_to_lyft_box 把预测的转了
                    annotation = {
                        'sample_token': sample_token,
                        'translation': sample_annotations[0][index][0:3],
                        'size': [abs(size[0]), abs(size[1]), abs(size[2])],
                        'rotation': quat.elements.tolist(),
                        'name': class_name,
                    }
                    all_annotations.append(annotation)

        return all_annotations

    def eval_core(self, res_path, output_dir, logger=None, NDS=True):
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
        mmcv.dump(gts, osp.join(output_dir, 'gts.pkl'))
        mmcv.dump(predictions, osp.join(output_dir, 'pre.pkl'))

        # gts = group_by_key(gts, 'name')
        # pre = group_by_key(predictions, 'name')
        dist_th_list = [0.5, 1, 2, 4]

        # class_name = 'car'
        # gts = gts['car']
        # pre = pre['car']
        # -----------------------------------
        # Step 1: Accumulate metric data for all classes and distance thresholds.
        # -----------------------------------
        metric_data_list = DetectionMetricDataList()

        for class_name in self.CLASSES:
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
        for class_name in self.CLASSES:
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
        metrics_summary['class_names'] = self.CLASSES
        return metrics_summary

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

        mmcv.dump(results,
                  '/mnt/cfs/algorithm/hao.lu/Code/BEVDepth_pg/work_dirs/UDA/resualt.pkl')
        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None

        result_files = self._format_bbox(results, metas_pre, jsonfile_prefix)

        if csv_savepath is not None:
            self.json2csv(result_files['pts_bbox'], csv_savepath)
        return result_files, tmp_dir

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

        if 'img_metas' in results[0].keys():
            results_pre, metas_pre = self.form_meta(results)
        else:
            results_pre = results

        result_files, tmp_dir = self.format_results(results_pre, metas_pre, jsonfile_prefix,
                                                    csv_savepath)
        # if isinstance(result_files, dict):
        #     print('!~!!#@!#$!@#!#!@#!#@!#@!')
        #     print('!~!!#@!#$!@#!#!@#!#@!#@!')
        #     print('!~!!#@!#$!@#!#!@#!#@!#@!')
        #     results_dict = dict()
        #     for name in result_names:
        #         print(f'Evaluating bboxes of {name}')
        #         ret_dict = self._evaluate_single(result_files[name])
        #     results_dict.update(ret_dict)
        # elif isinstance(result_files, str):
        results_dict = self._evaluate_single(result_files)

        if tmp_dir is not None:
            tmp_dir.cleanup()

        if show or out_dir:
            self.show(results, out_dir, show=show, pipeline=pipeline)
        return results_dict


    def json2csv(self, json_path, csv_savepath):
        """Convert the json file to csv format for submission.

        Args:
            json_path (str): Path of the result json file.
            csv_savepath (str): Path to save the csv file.
        """
        results = mmcv.load(json_path)['results']
        sample_list_path = osp.join(self.data_root, 'sample_submission.csv')
        data = pd.read_csv(sample_list_path)
        Id_list = list(data['Id'])
        pred_list = list(data['PredictionString'])
        cnt = 0
        print('Converting the json to csv...')
        for token in results.keys():
            cnt += 1
            predictions = results[token]
            prediction_str = ''
            for i in range(len(predictions)):
                prediction_str += \
                    str(predictions[i]['score']) + ' ' + \
                    str(predictions[i]['translation'][0]) + ' ' + \
                    str(predictions[i]['translation'][1]) + ' ' + \
                    str(predictions[i]['translation'][2]) + ' ' + \
                    str(predictions[i]['size'][0]) + ' ' + \
                    str(predictions[i]['size'][1]) + ' ' + \
                    str(predictions[i]['size'][2]) + ' ' + \
                    str(Quaternion(list(predictions[i]['rotation']))
                        .yaw_pitch_roll[0]) + ' ' + \
                    predictions[i]['name'] + ' '
            prediction_str = prediction_str[:-1]
            idx = Id_list.index(token)
            pred_list[idx] = prediction_str
        df = pd.DataFrame({'Id': Id_list, 'PredictionString': pred_list})
        mmcv.mkdir_or_exist(os.path.dirname(csv_savepath))
        df.to_csv(csv_savepath, index=False)

    def _build_default_pipeline(self):
        """Build the default pipeline for this dataset."""
        raise NotImplementedError('_build_default_pipeline is not implemented '
                                  f'for dataset {self.__class__.__name__}')

    def _get_pipeline(self, pipeline):
        """Get data loading pipeline in self.show/evaluate function.

        Args:
            pipeline (list[dict]): Input pipeline. If None is given,
                get from self.pipeline.
        """
        if pipeline is None:
            if not hasattr(self, 'pipeline') or self.pipeline is None:
                warnings.warn(
                    'Use default pipeline for data loading, this may cause '
                    'errors when data is on ceph')
                return self._build_default_pipeline()
            loading_pipeline = get_loading_pipeline(self.pipeline.transforms)
            return Compose(loading_pipeline)
        return Compose(pipeline)

    def _extract_data(self, index, pipeline, key, load_annos=False):
        """Load data using input pipeline and extract data according to key.

        Args:
            index (int): Index for accessing the target data.
            pipeline (:obj:`Compose`): Composed data loading pipeline.
            key (str | list[str]): One single or a list of data key.
            load_annos (bool): Whether to load data annotations.
                If True, need to set self.test_mode as False before loading.

        Returns:
            np.ndarray | torch.Tensor | list[np.ndarray | torch.Tensor]:
                A single or a list of loaded data.
        """
        assert pipeline is not None, 'data loading pipeline is not provided'
        # when we want to load ground-truth via pipeline (e.g. bbox, seg mask)
        # we need to set self.test_mode as False so that we have 'annos'
        if load_annos:
            original_test_mode = self.test_mode
            self.test_mode = False
        input_dict = self.get_data_info(index)
        self.pre_pipeline(input_dict)
        example = pipeline(input_dict)

        # extract data items according to keys
        if isinstance(key, str):
            data = extract_result_dict(example, key)
        else:
            data = [extract_result_dict(example, k) for k in key]
        if load_annos:
            self.test_mode = original_test_mode

        return data

    def __len__(self):
        """Return the length of data infos.

        Returns:
            int: Length of data infos.
        """
        return len(self.data_infos)

    def _rand_another(self, idx):
        """Randomly get another item with the same flag.

        Returns:
            int: Another index of item with the same flag.
        """
        pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(pool)

    def __getitem__(self, idx):
        """Get item from infos according to the given index.

        Returns:
            dict: Data dictionary of the corresponding index.
        """
        if self.test_mode:
            return self.prepare_test_data(idx)
        while True:
            data = self.prepare_train_data(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0. In 3D datasets, they are all the same, thus are all
        zeros.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)


    def pre_pipeline(self, results):
        """Initialization before data preparation.

        Args:
            results (dict): Dict before data preprocessing.

                - img_fields (list): Image fields.
                - bbox3d_fields (list): 3D bounding boxes fields.
                - pts_mask_fields (list): Mask fields of points.
                - pts_seg_fields (list): Mask fields of point segments.
                - bbox_fields (list): Fields of bounding boxes.
                - mask_fields (list): Fields of masks.
                - seg_fields (list): Segment fields.
                - box_type_3d (str): 3D box type.
                - box_mode_3d (str): 3D box mode.
        """
        results['img_fields'] = []
        results['bbox3d_fields'] = []
        results['pts_mask_fields'] = []
        results['pts_seg_fields'] = []
        results['bbox_fields'] = []
        results['mask_fields'] = []
        results['seg_fields'] = []
        results['box_type_3d'] = self.box_type_3d
        results['box_mode_3d'] = self.box_mode_3d

    def prepare_train_data(self, index):
        """Training data preparation.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict: Training data dict of the corresponding index.
        """
        input_dict = self.get_data_info(index)
        if input_dict is None:
            return None
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        if self.filter_empty_gt and \
                (example is None or
                    ~(example['gt_labels_3d']._data != -1).any()):
            return None
        return example

    def prepare_test_data(self, index):
        """Prepare data for testing.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict: Testing data dict of the corresponding index.
        """
        input_dict = self.get_data_info(index)
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        return example

def accumulate(gt_boxes: EvalBoxes,
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
    for ind in sortind:
        pred_box = pred_boxes_list[ind]
        pred_box = Box3D(**pred_box)
        min_dist = np.inf
        match_gt_idx = None
        if not (pred_box.sample_token in image_gts.keys()):
            print('Sample not found:', pred_box.sample_token)
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
            match_data['orient_err'].append(yaw_diff(gt_box_match, pred_box, period=period))
            match_data['attr_err'].append(1 - attr_acc(gt_box_match, pred_box))
            match_data['conf'].append(pred_box.score)
        else:
            # No match. Mark this as a false positive.
            tp.append(0)
            fp.append(1)
            conf.append(pred_box.score)
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