# Copyright (c) OpenMMLab. All rights reserved.
from os import path as osp
import tempfile
from os import path as osp
from typing import Callable


import mmcv
import numpy as np
from lyft_dataset_sdk.eval.detection.mAP_evaluation import (Box3D, get_ap,
                                                            get_class_names,
                                                            get_ious,
                                                            group_by_key,
                                                            wrap_in_box)
import os.path
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits
from tqdm import tqdm
import mmcv
import cv2
import numpy as np
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
from lyft_dataset_sdk.lyftdataset import LyftDataset as Lyft
from lyft_dataset_sdk.utils.data_classes import Box as LyftBox
from nuscenes.eval.common.config import config_factory
from nuscenes.eval.common.data_classes import EvalBox
from nuscenes.utils.data_classes import Box
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.common.utils import center_distance, scale_iou, yaw_diff, velocity_l2, attr_acc, cummean
from nuscenes.eval.detection.data_classes import DetectionMetricData
from nuscenes import NuScenes
from nuscenes.eval.common.config import config_factory
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.common.loaders import load_prediction, load_gt, add_center_dist, filter_eval_boxes
from nuscenes.eval.detection.algo import calc_ap, calc_tp
from nuscenes.eval.detection.data_classes import DetectionConfig, DetectionMetrics, DetectionBox, \
    DetectionMetricDataList
from nuscenes.eval.detection.render import summary_plot, class_pr_curve, class_tp_curve, dist_pr_curve, visualize_sample
from lyft_dataset_sdk.eval.detection.mAP_evaluation import (Box3D, get_ap,
                                                            get_class_names,
                                                            get_ious,
                                                            group_by_key,
                                                            wrap_in_box)



def center_distance(gt_box: EvalBox, pred_box: EvalBox) -> float:
    """
    L2 distance between the box centers (xy only).
    :param gt_box: GT annotation sample.
    :param pred_box: Predicted sample.
    :return: L2 distance.
    """
    return np.linalg.norm(np.array(pred_box.translation[:2]) - np.array(gt_box.translation[:2]))

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
    npos = len(gt_boxes)
    image_gts = group_by_key(gt_boxes, 'sample_token')
    image_gts = wrap_in_box(image_gts)
    # For missing classes in the GT, return a data structure corresponding to no predictions.
    if npos == 0:
        return DetectionMetricData.no_predictions()
    # Organize the predictions in a single list.
    pred_boxes_list = pred_boxes
    pred_confs = [box['score'] for box in pred_boxes_list]
    # Sort by confidence.
    sortind = [i for (v, i) in sorted((v, i) for (i, v) in enumerate(pred_confs))][::-1]
    # Do the actual matching.
    tp = []  # Accumulator of true positives.
    fp = []  # Accumulator of false positives.
    conf = []  # Accumulator of confidences.
    # match_data holds the extra metrics we calculate for each match.
    match_data = {'trans_err': [],  #  'vel_err': [],
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




cam_names = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT']



def load_lyft_gts(lyft, data_root, eval_split, logger=None):
    """Loads ground truth boxes from database.

    Args:
        lyft (:obj:`LyftDataset`): Lyft class in the sdk.
        data_root (str): Root of data for reading splits.
        eval_split (str): Name of the split for evaluation.
        logger (logging.Logger | str, optional): Logger used for printing
        related information during evaluation. Default: None.

    Returns:
        list[dict]: List of annotation dictionaries.
    """
    split_scenes = mmcv.list_from_file(
        osp.join(data_root, f'{eval_split}.txt'))

    # Read out all sample_tokens in DB.
    sample_tokens_all = [s['token'] for s in lyft.sample]
    assert len(sample_tokens_all) > 0, 'Error: Database has no samples!'

    if eval_split == 'test':
        # Check that you aren't trying to cheat :)
        assert len(lyft.sample_annotation) > 0, \
            'Error: You are trying to evaluate on the test set \
             but you do not have the annotations!'

    sample_tokens = []
    for sample_token in sample_tokens_all:
        scene_token = lyft.get('sample', sample_token)['scene_token']
        scene_record = lyft.get('scene', scene_token)
        if scene_record['name'] in split_scenes:
            sample_tokens.append(sample_token)

    all_annotations = []

    print_log('Loading ground truth annotations...', logger=logger)
    # Load annotations and filter predictions and annotations.
    for sample_token in mmcv.track_iter_progress(sample_tokens):
        sample = lyft.get('sample', sample_token)
        sample_annotation_tokens = sample['anns']
        for sample_annotation_token in sample_annotation_tokens:
            # Get label name in detection task and filter unused labels.
            sample_annotation = \
                lyft.get('sample_annotation', sample_annotation_token)
            detection_name = sample_annotation['category_name']
            if detection_name is None:
                continue
            annotation = {
                'sample_token': sample_token,
                'translation': sample_annotation['translation'],
                'size': sample_annotation['size'],
                'rotation': sample_annotation['rotation'],
                'name': detection_name,
            }
            all_annotations.append(annotation)

    return all_annotations


def load_lyft_predictions(res_path):
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

def form_meta(results_all):
    results = list()
    meta = list()
    for result in results_all:
        results.append(result['pts_bbox'])
        meta.append(result['img_metas'])
    return results, meta

def lyft_eval(lyft, data_root, res_path, eval_set, output_dir, logger=None, NDS=True):
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
    gts = load_lyft_gts(lyft, data_root, eval_set, logger)
    predictions = load_lyft_predictions(res_path)
    mmcv.dump(gts, osp.join(output_dir, 'gts.pkl'))
    mmcv.dump(predictions, osp.join(output_dir, 'pre.pkl'))
    if NDS==True:
        gts = group_by_key(gts, 'name')
        pre = group_by_key(predictions, 'name')
        dist_th_list = [0.5, 1, 2, 4]
        gts = gts['car']
        pre = pre['car']
        metric_data_list = DetectionMetricDataList()
        for dist_th in dist_th_list:
            md = accumulate(gts, pre, 'car', center_distance, dist_th)
            metric_data_list.set('car', dist_th, md)
        TP_METRICS = ['trans_err', 'scale_err', 'orient_err']
        eval_version = 'detection_cvpr_2019'
        eval_detection_configs = config_factory(eval_version)
        metrics = DetectionMetrics(eval_detection_configs)
        class_name = 'car'
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
        return metrics_summary
    else:
        class_names = get_class_names(gts)
        print('Calculating mAP@0.5:0.95...')

        iou_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        metrics = {}
        average_precisions = \
            get_classwise_aps(gts, predictions, class_names, iou_thresholds)
        APs_data = [['IOU', 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]]

        mAPs = np.mean(average_precisions, axis=0)
        mAPs_cate = np.mean(average_precisions, axis=1)
        final_mAP = np.mean(mAPs)

        metrics['average_precisions'] = average_precisions.tolist()
        metrics['mAPs'] = mAPs.tolist()
        metrics['Final mAP'] = float(final_mAP)
        metrics['class_names'] = class_names
        metrics['mAPs_cate'] = mAPs_cate.tolist()

        APs_data = [['class', 'mAP@0.5:0.95']]
        for i in range(len(class_names)):
            row = [class_names[i], round(mAPs_cate[i], 3)]
            APs_data.append(row)
        APs_data.append(['Overall', round(final_mAP, 3)])
        APs_table = AsciiTable(APs_data, title='mAPs@0.5:0.95')
        APs_table.inner_footing_row_border = True
        print_log(APs_table.table, logger=logger)

        res_path = osp.join(output_dir, 'lyft_metrics.json')
        mmcv.dump(metrics, res_path)
        return metrics


def get_classwise_aps(gt, predictions, class_names, iou_thresholds):
    """Returns an array with an average precision per class.

    Note: Ground truth and predictions should have the following format.

    .. code-block::

    gt = [{
        'sample_token': '0f0e3ce89d2324d8b45aa55a7b4f8207
                         fbb039a550991a5149214f98cec136ac',
        'translation': [974.2811881299899, 1714.6815014457964,
                        -23.689857123368846],
        'size': [1.796, 4.488, 1.664],
        'rotation': [0.14882026466054782, 0, 0, 0.9888642620837121],
        'name': 'car'
    }]

    predictions = [{
        'sample_token': '0f0e3ce89d2324d8b45aa55a7b4f8207
                         fbb039a550991a5149214f98cec136ac',
        'translation': [971.8343488872263, 1713.6816097857359,
                        -25.82534357061308],
        'size': [2.519726579986132, 7.810161372666739, 3.483438286096803],
        'rotation': [0.10913582721095375, 0.04099572636992043,
                     0.01927712319721745, 1.029328402625659],
        'name': 'car',
        'score': 0.3077029437237213
    }]

    Args:
        gt (list[dict]): list of dictionaries in the format described below.
        predictions (list[dict]): list of dictionaries in the format
            described below.
        class_names (list[str]): list of the class names.
        iou_thresholds (list[float]): IOU thresholds used to calculate
            TP / FN

    Returns:
        np.ndarray: an array with an average precision per class.
    """
    assert all([0 <= iou_th <= 1 for iou_th in iou_thresholds])

    gt_by_class_name = group_by_key(gt, 'name')
    pred_by_class_name = group_by_key(predictions, 'name')

    average_precisions = np.zeros((len(class_names), len(iou_thresholds)))

    for class_id, class_name in enumerate(class_names):
        if class_name in pred_by_class_name:
            recalls, precisions, average_precision = get_single_class_aps(
                gt_by_class_name[class_name], pred_by_class_name[class_name],
                iou_thresholds)
            average_precisions[class_id, :] = average_precision

    return average_precisions


def get_single_class_aps(gt, predictions, iou_thresholds):
    """Compute recall and precision for all iou thresholds. Adapted from
    LyftDatasetDevkit.

    Args:
        gt (list[dict]): list of dictionaries in the format described above.
        predictions (list[dict]): list of dictionaries in the format
            described below.
        iou_thresholds (list[float]): IOU thresholds used to calculate
            TP / FN

    Returns:
        tuple[np.ndarray]: Returns (recalls, precisions, average precisions)
            for each class.
    """
    num_gts = len(gt)
    image_gts = group_by_key(gt, 'sample_token')
    image_gts = wrap_in_box(image_gts)

    sample_gt_checked = {
        sample_token: np.zeros((len(boxes), len(iou_thresholds)))
        for sample_token, boxes in image_gts.items()
    }

    predictions = sorted(predictions, key=lambda x: x['score'], reverse=True)

    # go down dets and mark TPs and FPs
    num_predictions = len(predictions)
    tps = np.zeros((num_predictions, len(iou_thresholds)))
    fps = np.zeros((num_predictions, len(iou_thresholds)))
    fps_chongfu = np.zeros((num_predictions, len(iou_thresholds)))


    for prediction_index, prediction in enumerate(predictions):
        predicted_box = Box3D(**prediction)

        sample_token = prediction['sample_token']

        max_overlap = -np.inf
        jmax = -1

        if sample_token in image_gts:
            gt_boxes = image_gts[sample_token]
            # gt_boxes per sample
            gt_checked = sample_gt_checked[sample_token]
            # print(gt_boxes)
            # print(predicted_box)
            # gt flags per sample
        else:
            gt_boxes = []
            gt_checked = None

        if len(gt_boxes) > 0:
            overlaps = get_ious(gt_boxes, predicted_box)
            max_overlap = np.max(overlaps)

            jmax = np.argmax(overlaps)

        for i, iou_threshold in enumerate(iou_thresholds):
            if max_overlap > iou_threshold:
                if gt_checked[jmax, i] == 0:
                    tps[prediction_index, i] = 1.0
                    gt_checked[jmax, i] = 1
                else:
                    fps[prediction_index, i] = 1.0
                    fps_chongfu[prediction_index, i] = 1.0
            else:
                fps[prediction_index, i] = 1.0

    mmcv.dump(fps_chongfu, osp.join('/mnt/cfs/algorithm/hao.lu/Code/BEVDet/scripts/Eval', 'fps_chongfu.pkl'))
    mmcv.dump(fps, osp.join('/mnt/cfs/algorithm/hao.lu/Code/BEVDet/scripts/Eval', 'fps.pkl'))
    mmcv.dump(tps, osp.join('/mnt/cfs/algorithm/hao.lu/Code/BEVDet/scripts/Eval', 'tps.pkl'))
    mmcv.dump(num_gts, osp.join('/mnt/cfs/algorithm/hao.lu/Code/BEVDet/scripts/Eval', 'num_gts.pkl'))
    # compute precision recall
    fps = np.cumsum(fps, axis=0)
    tps = np.cumsum(tps, axis=0)

    recalls = tps / float(num_gts)
    # avoid divide by zero in case the first detection
    # matches a difficult ground truth
    precisions = tps / np.maximum(tps + fps, np.finfo(np.float64).eps)

    aps = []
    for i in range(len(iou_thresholds)):
        recall = recalls[:, i]
        precision = precisions[:, i]
        assert np.all(0 <= recall) & np.all(recall <= 1)
        assert np.all(0 <= precision) & np.all(precision <= 1)
        ap = get_ap(recall, precision)
        aps.append(ap)

    aps = np.array(aps)

    return recalls, precisions, aps
