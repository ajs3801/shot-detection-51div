"""Shared metrics operations."""
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.utils.metrics import ap_per_class
from .geometry import box_iou


def match_predictions(gt, pred):
    n = len(pred)
    if not len(gt) or not n:
        return np.zeros((n, 10), bool)
    matcher = SimpleNamespace(iouv=torch.linspace(0.5, 0.95, 10))
    return DetectionValidator.match_predictions(matcher, torch.zeros(n), torch.zeros(len(gt)), torch.from_numpy(box_iou(gt, pred))).numpy()

def aggregate_metrics(records, threshold, plots=None, prefix=''):
    tps = []
    scores = []
    ngt = 0
    tp = fp = fn = 0
    counts = []
    details = []
    empty_fp = 0
    empty_frames = 0
    for r in records:
        gt = np.asarray(r['gt_boxes'], np.float32).reshape(-1, 4)
        boxes = np.asarray(r['boxes'], np.float32).reshape(-1, 4)
        conf = np.asarray(r['scores'], np.float32)
        ngt += len(gt)
        matched = match_predictions(gt, boxes)
        tps.append(matched)
        scores.append(conf)
        keep = conf >= threshold
        selected = boxes[keep]
        fixed_match = match_predictions(gt, selected)
        itp = int(fixed_match[:, 0].sum())
        ifp = len(selected) - itp
        ifn = len(gt) - itp
        tp += itp
        fp += ifp
        fn += ifn
        counts.append(abs(len(selected) - len(gt)))
        if len(gt) == 0:
            empty_frames += 1
            empty_fp += len(selected)
        details.append({'sample': r['sample'], 'gt_count': len(gt), 'predicted_count': len(selected), 'tp': itp, 'fp': ifp, 'fn': ifn, 'count_error': len(selected) - len(gt)})
    alltp = np.concatenate(tps, axis=0) if tps else np.zeros((0, 10), bool)
    allscores = np.concatenate(scores) if scores else np.empty(0)
    ap = np.zeros((1, 10))
    if len(allscores) and ngt:
        if plots:
            Path(plots).mkdir(parents=True, exist_ok=True)
        values = ap_per_class(alltp, allscores, np.zeros(len(allscores)), np.zeros(ngt), plot=plots is not None, save_dir=Path(plots) if plots else Path('.'), names={0: 'impact'}, prefix=prefix)
        ap = values[5]
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {'images': len(records), 'ground_truth_impacts': ngt, 'threshold': float(threshold), 'iou_for_precision_recall': 0.5, 'precision': precision, 'recall': recall, 'f1': 2 * precision * recall / max(precision + recall, 1e-15), 'tp': tp, 'fp': fp, 'fn': fn, 'mAP50': float(ap[:, 0].mean()), 'mAP50_95': float(ap.mean()), 'count_MAE': float(np.mean(counts)) if counts else 0, 'exact_count_accuracy': float(np.mean(np.array(counts) == 0)) if counts else 0, 'zero_impact_images': empty_frames, 'false_detections_on_zero_impact_images': empty_fp, 'per_image': details}
