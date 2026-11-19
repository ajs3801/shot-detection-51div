"""Evaluate target polygons and both bullet crop pipelines on paired test images."""
import argparse
import csv
import json
import time
from pathlib import Path
from src.runtime import configure, add_model_options, config_from_args, output_directory, require_weights, save_json


def main(args):
    configure()
    import cv2
    import numpy as np
    from ultralytics import YOLO
    from src.datasets import check_pairs, paired_samples, write_data_yaml
    from src.geometry import order_quad, polygon_metrics, crop_polygon, shift_boxes
    from src.prediction import predict_target, predict_bullets
    from src.metrics import aggregate_metrics
    from src.visualization import evaluation_preview

    config = config_from_args(args)
    require_weights(config)
    counts = check_pairs(config)
    output = output_directory('test', args.output)
    save_json(output / 'config.json', config)
    save_json(output / 'split_check.json', counts)
    target, bullet = YOLO(config['target']['weights']), YOLO(config['bullet']['weights'])
    records_gt, records_pred, details = [], [], []
    started = time.perf_counter()
    tcfg, bcfg, icfg = config['target'], config['bullet'], config['inference']
    for sample in paired_samples(config, split='test'):
        full, quad_gt, truth = sample['full'], sample['gt_quad'], sample['gt_boxes']
        boxes_gt, scores_gt = predict_bullets(bullet, sample['gt_crop'], bcfg['device'], bcfg['imgsz'], icfg['ap_confidence_floor'], icfg['iou'])
        prediction = predict_target(
            target, full, tcfg['confidence'], tcfg['device'], tcfg['imgsz'], icfg['iou'],
            retina_masks=tcfg.get('retina_masks', True),
        )
        crop = rect = None
        boxes = np.empty((0, 4), np.float32)
        scores = np.empty(0, np.float32)
        if prediction is not None:
            crop, rect = crop_polygon(full, prediction['quad'], icfg['padding'])
            local, scores = predict_bullets(bullet, crop, bcfg['device'], bcfg['imgsz'], icfg['ap_confidence_floor'], icfg['iou'])
            boxes = shift_boxes(local, rect[:2])
            (output / 'predicted_crops').mkdir(exist_ok=True)
            cv2.imwrite(str(output / 'predicted_crops' / f'{sample["sample"]}.png'), crop)
        records_gt.append({'sample': sample['sample'], 'gt_boxes': truth,
                           'boxes': shift_boxes(boxes_gt, sample['gt_rect'][:2]), 'scores': scores_gt})
        # A missed paper produces no predictions, retaining every GT impact as FN.
        records_pred.append({'sample': sample['sample'], 'gt_boxes': truth, 'boxes': boxes, 'scores': scores})
        fitted = polygon_metrics(None if prediction is None else prediction['quad'], quad_gt, full.shape[1], full.shape[0])
        raw = polygon_metrics(None if prediction is None else prediction['contour'], quad_gt, full.shape[1], full.shape[0])
        detail = {'sample': sample['sample'], 'target_detected': prediction is not None,
                  'target_confidence': None if prediction is None else prediction['confidence'],
                  'quad_iou': fitted['iou'], 'mask_iou': raw['iou'], 'gt_area_retained': fitted['gt_area_recall'],
                  'corner_mean_error_px': None if prediction is None else float(np.linalg.norm(order_quad(prediction['quad'])-order_quad(quad_gt), axis=1).mean()),
                  'gt_impact_count': len(truth), 'gt_crop_predicted_count': int((scores_gt >= bcfg['confidence']).sum()),
                  'pred_crop_predicted_count': int((scores >= bcfg['confidence']).sum()),
                  'gt_polygon': quad_gt, 'pred_polygon': None if prediction is None else prediction['quad'],
                  'gt_crop_xyxy': sample['gt_rect'], 'pred_crop_xyxy': rect}
        details.append(detail)
        preview = evaluation_preview(sample, prediction, boxes_gt, scores_gt, crop, boxes, scores, bcfg['confidence'], rect)
        (output / 'visualizations').mkdir(exist_ok=True)
        cv2.imwrite(str(output / 'visualizations' / f'{sample["sample"]}.jpg'), preview)
        print(f'{sample["sample"]}: target IoU {fitted["iou"]:.3f}, impacts GT={len(truth)} GT-crop={detail["gt_crop_predicted_count"]} pred-crop={detail["pred_crop_predicted_count"]}', flush=True)
    gt_metrics = aggregate_metrics(records_gt, bcfg['confidence'], plots=output / 'plots', prefix='GT_crop_')
    pred_metrics = aggregate_metrics(records_pred, bcfg['confidence'], plots=output / 'plots', prefix='Predicted_crop_')
    detected = [d for d in details if d['target_detected']]
    target_metrics = {'test_images': len(details), 'detections': len(detected), 'missed_targets': len(details)-len(detected),
                      'mean_fitted_quad_IoU': float(np.mean([d['quad_iou'] for d in details])),
                      'mean_fitted_quad_IoU_detected_only': float(np.mean([d['quad_iou'] for d in detected])) if detected else None,
                      'mean_raw_mask_IoU': float(np.mean([d['mask_iou'] for d in details])),
                      'mean_corner_error_px_on_detected': float(np.mean([d['corner_mean_error_px'] for d in detected])) if detected else None,
                      'quad_recall_IoU50': float(np.mean([d['quad_iou'] >= .5 for d in details])),
                      'quad_recall_IoU75': float(np.mean([d['quad_iou'] >= .75 for d in details])),
                      'confidence_threshold': tcfg['confidence']}
    native = {}
    if not args.skip_native:
        for task, model in [('target', target), ('bullet', bullet)]:
            data = write_data_yaml(config, task, output / 'data')
            value = model.val(data=str(data), split='test', device=config[task]['device'],
                              imgsz=config[task]['imgsz'], batch=4, workers=0, plots=True,
                              project=str(output / 'native'), name=task, exist_ok=False,
                              conf=icfg['ap_confidence_floor'], iou=icfg['iou'],
                              max_det=10 if task == 'target' else 100, verbose=False)
            native[task] = {k: float(v) for k, v in value.results_dict.items()}
    summary = {'split': 'test', 'paired_test_images': len(details), 'target': target_metrics,
               'bullet_GT_crop': gt_metrics, 'bullet_predicted_polygon_crop': pred_metrics,
               'native_ultralytics_test_metrics': native, 'elapsed_s': time.perf_counter()-started,
               'threshold_source': config.get('threshold_provenance', 'Fixed config; no threshold selection on test.'),
               'coordinates': 'Both bullet pipelines are evaluated in original full-frame coordinates.',
               'empty_labels': 'Zero-impact test images remain included to measure false positives.',
               'target_failure_policy': 'Keep image; all its GT impacts count as FN.',
               'temporal_tracking': False}
    save_json(output / 'summary.json', summary)
    save_json(output / 'per_image.json', details)
    save_json(output / 'all_predictions.json', {'gt_crop': records_gt, 'predicted_crop': records_pred})
    with (output / 'comparison.csv').open('w', newline='') as stream:
        fields = ['pipeline', 'images', 'ground_truth_impacts', 'precision', 'recall', 'f1', 'tp', 'fp', 'fn', 'mAP50', 'mAP50_95', 'count_MAE']
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for name, metrics in [('GT_polygon_crop', gt_metrics), ('Predicted_polygon_crop', pred_metrics)]:
            writer.writerow({'pipeline': name, **metrics})
    print(f'Test results: {output}')
    print(json.dumps({'target': target_metrics, 'GT_crop_mAP50': gt_metrics['mAP50'], 'pred_crop_mAP50': pred_metrics['mAP50']}, ensure_ascii=False, indent=2))


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    add_model_options(value)
    value.add_argument('--output', type=Path)
    value.add_argument('--skip-native', action='store_true', help='Skip additional Ultralytics native test metrics.')
    return value


if __name__ == '__main__':
    main(parser().parse_args())
