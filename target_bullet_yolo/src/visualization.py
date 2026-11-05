"""Drawing only: never changes the pixels passed to either detector."""
import cv2
import numpy as np


def text(image, message, xy, size=.7, color=(245, 245, 245), thickness=2):
    cv2.putText(image, message, tuple(map(int, xy)), cv2.FONT_HERSHEY_SIMPLEX, size, (0, 0, 0), thickness+2, cv2.LINE_AA)
    cv2.putText(image, message, tuple(map(int, xy)), cv2.FONT_HERSHEY_SIMPLEX, size, color, thickness, cv2.LINE_AA)


def panel(image, rect, alpha=.82):
    x, y, width, height = map(int, rect)
    x, y = max(0, x), max(0, y)
    roi = image[y:min(y+height, image.shape[0]), x:min(x+width, image.shape[1])]
    if roi.size:
        cv2.addWeighted(np.full_like(roi, (28, 23, 18)), alpha, roi, 1-alpha, 0, roi)


def draw_tracks(image, tracks, offset=(0, 0), scale=1.):
    for track in tracks:
        box = np.array(track['box']) - np.tile(offset, 2)
        left, top, right, bottom = np.rint(box).astype(int)
        color = (0, 150, 255) if track.get('held') else (0, 235, 255)
        cv2.rectangle(image, (left, top), (right, bottom), color, max(1, round(2*scale)), cv2.LINE_AA)
        label = f'#{track["id"]}' if track.get('id') is not None else f'{track["score_ema"]:.2f}'
        if track.get('held'):
            label += ' HOLD'
        text(image, label, (left, max(12, top-6)), max(.3, .48*scale), color, 1)


def draw_boxes(image, boxes, scores=None, color=(0, 235, 255)):
    for index, box in enumerate(boxes):
        left, top, right, bottom = np.rint(box).astype(int)
        cv2.rectangle(image, (left, top), (right, bottom), color, 2, cv2.LINE_AA)
        if scores is not None:
            text(image, f'{scores[index]:.2f}', (left, max(18, top-5)), .5, color, 1)


def video_frame(image, quad, crop, rect, tracks, record, config, total):
    result = image.copy()
    h, w = result.shape[:2]
    scale = min(w/1920, h/1080)
    def point(x, y):
        return round(x*scale), round(y*scale)
    def put(message, x, y, size=.7, color=(245, 245, 245)):
        text(result, message, point(x, y), max(.28, size*scale), color, max(1, round(2*scale)))
    temporal = config['tracking']['enabled']
    if quad is not None:
        color = (80, 255, 80) if record['raw_target_detected'] else (255, 210, 60)
        cv2.polylines(result, [np.rint(quad).astype(np.int32)], True, color, max(1, round(3*scale)), cv2.LINE_AA)
        draw_tracks(result, tracks, scale=scale)
    panel(result, (*point(12, 12), *point(1170, 112)))
    title = 'TEMPORAL: target optical flow + impact consensus / tracking' if temporal else 'YOLO26s-seg target -> YOLO26n impacts (per frame)'
    put(title, 26, 42, .74)
    count = record['impact_count'] if quad is not None else 'N/A'
    raw = record['raw_impact_count'] if quad is not None else 'N/A'
    put(f'Frame {record["frame"]}/{total-1 if total else "?"} | {record["time_s"]:05.2f}s | Displayed impacts: {count} | Raw: {raw}', 26, 76, .66)
    hold = config['tracking']['impact_hold_s']
    put(f'Target: {record["target_tracking"]["state"]} | Yellow: updated  Orange: held (max {hold:g}s)' if temporal else 'Independent predictions; temporal stabilization disabled', 26, 107, .58)
    pw, ph = point(430, 865)
    px, py = w-round(450*scale), round(135*scale)
    ph = min(ph, h-py-round(12*scale))
    if min(pw, ph) < 20:
        return result
    panel(result, (px, py, pw, ph))
    text(result, 'STABILIZED POLYGON CROP' if temporal else 'PREDICTED POLYGON CROP',
         (px+round(12*scale), py+round(30*scale)), max(.28, .57*scale), thickness=1)
    if crop is not None:
        display = crop.copy()
        draw_tracks(display, tracks, rect[:2])
        available_w, available_h = pw-round(20*scale), ph-round(130*scale)
        ratio = min(available_w/display.shape[1], available_h/display.shape[0])
        if ratio > 0:
            small = cv2.resize(display, None, fx=ratio, fy=ratio)
            sx = px+(pw-small.shape[1])//2
            sy = py+round(50*scale)+(available_h-small.shape[0])//2
            result[sy:sy+small.shape[0], sx:sx+small.shape[1]] = small
        if temporal:
            settings = config['tracking']
            lines = [f'New tracks: {settings["confirmation_hits"]} hits / {settings["confirmation_window"]} frames',
                     f'Model target gap: {record["target_tracking"]["model_gap_frames"]/record["fps"]:.2f}s',
                     'Previous evidence is held briefly.']
        else:
            lines = [f'Target confidence >= {config["target"]["confidence"]:.2f}',
                     f'Impact confidence >= {config["bullet"]["confidence"]:.2f}', 'Independent current-frame detections.']
        for line, dy in zip(lines, [64, 39, 14]):
            text(result, line, (px+round(12*scale), py+ph-round(dy*scale)), max(.26, .52*scale), thickness=1)
    else:
        text(result, 'WAITING FOR TARGET', (px+round(12*scale), py+round(90*scale)), max(.3, .66*scale), (80, 80, 255), 1)
    return result


def evaluation_preview(sample, prediction, gt_boxes, gt_scores, crop, full_boxes, scores, threshold, pred_rect):
    canvas = np.full((900, 1920, 3), 24, np.uint8)
    full = sample['full'].copy()
    cv2.polylines(full, [np.rint(sample['gt_quad']).astype(np.int32)], True, (255, 180, 60), 3)
    if prediction is not None:
        cv2.polylines(full, [np.rint(prediction['quad']).astype(np.int32)], True, (80, 255, 80), 3)
    gt = sample['gt_crop'].copy()
    draw_boxes(gt, sample['gt_local_boxes'], color=(255, 180, 60))
    keep = gt_scores >= threshold
    draw_boxes(gt, gt_boxes[keep], gt_scores[keep])
    pred = None if crop is None else crop.copy()
    if pred is not None:
        local = full_boxes-np.tile(pred_rect[:2], 2)
        keep = scores >= threshold
        draw_boxes(pred, local[keep], scores[keep])
    titles = ['TARGET: blue GT / green prediction', 'BULLETS ON GT CROP', 'BULLETS ON PREDICTED CROP']
    for index, (title, view) in enumerate(zip(titles, [full, gt, pred])):
        x = index*640
        text(canvas, title, (x+12, 40), .64)
        if view is None:
            text(canvas, 'TARGET NOT DETECTED', (x+60, 400), .9, (80, 80, 255))
        else:
            ratio = min(620/view.shape[1], 780/view.shape[0])
            resized = cv2.resize(view, None, fx=ratio, fy=ratio)
            sx, sy = x+(640-resized.shape[1])//2, 65+(780-resized.shape[0])//2
            canvas[sy:sy+resized.shape[0], sx:sx+resized.shape[1]] = resized
    text(canvas, sample['sample']+' | blue boxes: GT; yellow boxes: predictions', (12, 883), .66)
    return canvas
