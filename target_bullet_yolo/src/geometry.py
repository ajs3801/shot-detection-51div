"""Shared geometry operations."""
from pathlib import Path
import cv2
import numpy as np


def order_quad(points):
    p = np.float32(points).reshape(4, 2)
    p = p[np.argsort(np.arctan2(p[:, 1] - p[:, 1].mean(), p[:, 0] - p[:, 0].mean()))]
    p = np.roll(p, -np.argmin(p.sum(1)), axis=0)
    if cv2.contourArea(p, oriented=True) < 0:
        p = p[[0, 3, 2, 1]]
    return p

def fit_quad(contour, width, height):
    hull = cv2.convexHull(np.float32(contour).reshape(-1, 1, 2))
    perimeter = cv2.arcLength(hull, True)
    quad = None
    for fraction in np.linspace(0.005, 0.1, 100):
        candidate = cv2.approxPolyDP(hull, float(fraction * perimeter), True)
        if len(candidate) == 4:
            quad = candidate.reshape(4, 2)
            break
    method = 'convex_hull_approx_4'
    if quad is None:
        quad = cv2.boxPoints(cv2.minAreaRect(hull))
        method = 'minimum_area_rectangle_fallback'
    quad = order_quad(quad)
    quad[:, 0] = np.clip(quad[:, 0], 0, width - 1)
    quad[:, 1] = np.clip(quad[:, 1], 0, height - 1)
    if cv2.contourArea(quad) < 100:
        return (None, 'invalid_small_quad')
    return (quad, method)

def crop_polygon(image, polygon, padding=6):
    (h, w) = image.shape[:2]
    polygon = np.float32(polygon)
    x0 = max(0, int(np.floor(polygon[:, 0].min())) - padding)
    y0 = max(0, int(np.floor(polygon[:, 1].min())) - padding)
    x1 = min(w, int(np.ceil(polygon[:, 0].max())) + padding + 1)
    y1 = min(h, int(np.ceil(polygon[:, 1].max())) + padding + 1)
    if x1 <= x0 or y1 <= y0:
        raise ValueError('Invalid crop')
    crop = image[y0:y1, x0:x1].copy()
    mask = np.zeros(crop.shape[:2], np.uint8)
    cv2.fillPoly(mask, [np.rint(polygon - [x0, y0]).astype('int32')], 255)
    crop[mask == 0] = 0
    return (crop, (x0, y0, x1, y1))

def shift_boxes(boxes, origin):
    result = np.asarray(boxes, dtype=np.float32).reshape(-1, 4).copy()
    if len(result):
        result += np.array([origin[0], origin[1], origin[0], origin[1]], np.float32)
    return result

def read_yolo_boxes(label, image_shape):
    (h, w) = image_shape[:2]
    out = []
    for line in Path(label).read_text().splitlines():
        if not line.strip():
            continue
        (cls, x, y, bw, bh) = map(float, line.split())
        assert cls == 0
        out.append([(x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h])
    return np.array(out, np.float32).reshape(-1, 4)

def box_iou(gt, pred):
    gt = np.asarray(gt, np.float32).reshape(-1, 4)
    pred = np.asarray(pred, np.float32).reshape(-1, 4)
    lt = np.maximum(gt[:, None, :2], pred[None, :, :2])
    rb = np.minimum(gt[:, None, 2:], pred[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(2)
    ga = np.clip(gt[:, 2:] - gt[:, :2], 0, None).prod(1)
    pa = np.clip(pred[:, 2:] - pred[:, :2], 0, None).prod(1)
    return inter / (ga[:, None] + pa[None, :] - inter + 1e-09)

def polygon_metrics(pred, gt, width=1920, height=1080):
    if pred is None:
        return {'iou': 0.0, 'gt_area_recall': 0.0, 'precision_area': 0.0}
    a = np.zeros((height, width), np.uint8)
    b = a.copy()
    cv2.fillPoly(a, [np.rint(pred).astype('int32')], 1)
    cv2.fillPoly(b, [np.rint(gt).astype('int32')], 1)
    inter = int(np.count_nonzero(a & b))
    union = int(np.count_nonzero(a | b))
    return {'iou': inter / max(union, 1), 'gt_area_recall': inter / max(int(b.sum()), 1), 'precision_area': inter / max(int(a.sum()), 1)}
