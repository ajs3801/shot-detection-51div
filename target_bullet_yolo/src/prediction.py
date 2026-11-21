"""Shared prediction operations."""
import numpy as np
from .geometry import fit_quad


def predict_target(model, image, confidence=0.25, device=0, imgsz=1280, iou=.7, *, retina_masks=True):
    result = model.predict(
        image, device=device, imgsz=imgsz, conf=confidence, iou=iou,
        max_det=10, retina_masks=retina_masks, verbose=False,
    )[0]
    if result.boxes is None or result.masks is None or (not len(result.boxes)):
        return None
    index = int(result.boxes.conf.argmax().item())
    contour = np.float32(result.masks[index:index + 1].xy[0])
    (h, w) = image.shape[:2]
    if len(contour) < 3:
        return None
    (quad, method) = fit_quad(contour, w, h)
    if quad is None:
        return None
    return {'confidence': float(result.boxes.conf[index]), 'contour': contour, 'quad': quad, 'quad_method': method}

def predict_bullets(model, image, device=1, imgsz=1024, confidence=0.001, iou=.7):
    r = model.predict(image, device=device, imgsz=imgsz, conf=confidence, iou=iou, max_det=100, verbose=False)[0]
    if r.boxes is None or not len(r.boxes):
        return (np.empty((0, 4), np.float32), np.empty(0, np.float32))
    return (r.boxes.xyxy.cpu().numpy().astype(np.float32), r.boxes.conf.cpu().numpy().astype(np.float32))
