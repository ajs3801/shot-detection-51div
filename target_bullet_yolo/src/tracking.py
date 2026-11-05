"""Causal motion compensation and detection consensus; no labels or future frames."""
from collections import deque
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


class TargetTracker:
    def __init__(self, fps=30, measurement_alpha=.15, max_model_gap_s=3., freeze_s=.35):
        self.alpha = measurement_alpha
        self.max_gap = round(max_model_gap_s * fps)
        self.freeze = round(freeze_s * fps)
        self.gray = self.quad = None
        self.last_detection = self.last_motion = -100000
        self.generation = 0

    def update(self, image, measurement, frame):
        gray = cv2.cvtColor(cv2.resize(image, None, fx=.5, fy=.5), cv2.COLOR_BGR2GRAY)
        measured = None if measurement is None else np.float32(measurement).reshape(4, 2)
        previous = None if self.quad is None else self.quad.copy()
        propagated = None
        flow_info = {'valid': False, 'inliers': 0}
        if previous is not None and self.gray is not None:
            mask = np.zeros_like(self.gray)
            cv2.fillConvexPoly(mask, np.rint(previous * .5).astype(np.int32), 255)
            points = cv2.goodFeaturesToTrack(self.gray, 180, .015, 7, mask=mask, blockSize=5)
            if points is not None and len(points) >= 10:
                nxt, status, _ = cv2.calcOpticalFlowPyrLK(self.gray, gray, points, None, winSize=(21, 21), maxLevel=3)
                back, reverse, _ = cv2.calcOpticalFlowPyrLK(gray, self.gray, nxt, None, winSize=(21, 21), maxLevel=3)
                good = status.ravel().astype(bool) & reverse.ravel().astype(bool)
                good &= np.linalg.norm(points.reshape(-1, 2) - back.reshape(-1, 2), axis=1) < 1.
                if good.sum() >= 10:
                    affine, inliers = cv2.estimateAffinePartial2D(points[good], nxt[good], method=cv2.RANSAC, ransacReprojThreshold=1.5)
                    if affine is not None:
                        scale = np.linalg.norm(affine[:, 0])
                        candidate = cv2.transform((previous * .5)[None], affine)[0] * 2
                        fraction = float(inliers.mean())
                        jump = np.linalg.norm(candidate - previous, axis=1).max()
                        if .96 < scale < 1.04 and fraction > .65 and jump < 80:
                            propagated = candidate
                            self.last_motion = frame
                            flow_info = {'valid': True, 'inliers': int(inliers.sum()), 'inlier_fraction': fraction, 'max_motion_px': float(jump)}
        scene_cut = (previous is not None and propagated is None and self.gray is not None
                     and float(cv2.absdiff(gray, self.gray).mean()) > 40.)
        reset = bool(scene_cut)
        if scene_cut:
            previous = self.quad = None
            self.last_motion = -100000
        if measured is not None:
            if previous is None:
                self.last_detection = frame
                self.generation += 1
                reset = True
                self.quad = measured.copy()
                self.last_motion = frame
                state = 'acquired'
            else:
                reference = propagated if propagated is not None else previous
                discrepancy = np.linalg.norm(measured - reference, axis=1).mean()
                diameter = np.linalg.norm(reference[0] - reference[2])
                if discrepancy > max(55., .09 * diameter):
                    # A large jump without reliable image correspondence starts a new paper.
                    if propagated is None:
                        self.last_detection = frame
                        self.generation += 1
                        reset = True
                        self.quad = measured.copy()
                        state = 'reacquired'
                    else:
                        self.quad = propagated
                        state = 'flow; measurement rejected'
                else:
                    self.last_detection = frame
                    self.quad = (1 - self.alpha) * reference + self.alpha * measured
                    state = 'detected + flow' if propagated is not None else 'detected + EMA'
        elif previous is not None and frame - self.last_detection <= self.max_gap:
            if propagated is not None:
                self.quad = propagated
                state = 'flow hold'
            elif frame - self.last_motion <= self.freeze:
                self.quad = previous
                state = 'short freeze'
            else:
                self.quad = None
                state = 'lost'
                reset = True
        else:
            self.quad = None
            state = 'waiting' if previous is None else 'expired'
            reset = reset or previous is not None
        if self.quad is not None:
            self.quad[:, 0] = np.clip(self.quad[:, 0], 0, image.shape[1] - 1)
            self.quad[:, 1] = np.clip(self.quad[:, 1], 0, image.shape[0] - 1)
        self.gray = gray
        return self.quad, {'state': state, 'reset': reset, 'generation': self.generation,
                           'model_gap_frames': frame - self.last_detection, 'flow': flow_info,
                           'scene_cut': bool(scene_cut)}


PLANE = np.float32([[0, 0], [1000, 0], [1000, 1400], [0, 1400]])


def map_boxes(boxes, homography):
    boxes = np.float32(boxes).reshape(-1, 4)
    if not len(boxes):
        return boxes.copy()
    corners = np.stack([boxes[:, [0, 1]], boxes[:, [2, 1]], boxes[:, [2, 3]], boxes[:, [0, 3]]], axis=1)
    transformed = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), homography).reshape(-1, 4, 2)
    return np.concatenate([transformed.min(1), transformed.max(1)], axis=1)


def iou_matrix(a, b):
    a = np.float32(a).reshape(-1, 4); b = np.float32(b).reshape(-1, 4)
    left = np.maximum(a[:, None, :2], b[None, :, :2])
    right = np.minimum(a[:, None, 2:], b[None, :, 2:])
    intersection = np.maximum(0, right - left).prod(2)
    area_a = np.maximum(0, a[:, 2:] - a[:, :2]).prod(1)
    area_b = np.maximum(0, b[:, 2:] - b[:, :2]).prod(1)
    return intersection / (area_a[:, None] + area_b[None, :] - intersection + 1e-9)


class ImpactTracker:
    def __init__(self, fps=30, high=.45, low=.20, min_hits=5, window=8, hold_s=1.5):
        self.high, self.low = high, low
        self.min_hits, self.window = min_hits, window
        self.hold_frames = round(fps * hold_s)
        self.tracks = []
        self.next_id = 1

    def reset(self):
        self.tracks = []

    def update(self, full_boxes, scores, quad, frame):
        to_plane = cv2.getPerspectiveTransform(np.float32(quad), PLANE)
        boxes = map_boxes(full_boxes, to_plane)
        scores = np.float32(scores)
        # Only boxes whose centers lie inside the paper can become impact tracks.
        centers = (boxes[:, :2] + boxes[:, 2:]) / 2
        keep = ((centers >= 0) & (centers <= [1000, 1400])).all(1) & (scores >= self.low)
        boxes, scores = boxes[keep], scores[keep]
        # Detector NMS (.7) sometimes leaves two boxes for one blurred hole.
        # Apply a stricter overlap check before temporal association; nearby,
        # distinct boxes with modest overlap remain independent.
        remaining = list(np.argsort(-scores))
        selected = []
        while remaining:
            index = remaining.pop(0)
            selected.append(index)
            if remaining:
                overlaps = iou_matrix(boxes[[index]], boxes[remaining])[0]
                remaining = [j for j, overlap in zip(remaining, overlaps) if overlap <= .60]
        boxes, scores = boxes[selected], scores[selected]
        self.tracks = [t for t in self.tracks if frame - t['last_seen'] <= (self.hold_frames if t['confirmed'] else self.window)]
        matches = []
        if self.tracks and len(boxes):
            previous = np.array([t['box'] for t in self.tracks])
            overlap = iou_matrix(previous, boxes)
            pc = (previous[:, :2] + previous[:, 2:]) / 2
            bc = (boxes[:, :2] + boxes[:, 2:]) / 2
            distance = np.linalg.norm(pc[:, None] - bc[None], axis=2)
            diameter = np.maximum(np.linalg.norm(previous[:, 2:] - previous[:, :2], axis=1)[:, None],
                                  np.linalg.norm(boxes[:, 2:] - boxes[:, :2], axis=1)[None])
            allowed = (overlap >= .12) | (distance < .42 * diameter)
            cost = 1 - overlap + .2 * distance / np.maximum(diameter, 1)
            cost[~allowed] = 1000
            rows, cols = linear_sum_assignment(cost)
            matches = [(int(i), int(j)) for i, j in zip(rows, cols) if cost[i, j] < 1000]
        used = set()
        for i, j in matches:
            t = self.tracks[i]
            # Weak measurements keep established tracks alive, but cannot confirm new ones.
            if not t['confirmed'] and scores[j] < self.high:
                continue
            t['box'] = .75 * t['box'] + .25 * boxes[j]
            t['score'] = .8 * t['score'] + .2 * float(scores[j])
            t['last_seen'] = frame
            if scores[j] >= self.high:
                t['hits'].append(frame)
            used.add(j)
        for j in range(len(boxes)):
            if j not in used and scores[j] >= self.high:
                self.tracks.append({'id': self.next_id, 'box': boxes[j].copy(), 'score': float(scores[j]),
                                    'hits': deque([frame]), 'last_seen': frame, 'confirmed': False})
                self.next_id += 1
        # Alternating box offsets can create parallel tracks even if individual
        # frames passed NMS. Merge only tracks that converge to near-identical
        # boxes; do not force the number of shots to grow monotonically.
        merged = []
        for track in sorted(self.tracks, key=lambda t: t['id']):
            duplicate = next((old for old in merged if
                              iou_matrix([old['box']], [track['box']])[0, 0] > .70), None)
            if duplicate is None:
                merged.append(track)
            else:
                duplicate['last_seen'] = max(duplicate['last_seen'], track['last_seen'])
                duplicate['confirmed'] |= track['confirmed']
                duplicate['hits'] = deque(sorted(set(duplicate['hits']) | set(track['hits'])))
                duplicate['box'] = (duplicate['box'] + track['box']) / 2
                duplicate['score'] = max(duplicate['score'], track['score'])
        self.tracks = merged
        results = []
        to_image = cv2.getPerspectiveTransform(PLANE, np.float32(quad))
        for t in self.tracks:
            while t['hits'] and t['hits'][0] < frame - self.window + 1:
                t['hits'].popleft()
            if len(t['hits']) >= self.min_hits:
                t['confirmed'] = True
            if t['confirmed']:
                results.append({'id': t['id'], 'box': map_boxes([t['box']], to_image)[0].tolist(),
                                'score_ema': t['score'], 'held': t['last_seen'] != frame,
                                'gap_frames': frame - t['last_seen']})
        return results
