"""Causal motion compensation and detection consensus; no labels or future frames."""
from collections import deque
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


class TargetTracker:
    def __init__(self, fps=30, measurement_alpha=.15, max_model_gap_s=3., freeze_s=.35, *, reuse_features=False):
        self.alpha = measurement_alpha
        self.max_gap = round(max_model_gap_s * fps)
        self.freeze = round(freeze_s * fps)
        self.gray = self.quad = None
        self.last_detection = self.last_motion = -100000
        self.generation = 0
        self.reuse_features = reuse_features
        self.feature_refresh_frames = max(1, round(.2 * fps))
        self.features = None
        self.last_feature_refresh = -100000

    @staticmethod
    def _detect_features(gray, quad):
        polygon = np.rint(quad * .5).astype(np.int32)
        x, y, width, height = cv2.boundingRect(polygon)
        # The border keeps corner responses identical to the full-frame masked search.
        left, top = max(0, x - 12), max(0, y - 12)
        right, bottom = min(gray.shape[1], x + width + 12), min(gray.shape[0], y + height + 12)
        if left >= right or top >= bottom:
            return None
        region = gray[top:bottom, left:right]
        mask = np.zeros(region.shape, np.uint8)
        cv2.fillConvexPoly(mask, polygon - [left, top], 255)
        points = cv2.goodFeaturesToTrack(region, 180, .015, 7, mask=mask, blockSize=5)
        return None if points is None else points + np.float32([left, top])

    @staticmethod
    def _distributed_features(points, quad):
        if points is None or len(points) < 24:
            return False
        coordinates = points.reshape(-1, 2)
        span = np.ptp(quad * .5, axis=0)
        covered = np.ptp(coordinates, axis=0)
        hull_area = cv2.contourArea(cv2.convexHull(coordinates))
        return bool((covered >= .35 * span).all() and hull_area >= .1 * cv2.contourArea(quad * .5))

    @staticmethod
    def scene_changed(gray, previous_gray):
        return gray is not None and previous_gray is not None and float(cv2.absdiff(gray, previous_gray).mean()) > 40.

    @staticmethod
    def _compatible_quads(measured, reference):
        discrepancy = np.linalg.norm(measured - reference, axis=1).mean()
        diameter = np.linalg.norm(reference[0] - reference[2])
        return discrepancy <= max(55., .09 * diameter)

    def reconnect(self, previous_gray, previous_quad, measured):
        """Project the last visible paper onto the recovered frame using its image features."""
        points = self._detect_features(previous_gray, previous_quad)
        candidate, _, info = self._propagate(self.gray, previous_quad, points, previous_gray=previous_gray,
                                              min_inlier_fraction=.55)
        if candidate is None or not self._compatible_quads(measured, candidate):
            return None
        if info['inlier_fraction'] <= .65:
            # Across missing frames, accept weaker flow only if the paper still looks similar.
            x, y, width, height = cv2.boundingRect(previous_quad * .5)
            paper = np.s_[y:y+height, x:x+width]
            if float(cv2.absdiff(self.gray[paper], previous_gray[paper]).mean()) >= 12.:
                return None
        return candidate

    def _propagate(self, gray, previous, points, *, previous_gray=None, min_inlier_fraction=.65):
        info = {'valid': False, 'inliers': 0}
        if points is None or len(points) < 10:
            return None, None, info
        previous_gray = self.gray if previous_gray is None else previous_gray
        nxt, status, _ = cv2.calcOpticalFlowPyrLK(previous_gray, gray, points, None, winSize=(21, 21), maxLevel=3)
        if nxt is None or status is None:
            return None, None, info
        back, reverse, _ = cv2.calcOpticalFlowPyrLK(gray, previous_gray, nxt, None, winSize=(21, 21), maxLevel=3)
        if back is None or reverse is None:
            return None, None, info
        good = status.ravel().astype(bool) & reverse.ravel().astype(bool)
        good &= np.isfinite(nxt).all(axis=(1, 2)) & np.isfinite(back).all(axis=(1, 2))
        good &= np.linalg.norm(points.reshape(-1, 2) - back.reshape(-1, 2), axis=1) < 1.
        if good.sum() < 10:
            return None, None, info
        affine, inliers = cv2.estimateAffinePartial2D(points[good], nxt[good], method=cv2.RANSAC, ransacReprojThreshold=1.5)
        if affine is None or inliers is None:
            return None, None, info
        scale = np.linalg.norm(affine[:, 0])
        candidate = cv2.transform((previous * .5)[None], affine)[0] * 2
        fraction = float(inliers.mean())
        jump = np.linalg.norm(candidate - previous, axis=1).max()
        if not (.96 < scale < 1.04 and fraction > min_inlier_fraction and jump < 80):
            return None, None, info
        tracked = nxt[good][inliers.ravel().astype(bool)]
        # Features leaving the paper cannot support a later target pose.
        polygon = candidate * .5
        inside = np.array([cv2.pointPolygonTest(polygon, tuple(map(float, point)), False) >= 0
                           for point in tracked.reshape(-1, 2)])
        tracked = tracked[inside]
        info = {'valid': True, 'inliers': int(inliers.sum()), 'inlier_fraction': fraction, 'max_motion_px': float(jump)}
        return candidate, tracked, info

    def update(self, image, measurement, frame):
        gray = cv2.cvtColor(cv2.resize(image, None, fx=.5, fy=.5), cv2.COLOR_BGR2GRAY)
        measured = None if measurement is None else np.float32(measurement).reshape(4, 2)
        previous = None if self.quad is None else self.quad.copy()
        propagated = None
        flow_info = {'valid': False, 'inliers': 0}
        tracked_features = None
        reused = refreshed = False
        if previous is not None and self.gray is not None:
            reused = (self.reuse_features and frame - self.last_feature_refresh < self.feature_refresh_frames
                      and self._distributed_features(self.features, previous))
            points = self.features if reused else self._detect_features(self.gray, previous)
            refreshed = not reused
            propagated, tracked_features, flow_info = self._propagate(gray, previous, points)
            if reused and (propagated is None or not self._distributed_features(tracked_features, propagated)):
                points = self._detect_features(self.gray, previous)
                propagated, tracked_features, flow_info = self._propagate(gray, previous, points)
                refreshed = True
            if refreshed:
                self.last_feature_refresh = frame
            if propagated is not None:
                self.last_motion = frame
        scene_cut = previous is not None and propagated is None and self.scene_changed(gray, self.gray)
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
                if not self._compatible_quads(measured, reference):
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
        if self.reuse_features:
            accepted_measurement = measured is not None and state != 'flow; measurement rejected'
            self.features = (tracked_features if not reset and not accepted_measurement
                             and self.quad is not None and self._distributed_features(tracked_features, self.quad)
                             else None)
            flow_info.update(features_reused=bool(reused), features_refreshed=bool(refreshed))
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
                                'center_plane': ((t['box'][:2]+t['box'][2:])/2).tolist(),
                                'box_plane': t['box'].tolist(),
                                'score_ema': t['score'], 'held': t['last_seen'] != frame,
                                'gap_frames': frame - t['last_seen']})
        return results
