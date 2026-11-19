"""Grid-based planar calibration and distances between accumulated impact centers."""
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from .tracking import PLANE, map_boxes


def transform_points(points, matrix):
    points = np.asarray(points, np.float32).reshape(-1, 1, 2)
    if not len(points):
        return np.empty((0, 2), np.float32)
    return cv2.perspectiveTransform(points, np.asarray(matrix, np.float64)).reshape(-1, 2)


def group_metrics(points, ids):
    points = np.asarray(points, np.float32).reshape(-1, 2)
    pairs = [{'ids': [ids[i], ids[j]], 'distance_cm': float(np.linalg.norm(points[i]-points[j]))}
             for i in range(len(points)) for j in range(i+1, len(points))]
    maximum = max(pairs, key=lambda pair: pair['distance_cm']) if pairs else None
    circle = None
    if len(points):
        center, _ = cv2.minEnclosingCircle(points)
        # Recompute radius: OpenCV deliberately adds a small numerical epsilon.
        radius = float(np.linalg.norm(points-np.float32(center), axis=1).max())
        circle = {'center_cm': list(center), 'radius_cm': radius, 'diameter_cm': 2*radius}
    return {'pairs': pairs, 'max_pair': maximum, 'circle': circle}


def _lattice(signal, start):
    residual = gaussian_filter1d(signal.astype(float), 10)-gaussian_filter1d(signal.astype(float), 1.2)
    correlation = np.correlate(residual, residual, 'full')[len(residual)-1:]
    candidates, _ = find_peaks(correlation[14:100], distance=8)
    if not len(candidates) or correlation[0] <= 0:
        return None
    candidates += 14
    period = float(candidates[np.argmax(correlation[candidates])])
    if correlation[int(period)]/correlation[0] < .20:
        return None
    peaks, _ = find_peaks(residual, prominence=max(1., np.std(residual)*.35), distance=period*.6)
    if len(peaks) < 8:
        return None
    best = max((np.abs(peaks-origin-np.rint((peaks-origin)/period)*period) < period*.17
                for origin in peaks), key=np.count_nonzero)
    positions = peaks[best].astype(float)
    if len(positions) < 8:
        return None
    origin = positions[0]
    indices = np.rint((positions-origin)/period)
    spacing, intercept = np.polyfit(indices, positions, 1)
    error = float(np.sqrt(np.mean((positions-indices*spacing-intercept)**2)))
    if not 14 <= spacing <= 100 or error > spacing*.10:
        return None
    return positions+start, float(spacing), float(intercept+start), error


def estimate_grid(plane_image, cell_cm=1.):
    """Find repeated grid lines in four clear strips; return plane -> cm homography.

    The strips avoid the central printed silhouette. Reject weak, irregular, or
    inconsistent grids rather than displaying an uncalibrated cm measurement.
    """
    gray = cv2.cvtColor(plane_image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    vertical, horizontal = [], []
    for fraction in [.20, .70]:
        y, x0, x1 = round(fraction*h), round(.12*w), round(.84*w)
        result = _lattice(np.median(gray[y-20:y+21, x0:x1], axis=0), x0)
        if result is None:
            return None
        vertical.append((*result, y))
    for fraction in [.20, .78]:
        x, y0, y1 = round(fraction*w), round(.14*h), round(.75*h)
        result = _lattice(np.median(gray[y0:y1, x-12:x+13], axis=1), y0)
        if result is None:
            return None
        horizontal.append((*result, x))
    for a, b in [vertical, horizontal]:
        if abs(a[1]/b[1]-1) > .08:
            return None

    def matched_lines(strips, is_vertical):
        a, b = strips
        positions, period, origin, _, fixed = a
        other, _, _, _, other_fixed = b
        lines, used = [], set()
        for position in positions:
            index = int(np.argmin(abs(other-position)))
            if index in used or abs(other[index]-position) > period*.43:
                continue
            used.add(index)
            lattice_index = int(round((position-origin)/period))
            p = [position, fixed, 1] if is_vertical else [fixed, position, 1]
            q = [other[index], other_fixed, 1] if is_vertical else [other_fixed, other[index], 1]
            lines.append((lattice_index, np.cross(p, q)))
        return lines

    vertical_lines, horizontal_lines = matched_lines(vertical, True), matched_lines(horizontal, False)
    if min(len(vertical_lines), len(horizontal_lines)) < 8:
        return None
    source, destination = [], []
    for ix, vx in vertical_lines:
        for iy, hy in horizontal_lines:
            point = np.cross(vx, hy)
            if abs(point[2]) < 1e-8:
                continue
            source.append(point[:2]/point[2])
            destination.append([ix*cell_cm, iy*cell_cm])
    matrix, inliers = cv2.findHomography(np.float32(source), np.float32(destination), cv2.RANSAC, .12*cell_cm)
    if matrix is None or float(inliers.mean()) < .85 or not np.isfinite(matrix).all():
        return None
    errors = np.linalg.norm(transform_points(source, matrix)-destination, axis=1)
    if float(np.median(errors)) > .07*cell_cm:
        return None
    # A stable arbitrary origin, independent of which grid lines were visible.
    anchor = transform_points([[w/2, h/2]], matrix)[0]
    matrix = np.array([[1, 0, -anchor[0]], [0, 1, -anchor[1]], [0, 0, 1.]]) @ matrix
    corners = transform_points(PLANE, matrix)
    if not np.isfinite(corners).all() or not cv2.isContourConvex(corners.astype(np.float32)):
        return None
    return matrix, {'cell_cm': cell_cm, 'vertical_lines': len(vertical_lines),
                    'horizontal_lines': len(horizontal_lines), 'inlier_fraction': float(inliers.mean()),
                    'median_grid_fit_error_cm': float(np.median(errors)),
                    'grid_period_plane_px': {'x': float(np.mean([s[1] for s in vertical])),
                                             'y': float(np.mean([s[1] for s in horizontal]))}}


class GridCalibrator:
    def __init__(self, fps=30, cell_cm=1.):
        if not np.isfinite(cell_cm) or cell_cm <= 0:
            raise ValueError('Grid cell size must be a finite positive number.')
        self.fps, self.cell_cm = fps, cell_cm
        self.reset()

    def reset(self):
        self.matrix = self.details = None
        self.last_attempt = self.last_success = -100000

    def update(self, image, quad, frame):
        interval = max(1, round(self.fps*(1. if self.matrix is not None else .2)))
        if frame-self.last_attempt < interval:
            return
        self.last_attempt = frame
        to_plane = cv2.getPerspectiveTransform(np.float32(quad), PLANE)
        plane = cv2.warpPerspective(image, to_plane, (1000, 1400))
        result = estimate_grid(plane, self.cell_cm)
        if result is None:
            return
        candidate, details = result
        if self.matrix is not None:
            previous = transform_points(PLANE, self.matrix)
            measured = transform_points(PLANE, candidate)
            if np.linalg.norm(previous-measured, axis=1).max() > 1.5*self.cell_cm:
                return
            candidate = cv2.getPerspectiveTransform(PLANE, np.float32(.85*previous+.15*measured))
        self.matrix, self.details, self.last_success = candidate, details, frame


class ImpactGroup:
    """Accumulate confirmed observations on one paper, including temporarily missing holes."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.shots = []
        self.next_id = 1

    def rebase(self, projected_quad, current_quad):
        """Keep the same impacts when reacquisition changes the paper's coordinate basis."""
        to_image = cv2.getPerspectiveTransform(PLANE, np.float32(projected_quad))
        to_plane = cv2.getPerspectiveTransform(np.float32(current_quad), PLANE)
        mapping = to_plane @ to_image
        points = transform_points([shot['point_plane'] for shot in self.shots], mapping)
        boxes = map_boxes([shot['box_plane'] for shot in self.shots], mapping)
        for shot, point, box in zip(self.shots, points, boxes):
            shot['point_plane'], shot['box_plane'] = point, box

    @staticmethod
    def _matching_distance(shot, point, plane_box):
        size = min(np.linalg.norm(shot['box_plane'][2:]-shot['box_plane'][:2]),
                   np.linalg.norm(plane_box[2:]-plane_box[:2]))
        distance = float(np.linalg.norm(point-shot['point_plane']))
        return distance if distance < max(3., .35*size) else None

    def has_new_candidate(self, boxes, quad):
        if not len(boxes):
            return False
        to_plane = cv2.getPerspectiveTransform(np.float32(quad), PLANE)
        for box in map_boxes(boxes, to_plane):
            center = (box[:2]+box[2:])/2
            if ((center >= PLANE[0]) & (center <= PLANE[2])).all():
                if all(self._matching_distance(shot, center, box) is None for shot in self.shots):
                    return True
        return False

    def projected_tracks(self, quad, frame):
        if not self.shots:
            return []
        to_image = cv2.getPerspectiveTransform(PLANE, np.float32(quad))
        points = [shot['point_plane'] for shot in self.shots]
        # Center each display box on the accumulated position, not its most recent observation.
        boxes = [shot['box_plane'] + np.tile(point - (shot['box_plane'][:2]+shot['box_plane'][2:])/2, 2)
                 for shot, point in zip(self.shots, points)]
        return [{'id': shot['id'], 'box': image_box.tolist(), 'box_plane': box.tolist(),
                 'center_plane': point.tolist(), 'score_ema': shot['score_ema'],
                 'held': shot['last_seen_frame'] != frame, 'gap_frames': frame-shot['last_seen_frame']}
                for shot, point, box, image_box in zip(self.shots, points, boxes, map_boxes(boxes, to_image))]

    def update(self, tracks, quad, frame):
        to_plane = cv2.getPerspectiveTransform(np.float32(quad), PLANE)
        used = set()
        for track in tracks:
            if track.get('held'):
                continue
            box = np.float32(track['box'])
            point = np.float32(track['center_plane']) if 'center_plane' in track else transform_points([(box[:2]+box[2:])/2], to_plane)[0]
            plane_box = np.float32(track['box_plane']) if 'box_plane' in track else np.r_[point-8, point+8]
            track_id = track.get('id')
            match = next((shot for shot in self.shots if track_id is not None and track_id in shot['track_ids']), None)
            if match is None:
                candidates = []
                for shot in self.shots:
                    if shot['id'] in used:
                        continue
                    distance = self._matching_distance(shot, point, plane_box)
                    if distance is not None:
                        candidates.append((distance, shot))
                if candidates:
                    match = min(candidates, key=lambda item: item[0])[1]
            if match is None:
                match = {'id': self.next_id, 'track_ids': set(), 'point_plane': point.copy(),
                         'box_plane': plane_box.copy(), 'first_seen_frame': frame, 'last_seen_frame': frame, 'observations': 0}
                self.next_id += 1
                self.shots.append(match)
            if track_id is not None:
                match['track_ids'].add(track_id)
            match['point_plane'] = .8*match['point_plane']+.2*point
            match['box_plane'] = plane_box.copy()
            match['score_ema'] = track.get('score_ema')
            match['last_seen_frame'] = frame
            match['observations'] += 1
            used.add(match['id'])

    def measurements(self, calibration, quad, frame):
        state = 'calibrated' if calibration.matrix is not None else 'waiting for grid'
        result = {'state': state if quad is not None else 'waiting for target',
                  'impact_count': len(self.shots), 'grid_cell_cm': calibration.cell_cm,
                  'measurement_basis': 'distances between detected impact centers',
                  'accumulated': True, 'points': []}
        if quad is None or calibration.matrix is None:
            return result
        plane_points = [shot['point_plane'] for shot in self.shots]
        metric = transform_points(plane_points, calibration.matrix)
        to_image = cv2.getPerspectiveTransform(PLANE, np.float32(quad))
        image_points = transform_points(plane_points, to_image)
        result['points'] = [{'id': shot['id'], 'xy_cm': cm.tolist(), 'xy_image': image.tolist(),
                             'held': shot['last_seen_frame'] != frame, 'last_seen_frame': shot['last_seen_frame']}
                            for shot, cm, image in zip(self.shots, metric, image_points)]
        result.update(group_metrics(metric, [shot['id'] for shot in self.shots]))
        if result['circle'] is not None:
            circle = result['circle']
            theta = np.linspace(0, 2*np.pi, 97)
            boundary = np.float32(circle['center_cm'])+circle['radius_cm']*np.column_stack([np.cos(theta), np.sin(theta)])
            result['circle']['boundary_image'] = transform_points(boundary, to_image @ np.linalg.inv(calibration.matrix)).tolist()
        result['calibration'] = {**calibration.details, 'last_success_frame': calibration.last_success,
                                 'age_s': (frame-calibration.last_success)/calibration.fps,
                                 'plane_to_cm': calibration.matrix.tolist()}
        return result
