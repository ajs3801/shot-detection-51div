"""Metric geometry, grid calibration, persistence, and duplicate handling."""
import unittest
import cv2
import numpy as np
from src.grouping import GridCalibrator, ImpactGroup, estimate_grid, group_metrics, transform_points
from src.tracking import PLANE


class GroupingTests(unittest.TestCase):
    def test_three_four_five_triangle_and_circle(self):
        result = group_metrics([[0, 0], [3, 0], [0, 4]], [1, 2, 3])
        self.assertEqual([pair['distance_cm'] for pair in result['pairs']], [3., 4., 5.])
        self.assertAlmostEqual(result['circle']['diameter_cm'], 5., places=5)

    def test_circle_diameter_and_max_pair_are_distinct(self):
        result = group_metrics([[0, 0], [2, 0], [1, np.sqrt(3)]], [1, 2, 3])
        self.assertAlmostEqual(result['max_pair']['distance_cm'], 2., places=5)
        self.assertAlmostEqual(result['circle']['diameter_cm'], 4/np.sqrt(3), places=5)

    def test_zero_and_one_impact(self):
        self.assertIsNone(group_metrics([], [])['circle'])
        result = group_metrics([[7, 11]], [1])
        self.assertEqual(result['pairs'], [])
        self.assertEqual(result['circle']['diameter_cm'], 0.)

    def test_grid_calibration_with_perspective(self):
        image = np.full((1400, 1000, 3), 220, np.uint8)
        for x in range(80, 900, 36):
            cv2.line(image, (x, 100), (x, 1120), (95, 95, 95), 3)
        for y in range(100, 1120, 39):
            cv2.line(image, (80, y), (900, y), (95, 95, 95), 3)
        distortion = cv2.getPerspectiveTransform(PLANE, np.float32([[5, 15], [985, 0], [999, 1385], [15, 1395]]))
        warped = cv2.warpPerspective(image, distortion, (1000, 1400), borderValue=(220, 220, 220))
        estimate = estimate_grid(warped)
        self.assertIsNotNone(estimate)
        matrix, details = estimate
        points = transform_points([[200, 300], [200+3*36, 300+4*39]], distortion)
        metric = transform_points(points, matrix)
        self.assertAlmostEqual(float(np.linalg.norm(metric[0]-metric[1])), 5., delta=.10)
        self.assertGreaterEqual(details['inlier_fraction'], .85)
        doubled, _ = estimate_grid(warped, cell_cm=2.)
        metric = transform_points(points, doubled)
        self.assertAlmostEqual(float(np.linalg.norm(metric[0]-metric[1])), 10., delta=.20)

    def test_no_grid_does_not_report_cm(self):
        self.assertIsNone(estimate_grid(np.full((1400, 1000, 3), 180, np.uint8)))
        result = ImpactGroup().measurements(GridCalibrator(), PLANE, 0)
        self.assertEqual(result['state'], 'waiting for grid')
        self.assertNotIn('pairs', result)

    def test_accumulate_missing_and_reacquired_tracks_without_duplicates(self):
        group = ImpactGroup()
        def track(identifier, x, y):
            return {'id': identifier, 'box': [x-10, y-10, x+10, y+10],
                    'box_plane': [x-10, y-10, x+10, y+10], 'center_plane': [x, y], 'held': False}
        group.update([track(1, 100, 200), track(2, 125, 200)], PLANE, 0)
        group.update([], PLANE, 100)
        self.assertEqual(len(group.shots), 2)
        group.update([track(9, 102, 201), track(10, 126, 200)], PLANE, 101)
        self.assertEqual(len(group.shots), 2)
        calibration = GridCalibrator()
        calibration.matrix = np.diag([.1, .1, 1.])
        calibration.details = {}
        calibration.last_success = 101
        result = group.measurements(calibration, PLANE, 101)
        self.assertAlmostEqual(result['max_pair']['distance_cm'], 2.48, delta=.03)
        group.reset()
        self.assertEqual(group.measurements(calibration, PLANE, 102)['impact_count'], 0)

    def test_invalid_grid_size_rejected(self):
        for value in [0, -1, float('nan'), float('inf')]:
            with self.assertRaises(ValueError):
                GridCalibrator(cell_cm=value)


if __name__ == '__main__':
    unittest.main()
