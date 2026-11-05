"""Behavior checks for bounded holds, confirmation, association and camera motion."""
import unittest
import cv2
import numpy as np
from src.tracking import ImpactTracker, TargetTracker, PLANE


class TrackingTests(unittest.TestCase):
    def test_one_frame_false_detection_never_confirmed(self):
        tracker = ImpactTracker()
        for frame in range(15):
            boxes, scores = ([[100, 100, 125, 130]], [.9]) if frame == 0 else ([], [])
            self.assertEqual(tracker.update(boxes, scores, PLANE, frame), [])

    def test_two_close_impacts_and_temporary_gap(self):
        tracker = ImpactTracker(fps=30)
        boxes = [[100, 100, 125, 130], [120, 100, 145, 130]]
        for frame in range(5):
            tracks = tracker.update(boxes, [.9, .85], PLANE, frame)
        self.assertEqual(len(tracks), 2)
        ids = [t['id'] for t in tracks]
        tracks = tracker.update([], [], PLANE, 5)
        self.assertEqual([t['id'] for t in tracks], ids)
        self.assertTrue(all(t['held'] for t in tracks))
        tracks = tracker.update(boxes[::-1], [.9, .85], PLANE, 6)
        self.assertEqual([t['id'] for t in tracks], ids)
        self.assertTrue(all(not t['held'] for t in tracks))
        self.assertEqual(tracker.update([], [], PLANE, 52), [])

    def test_weak_detections_do_not_create_tracks(self):
        tracker = ImpactTracker()
        for frame in range(30):
            self.assertEqual(tracker.update([[100, 100, 125, 130]], [.3], PLANE, frame), [])

    def test_near_duplicate_boxes_do_not_multiply_tracks(self):
        tracker = ImpactTracker()
        for frame in range(30):
            # IOU .66: passes detector NMS .7 but describes the same blurred hole.
            tracks = tracker.update([[100, 100, 130, 140], [106, 100, 136, 140]], [.9, .55], PLANE, frame)
        self.assertEqual(len(tracks), 1)

    def test_motion_compensation_and_gap_timeout(self):
        rng = np.random.default_rng(22)
        texture = rng.integers(0, 256, (400, 600, 3), dtype=np.uint8)
        quad = np.float32([[100, 60], [480, 60], [480, 350], [100, 350]])
        tracker = TargetTracker(fps=30)
        first, state = tracker.update(texture, quad, 0)
        self.assertTrue(state['reset'])
        shifted = cv2.warpAffine(texture, np.float32([[1, 0, 4], [0, 1, 2]]), (600, 400))
        next_quad, state = tracker.update(shifted, None, 1)
        self.assertEqual(state['state'], 'flow hold')
        np.testing.assert_allclose(next_quad, quad + [4, 2], atol=.3)
        lost, state = tracker.update(shifted, None, 91)
        self.assertIsNone(lost)
        self.assertTrue(state['reset'])

    def test_reset_purges_confirmed_impacts(self):
        tracker = ImpactTracker()
        for frame in range(5):
            tracks = tracker.update([[100, 100, 125, 130]], [.9], PLANE, frame)
        self.assertEqual(len(tracks), 1)
        tracker.reset()
        self.assertEqual(tracker.update([], [], PLANE, 5), [])

    def test_abrupt_scene_change_resets_even_with_same_detected_quad(self):
        tracker = TargetTracker()
        quad = np.float32([[40, 40], [500, 40], [500, 350], [40, 350]])
        tracker.update(np.zeros((400, 600, 3), np.uint8), quad, 0)
        _, state = tracker.update(np.full((400, 600, 3), 255, np.uint8), quad, 1)
        self.assertTrue(state['reset'])
        self.assertTrue(state['scene_cut'])
        self.assertEqual(state['generation'], 2)


if __name__ == '__main__':
    unittest.main()
