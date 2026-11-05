"""Checks for masking, paired coordinates, split routing and failure accounting."""
import argparse
import locale
import tempfile
import unittest
from pathlib import Path
import numpy as np
from src.geometry import crop_polygon
from src.metrics import aggregate_metrics
from src.runtime import read_config
from train import make_plan


class PipelineTests(unittest.TestCase):
    def test_polygon_exterior_is_exact_zero(self):
        image = np.full((100, 140, 3), 137, np.uint8)
        quad = np.float32([[40, 12], [110, 25], [95, 85], [20, 70]])
        crop, rect = crop_polygon(image, quad, 6)
        import cv2
        mask = np.zeros(crop.shape[:2], np.uint8)
        cv2.fillPoly(mask, [np.rint(quad-rect[:2]).astype(np.int32)], 1)
        self.assertTrue(np.all(crop[mask == 0] == 0))
        self.assertTrue(np.all(crop[mask == 1] == 137))

    def test_target_miss_counts_all_impacts_as_false_negatives(self):
        result = aggregate_metrics([{'sample': 'missing-paper', 'gt_boxes': [[10, 10, 20, 20], [30, 30, 40, 40]],
                                     'boxes': [], 'scores': []}], .45)
        self.assertEqual((result['tp'], result['fp'], result['fn']), (0, 0, 2))

    def test_empty_label_image_counts_false_positives(self):
        result = aggregate_metrics([{'sample': 'empty', 'gt_boxes': [], 'boxes': [[10, 10, 20, 20]], 'scores': [.9]}], .45)
        self.assertEqual(result['zero_impact_images'], 1)
        self.assertEqual(result['fp'], 1)

    def test_train_plan_uses_train_and_val_only(self):
        args = argparse.Namespace(task='all', epochs=1, workers=0, target_batch=2, bullet_batch=2)
        plan = make_plan(read_config(), args, Path('/tmp/plan-only'))
        self.assertEqual(set(plan), {'target', 'bullet'})
        for item in plan.values():
            self.assertEqual(item['training_split'], 'train')
            self.assertEqual(item['model_selection_split'], 'val')
            self.assertNotIn('test', item['options'])
            self.assertEqual(item['options']['epochs'], 1)

    def test_config_paths_are_relative_to_config_not_cwd(self):
        config = read_config()
        self.assertTrue(Path(config['target']['weights']).is_absolute())
        self.assertTrue(Path(config['datasets']['target']).is_dir())

    def test_dataset_yaml_is_utf8_even_after_locale_change(self):
        from src.datasets import write_data_yaml
        config = read_config()
        original = locale.setlocale(locale.LC_CTYPE)
        try:
            locale.setlocale(locale.LC_CTYPE, 'C')
            with tempfile.TemporaryDirectory() as directory:
                path = write_data_yaml(config, 'bullet', directory)
                self.assertIn(config['datasets']['bullet'], path.read_text(encoding='utf-8'))
        finally:
            locale.setlocale(locale.LC_CTYPE, original)


if __name__ == '__main__':
    unittest.main()
