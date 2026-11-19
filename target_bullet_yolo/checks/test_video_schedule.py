"""Checks for source-frame output and independent adaptive model schedules."""
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

os.environ.setdefault('YOLO_OFFLINE', 'true')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('ULTRALYTICS_HUB_NO_CHECKS', '1')

import cv2
import numpy as np

from src import video
from src.runtime import read_config
import run


class AdaptivePresetTests(unittest.TestCase):
    def test_presets_apply_the_documented_values(self):
        cases = {
            'budget27': ((30, .5, 3, 4), (1280, 896), (3, 5)),
            'adaptive30': ((30, 2, 5, 8), (1280, 1024), (5, 8)),
        }
        for preset, (schedule, image_sizes, confirmation) in cases.items():
            with self.subTest(preset=preset), patch.object(run, 'configure'), patch.object(video, 'run_video') as run_video:
                args = run.parser().parse_args(['--source', 'sample.mp4', '--device', 'cpu', '--preset', preset])
                run.main(args)
                config = run_video.call_args.args[2]
                self.assertTrue(run_video.call_args.kwargs['adaptive_inference'])
                self.assertEqual(tuple(config['inference'][key] for key in
                                       ('analysis_fps', 'target_hz', 'bullet_hz', 'bullet_burst_frames')), schedule)
                self.assertEqual((config['target']['imgsz'], config['bullet']['imgsz']), image_sizes)
                self.assertEqual((config['tracking']['confirmation_hits'],
                                  config['tracking']['confirmation_window']), confirmation)

    def test_explicit_flags_override_preset_values(self):
        with patch.object(run, 'configure'), patch.object(video, 'run_video') as run_video:
            args = run.parser().parse_args(['--source', 'sample.mp4', '--device', 'cpu', '--preset', 'budget27',
                                          '--bullet-hz', '2', '--bullet-imgsz', '1024', '--confirmation-hits', '4'])
            run.main(args)
            config = run_video.call_args.args[2]
            self.assertEqual(config['inference']['bullet_hz'], 2)
            self.assertEqual(config['bullet']['imgsz'], 1024)
            self.assertEqual(config['tracking']['confirmation_hits'], 4)


class AdaptiveVideoTests(unittest.TestCase):
    def setUp(self):
        self.config = read_config()
        self.config['tracking']['enabled'] = True
        self.config['grouping'] = {'enabled': False}
        self.image = np.random.default_rng(11).integers(0, 256, (240, 320, 3), dtype=np.uint8)
        self.quad = np.float32([[30, 30], [290, 30], [290, 210], [30, 210]])
        for attr in ('require_weights', 'YOLO'):
            patcher = patch.object(video, attr)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_preserves_source_output_and_runs_only_the_candidate_burst(self):
        source_frames = 72
        source_index = -1
        target_frames, bullet_frames = [], []
        shifted_image = cv2.warpAffine(self.image, np.float32([[1, 0, 4], [0, 1, 2]]), (320, 240))

        def read():
            nonlocal source_index
            source_index += 1
            if source_index >= source_frames:
                return False, None
            return True, self.image if source_index < 12 else shifted_image

        def predict_target(*args, **kwargs):
            target_frames.append(source_index)
            quad = self.quad if source_index < 12 else self.quad + np.float32([4, 2])
            return {'quad': quad, 'confidence': .9}

        def predict_bullets(*args, **kwargs):
            bullet_frames.append(source_index)
            if source_index == 0 or 24 <= source_index < 66:
                return np.empty((0, 4), np.float32), np.empty(0, np.float32)
            score = .25 if source_index == 6 else .9
            return np.float32([[110, 100, 127, 116]]), np.float32([score])

        capture = Mock(isOpened=Mock(return_value=True), read=Mock(side_effect=read))
        capture.get.side_effect = {cv2.CAP_PROP_FRAME_WIDTH: 320, cv2.CAP_PROP_FRAME_HEIGHT: 240,
                                   cv2.CAP_PROP_FPS: 30, cv2.CAP_PROP_FRAME_COUNT: source_frames}.__getitem__
        encoder = Mock(wait=Mock(return_value=0))

        def open_encoder(command, **kwargs):
            self.assertEqual(float(command[command.index('-framerate') + 1]), 30.)
            Path(command[-1]).touch()
            return encoder

        with tempfile.TemporaryDirectory(dir=Path.cwd()) as folder, \
                patch.object(video.cv2, 'VideoCapture', return_value=capture), \
                patch.object(video.cv2, 'imwrite'), \
                patch.object(video, 'find_ffmpeg', return_value='ffmpeg'), \
                patch.object(video, 'video_frame', side_effect=lambda image, *_: image) as render, \
                patch.object(video.subprocess, 'Popen', side_effect=open_encoder), \
                patch.object(video, 'predict_target', side_effect=predict_target), \
                patch.object(video, 'predict_bullets', side_effect=predict_bullets), \
                patch.object(video.ImpactTracker, 'update', autospec=True,
                             side_effect=video.ImpactTracker.update) as update_impacts:
            source = Path(folder) / 'source.mp4'
            source.touch()
            output = Path(folder) / 'out.mp4'
            summary = video.run_video(source, output, self.config, audio=False, adaptive_inference=True)
            predictions = output.with_suffix('') / 'predictions.jsonl'
            rows = [json.loads(row) for row in predictions.read_text(encoding='utf-8').splitlines()]
            self.assertTrue(output.is_file())

        self.assertEqual((summary['frames'], summary['source_frames'], summary['fps']),
                         (source_frames, source_frames, 30))
        self.assertEqual([row['source_frame'] for row in rows], list(range(source_frames)))
        np.testing.assert_allclose([row['source_time_s'] for row in rows], np.arange(source_frames) / 30)
        self.assertEqual((render.call_count, encoder.stdin.write.call_count), (source_frames, source_frames))
        capture.grab.assert_not_called()
        self.assertEqual(target_frames, list(range(0, source_frames, 15)))
        self.assertEqual(bullet_frames, [0, *range(6, 15), *range(18, source_frames, 6)])
        self.assertEqual([call.args[-1] for call in update_impacts.call_args_list], bullet_frames)
        self.assertEqual({key: summary['inference_calls'][key] for key in ('target', 'bullet', 'bullet_bursts')},
                         {'target': 5, 'bullet': 19, 'bullet_bursts': 1})
        self.assertEqual([row['frame'] for row in rows if row['bullet_burst_started']], [6])
        self.assertEqual([row['frame'] for row in rows if row['bullet_inference_ran']], bullet_frames)
        self.assertEqual([row['frame'] for row in rows if row['target_inference_ran']], target_frames)
        for model, model_frames in (('target', target_frames), ('bullet', bullet_frames)):
            indexes = {frame: index for index, frame in enumerate(model_frames)}
            self.assertEqual([row[f'{model}_inference_index'] for row in rows],
                             [indexes.get(frame) for frame in range(source_frames)])
        self.assertEqual(rows[0]['raw_impact_count'], 0)
        self.assertIsNone(rows[1]['raw_impact_count'])
        self.assertEqual(rows[10]['impact_count'], 0)
        self.assertEqual(rows[11]['impact_count'], 1)
        self.assertFalse(rows[11]['impact_tracks'][0]['held'])
        self.assertIsNone(rows[16]['raw_impact_count'])
        self.assertTrue(rows[16]['impact_tracks'][0]['held'])
        self.assertEqual(rows[16]['impact_tracks'][0]['gap_frames'], 2)
        self.assertTrue(all(row['impact_count'] == 1 for row in rows[11:]))
        self.assertEqual(rows[-1]['impact_tracks'][0]['id'], rows[11]['impact_tracks'][0]['id'])
        np.testing.assert_allclose(rows[12]['target_polygon_TL_TR_BR_BL'], self.quad + [4, 2], atol=.4)
        before, after = (np.array(rows[frame]['impact_tracks'][0]['box']) for frame in (11, 16))
        np.testing.assert_allclose(after - before, [4, 2, 4, 2], atol=1.5)

    def test_five_hz_samples_nearest_frame_at_twenty_four_fps(self):
        with patch.object(video, 'predict_target', return_value={'quad': self.quad, 'confidence': .9}), \
                patch.object(video, 'predict_bullets', return_value=(np.empty((0, 4)), np.empty(0))):
            predictor = video.VideoPredictor(self.config, 24, adaptive_inference=True)
            records = [predictor.predict(self.image, frame)[0] for frame in range(48)]

        self.assertEqual([row['frame'] for row in records if row['target_inference_ran']], [0, 12, 24, 36])
        self.assertEqual([row['frame'] for row in records if row['bullet_inference_ran']],
                         [0, 5, 10, 14, 19, 24, 29, 34, 38, 43])

    def test_sixty_fps_output_caps_analysis_and_prioritizes_burst(self):
        source_frames = 60
        source_index = -1
        target_frames, bullet_frames = [], []

        def read():
            nonlocal source_index
            source_index += 1
            return (True, self.image) if source_index < source_frames else (False, None)

        def predict_target(*args, **kwargs):
            target_frames.append(source_index)
            return {'quad': self.quad, 'confidence': .9}

        def predict_bullets(*args, **kwargs):
            bullet_frames.append(source_index)
            if source_index < 24:
                return np.empty((0, 4), np.float32), np.empty(0, np.float32)
            score = .25 if source_index == 24 else .9
            return np.float32([[110, 100, 127, 116]]), np.float32([score])

        capture = Mock(isOpened=Mock(return_value=True), read=Mock(side_effect=read))
        capture.get.side_effect = {cv2.CAP_PROP_FRAME_WIDTH: 320, cv2.CAP_PROP_FRAME_HEIGHT: 240,
                                   cv2.CAP_PROP_FPS: 60, cv2.CAP_PROP_FRAME_COUNT: source_frames}.__getitem__
        encoder = Mock(wait=Mock(return_value=0))

        def open_encoder(command, **kwargs):
            self.assertEqual(float(command[command.index('-framerate') + 1]), 60.)
            Path(command[-1]).touch()
            return encoder

        with tempfile.TemporaryDirectory(dir=Path.cwd()) as folder, \
                patch.object(video.cv2, 'VideoCapture', return_value=capture), \
                patch.object(video.cv2, 'imwrite'), \
                patch.object(video, 'find_ffmpeg', return_value='ffmpeg'), \
                patch.object(video, 'video_frame', side_effect=lambda image, *_: image) as render, \
                patch.object(video.subprocess, 'Popen', side_effect=open_encoder), \
                patch.object(video, 'predict_target', side_effect=predict_target), \
                patch.object(video, 'predict_bullets', side_effect=predict_bullets), \
                patch.object(video.VideoPredictor, 'predict', autospec=True,
                             side_effect=video.VideoPredictor.predict) as predict:
            source = Path(folder) / 'source.mp4'
            source.touch()
            output = Path(folder) / 'out.mp4'
            summary = video.run_video(source, output, self.config, audio=False, adaptive_inference=True)
            predictions = output.with_suffix('') / 'predictions.jsonl'
            rows = [json.loads(row) for row in predictions.read_text(encoding='utf-8').splitlines()]

        self.assertEqual((summary['frames'], summary['source_frames'], summary['fps']), (60, 60, 60))
        self.assertEqual((render.call_count, encoder.stdin.write.call_count), (60, 60))
        self.assertEqual([call.args[2] for call in predict.call_args_list], list(range(30)))
        self.assertTrue(all(call.args[0].fps == 30 for call in predict.call_args_list))
        self.assertEqual([row['frame'] for row in rows], list(range(source_frames)))
        self.assertEqual([row['source_frame'] for row in rows], list(range(source_frames)))
        np.testing.assert_allclose([row['time_s'] for row in rows], np.arange(source_frames)/60)
        np.testing.assert_allclose([row['source_time_s'] for row in rows], np.arange(source_frames)/60)
        self.assertEqual([row['analysis_frame'] for row in rows],
                         [frame//2 if frame % 2 == 0 else None for frame in range(source_frames)])
        self.assertEqual([row['analysis_ran'] for row in rows], [frame % 2 == 0 for frame in range(source_frames)])
        self.assertTrue(all(row['analysis_fps'] == 30 for row in rows))
        self.assertEqual(target_frames, [0, 42])
        self.assertEqual(bullet_frames, [0, 12, *range(24, 42, 2), 48])
        for model, model_frames in (('target', target_frames), ('bullet', bullet_frames)):
            indexes = {frame: index for index, frame in enumerate(model_frames)}
            self.assertEqual([row[f'{model}_inference_index'] for row in rows],
                             [indexes.get(frame) for frame in range(source_frames)])
        self.assertEqual([row['source_frame'] for row in rows if row['bullet_burst_started']], [24])
        self.assertEqual(rows[33]['impact_count'], 0)
        self.assertEqual(rows[34]['impact_count'], 1)
        self.assertTrue(rows[35]['impact_tracks'][0]['held'])
        self.assertEqual(rows[34]['impact_tracks'][0]['id'], rows[35]['impact_tracks'][0]['id'])
        self.assertEqual(rows[34]['impact_tracks'][0]['box'], rows[35]['impact_tracks'][0]['box'])
        for row in rows[1::2]:
            self.assertFalse(row['target_inference_ran'])
            self.assertFalse(row['bullet_inference_ran'])
            self.assertFalse(row['bullet_burst_started'])
            self.assertIsNone(row['raw_target_detected'])
            self.assertIsNone(row['raw_target_confidence'])
            self.assertIsNone(row['raw_target_polygon'])
            self.assertIsNone(row['raw_impact_count'])
            self.assertEqual(row['raw_impact_boxes_original_xyxy'], [])
            self.assertEqual(row['raw_impact_scores'], [])
        capture.grab.assert_not_called()

    def test_target_outage_keeps_count_until_the_paper_identity_changes(self):
        self.config['tracking']['max_model_gap_s'] = .2
        self.config['inference'].update(target_hz=30, bullet_hz=30, bullet_burst_frames=1)
        shift = np.float32([4, 2])
        shifted_image = cv2.warpAffine(self.image, np.float32([[1, 0, 4], [0, 1, 2]]), (320, 240))
        different_image = np.full_like(self.image, 255)
        different_quad = np.float32([[125, 8], [315, 8], [315, 230], [125, 230]])

        for same_paper in (True, False):
            with self.subTest(same_paper=same_paper):
                frame = -1

                def predict_target(*args, **kwargs):
                    if frame == 0:
                        return {'quad': self.quad, 'confidence': .9}
                    if frame < 12:
                        return None
                    quad = self.quad + shift if same_paper else different_quad
                    return {'quad': quad, 'confidence': .9}

                def predict_bullets(*args, **kwargs):
                    if not same_paper and frame >= 12:
                        return np.empty((0, 4), np.float32), np.empty(0, np.float32)
                    boxes = [[110, 100, 127, 116]]
                    if frame >= 18:
                        boxes.append([180, 100, 197, 116])
                    return np.float32(boxes), np.full(len(boxes), .9, np.float32)

                with patch.object(video, 'predict_target', side_effect=predict_target), \
                        patch.object(video, 'predict_bullets', side_effect=predict_bullets):
                    predictor = video.VideoPredictor(self.config, 30, adaptive_inference=True)
                    records = []
                    results = []
                    for frame in range(24):
                        image = self.image if frame < 12 else (shifted_image if same_paper else different_image)
                        result = predictor.predict(image, frame)
                        records.append(result[0])
                        results.append(result)

                self.assertEqual(records[4]['impact_count'], 1)
                original_id = records[4]['impact_tracks'][0]['id']
                self.assertEqual(records[7]['target_tracking']['state'], 'expired')
                self.assertTrue(records[7]['target_tracking']['reset'])
                for record, result in zip(records[7:12], results[7:12]):
                    self.assertFalse(record['target_available'])
                    self.assertIsNone(record['target_polygon_TL_TR_BR_BL'])
                    self.assertEqual(record['impact_count'], 1)
                    self.assertTrue(record['impact_count_held'])
                    self.assertEqual(record['impact_tracks'], [])
                    self.assertIsNone(result[1])
                    self.assertIsNone(result[2])
                    self.assertIsNone(result[3])
                    self.assertEqual(result[4], [])
                    self.assertFalse(record['bullet_inference_ran'])
                self.assertTrue(records[12]['target_tracking']['reset'])
                if same_paper:
                    self.assertTrue(records[12]['target_available'])
                    self.assertFalse(records[12]['impact_count_held'])
                    self.assertTrue(all(record['impact_count'] == 1 for record in records[12:22]))
                    self.assertEqual(records[16]['impact_tracks'][0]['id'], original_id)
                    self.assertEqual(records[22]['impact_count'], 2)
                    self.assertEqual({track['id'] for track in records[22]['impact_tracks']}, {original_id, original_id + 1})
                    np.testing.assert_allclose(records[12]['target_polygon_TL_TR_BR_BL'], self.quad + shift, atol=.4)
                else:
                    self.assertTrue(all(record['impact_count'] == 0 for record in records[12:]))
                    self.assertFalse(records[12]['impact_count_held'])
                    self.assertEqual(records[12]['impact_tracks'], [])


if __name__ == '__main__':
    unittest.main()
