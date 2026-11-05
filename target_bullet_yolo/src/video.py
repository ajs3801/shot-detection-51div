"""Fresh two-stage video inference and H.264/AAC-compatible audio muxing."""
import json
import os
import shutil
import subprocess
import time
from collections import Counter
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
from .geometry import crop_polygon, shift_boxes
from .prediction import predict_target, predict_bullets
from .runtime import ROOT, save_json, require_weights
from .tracking import TargetTracker, ImpactTracker
from .visualization import video_frame, text


def find_ffmpeg():
    candidates = []
    if os.environ.get('IMAGEIO_FFMPEG_EXE'):
        candidates.append(os.environ['IMAGEIO_FFMPEG_EXE'])
    if (ROOT / 'bin/ffmpeg').is_file():
        candidates.append(str(ROOT / 'bin/ffmpeg'))
    try:
        import imageio_ffmpeg
        candidates.append(imageio_ffmpeg.get_ffmpeg_exe())
    except ImportError:
        pass
    if shutil.which('ffmpeg'):
        candidates.append(shutil.which('ffmpeg'))
    for binary in dict.fromkeys(candidates):
        try:
            result = subprocess.run([binary, '-hide_banner', '-encoders'], capture_output=True, text=True, check=True)
            if 'libx264' in result.stdout:
                return binary
        except (OSError, subprocess.CalledProcessError):
            continue
    raise RuntimeError('No FFmpeg with libx264. Install imageio-ffmpeg or set IMAGEIO_FFMPEG_EXE.')


class VideoPredictor:
    def __init__(self, config, fps):
        require_weights(config)
        self.config, self.fps = config, fps
        self.target = YOLO(config['target']['weights'])
        self.bullet = YOLO(config['bullet']['weights'])
        settings = config['tracking']
        self.target_tracker = TargetTracker(fps, settings['target_measurement_alpha'], settings['max_model_gap_s'], settings['freeze_s'])
        self.impacts = ImpactTracker(fps, config['bullet']['confidence'], min(config['inference']['bullet_low_confidence'], config['bullet']['confidence']),
                                     settings['confirmation_hits'], settings['confirmation_window'], settings['impact_hold_s'])

    def predict(self, image, frame):
        tcfg, bcfg, icfg = self.config['target'], self.config['bullet'], self.config['inference']
        temporal = self.config['tracking']['enabled']
        start = time.perf_counter()
        target = predict_target(self.target, image, tcfg['confidence'], tcfg['device'], tcfg['imgsz'], icfg['iou'])
        after_target = time.perf_counter()
        measurement = target['quad'] if target is not None else None
        if temporal:
            quad, status = self.target_tracker.update(image, measurement, frame)
            if status['reset'] or quad is None:
                self.impacts.reset()
        else:
            quad = measurement
            status = {'state': 'detected' if quad is not None else 'not detected', 'reset': False, 'model_gap_frames': 0}
        after_flow = time.perf_counter()
        crop = rect = None
        boxes = np.empty((0, 4), np.float32)
        scores = np.empty(0, np.float32)
        if quad is not None:
            crop, rect = crop_polygon(image, quad, icfg['padding'])
            floor = min(icfg['bullet_low_confidence'], bcfg['confidence']) if temporal else bcfg['confidence']
            local, scores = predict_bullets(self.bullet, crop, bcfg['device'], bcfg['imgsz'], floor, icfg['iou'])
            boxes = shift_boxes(local, rect[:2])
        after_bullet = time.perf_counter()
        tracks = []
        if quad is not None:
            if temporal:
                tracks = self.impacts.update(boxes, scores, quad, frame)
            else:
                tracks = [{'id': None, 'box': box.tolist(), 'score_ema': float(score), 'held': False, 'gap_frames': 0}
                          for box, score in zip(boxes, scores) if score >= bcfg['confidence']]
        end = time.perf_counter()
        record = {'frame': frame, 'time_s': frame/self.fps, 'fps': self.fps,
                  'raw_target_detected': target is not None,
                  'raw_target_confidence': target['confidence'] if target is not None else None,
                  'raw_target_polygon': None if target is None else target['quad'].tolist(),
                  'target_available': quad is not None, 'target_tracking': status,
                  'target_polygon_TL_TR_BR_BL': None if quad is None else quad.tolist(),
                  'crop_xyxy': rect, 'raw_impact_count': int((scores >= bcfg['confidence']).sum()) if quad is not None else None,
                  'raw_impact_boxes_original_xyxy': boxes.tolist(), 'raw_impact_scores': scores.tolist(),
                  'impact_count': len(tracks) if quad is not None else None, 'impact_tracks': tracks,
                  'timing_ms': {'target': (after_target-start)*1000, 'flow': (after_flow-after_target)*1000,
                                'crop_bullet': (after_bullet-after_flow)*1000, 'impact_tracking': (end-after_bullet)*1000,
                                'total': (end-start)*1000}}
        return record, quad, crop, rect, tracks


def run_video(source, output, config, max_frames=None, audio=True):
    source, output = Path(source).expanduser().resolve(), Path(output).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if output == source:
        raise ValueError('Input and output paths must differ.')
    if output.suffix.lower() != '.mp4':
        raise ValueError('Output must end with .mp4')
    if output.exists():
        raise FileExistsError(f'Choose a new output file: {output}')
    artifacts = output.with_suffix('')
    if artifacts.exists() and any(artifacts.iterdir()):
        raise FileExistsError(f'Choose a new output name; artifact directory already contains files: {artifacts}')
    if max_frames is not None and max_frames <= 0:
        raise ValueError('--max-frames must be positive')
    capture = cv2.VideoCapture(str(source), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1])
    if not capture.isOpened():
        raise ValueError(f'Cannot open video: {source}')
    width, height = int(capture.get(3)), int(capture.get(4))
    fps = capture.get(cv2.CAP_PROP_FPS)
    expected = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if not np.isfinite(fps) or fps <= 0 or min(width, height) < 64:
        capture.release()
        raise ValueError('Video must have a valid FPS and dimensions >= 64 pixels.')
    limit = min(expected, max_frames) if expected and max_frames else max_frames or expected
    output.parent.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(parents=True, exist_ok=True)
    save_json(artifacts / 'config.json', config)
    encoder = None
    log = None
    started = time.perf_counter()
    n = 0
    raw_found = available = 0
    states, histogram = Counter(), Counter()
    time_sums, confirmed_ids = Counter(), set()
    count_changes = count_drops = valid_pairs = 0
    previous = None
    preview_indices = {0}
    if limit:
        preview_indices.update(round((limit-1)*fraction) for fraction in [.2, .4, .6, .8, .98])
    previews = []

    def progress(status, **extra):
        save_json(artifacts / 'progress.json', {'status': status, 'source': str(source), 'output': str(output),
                  'frames_processed': n, 'expected_frames': limit or None,
                  'elapsed_s': time.perf_counter()-started, **extra})

    try:
        progress('initializing')
        predictor = VideoPredictor(config, fps)
        binary = find_ffmpeg()
        temporary = output.with_name(output.stem + '.partial.mp4')
        command = [binary, '-hide_banner', '-loglevel', 'warning', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
                   '-video_size', f'{width}x{height}', '-framerate', f'{fps:.9f}', '-i', 'pipe:0']
        if audio:
            command += ['-i', str(source), '-map', '0:v:0', '-map', '1:a?', '-c:a', 'copy']
        else:
            command += ['-map', '0:v:0', '-an']
        if max_frames is not None:
            command += ['-t', str(limit/fps)]
        # Odd-sized videos get one-pixel encoder padding, without changing inference.
        if width % 2 or height % 2:
            command += ['-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2']
        command += ['-c:v', 'libx264', '-threads', '2', '-preset', 'fast', '-crf', '20',
                    '-pix_fmt', 'yuv420p', '-movflags', '+faststart', '-y', str(temporary)]
        log = (artifacts / 'ffmpeg.log').open('w')
        encoder = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=log)
        with (artifacts / 'predictions.jsonl').open('w') as predictions:
            while max_frames is None or n < max_frames:
                ok, image = capture.read()
                if not ok:
                    break
                row, quad, crop, rect, tracks = predictor.predict(image, n)
                annotated = video_frame(image, quad, crop, rect, tracks, row, config, limit)
                predictions.write(json.dumps(row) + '\n')
                encoder.stdin.write(annotated.tobytes())
                raw_found += int(row['raw_target_detected'])
                available += int(row['target_available'])
                states[row['target_tracking']['state']] += 1
                histogram[str(row['impact_count']) if row['impact_count'] is not None else 'no_target'] += 1
                time_sums.update(row['timing_ms'])
                confirmed_ids.update(t['id'] for t in tracks if t['id'] is not None)
                if previous is not None and row['impact_count'] is not None:
                    valid_pairs += 1
                    count_changes += int(previous != row['impact_count'])
                    count_drops += int(previous > row['impact_count'])
                previous = row['impact_count']
                if n in preview_indices:
                    cv2.imwrite(str(artifacts / f'preview_frame_{n:05d}.jpg'), annotated)
                    previews.append((n, cv2.resize(annotated, (640, round(640*height/width)))))
                n += 1
                if n % 60 == 0:
                    progress('predicting')
                    print(f'{n}/{limit or "?"} frames', flush=True)
        progress('finalizing')
        encoder.stdin.close()
        if encoder.wait() != 0:
            raise RuntimeError(f'FFmpeg failed; see {artifacts / "ffmpeg.log"}')
        if limit and n != limit:
            raise RuntimeError(f'Decoded {n}, expected {limit} frames')
        if not n:
            raise RuntimeError('No frames decoded')
        temporary.replace(output)
        tile_h = previews[0][1].shape[0]+30
        montage = np.full((((len(previews)+1)//2)*tile_h, 1280, 3), 24, np.uint8)
        for i, (frame, view) in enumerate(previews):
            x, y = (i % 2)*640, (i//2)*tile_h
            montage[y+30:y+tile_h, x:x+640] = view
            text(montage, f'Frame {frame} / {frame/fps:.2f}s', (x+10, y+22), .55)
        cv2.imwrite(str(artifacts / 'preview_montage.jpg'), montage)
        summary = {'source': str(source), 'output': str(output), 'frames': n, 'fps': fps, 'duration_s': n/fps,
                   'input_width': width, 'input_height': height, 'raw_target_detected_frames': raw_found,
                   'target_available_frames': available, 'target_missing_frames': n-available,
                   'target_states': dict(states), 'impact_count_histogram': dict(histogram),
                   'confirmed_track_ids': sorted(confirmed_ids),
                   'track_count_changes': {'changes': count_changes, 'drops': count_drops, 'valid_pairs': valid_pairs},
                   'mean_inference_timing_ms': {key: value/n for key, value in time_sums.items()},
                   'elapsed_s': time.perf_counter()-started, 'temporal_tracking': config['tracking']['enabled'],
                   'prediction_cache_used': False, 'GT_used': False, 'audio': 'copy when present' if audio else 'disabled'}
        save_json(artifacts / 'summary.json', summary)
        progress('completed', summary=summary)
        print(f'Video: {output}\nPredictions / previews: {artifacts}')
        return summary
    except BaseException as error:
        if encoder is not None and encoder.poll() is None:
            encoder.kill()
            encoder.wait()
        progress('failed', error=repr(error))
        raise
    finally:
        capture.release()
        if log is not None:
            log.close()
