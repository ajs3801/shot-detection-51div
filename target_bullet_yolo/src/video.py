"""Fresh two-stage video inference and H.264/AAC-compatible audio muxing."""
import json
import math
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
from .grouping import GridCalibrator, ImpactGroup
from .visualization import video_frame, text


ADAPTIVE_TARGET_HZ = 2
ADAPTIVE_BULLET_HZ = 5
ADAPTIVE_BURST_FRAMES = 8
ADAPTIVE_MAX_ANALYSIS_HZ = 30


def adaptive_schedule(config, fps):
    settings = config['inference']
    rates = {'analysis_fps': settings.get('analysis_fps', ADAPTIVE_MAX_ANALYSIS_HZ),
             'target_hz': settings.get('target_hz', ADAPTIVE_TARGET_HZ),
             'bullet_hz': settings.get('bullet_hz', ADAPTIVE_BULLET_HZ)}
    for name, value in rates.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f'inference.{name} must be finite and positive.')
    if rates['analysis_fps'] > ADAPTIVE_MAX_ANALYSIS_HZ:
        raise ValueError(f'inference.analysis_fps must be at most {ADAPTIVE_MAX_ANALYSIS_HZ}.')
    if max(rates['target_hz'], rates['bullet_hz']) > rates['analysis_fps']:
        raise ValueError('Adaptive target and bullet rates must not exceed inference.analysis_fps.')
    burst = settings.get('bullet_burst_frames', ADAPTIVE_BURST_FRAMES)
    if type(burst) is not int or burst < 1:
        raise ValueError('inference.bullet_burst_frames must be a positive integer.')
    analysis_fps = min(fps, rates['analysis_fps'])
    return {'mode': 'adaptive', 'analysis_fps': analysis_fps,
            'target_hz': min(rates['target_hz'], analysis_fps),
            'bullet_hz': min(rates['bullet_hz'], analysis_fps), 'bullet_burst_frames': burst}


def sample_due(frame, source_fps, rate):
    # A frame is selected if its half-frame time interval contains a scheduled tick.
    return math.floor((frame + .5)*rate/source_fps) > math.floor((frame - .5)*rate/source_fps)


def held_analysis_row(previous):
    """Keep the last projected result for a display frame between analysis ticks."""
    row = dict(previous, crop_xyxy=None, raw_target_detected=None, raw_target_confidence=None,
               raw_target_polygon=None, raw_impact_count=None, raw_impact_boxes_original_xyxy=[],
               raw_impact_scores=[], target_inference_ran=False, bullet_inference_ran=False,
               bullet_burst_started=False, target_refresh_deferred=False,
               timing_ms=dict.fromkeys(previous['timing_ms'], 0.))
    row['target_tracking'] = dict(previous['target_tracking'], state='analysis hold', reset=False,
                                  scene_cut=False, flow={'valid': None, 'inliers': 0})
    row['impact_tracks'] = [dict(track, held=True) for track in previous['impact_tracks']]
    if previous.get('grouping') is not None:
        group = previous['grouping']
        row['grouping'] = dict(group, points=[dict(point, held=True) for point in group['points']])
    return row


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
    def __init__(self, config, fps, *, adaptive_inference=False):
        require_weights(config)
        self.config, self.fps = config, fps
        self.adaptive = adaptive_inference
        self.schedule = adaptive_schedule(config, fps) if self.adaptive else None
        self.devices = {task: config[task]['device'] for task in ('target', 'bullet')}
        cpu_threads = config['inference'].get('cpu_threads')
        if cpu_threads is not None and any(str(device).lower() == 'cpu' for device in self.devices.values()):
            import torch
            torch.set_num_threads(cpu_threads)
            for task, device in self.devices.items():
                if str(device).lower() == 'cpu':
                    # A typed CPU device keeps Ultralytics from resetting the selected thread count.
                    self.devices[task] = torch.device('cpu')
        self.target = YOLO(config['target']['weights'])
        self.bullet = YOLO(config['bullet']['weights'])
        settings = config['tracking']
        self.target_tracker = TargetTracker(fps, settings['target_measurement_alpha'], settings['max_model_gap_s'],
                                            settings['freeze_s'], reuse_features=self.adaptive)
        self.impacts = ImpactTracker(fps, config['bullet']['confidence'], min(config['inference']['bullet_low_confidence'], config['bullet']['confidence']),
                                     settings['confirmation_hits'], settings['confirmation_window'], settings['impact_hold_s'])
        self.grouping = config.get('grouping', {}).get('enabled', False)
        if self.grouping:
            self.grid = GridCalibrator(fps, config.get('grouping', {}).get('grid_cell_cm', 1.))
        if self.grouping or self.adaptive:
            self.group = ImpactGroup()
        if self.adaptive:
            self.last_target_call = -math.inf
            self.flow_failures = 0
            self.burst_until = -1
            self.target_refresh_pending = False
            self.suspended_target = None

    def target_due(self, frame):
        if sample_due(frame, self.fps, self.schedule['target_hz']):
            self.target_refresh_pending = True
        gap = frame - self.last_target_call
        recovery_due = ((self.target_tracker.quad is None and gap >= max(1, round(self.fps/15)))
                        or (self.flow_failures >= 3 and gap >= max(1, round(self.fps*.2))))
        healthy_burst = frame <= self.burst_until and self.target_tracker.quad is not None and self.flow_failures == 0
        return recovery_due or (self.target_refresh_pending and not healthy_burst)

    def _update_accumulated_target(self, quad, status, frame, previous_quad, previous_gray):
        cut_while_waiting = (self.suspended_target is not None and previous_quad is None
                             and self.target_tracker.scene_changed(self.target_tracker.gray, previous_gray))
        if status['scene_cut'] or cut_while_waiting:
            self.group.reset()
            self.suspended_target = None
        elif quad is None:
            if previous_quad is not None and self.group.shots:
                self.suspended_target = (frame, previous_quad, previous_gray)
        elif status['reset']:
            projected = None
            if self.suspended_target is not None:
                lost_frame, last_quad, last_gray = self.suspended_target
                if frame - lost_frame <= self.impacts.hold_frames:
                    projected = self.target_tracker.reconnect(last_gray, last_quad, quad)
            if projected is None:
                self.group.reset()
            else:
                self.group.rebase(projected, quad)
            self.suspended_target = None

    def predict(self, image, frame):
        tcfg, bcfg, icfg = self.config['target'], self.config['bullet'], self.config['inference']
        temporal = self.config['tracking']['enabled']
        start = time.perf_counter()
        target_ran = not self.adaptive or self.target_due(frame)
        target = None
        if target_ran:
            target = predict_target(
                self.target, image, tcfg['confidence'], self.devices['target'], tcfg['imgsz'], icfg['iou'],
                retina_masks=tcfg.get('retina_masks', True),
            )
            if self.adaptive:
                self.last_target_call = frame
                self.target_refresh_pending = False
        after_target = time.perf_counter()
        measurement = target['quad'] if target is not None else None
        if temporal:
            previous_quad, previous_gray = self.target_tracker.quad, self.target_tracker.gray
            quad, status = self.target_tracker.update(image, measurement, frame)
            if status['reset'] or quad is None:
                self.impacts.reset()
        else:
            quad = measurement
            status = {'state': 'detected' if quad is not None else 'not detected', 'reset': False, 'model_gap_frames': 0}
        if self.adaptive:
            self.flow_failures = 0 if status['flow']['valid'] else self.flow_failures + 1
            if status['reset'] or quad is None:
                self.burst_until = -1
            self._update_accumulated_target(quad, status, frame, previous_quad, previous_gray)
        if self.grouping and (status['reset'] or quad is None):
            if not self.adaptive:
                self.group.reset()
            self.grid.reset()
        after_flow = time.perf_counter()
        crop = rect = None
        boxes = np.empty((0, 4), np.float32)
        scores = np.empty(0, np.float32)
        bullet_sample = not self.adaptive or sample_due(frame, self.fps, self.schedule['bullet_hz'])
        in_burst = self.adaptive and frame <= self.burst_until
        bullet_ran = quad is not None and (bullet_sample or in_burst)
        if bullet_ran:
            crop, rect = crop_polygon(image, quad, icfg['padding'])
            floor = min(icfg['bullet_low_confidence'], bcfg['confidence']) if temporal else bcfg['confidence']
            local, scores = predict_bullets(self.bullet, crop, self.devices['bullet'], bcfg['imgsz'], floor, icfg['iou'])
            boxes = shift_boxes(local, rect[:2])
        after_bullet = time.perf_counter()
        tracks = []
        burst_started = False
        if bullet_ran:
            if self.adaptive and not in_burst and self.group.has_new_candidate(boxes[scores >= floor], quad):
                self.burst_until = frame + self.schedule['bullet_burst_frames']
                burst_started = True
            if temporal:
                tracks = self.impacts.update(boxes, scores, quad, frame)
            else:
                tracks = [{'id': None, 'box': box.tolist(), 'score_ema': float(score), 'held': False, 'gap_frames': 0}
                          for box, score in zip(boxes, scores) if score >= bcfg['confidence']]
        after_tracking = time.perf_counter()
        group_result = None
        held_count = len(self.group.shots) if self.adaptive and quad is None and self.group.shots else None
        if quad is not None:
            if self.grouping:
                self.grid.update(image, quad, frame)
            if bullet_ran and (self.grouping or self.adaptive):
                self.group.update(tracks, quad, frame)
            if self.adaptive:
                tracks = self.group.projected_tracks(quad, frame)
            if self.grouping:
                group_result = self.group.measurements(self.grid, quad, frame)
        elif self.grouping and held_count is not None:
            group_result = self.group.measurements(self.grid, None, frame)
        end = time.perf_counter()
        record = {'frame': frame, 'time_s': frame/self.fps, 'fps': self.fps,
                  'raw_target_detected': (target is not None) if target_ran else None,
                  'raw_target_confidence': target['confidence'] if target is not None else None,
                  'raw_target_polygon': None if target is None else target['quad'].tolist(),
                  'target_available': quad is not None, 'target_tracking': status,
                  'target_polygon_TL_TR_BR_BL': None if quad is None else quad.tolist(),
                  'crop_xyxy': rect, 'raw_impact_count': int((scores >= bcfg['confidence']).sum()) if bullet_ran else None,
                  'raw_impact_boxes_original_xyxy': boxes.tolist(), 'raw_impact_scores': scores.tolist(),
                  'impact_count': len(tracks) if quad is not None else held_count, 'impact_tracks': tracks,
                  'timing_ms': {'target': (after_target-start)*1000, 'flow': (after_flow-after_target)*1000,
                                'crop_bullet': (after_bullet-after_flow)*1000, 'impact_tracking': (after_tracking-after_bullet)*1000,
                                'total': (end-start)*1000}}
        if self.grouping:
            record['grouping'] = group_result
            record['timing_ms']['grouping'] = (end-after_tracking)*1000
        if self.adaptive:
            record.update(inference_mode='adaptive', target_inference_ran=target_ran,
                          bullet_inference_ran=bullet_ran, bullet_burst_started=burst_started,
                          impact_count_held=held_count is not None,
                          target_refresh_deferred=self.target_refresh_pending)
        return record, quad, crop, rect, tracks


def run_video(source, output, config, max_frames=None, audio=True, *, inference_fps=None, save_video=True,
              adaptive_inference=False):
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
    if inference_fps is not None and (not math.isfinite(inference_fps) or inference_fps <= 0):
        raise ValueError('--inference-fps must be finite and positive')
    if adaptive_inference and (inference_fps is not None or not config['tracking']['enabled']):
        raise ValueError('--adaptive-inference requires temporal tracking and cannot be combined with --inference-fps')
    cpu_threads = config['inference'].get('cpu_threads')
    if cpu_threads is None and adaptive_inference and any(str(config[task]['device']).lower() == 'cpu' for task in ('target', 'bullet')):
        cpu_threads = min(4, os.cpu_count() or 1)
    if cpu_threads is not None:
        if type(cpu_threads) is not int or cpu_threads < 1:
            raise ValueError('CPU threads must be a positive integer.')
        config = {**config, 'inference': {**config['inference'], 'cpu_threads': cpu_threads}}
    capture = cv2.VideoCapture(str(source), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1])
    if not capture.isOpened():
        raise ValueError(f'Cannot open video: {source}')
    width, height = int(capture.get(3)), int(capture.get(4))
    source_fps = capture.get(cv2.CAP_PROP_FPS)
    expected = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if not math.isfinite(source_fps) or source_fps <= 0 or min(width, height) < 64:
        capture.release()
        raise ValueError('Video must have a valid FPS and dimensions >= 64 pixels.')
    fps = min(source_fps, inference_fps) if inference_fps is not None else source_fps
    try:
        inference_schedule = adaptive_schedule(config, fps) if adaptive_inference else None
    except ValueError:
        capture.release()
        raise
    analysis_fps = inference_schedule['analysis_fps'] if adaptive_inference else fps
    sampled = fps < source_fps
    source_limit = min(expected, max_frames) if expected and max_frames else max_frames or expected
    limit = max(1, math.ceil((source_limit - .5) * fps / source_fps)) if source_limit else 0
    output.parent.mkdir(parents=True, exist_ok=True)
    artifacts.mkdir(parents=True, exist_ok=True)
    save_json(artifacts / 'config.json', {**config, 'video_inference': inference_schedule} if adaptive_inference else config)
    encoder = None
    log = None
    started = time.perf_counter()
    n = source_n = analysis_n = 0
    raw_found = available = 0
    states, histogram = Counter(), Counter()
    time_sums, confirmed_ids = Counter(), set()
    inference_calls = dict(target=0, bullet=0, bullet_bursts=0)
    count_changes = count_drops = valid_pairs = 0
    previous = None
    last_analysis_row = last_analysis_quad = None
    deferred_target_frames = 0
    preview_indices = {0}
    if limit:
        preview_indices.update(round((limit-1)*fraction) for fraction in [.2, .4, .6, .8, .98])
    previews = []
    output_times = dict(render=0., pipe_write=0., encoder_finalize=0., previews=0.)
    latest_group = None
    calibrated_frames = 0

    def progress(status, **extra):
        save_json(artifacts / 'progress.json', {'status': status, 'source': str(source),
                  'output': str(output) if save_video else None,
                  'frames_processed': n, 'expected_frames': limit or None,
                  'source_frames_processed': source_n, 'expected_source_frames': source_limit or None,
                  'elapsed_s': time.perf_counter()-started, **extra})

    try:
        progress('initializing')
        predictor = VideoPredictor(config, analysis_fps, adaptive_inference=adaptive_inference)
        if save_video:
            temporary = output.with_name(output.stem + '.partial.mp4')
            command = [find_ffmpeg(), '-hide_banner', '-loglevel', 'warning', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
                       '-video_size', f'{width}x{height}', '-framerate', str(fps), '-i', 'pipe:0']
            if audio:
                if max_frames is not None and source_limit:
                    command += ['-t', str(source_limit/source_fps)]
                command += ['-i', str(source), '-map', '0:v:0', '-map', '1:a?', '-c:a', 'copy']
            else:
                command += ['-map', '0:v:0', '-an']
            # Odd-sized videos get one-pixel encoder padding, without changing inference.
            if width % 2 or height % 2:
                command += ['-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2']
            command += ['-c:v', 'libx264', '-threads', '2', '-preset', 'fast', '-crf', '20',
                        '-pix_fmt', 'yuv420p', '-movflags', '+faststart', '-y', str(temporary)]
            log = (artifacts / 'ffmpeg.log').open('w')
            encoder = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=log)
        with (artifacts / 'predictions.jsonl').open('w') as predictions:
            while max_frames is None or source_n < max_frames:
                # Select the source frame nearest the next sample time. Decode skipped
                # frames without retrieving their pixel arrays from OpenCV.
                if sampled and n and (source_n + .5) * fps <= n * source_fps:
                    if not capture.grab():
                        break
                    source_n += 1
                    continue
                ok, image = capture.read()
                if not ok:
                    break
                source_frame = source_n
                source_n += 1
                analyze = not adaptive_inference or sample_due(source_frame, source_fps, analysis_fps)
                analysis_frame = analysis_n if analyze else None
                if analyze:
                    row, quad, crop, rect, tracks = predictor.predict(image, analysis_frame)
                    analysis_n += 1
                    last_analysis_row, last_analysis_quad = row, quad
                else:
                    row = held_analysis_row(last_analysis_row)
                    quad = last_analysis_quad
                    crop = rect = None
                    tracks = row['impact_tracks']
                if sampled or adaptive_inference:
                    row.update(source_frame=source_frame, source_time_s=source_frame/source_fps)
                if adaptive_inference:
                    row.update(frame=n, time_s=n/fps, fps=fps, analysis_fps=analysis_fps,
                               analysis_frame=analysis_frame, analysis_ran=analyze)
                    for model in ('target', 'bullet'):
                        ran = row[f'{model}_inference_ran']
                        row[f'{model}_inference_index'] = inference_calls[model] if ran else None
                        inference_calls[model] += ran
                    inference_calls['bullet_bursts'] += row['bullet_burst_started']
                    deferred_target_frames += row['target_refresh_deferred']
                if row.get('grouping') is not None:
                    latest_group = row['grouping']
                    calibrated_frames += int(latest_group['state'] == 'calibrated')
                predictions.write(json.dumps(row) + '\n')
                if encoder is not None:
                    before_render = time.perf_counter()
                    annotated = video_frame(image, quad, crop, rect, tracks, row, config, limit)
                    before_write = time.perf_counter()
                    encoder.stdin.write(annotated.tobytes())
                    output_times['render'] += before_write - before_render
                    output_times['pipe_write'] += time.perf_counter() - before_write
                    if n in preview_indices:
                        before_preview = time.perf_counter()
                        cv2.imwrite(str(artifacts / f'preview_frame_{n:05d}.jpg'), annotated)
                        tile = cv2.resize(annotated, (640, round(640*height/width)))
                        previews.append((source_frame, tile))
                        output_times['previews'] += time.perf_counter() - before_preview
                raw_found += bool(row['raw_target_detected'])
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
                n += 1
                if n % 60 == 0:
                    progress('predicting')
                    print(f'{n}/{limit or "?"} frames', flush=True)
        progress('finalizing')
        if source_limit and source_n != source_limit:
            raise RuntimeError(f'Decoded {source_n}, expected {source_limit} input frames')
        if not n:
            raise RuntimeError('No frames decoded')
        if encoder is not None:
            before_finalize = time.perf_counter()
            encoder.stdin.close()
            if encoder.wait() != 0:
                raise RuntimeError(f'FFmpeg failed; see {artifacts / "ffmpeg.log"}')
            temporary.replace(output)
            before_preview = time.perf_counter()
            output_times['encoder_finalize'] = before_preview - before_finalize
            tile_h = previews[0][1].shape[0]+30
            montage = np.full((((len(previews)+1)//2)*tile_h, 1280, 3), 24, np.uint8)
            for i, (frame, view) in enumerate(previews):
                x, y = (i % 2)*640, (i//2)*tile_h
                montage[y+30:y+tile_h, x:x+640] = view
                text(montage, f'Frame {frame} / {frame/source_fps:.2f}s', (x+10, y+22), .55)
            cv2.imwrite(str(artifacts / 'preview_montage.jpg'), montage)
            output_times['previews'] += time.perf_counter() - before_preview
        summary = {'source': str(source), 'output': str(output) if save_video else None,
                   'frames': n, 'fps': fps, 'duration_s': n/fps,
                   'source_fps': source_fps, 'source_frames': source_n, 'source_duration_s': source_n/source_fps,
                   'input_width': width, 'input_height': height, 'raw_target_detected_frames': raw_found,
                   'target_available_frames': available, 'target_missing_frames': n-available,
                   'target_states': dict(states), 'impact_count_histogram': dict(histogram),
                   'confirmed_track_ids': sorted(confirmed_ids),
                   'track_count_changes': {'changes': count_changes, 'drops': count_drops, 'valid_pairs': valid_pairs},
                   'mean_inference_timing_ms': {key: value/n for key, value in time_sums.items()},
                   'output_timing_s': output_times,
                   'elapsed_s': time.perf_counter()-started, 'temporal_tracking': config['tracking']['enabled'],
                   'prediction_cache_used': False, 'GT_used': False,
                   'audio': 'copy when present' if audio and save_video else 'disabled'}
        if config.get('grouping', {}).get('enabled', False):
            summary['grouping'] = {'calibrated_frames': calibrated_frames, 'latest': latest_group,
                                   'note': 'Grid-derived estimates between detected impact centers; no target-center reference.'}
        if adaptive_inference:
            summary.update(inference_schedule=inference_schedule, inference_calls=inference_calls,
                           analysis_frames=analysis_n, analysis_fps=analysis_fps,
                           analysis_elapsed_s=time_sums['total']/1000,
                           target_refresh_deferred_frames=deferred_target_frames,
                           mean_analysis_timing_ms={key: value/analysis_n for key, value in time_sums.items()})
        save_json(artifacts / 'summary.json', summary)
        progress('completed', summary=summary)
        if save_video:
            print(f'Video: {output}')
        print(f'Predictions: {artifacts}')
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
