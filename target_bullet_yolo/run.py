"""Predict target polygons and impacts from a video and optionally render an MP4."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
from src.runtime import ROOT, configure, add_model_options, config_from_args


ADAPTIVE_PRESETS = {
    'budget27': {
        'inference': {'analysis_fps': 30, 'target_hz': 0.5, 'bullet_hz': 3, 'bullet_burst_frames': 4},
        'target': {'imgsz': 1280},
        'bullet': {'imgsz': 896},
        'tracking': {'confirmation_hits': 3, 'confirmation_window': 5},
    },
    'adaptive30': {
        'inference': {'analysis_fps': 30, 'target_hz': 2, 'bullet_hz': 5, 'bullet_burst_frames': 8},
        'target': {'imgsz': 1280},
        'bullet': {'imgsz': 1024},
        'tracking': {'confirmation_hits': 5, 'confirmation_window': 8},
    },
}


def main(args):
    configure()
    from src.video import run_video
    config = config_from_args(args)
    preset = getattr(args, 'preset', None)
    adaptive_inference = getattr(args, 'adaptive_inference', False) or preset is not None
    if preset is not None:
        for section, settings in ADAPTIVE_PRESETS[preset].items():
            config[section].update(settings)
    adaptive_options = ('analysis_fps', 'target_hz', 'bullet_hz', 'bullet_burst_frames')
    for key in adaptive_options:
        value = getattr(args, key, None)
        if value is not None:
            if not adaptive_inference:
                raise ValueError('Adaptive schedule options require --preset or --adaptive-inference.')
            config['inference'][key] = value
    if getattr(args, 'cpu_threads', None) is not None:
        config['inference']['cpu_threads'] = args.cpu_threads
    if getattr(args, 'grouping', False):
        config.setdefault('grouping', {})['enabled'] = True
    if getattr(args, 'grid_cell_cm', None) is not None:
        config.setdefault('grouping', {})['grid_cell_cm'] = args.grid_cell_cm
    if args.no_temporal:
        config['tracking']['enabled'] = False
    if args.target_conf is not None:
        config['target']['confidence'] = args.target_conf
    if args.bullet_conf is not None:
        config['bullet']['confidence'] = args.bullet_conf
    bullet_imgsz = getattr(args, 'bullet_imgsz', None)
    if bullet_imgsz is not None:
        if bullet_imgsz < 1:
            raise ValueError('--bullet-imgsz must be positive.')
        config['bullet']['imgsz'] = bullet_imgsz
    tracking = config['tracking']
    for key in ('confirmation_hits', 'confirmation_window'):
        value = getattr(args, key, None)
        if value is not None:
            tracking[key] = value
    if not 1 <= tracking['confirmation_hits'] <= tracking['confirmation_window']:
        raise ValueError('Confirmation hits must be between 1 and the confirmation window.')
    if not 0 < config['target']['confidence'] <= 1 or not 0 < config['bullet']['confidence'] <= 1:
        raise ValueError('Confidence thresholds must be in (0, 1].')
    if args.max_frames is not None and args.max_frames <= 0:
        raise ValueError('--max-frames must be positive')
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
    output = args.output or ROOT / 'outputs/video' / f'{args.source.stem}_{timestamp}.mp4'
    run_video(args.source, output, config, args.max_frames, audio=not args.no_audio,
              inference_fps=getattr(args, 'inference_fps', None), save_video=not getattr(args, 'no_video', False),
              adaptive_inference=adaptive_inference)


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    add_model_options(value)
    value.add_argument('--source', required=True, type=Path, help='Input video file.')
    value.add_argument('--output', type=Path, help='New MP4 path used to name the result directory, even with --no-video.')
    value.add_argument('--no-temporal', action='store_true', help='Disable optical flow, consensus and impact holding.')
    value.add_argument('--no-audio', action='store_true')
    value.add_argument('--target-conf', type=float)
    value.add_argument('--bullet-conf', type=float)
    value.add_argument('--bullet-imgsz', type=int, help='Override the impact detector input size.')
    value.add_argument('--confirmation-hits', type=int, help='Impact observations required for confirmation.')
    value.add_argument('--confirmation-window', type=int, help='Confirmation window in analysis frames.')
    value.add_argument('--max-frames', type=int, help='Optional limit on input frames for a quick check.')
    sampling = value.add_mutually_exclusive_group()
    sampling.add_argument('--preset', choices=ADAPTIVE_PRESETS,
                          help='Use the Budget27 or Adaptive30 adaptive inference settings.')
    sampling.add_argument('--inference-fps', type=float, help='Sample and output this many frames per second, up to the input FPS.')
    sampling.add_argument('--adaptive-inference', action='store_true',
                          help='Keep source FPS with analysis capped at 30 Hz and denser impact sampling around candidates.')
    value.add_argument('--analysis-fps', type=float, help='Adaptive analysis rate, at most 30 Hz. Default: 30.')
    value.add_argument('--target-hz', type=float, help='Adaptive regular target detection rate. Default: 2.')
    value.add_argument('--bullet-hz', type=float, help='Adaptive baseline impact detection rate. Default: 5.')
    value.add_argument('--bullet-burst-frames', type=int, help='Additional analysis frames per new impact candidate. Default: 8.')
    value.add_argument('--cpu-threads', type=int, help='Override the PyTorch CPU thread count for video prediction.')
    value.add_argument('--no-video', action='store_true', help='Save predictions and timing without rendering or encoding video.')
    value.add_argument('--grouping', action='store_true', help='Measure impact-to-impact distances and draw a group circle using the printed grid.')
    value.add_argument('--grid-cell-cm', type=float, help='Physical size of one square grid cell; default 1 cm.')
    return value


if __name__ == '__main__':
    main(parser().parse_args())
