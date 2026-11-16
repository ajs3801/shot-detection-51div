"""Predict target polygons and impacts from a video; write an annotated MP4."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
from src.runtime import ROOT, configure, add_model_options, config_from_args


def main(args):
    configure()
    from src.video import run_video
    config = config_from_args(args)
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
    if not 0 < config['target']['confidence'] <= 1 or not 0 < config['bullet']['confidence'] <= 1:
        raise ValueError('Confidence thresholds must be in (0, 1].')
    if args.max_frames is not None and args.max_frames <= 0:
        raise ValueError('--max-frames must be positive')
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
    output = args.output or ROOT / 'outputs/video' / f'{args.source.stem}_{timestamp}.mp4'
    run_video(args.source, output, config, args.max_frames, audio=not args.no_audio)


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    add_model_options(value)
    value.add_argument('--source', required=True, type=Path, help='Input video file.')
    value.add_argument('--output', type=Path, help='New output MP4; default outputs/video/<source>_<timestamp>.mp4.')
    value.add_argument('--no-temporal', action='store_true', help='Disable optical flow, consensus and impact holding.')
    value.add_argument('--no-audio', action='store_true')
    value.add_argument('--target-conf', type=float)
    value.add_argument('--bullet-conf', type=float)
    value.add_argument('--max-frames', type=int, help='Optional frame limit for a quick check.')
    value.add_argument('--grouping', action='store_true', help='Measure impact-to-impact distances and draw a group circle using the printed grid.')
    value.add_argument('--grid-cell-cm', type=float, help='Physical size of one square grid cell; default 1 cm.')
    return value


if __name__ == '__main__':
    main(parser().parse_args())
