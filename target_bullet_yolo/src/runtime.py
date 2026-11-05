"""Configuration, local paths and runtime setup shared by the three commands."""
import copy
import json
import os
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / 'config.json'


def configure():
    for key, value in {
        'YOLO_CONFIG_DIR': str(ROOT / '.cache/ultralytics'),
        'MPLCONFIGDIR': str(ROOT / '.cache/matplotlib'),
        'XDG_CACHE_HOME': str(ROOT / '.cache'),
        'TORCH_HOME': str(ROOT / '.cache/torch'),
        'OMP_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '2',
    }.items():
        os.environ.setdefault(key, value)
    import cv2
    import torch
    cv2.setNumThreads(1)
    torch.set_num_threads(2)


def read_config(path=DEFAULT_CONFIG):
    path = Path(path).expanduser().resolve()
    config = json.loads(path.read_text(encoding='utf-8'))
    for key in ['target', 'bullet', 'crop_geometry']:
        if config['datasets'].get(key):
            config['datasets'][key] = str(resolve(config['datasets'][key], path.parent))
    for task in ['target', 'bullet']:
        for key in ['weights', 'pretrained']:
            config[task][key] = str(resolve(config[task][key], path.parent))
    return config


def resolve(value, base=ROOT):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def save_json(path, value):
    import numpy as np
    def default(item):
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, Path):
            return str(item)
        raise TypeError(type(item).__name__)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=default) + '\n', encoding='utf-8')
    temporary.replace(path)


def devices(config):
    import torch
    result = copy.deepcopy(config)
    for index, task in enumerate(['target', 'bullet']):
        if str(result[task]['device']) == 'auto':
            count = torch.cuda.device_count()
            result[task]['device'] = str(min(index, count - 1)) if count else 'cpu'
    return result


def add_model_options(parser):
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--target-weights', type=Path)
    parser.add_argument('--bullet-weights', type=Path)
    parser.add_argument('--device', help='Device for both models: 0, 1 or cpu. Default: auto.')
    parser.add_argument('--target-device')
    parser.add_argument('--bullet-device')


def config_from_args(args):
    config = read_config(args.config)
    for task in ['target', 'bullet']:
        weights = getattr(args, f'{task}_weights', None)
        if weights is not None:
            config[task]['weights'] = str(weights.expanduser().resolve())
        device = getattr(args, f'{task}_device', None) or getattr(args, 'device', None)
        if device is not None:
            config[task]['device'] = device
    return devices(config)


def output_directory(kind, supplied=None):
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
    path = Path(supplied).expanduser().resolve() if supplied else ROOT / 'outputs' / kind / timestamp
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f'Choose a new, empty output directory: {path}')
    path.mkdir(parents=True, exist_ok=True)
    return path


def require_weights(config, pretrained=False):
    key = 'pretrained' if pretrained else 'weights'
    for task in ['target', 'bullet']:
        if not Path(config[task][key]).is_file():
            raise FileNotFoundError(f'{task} {key}: {config[task][key]}')
