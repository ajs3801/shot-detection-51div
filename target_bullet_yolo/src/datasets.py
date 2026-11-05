"""Paired YOLO datasets; evaluation never silently drops unmatched test images."""
import json
from pathlib import Path
import cv2
import numpy as np
import yaml
from .geometry import crop_polygon, read_yolo_boxes, shift_boxes

EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


def images(root, split):
    directory = Path(root) / 'images' / split
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    result = {}
    for image in sorted(directory.iterdir()):
        if image.suffix.lower() in EXTENSIONS:
            if image.stem in result:
                raise ValueError(f'Duplicate image stem: {image.stem} in {directory}')
            result[image.stem] = image
    if not result:
        raise ValueError(f'No images: {directory}')
    return result


def check_pairs(config):
    counts, seen = {}, set()
    for split in ['train', 'val', 'test']:
        target = images(config['datasets']['target'], split)
        bullet = images(config['datasets']['bullet'], split)
        if target.keys() != bullet.keys():
            raise ValueError(f'{split} pair mismatch: target-only={sorted(target.keys()-bullet.keys())}, bullet-only={sorted(bullet.keys()-target.keys())}')
        repeated = seen.intersection(target)
        if repeated:
            raise ValueError(f'Image stems repeated across splits: {sorted(repeated)}')
        seen.update(target)
        empty = 0
        for stem in target:
            for task in ['target', 'bullet']:
                label = Path(config['datasets'][task]) / 'labels' / split / f'{stem}.txt'
                if not label.is_file():
                    raise FileNotFoundError(label)
                if task == 'target':
                    lines = [line.split() for line in label.read_text(encoding='utf-8').splitlines() if line.strip()]
                    if len(lines) != 1 or len(lines[0]) != 9 or lines[0][0] != '0':
                        raise ValueError(f'Expected one class-0 four-point polygon: {label}')
                elif not label.read_text(encoding='utf-8').strip():
                    empty += 1
        counts[split] = {'paired_images': len(target), 'empty_bullet_labels': empty}
    return counts


def write_data_yaml(config, task, directory):
    """Resolve paths locally instead of relying on absolute paths in old YAMLs."""
    root = Path(config['datasets'][task])
    original = root / 'data.yaml'
    names = yaml.safe_load(original.read_text(encoding='utf-8'))['names'] if original.is_file() else {0: 'target' if task == 'target' else 'impact'}
    value = {'path': str(root), 'train': 'images/train', 'val': 'images/val',
             'test': 'images/test', 'nc': 1, 'names': names}
    path = Path(directory) / f'{task}_data.yaml'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, allow_unicode=True, sort_keys=False), encoding='utf-8')
    return path


def read_image(path):
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f'Cannot decode image: {path}')
    return image


def paired_samples(config, split='test'):
    target = images(config['datasets']['target'], split)
    bullet = images(config['datasets']['bullet'], split)
    if target.keys() != bullet.keys():
        raise ValueError(f'{split}: target and bullet stems differ')
    geometry_path = config['datasets'].get('crop_geometry')
    geometry = json.loads(Path(geometry_path).read_text(encoding='utf-8'))['samples'] if geometry_path else {}
    for stem, full_path in target.items():
        full, crop = read_image(full_path), read_image(bullet[stem])
        label = Path(config['datasets']['target']) / 'labels' / split / f'{stem}.txt'
        values = list(map(float, label.read_text(encoding='utf-8').split()))
        if len(values) != 9 or values[0] != 0:
            raise ValueError(f'Expected one four-point target polygon: {label}')
        quad = np.float32(values[1:]).reshape(4, 2) * [full.shape[1], full.shape[0]]
        entry = geometry.get(stem)
        if entry:
            if entry['split'] != split:
                raise ValueError(f'Crop geometry split mismatch: {stem}')
            rect = tuple(entry['crop_xyxy'])
        else:
            _, rect = crop_polygon(full, quad, config['inference']['padding'])
        x0, y0, x1, y1 = rect
        if (crop.shape[1], crop.shape[0]) != (x1-x0, y1-y0):
            raise ValueError(f'GT crop geometry does not match {stem}: crop={crop.shape[:2]}, rect={rect}. Supply datasets.crop_geometry for custom crops.')
        if not (0 <= x0 < x1 <= full.shape[1] and 0 <= y0 < y1 <= full.shape[0]):
            raise ValueError(f'GT crop lies outside full image: {stem}')
        bullet_label = Path(config['datasets']['bullet']) / 'labels' / split / f'{stem}.txt'
        local = read_yolo_boxes(bullet_label, crop.shape)
        yield {'sample': stem, 'full_path': full_path, 'crop_path': bullet[stem],
               'full': full, 'gt_crop': crop, 'gt_quad': np.float32(quad),
               'gt_rect': rect, 'gt_local_boxes': local, 'gt_boxes': shift_boxes(local, rect[:2])}
