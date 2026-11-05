"""Train YOLO26s-seg target and YOLO26n impact models on images/train."""
import argparse
import copy
import json
import shutil
import time
import traceback
from pathlib import Path
from src.runtime import configure, add_model_options, config_from_args, output_directory, save_json


def make_plan(config, args, output):
    tasks = ['target', 'bullet'] if args.task == 'all' else [args.task]
    plan = {}
    for task in tasks:
        options = {**copy.deepcopy(config['training']['common']), **config['training'][task]}
        if args.epochs is not None:
            options['epochs'] = args.epochs
        if args.workers is not None:
            options['workers'] = args.workers
        batch = getattr(args, f'{task}_batch')
        if batch is not None:
            options['batch'] = batch
        options.update(imgsz=config[task]['imgsz'], device=config[task]['device'],
                       project=str(output / 'training'), name=task, exist_ok=False)
        plan[task] = {'model': config[task]['pretrained'], 'dataset_root': config['datasets'][task],
                      'training_split': 'train', 'model_selection_split': 'val', 'options': options}
    return plan


def main(args):
    configure()
    if args.epochs is not None and args.epochs <= 0:
        raise ValueError('--epochs must be positive')
    from src.datasets import check_pairs, write_data_yaml
    config = config_from_args(args)
    for task in ['target', 'bullet']:
        if getattr(args, f'{task}_weights') is not None:
            config[task]['pretrained'] = config[task]['weights']
    counts = check_pairs(config)
    # Dry runs validate without importing models, creating runs, or training.
    planned_output = args.output.expanduser().resolve() if args.output else Path(__file__).parent / 'outputs/train/<new-run>'
    plan = make_plan(config, args, planned_output)
    for task, item in plan.items():
        if not Path(item['model']).is_file():
            raise FileNotFoundError(item['model'])
    if args.dry_run:
        print(json.dumps({'dry_run': True, 'splits': counts, 'plan': plan}, indent=2, ensure_ascii=False))
        return
    from ultralytics import YOLO, settings
    import torch
    settings.update({key: False for key in ['wandb', 'mlflow', 'comet', 'clearml', 'neptune', 'tensorboard', 'sync']})
    torch.backends.cuda.matmul.allow_tf32 = True
    output = output_directory('train', args.output)
    plan = make_plan(config, args, output)
    save_json(output / 'training_plan.json', {'splits': counts, 'plan': plan})
    started = time.perf_counter()
    status = {'status': 'training', 'tasks': {}}
    save_json(output / 'status.json', status)
    try:
        for task, item in plan.items():
            model = YOLO(item['model'])
            def report(trainer, task=task):
                status['tasks'][task] = {'status': 'training', 'epoch': trainer.epoch + 1,
                                          'metrics': {k: float(v) for k, v in trainer.metrics.items()}}
                save_json(output / 'status.json', status)
            model.add_callback('on_fit_epoch_end', report)
            data = write_data_yaml(config, task, output / 'data')
            # YOLO train reads 'train'; val is used only for monitoring / best.pt.
            model.train(data=str(data), **item['options'])
            trained = output / 'models' / f'{task}.pt'
            trained.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(model.trainer.best, trained)
            config[task]['weights'] = str(trained)
            status['tasks'][task] = {'status': 'completed', 'best_weights': str(trained),
                                      'epochs_completed': model.trainer.epoch + 1}
            save_json(output / 'config.json', config)
            save_json(output / 'status.json', status)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    except BaseException as error:
        status.update(status='failed', error=repr(error), traceback=traceback.format_exc())
        save_json(output / 'status.json', status)
        raise
    status.update(status='completed', elapsed_s=time.perf_counter() - started)
    save_json(output / 'status.json', status)
    print(f'Training complete: {output}')
    print(f'Test: python test.py --config "{output / "config.json"}"')


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    add_model_options(value)
    value.add_argument('--task', choices=['all', 'target', 'bullet'], default='all')
    value.add_argument('--output', type=Path, help='New/empty run directory; default outputs/train/<timestamp>.')
    value.add_argument('--epochs', type=int)
    value.add_argument('--target-batch', type=int)
    value.add_argument('--bullet-batch', type=int)
    value.add_argument('--workers', type=int)
    value.add_argument('--dry-run', action='store_true', help='Check dataset and show training arguments without training.')
    return value


if __name__ == '__main__':
    main(parser().parse_args())
