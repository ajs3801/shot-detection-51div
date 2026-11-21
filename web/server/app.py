"""탄착군 판정 웹 서버: 발표 페이지를 제공하고, 업로드한 영상을 분석 엔진(run.py)으로 처리합니다."""
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

BASE = Path(__file__).resolve().parents[1]
ENGINE = Path(os.environ.get('SHOT_ENGINE', BASE.parent / 'target_bullet_yolo')).resolve()
STATIC = BASE / 'static'
JOBS = Path(os.environ.get('SHOT_JOBS', BASE / 'jobs')).resolve()
DEVICE = os.environ.get('SHOT_DEVICE', 'cpu')
CPU_THREADS = os.environ.get('SHOT_CPU_THREADS')
MAX_MB = int(os.environ.get('SHOT_MAX_UPLOAD_MB', '500'))
KEEP_HOURS = float(os.environ.get('SHOT_KEEP_HOURS', '24'))
MODES = {'realtime': 'budget27', 'saved': 'adaptive30'}
VIDEO_EXT = {'.mp4', '.mov', '.m4v', '.avi', '.mkv', '.webm'}

JOBS.mkdir(parents=True, exist_ok=True)
app = FastAPI(title='탄착군 판정')
work = queue.Queue()
jobs = {}
lock = threading.Lock()


def ffmpeg():
    exe = shutil.which('ffmpeg')
    if exe:
        return exe
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def save_state(job_id, **changes):
    with lock:
        job = jobs[job_id]
        job.update(changes, updated=time.time())
        (JOBS / job_id / 'state.json').write_text(json.dumps(job, ensure_ascii=False), encoding='utf-8')


def compact(predictions, summary):
    r1 = lambda v, n=1: round(float(v), n)
    frames = []
    with open(predictions, encoding='utf-8') as handle:
        for line in handle:
            row = json.loads(line)
            frame = {}
            quad = row.get('target_polygon_TL_TR_BR_BL')
            if quad:
                frame['q'] = [r1(c) for point in quad for c in point]
            frame['ts'] = (row.get('target_tracking') or {}).get('state')
            frame['tr'] = int(bool(row.get('target_inference_ran')))
            frame['br'] = int(bool(row.get('bullet_inference_ran')))
            frame['n'] = row.get('impact_count')
            frame['tk'] = [[t['id'], *[r1(c) for c in t['box']], int(bool(t.get('held')))]
                           for t in (row.get('impact_tracks') or [])]
            frame['rb'] = ([[r1(c) for c in box] for box in (row.get('raw_impact_boxes_original_xyxy') or [])]
                           if row.get('bullet_inference_ran') else None)
            group = row.get('grouping')
            if group:
                out = {'s': group.get('state')}
                if group.get('points'):
                    out['p'] = [[p['id'], r1(p['xy_cm'][0], 3), r1(p['xy_cm'][1], 3), r1(p['xy_image'][0]),
                                 r1(p['xy_image'][1]), int(bool(p.get('held')))] for p in group['points']]
                circle = group.get('circle')
                if circle:
                    boundary = circle.get('boundary_image') or []
                    step = max(1, len(boundary) // 36)
                    out['c'] = [r1(circle['diameter_cm'], 3), [r1(v) for p in boundary[::step] for v in p]]
                pair = group.get('max_pair')
                if pair:
                    out['m'] = [pair['ids'][0], pair['ids'][1], r1(pair['distance_cm'], 3)]
                if group.get('pairs'):
                    out['pr'] = [[p['ids'][0], p['ids'][1], r1(p['distance_cm'], 3)] for p in group['pairs']]
                frame['g'] = out
            frames.append(frame)
    return {'fps': summary.get('source_fps') or summary.get('fps'), 'w': summary['input_width'],
            'h': summary['input_height'], 'n': len(frames), 'frames': frames}


def tail(path, lines=8):
    try:
        return '\n'.join(path.read_text(encoding='utf-8', errors='replace').strip().splitlines()[-lines:])
    except OSError:
        return ''


def process(job_id):
    folder = JOBS / job_id
    job = jobs[job_id]
    source = folder / job['upload']
    try:
        save_state(job_id, status='preparing', progress=0)
        preview = folder / 'preview.mp4'
        command = [ffmpeg(), '-hide_banner', '-loglevel', 'error', '-y', '-i', str(source), '-an',
                   '-vf', "scale='trunc(min(1920,iw)/2)*2':-2", '-fps_mode', 'passthrough',
                   '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '23', '-pix_fmt', 'yuv420p',
                   '-movflags', '+faststart', str(preview)]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0 or not preview.exists():
            raise RuntimeError('영상을 읽지 못했습니다. 다른 형식(mp4 권장)으로 올려 주세요.')

        save_state(job_id, status='analyzing', progress=0)
        output = folder / 'run.mp4'
        command = [sys.executable, 'run.py', '--source', str(source), '--output', str(output),
                   '--preset', MODES[job['mode']], '--grouping', '--no-video', '--no-audio', '--device', DEVICE]
        if CPU_THREADS:
            command += ['--cpu-threads', CPU_THREADS]
        with (folder / 'engine.log').open('w', encoding='utf-8') as log:
            env = {**os.environ, 'PYTHONUTF8': '1', 'PYTHONIOENCODING': 'utf-8'}
            engine = subprocess.Popen(command, cwd=ENGINE, stdout=log, stderr=subprocess.STDOUT, env=env)
            progress_file = folder / 'run' / 'progress.json'
            while engine.poll() is None:
                time.sleep(1)
                try:
                    state = json.loads(progress_file.read_text(encoding='utf-8'))
                    done, total = state.get('source_frames_processed') or 0, state.get('expected_source_frames') or 0
                    if total:
                        save_state(job_id, progress=min(.99, done / total))
                except (OSError, ValueError):
                    pass
        if engine.returncode != 0:
            raise RuntimeError('분석 중 오류가 났습니다.\n' + tail(folder / 'engine.log'))

        summary = json.loads((folder / 'run' / 'summary.json').read_text(encoding='utf-8'))
        data = compact(folder / 'run' / 'predictions.jsonl', summary)
        (folder / 'result.json').write_text(json.dumps(data, separators=(',', ':')), encoding='utf-8')
        save_state(job_id, status='done', progress=1, elapsed_s=round(summary.get('elapsed_s', 0), 1))
    except Exception as error:  # 사용자에게 보여 줄 메시지로 바꿔 저장합니다.
        save_state(job_id, status='failed', error=str(error))


def worker():
    while True:
        job_id = work.get()
        try:
            process(job_id)
        finally:
            work.task_done()


def cleaner():
    while True:
        limit = time.time() - KEEP_HOURS * 3600
        for folder in JOBS.iterdir():
            job = jobs.get(folder.name)
            if folder.is_dir() and folder.stat().st_mtime < limit and (not job or job['status'] in ('done', 'failed')):
                shutil.rmtree(folder, ignore_errors=True)
                with lock:
                    jobs.pop(folder.name, None)
        time.sleep(1800)


def restore():
    """서버를 다시 켜도 끝난 작업의 결과는 계속 볼 수 있게 상태를 읽어 옵니다."""
    for state in JOBS.glob('*/state.json'):
        try:
            job = json.loads(state.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            continue
        if job.get('status') not in ('done', 'failed'):
            job.update(status='failed', error='서버가 다시 시작되어 분석이 중단되었습니다. 다시 올려 주세요.')
        jobs[job['id']] = job


restore()
threading.Thread(target=worker, daemon=True).start()
threading.Thread(target=cleaner, daemon=True).start()


@app.get('/api/health')
def health():
    return {'ok': True, 'modes': list(MODES), 'max_upload_mb': MAX_MB, 'device': DEVICE}


@app.post('/api/jobs')
async def create_job(video: UploadFile = File(...), mode: str = Form('realtime')):
    if mode not in MODES:
        raise HTTPException(400, '분석 모드는 realtime 또는 saved 입니다.')
    suffix = Path(video.filename or '').suffix.lower()
    if suffix not in VIDEO_EXT:
        raise HTTPException(400, '영상 파일(mp4, mov, avi, mkv, webm)만 올릴 수 있습니다.')
    job_id = uuid.uuid4().hex[:12]
    folder = JOBS / job_id
    folder.mkdir(parents=True)
    target, size = folder / f'upload{suffix}', 0
    with target.open('wb') as handle:
        while chunk := await video.read(1 << 20):
            size += len(chunk)
            if size > MAX_MB * 1024 * 1024:
                handle.close()
                shutil.rmtree(folder, ignore_errors=True)
                raise HTTPException(413, f'영상이 {MAX_MB}MB를 넘습니다.')
            handle.write(chunk)
    with lock:
        jobs[job_id] = {'id': job_id, 'mode': mode, 'name': video.filename, 'upload': target.name,
                        'status': 'queued', 'progress': 0, 'created': time.time()}
    save_state(job_id)
    work.put(job_id)
    return {'id': job_id}


@app.get('/api/jobs/{job_id}')
def job_status(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, '작업을 찾을 수 없습니다.')
    ahead = sum(1 for j in jobs.values() if j['status'] == 'queued' and j['created'] < job['created'])
    return {**job, 'queue_ahead': ahead if job['status'] == 'queued' else 0}


@app.get('/api/jobs/{job_id}/result')
def job_result(job_id: str):
    path = JOBS / job_id / 'result.json'
    if job_id not in jobs or not path.exists():
        raise HTTPException(404, '결과가 아직 없습니다.')
    return FileResponse(path, media_type='application/json')


@app.get('/api/jobs/{job_id}/video')
def job_video(job_id: str):
    path = JOBS / job_id / 'preview.mp4'
    if job_id not in jobs or not path.exists():
        raise HTTPException(404, '영상이 아직 없습니다.')
    return FileResponse(path, media_type='video/mp4')


app.mount('/', StaticFiles(directory=STATIC, html=True), name='static')
