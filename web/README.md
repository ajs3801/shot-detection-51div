# 실내 사격장 자동 탄착군 분석 시스템: 웹페이지

결과 발표 페이지와, 새 영상을 올려 같은 모델로 분석하는 웹 서버입니다.

- 예시 영상(clip_01_04)의 분석 결과를 실시간 / 저장 영상 모드로 보여 줍니다.
- 서버로 열면 **내 영상 분석하기**에서 영상을 올려 분석할 수 있습니다. 분석이 끝나면 올린 영상 위에 표적지, 탄착, 탄착군 원이 그려지고 합격 여부가 나옵니다.

## 구성

```text
web/
├── server/app.py      웹 서버 (FastAPI). 페이지 제공 + 업로드 분석 API
├── static/
│   ├── index.html     발표 페이지
│   └── data.json      예시 영상의 프레임별 분석 결과 (실시간 / 저장 영상)
├── requirements.txt
└── Dockerfile
```

분석 엔진은 저장소의 `target_bullet_yolo/` 를 그대로 사용합니다 (가중치 `target_bullet_yolo/models/` 포함).

| 페이지 표기 | 엔진 프리셋 |
|---|---|
| 실시간 | `--preset budget27` |
| 저장 영상 | `--preset adaptive30` |

### 예시 영상

저장소와 서버에는 분석에서 나온 파생 결과(`web/static/data.json`, 좌표와 수치)만 둡니다. 원본 영상 `clip_01_04.mp4` 는 팀 비공개 Google Drive에만 보관합니다.

- 페이지는 기본으로 분석 결과(표적지, 탄착, 탄착군 표시와 판정 카드)만 보여 줍니다.
- 영상과 함께 보려면 페이지의 **PC에 있는 clip_01_04.mp4로 재생** 을 눌러 Drive에서 받은 파일을 고릅니다. 영상은 보는 사람의 브라우저 안에서만 재생되고 서버로 올라가지 않습니다.
- `web/static/` 에 영상을 넣지 마세요. 넣더라도 서버는 영상 파일(mp4, mov 등)을 내보내지 않고, Docker 이미지에는 `index.html` 과 `data.json` 만 들어갑니다.

## 실행 방법 1: Python (Windows PowerShell)

Python 3.10, 3.11, 3.12에서 동작합니다. 저장소를 OneDrive처럼 동기화되는 폴더 밖에 두는 편이 빠릅니다.

```powershell
cd shot-detection-51div\web
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
$env:PYTHONUTF8=1

pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.3,<3" "torchvision>=0.18,<1"
pip install -r requirements.txt

python -m uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 1
```

다음에 다시 켤 때는 `.venv\Scripts\Activate.ps1`, `$env:PYTHONUTF8=1`, 마지막 줄만 실행하면 됩니다.

## 실행 방법 2: Python (Linux, macOS)

```bash
cd shot-detection-51div/web
python -m venv .venv
source .venv/bin/activate

pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.3,<3" "torchvision>=0.18,<1"
pip install -r requirements.txt

python -m uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 1
```

브라우저에서 `http://서버주소:8000` 을 엽니다. GPU 서버라면 PyTorch를 CUDA용으로 설치하고 `SHOT_DEVICE=0` 으로 실행합니다.

## 실행 방법 3: Docker

저장소 루트에서 빌드합니다.

```bash
docker build -f web/Dockerfile -t shot-group-web .
docker run -d --name shot-group-web -p 8000:8000 -v shot-jobs:/data shot-group-web
```

## 환경 변수

| 이름 | 기본값 | 설명 |
|---|---|---|
| `SHOT_ENGINE` | `../target_bullet_yolo` | 분석 엔진 폴더 |
| `SHOT_DEVICE` | `cpu` | 분석 장치. GPU는 `0` |
| `SHOT_CPU_THREADS` | 4와 CPU 수 중 작은 값 | CPU 분석 스레드 수 |
| `SHOT_MAX_UPLOAD_MB` | `500` | 업로드 최대 크기 |
| `SHOT_KEEP_HOURS` | `24` | 올린 영상과 결과를 보관하는 시간. 지나면 자동 삭제 |
| `SHOT_JOBS` | `web/jobs` | 업로드 영상과 결과를 저장하는 폴더 |

## nginx 뒤에 둘 때

업로드 크기와 대기 시간을 늘려 주세요. 페이지는 상대 경로를 쓰므로 하위 경로(예: `/shot/`)에 두어도 됩니다.

```nginx
location /shot/ {
    proxy_pass http://127.0.0.1:8000/;
    client_max_body_size 600m;
    proxy_read_timeout 300s;
}
```

## 정적 호스팅만 할 때

`web/static/` 폴더만 GitHub Pages, Netlify 같은 곳에 올려도 페이지와 예시 결과는 동작합니다. 이때는 분석 서버가 없으므로 **내 영상 분석하기**가 꺼진 상태로 보입니다. `index.html` 을 더블클릭해 열면 결과 데이터를 불러오지 못하니 반드시 웹 서버로 여세요.

## 분석 시간

2코어 CPU 서버에서 측정한 값입니다 (영상 확인 단계 포함).

| 영상 | 모드 | 전체 소요 |
|---|---|---|
| clip_01_04 (26.8초, 30fps) | 실시간 | 약 40초 |
| clip_03_04 (32.5초, 60fps) | 저장 영상 | 약 85초 |

분석은 한 번에 하나씩 순서대로 처리하고, 기다리는 작업은 페이지에 대기 순서로 표시됩니다.

## 알아 둘 점

- **엔진 환경**: `web/requirements.txt` 는 `target_bullet_yolo/requirements.txt` 를 그대로 불러 같은 환경(ultralytics 8.4.104)을 씁니다.
- **예시 결과와의 차이**: `static/data.json` 은 실험 PC에서 ultralytics 8.4.78로 만든 결과입니다. 같은 clip_01_04를 이 서버(8.4.104)에서 분석하면 4발 모두 찾고 합격(기준 5.5cm)이지만, 수치가 다릅니다.

  | clip_01_04 | data.json (8.4.78) | 서버 (8.4.104) |
  |---|---|---|
  | 실시간: 탄착 확정 시각 | 6.1 · 10.4 · 13.4 · 16.5초 | 10.5 · 13.5 · 14.7 · 16.7초 |
  | 실시간: 탄착군 지름 | 4.17cm | 4.30cm |
  | 저장 영상: 탄착 확정 시각 | 6.2 · 10.6 · 13.4 · 16.6초 | 6.2 · 10.5 · 13.4 · 16.6초 |
  | 저장 영상: 탄착군 지름 | 4.26cm | 4.22cm |
- **워커 1개**: 작업 상태를 서버 메모리에 두므로 `--workers 1` 로 실행하세요. 서버를 다시 켜면 끝난 작업의 결과는 남고, 진행 중이던 작업은 실패로 표시됩니다.
- **접근 제한**: 업로드에 로그인 기능이 없습니다. 외부에 공개할 때는 내부망에 두거나 nginx의 basic auth 같은 접근 제한을 걸어 주세요.
- **영상 형식**: 업로드한 영상은 서버에서 브라우저 재생용 H.264 mp4로 한 번 변환해 보여 줍니다. mp4를 권장합니다.

## API

| 메서드 | 경로 | 설명 |
|---|---|---|
| GET | `/api/health` | 서버 상태, 최대 업로드 크기 |
| POST | `/api/jobs` | `video`(파일), `mode`(`realtime` 또는 `saved`)로 분석 요청. `{id}` 반환 |
| GET | `/api/jobs/{id}` | 상태(`queued`, `preparing`, `analyzing`, `done`, `failed`)와 진행률 |
| GET | `/api/jobs/{id}/result` | 프레임별 결과 JSON |
| GET | `/api/jobs/{id}/video` | 재생용 영상 |
