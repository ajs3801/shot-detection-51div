# 표적지 polygon + 탄착 bbox YOLO


## 1. 파일 구성

```text
train.py       train split으로 학습, val split으로 학습 상태 확인 / best.pt 선택
test.py        test split에서 표적지 + 탄착 성능 평가
run.py         영상 입력 → 표적지 예측 → polygon crop → 탄착 예측 → 안정화 → 시각화 MP4
run_cpu.py     같은 영상 파이프라인을 CPU에서만 실행
config.json    데이터 경로, 모델, confidence, 학습 및 안정화 옵션
models/        기존 학습을 완료한 YOLO26s-seg / YOLO26n 가중치
weights/       새 학습의 시작점인 pretrained 가중치
src/           공통 geometry, prediction, metrics, tracking, visualization, video 코드
src/crop_geometry.json
               기존 데이터의 GT crop 원점 정보. 일부 초기 crop의 padding 차이를 보존함
data/target_dataset/.gitkeep  표적지 데이터셋 자리
data/bullet_dataset/.gitkeep  탄착 데이터셋 자리
checks/        추적 및 평가 동작 검사
bin/ffmpeg     선택적으로 둘 수 있는 로컬 H.264 인코더
outputs/       새 학습, 테스트, 영상 결과가 생성되는 폴더
```

기존 yolo_two_stage_20261104 폴더나 기존 영상 결과를 변경하지 않는 별도 실행 프로젝트입니다.
학습 완료 모델로 학습을 다시 하지 않고도 `test.py`와 `run.py`를 실행할 수 있습니다. `models/`와 `weights/`의 `.pt` 가중치를 코드와 함께 Git에서 관리합니다.
외부 labeling_review/manifest.json이나 이전 predictions.jsonl에 의존하지 않습니다.

## GitHub 업로드 구성

| 항목 | 용도 | Git에 포함 |
|---|---|---|
| `train.py`, `test.py`, `run.py`, `src/` | 실행 코드 | 포함 |
| `config.json`, `requirements.txt`, `README.md`, `.gitignore` | 설정과 실행 안내 | 포함 |
| `src/crop_geometry.json`, `weights_sha256.json` | GT crop 좌표 및 가중치 식별 정보 | 포함 |
| `data/target_dataset/`, `data/bullet_dataset/` | 실제 데이터셋 위치 | `.gitkeep`만 포함. 이미지·라벨·data.yaml은 제외 |
| `checks/` | 동작을 검증하는 자동 검사 **소스 코드** | 포함 권장. 학습·추론 실행에는 불필요 |
| `.cache/`, `__pycache__/` | 라이브러리 캐시·Python 컴파일 파일 | 제외. 자동 재생성 |
| `bin/` | 로컬 FFmpeg 실행 파일 | 제외. 설치된 인코더로 대체 가능 |
| `outputs/` | 생성한 모델·평가 지표·영상·실행 검사 결과 | 제외 |
| `models/*.pt`, `weights/*.pt` | 학습 완료·사전 학습 모델 바이너리 | 포함 |

위 제외 규칙을 `.gitignore`에 설정했습니다. 현재 서버의 파일은 유지합니다.
이미 Git에 추적 중인 파일에는 `.gitignore`가 소급 적용되지 않습니다.

`checks/`와 `outputs/checks/`는 다릅니다. 전자는 Git에 포함할 검사 코드이고, 후자는 제외할 실행 결과입니다.

## 2. 환경

현재 서버에서는 /opt/conda/bin/python으로 실행합니다.
새 환경에서는 GPU에 맞는 PyTorch / torchvision을 먼저 설치한 뒤:

```bash
python -m pip install -r requirements.txt
```

config.json의 상대 경로는 config 파일 위치를 기준으로 해석합니다.
--source, --output 등 CLI의 상대 경로는 현재 작업 디렉터리 기준입니다.
기본 데이터셋 위치는 프로젝트 내부의 `data/target_dataset`, `data/bullet_dataset`입니다.
저장소에는 각 폴더의 `.gitkeep`만 포함합니다. 학습·평가 전 실제 데이터셋을 해당 위치에 준비합니다.

실제 데이터를 넣은 뒤의 구조:

```text
data/
├── target_dataset/
│   ├── .gitkeep
│   ├── data.yaml
│   ├── images/{train,val,test}/
│   └── labels/{train,val,test}/
└── bullet_dataset/
    ├── .gitkeep
    ├── data.yaml
    ├── images/{train,val,test}/
    └── labels/{train,val,test}/
```

각 폴더에 데이터셋 자체의 내용물을 넣습니다. `data/target_dataset/target_dataset/`처럼 한 단계 더 중첩하지 않습니다.
GT crop 원점 정보는 `src/crop_geometry.json`에서 읽습니다.
다른 위치로 옮기면 config.json의 datasets 경로를 수정합니다.
device=auto는 GPU가 2개이면 target=0 / bullet=1, 1개이면 둘 다 0, GPU가 없으면 CPU입니다.
--device 0 또는 --device cpu로 두 모델을 같은 장치에 지정할 수 있습니다.

## 인코더와 가중치 준비

영상 저장에는 **libx264를 지원하는 FFmpeg**가 필요합니다. 코드는 다음 순서로 사용 가능한 실행 파일을 찾습니다.

1. `IMAGEIO_FFMPEG_EXE` 환경 변수
2. 로컬 `bin/ffmpeg`가 있으면 해당 파일
3. `requirements.txt`의 `imageio-ffmpeg` 패키지가 제공하는 실행 파일
4. 시스템 `PATH`의 `ffmpeg`

따라서 `bin/`을 GitHub에 올릴 필요는 없습니다. 의존성을 설치하거나 실행 파일을 지정하면 됩니다.

```bash
export IMAGEIO_FFMPEG_EXE=/path/to/ffmpeg
```

기본 설정에서 사용하는 가중치 경로는 다음과 같습니다. 이 `.pt` 파일들은 Git에 포함할 수 있도록 설정했습니다. 다른 위치의 파일은 `config.json`이나 `--target-weights`, `--bullet-weights`로 지정할 수 있습니다.

```text
models/target_yolo26s_seg.pt   # test/run: 학습 완료 표적지 모델
models/bullet_yolo26n.pt       # test/run: 학습 완료 탄착 모델
weights/yolo26s-seg.pt         # train: 사전 학습 표적지 모델
weights/yolo26n.pt             # train: 사전 학습 탄착 모델
```

`models/`, `weights/`의 `.gitkeep`은 빈 폴더 구조를 유지하는 파일입니다. 두 폴더의 `.pt` 파일은 Git에 포함하며, `outputs/`에 생성되는 학습 중간 결과는 계속 제외합니다.

## 3. 학습: train.py

```bash
cd target_bullet_yolo
```

```bash
python train.py --dry-run
python train.py --task all --output outputs/train/exp01
```

표적지 또는 탄착만 학습:

```bash
python train.py --task target --output outputs/train/target01
python train.py --task bullet --output outputs/train/bullet01
```

옵션 예시:

```bash
python train.py --task all --epochs 200 --device 0 --workers 2 --output outputs/train/exp02
```

실제 optimizer 학습 입력은 images/train입니다. images/val은 학습 모니터링 / early stopping /
best.pt 선택용입니다. train.py는 test split 성능을 보면서 학습하거나 threshold를 조정하지 않습니다.
target와 bullet은 순서대로 학습합니다. 기본 학습 옵션은 기존 학습 설정과 동일합니다.
--target-weights / --bullet-weights를 주면 해당 가중치에서 새 학습을 시작할 수 있습니다.

학습 결과:

```text
outputs/train/exp01/training/target/weights/best.pt
outputs/train/exp01/training/bullet/weights/best.pt
outputs/train/exp01/models/target.pt
outputs/train/exp01/models/bullet.pt
outputs/train/exp01/config.json
```

생성된 config.json은 새 모델 경로를 가리킵니다. 후속 test/run에 반드시 이 config를 전달하면 됩니다.
한 종류만 학습하면 다른 모델은 입력 config의 가중치를 유지합니다.
기존 models/ 가중치는 학습 실행으로 덮어쓰지 않습니다.

## 4. 테스트: test.py

기존 학습 완료 모델 평가:

```bash
python test.py --output outputs/test/existing_models
```

새로 학습한 모델 평가:

```bash
python test.py --config outputs/train/exp01/config.json --output outputs/test/exp01
```

test.py는 test split만 평가하며 다음을 저장합니다.

1. 표적지 polygon: 검출 수, polygon IoU, corner 오차 등
2. bullet_GT_crop: 기존 GT polygon crop 이미지에서의 탄착 성능
3. bullet_predicted_polygon_crop: 예측 polygon으로 새로 crop한 이미지에서의 탄착 성능
4. Ultralytics native target / bullet test mAP

target와 bullet의 test 이미지 stem이 같은지 검사합니다.
두 탄착 평가 모두 원본 전체 프레임 좌표계에서 같은 GT와 비교합니다.
표적지 검출 실패 이미지를 제외하지 않습니다. 해당 탄착 GT는 false negative로 계산합니다.
탄착이 없는 test 이미지도 유지해 false positive를 평가합니다.
독립적인 test 이미지에는 시간 순서가 없으므로 optical flow / 탄착 추적을 적용하지 않습니다.
confidence는 config의 고정값을 사용하며 test 결과로 재조정하지 않습니다.
현재 기본값 target=.25 / bullet=.45는 기존 val split에서 선택한 값입니다.
새 모델에 대한 threshold 재보정은 자동으로 하지 않습니다.

주요 결과:

```text
outputs/test/exp01/summary.json
outputs/test/exp01/comparison.csv
outputs/test/exp01/per_image.json
outputs/test/exp01/visualizations/*.jpg
outputs/test/exp01/predicted_crops/*.png
outputs/test/exp01/plots/
outputs/test/exp01/native/
```

추가 native 평가만 생략하려면 --skip-native를 사용합니다.

## 5. 영상 추론: run.py

CPU 전용 실행은 `run_cpu.py`를 사용합니다. 설정 파일이나 CLI의 device 값과 관계없이
두 모델 모두 CPU로 고정하며, 나머지 옵션은 `run.py`와 같습니다.

```bash
python run_cpu.py --source /workspace/KIPOT/군학AI/video/clip_02_02.mp4 --output outputs/video/clip_02_02_cpu.mp4
```

기존 학습 완료 모델 + 기본 안정화 사용:

```bash
python run.py --source ../video/clip_02_02.mp4 --output outputs/video/clip_02_02.mp4
```

새로 학습한 모델 사용:

```bash
python run.py --source ../video/clip_02_02.mp4 --config outputs/train/exp01/config.json --output outputs/video/exp01.mp4
```

안정화 없이 프레임별 예측:

```bash
python run.py --source ../video/clip_02_02.mp4 --no-temporal --output outputs/video/clip_02_02_raw.mp4
```

빠른 실행 확인 (90프레임만 처리):

```bash
python run.py --source ../video/clip_02_02.mp4 --max-frames 90 --output outputs/video/check90.mp4
```

격자 기준 탄착군 시각화:

```bash
python run.py --config config.local.json --grouping --grid-cell-cm 1 \
  --source /workspace/KIPOT/군학AI/video/clip_02_02.mp4 \
  --output outputs/video/clip_02_02_gpu_grouping.mp4
```

`--grouping`은 확정된 탄착 중심들을 표적별로 누적하고, 각 탄착 사이의 거리(cm),
최대 탄착 간 거리, 최소 포괄원의 지름을 표시합니다. 표적 중심은 측정에 사용하지 않습니다.
분홍색 선은 최소 포괄원, 하늘색 선은 가장 먼 두 탄착을 연결합니다.
촬영 각도에 따라 실제 원이 원본 영상에서는 타원처럼 투영될 수 있습니다.
회색 점은 현재 검출되지 않지만 같은 표적에서 이전에 확정된 탄착입니다.
표적지 추적이 초기화되면 누적 탄착과 격자 보정을 함께 초기화합니다.

cm 환산은 추가 학습 모델 없이 OpenCV로 인식한 격자 선의 반복 간격과 원근 변환을 사용합니다.
`--grid-cell-cm`은 실제 격자 한 칸의 길이이며 기본값은 1cm입니다.
현재 자동 보정은 샘플 표적처럼 중앙 무늬의 양옆에 반복 격자가 보이는 구조를 대상으로 합니다.
규칙적인 격자가 확인되지 않으면 cm 값을 표시하지 않고 보정 대기 상태를 표시합니다.
보정 후 흐린 프레임에서는 최근 보정값과 추적한 표적 위치를 사용합니다.
측정값은 검출 bbox 중심과 격자 보정에 기반한 추정값이며, 실제 탄착 중심과의 오차는 별도 검증이 필요합니다.
최소 포괄원의 지름은 가장 먼 두 탄착 사이 거리와 다를 수 있습니다.
전체 탄착 쌍의 거리와 보정 정보는 프레임별 `predictions.jsonl`의 `grouping`에 저장합니다.
이 옵션은 `run_cpu.py`에서도 사용할 수 있습니다.

--output 생략 시 outputs/video/<영상이름>_<시간>.mp4로 저장합니다.
기존 결과는 덮어쓰지 않으므로 재실행 시 새 output 이름을 사용합니다.
기본 출력은 원본 해상도 / FPS의 H.264 MP4이며, 원본 오디오가 있으면 복사합니다.
입력 크기가 홀수이면 인코더 호환을 위해 오른쪽/아래에 최대 1px를 추가합니다.
MP4에서 복사할 수 없는 오디오 코덱의 입력은 --no-audio로 실행할 수 있습니다.
출력 영상과 같은 이름의 폴더에 예측 JSONL, 처리 상태, 요약, 미리보기 이미지도 저장합니다.

```text
outputs/video/clip_02_02.mp4
outputs/video/clip_02_02/predictions.jsonl
outputs/video/clip_02_02/summary.json
outputs/video/clip_02_02/progress.json
outputs/video/clip_02_02/preview_montage.jpg
```

실행 시 두 모델을 직접 불러 매 프레임 추론합니다. 기존 예측 캐시나 GT는 사용하지 않습니다.
표적지 polygon bbox에 6px padding을 주어 crop하고, polygon 외부 픽셀은 정확히 (0,0,0)으로 마스킹합니다.

## 6. 기본 안정화

```text
표적지: half resolution (960x540 at 1920x1080 input) sparse LK optical flow + polygon EMA
flow: 최대 180개 특징점, 정방향/역방향 검사, RANSAC affine motion
표적지 모델 누락 유지: flow가 유효하면 최대 3초, flow도 불확실하면 짧게 freeze
탄착: 8프레임 중 confidence >= .45로 5회 확인 후 track 확정
이미 확정된 탄착: 낮은 confidence >= .20로도 이어가며, 검출 누락 시 최대 1.5초 유지
중복 박스 억제, 같은 탄착 track ID 유지, 큰 장면 변화나 추적 유실 시 track 초기화
노란 박스: 현재 프레임에서 갱신 / 주황 박스와 HOLD: 이전 관측을 유지 중
```

config.json의 tracking 옵션으로 수정할 수 있습니다.
안정화는 표시 끊김을 줄이며, 반복되는 오검출도 유지될 수 있습니다.
8프레임 중 5회 확인에 따른 관찰 지연은 optical flow 계산 비용과 별개입니다.

## 7. 검사

```bash
python -m unittest discover -s checks -v
python train.py --dry-run
```

검사 결과 및 실행 예제는 outputs/checks/ 하위에 저장합니다.
outputs/checks/train_smoke_utf8/는 실행 연결을 확인하기 위해 320px로 1 epoch만 학습한 결과입니다.
실제 예측용 모델은 기본 models/ 또는 별도로 충분히 학습한 outputs/train/.../models/를 사용합니다.
