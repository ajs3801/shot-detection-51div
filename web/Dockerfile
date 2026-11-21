# 저장소 루트에서 빌드합니다:
#   docker build -f web/Dockerfile -t shot-group-web .
FROM python:3.11-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY web/requirements.txt web/requirements.txt
# CPU용 PyTorch를 먼저 설치합니다. GPU 서버라면 index-url을 CUDA용으로 바꾸세요.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.7.1 torchvision==0.22.1 \
 && pip install --no-cache-dir -r web/requirements.txt

COPY target_bullet_yolo target_bullet_yolo
COPY web/server web/server
COPY web/static web/static

ENV SHOT_DEVICE=cpu \
    SHOT_JOBS=/data/jobs
VOLUME /data
EXPOSE 8000
WORKDIR /app/web

# 작업 상태를 메모리에 두므로 워커는 1개로 실행합니다.
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
