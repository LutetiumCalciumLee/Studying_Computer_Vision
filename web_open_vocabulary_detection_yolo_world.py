"""YOLO-World Flask 웹캠 스트림·이미지 업로드 객체 탐지.

웹캠 프레임은 좌우 반전 후 탐지하며, 박스와 글자는 정상 방향으로 표시합니다.

모델:
    기본 가중치: 이 파일 옆 00_Models/yolov8x-worldv2.pt.
    Git LFS로 받은 실제 가중치가 필요합니다. 포인터만 있으면 git lfs pull을 실행하세요.
    --model 또는 YOLO_WORLD_MODEL 환경변수로 다른 가중치를 지정할 수 있습니다.
    상대 모델 경로는 이 파일의 폴더 기준입니다.
    영어 객체 이름을 set_classes()로 적용하며, 최초 사용 시 CLIP 가중치도 다운로드합니다.

준비 (Conda vision_ai / Python 3.12에서 확인):
    conda activate vision_ai
    PyTorch와 torchvision은 GPU·드라이버에 맞는 CUDA 지원 빌드로 별도 설치합니다.
    CUDA Toolkit 설치만으로 CPU 전용 PyTorch가 GPU를 사용하지는 않습니다.
    python -m pip install ultralytics opencv-python numpy pillow flask
    python -m pip install "https://github.com/ultralytics/CLIP/archive/refs/heads/main.zip"
    모델 최초 다운로드에는 인터넷 연결이 필요합니다.

실행 (이 파일이 있는 폴더에서):
    python web_open_vocabulary_detection_yolo_world.py
    python web_open_vocabulary_detection_yolo_world.py --classes "person, cup, head" --device cuda
    브라우저: http://127.0.0.1:5000 (포트 변경: --port 5002)
    기본 실행은 클래스가 비어 있으며, 웹에서 객체 적용 후 웹캠 시작 또는 이미지 업로드.
    --classes 지정 시 시작할 때 해당 탐지 대상을 설정합니다.
    입력을 비우고 적용하면 추론을 중지합니다. 웹캠 스트림은 자동 종료되지 않습니다.
    이미지 결과는 화면에 반환하며 파일로 저장하지 않습니다. 요청 용량 제한: 10MiB.
    한 서버의 카메라 스트림은 한 개만 허용합니다. 서버 종료: Ctrl+C.

장치·카메라:
    --device auto(기본): CUDA를 사용할 수 있으면 GPU, 아니면 CPU.
    --device cuda: CUDA를 사용할 수 없으면 진단 메시지와 함께 중단.
    --camera 1: 다른 카메라 선택. 카메라를 사용하는 프로그램은 하나씩 실행하세요.
    --conf: 탐지 신뢰도 임계값(0~1). 전체 옵션: python web_open_vocabulary_detection_yolo_world.py --help

FPS 표시:
    카메라 처리 루프의 프레임 간격으로 계산합니다.
    캡처·추론·표시/전송 대기의 영향을 받으며, 순수 추론 FPS와 다릅니다.
    화면에 표시되는 서버 루프 FPS는 브라우저에서 실제 재생되는 FPS와 다릅니다.
    첫 프레임과 초기 워밍업 구간은 모델 초기화 때문에 느릴 수 있습니다.

출처:
    원본 실습: https://github.com/Tory-Hwang/05_Open_Vocabulary_Object_Detection
"""
import argparse
import os
import time
from pathlib import Path
from threading import Lock, RLock

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = BASE_DIR / "00_Models" / "yolov8x-worldv2.pt"


def parse_classes(text):
    """쉼표로 구분한 객체 이름에서 공백·빈 항목·중복을 제거합니다."""
    return list(dict.fromkeys(name.strip() for name in text.split(",") if name.strip()))


def select_device(torch, requested="auto"):
    """CUDA 사용 가능 여부로 장치를 선택하며, cuda를 명시한 경우 CPU로 대체하지 않습니다."""
    available = torch.cuda.is_available()
    if requested == "cuda" and not available:
        raise RuntimeError(
            "CUDA를 사용할 수 없습니다. CUDA용 PyTorch와 NVIDIA 드라이버를 확인하세요. "
            f"PyTorch={torch.__version__}, wheel CUDA={torch.version.cuda}"
        )
    return ("cuda" if available else "cpu") if requested == "auto" else requested


TITLE = 'Real Time Object Detection (YOLO-World)'
BACKEND = 'YOLO-World'
DEFAULT_CLASSES = 'person, cup, head'
DEFAULT_CONFIDENCE = 0.25
DEFAULT_PORT = 5000


def resolve_model_path(model_file=None):
    """동봉된 모델 경로와 Git LFS 다운로드 상태를 확인합니다."""
    path = Path(model_file or os.environ.get("YOLO_WORLD_MODEL") or DEFAULT_MODEL).expanduser()
    if not path.is_absolute():
        path = BASE_DIR / path
    if not path.is_file():
        raise FileNotFoundError(
            f"모델 파일이 없습니다: {path}\n"
            "저장소를 Git LFS로 clone하고 저장소 폴더에서 git lfs pull을 실행하세요. "
            "다른 모델은 --model로 지정할 수 있습니다."
        )
    with path.open("rb") as stream:
        prefix = stream.read(128)
    if prefix.startswith(b"version https://git-lfs.github.com/spec/v1"):
        raise ValueError(
            f"실제 모델 대신 Git LFS 포인터 파일이 있습니다: {path}\n"
            "Git LFS 설치 후 저장소 폴더에서 git lfs pull을 실행하세요. "
            "GitHub ZIP 다운로드에 모델이 포함되지 않았다면 git clone으로 받아야 합니다."
        )
    return path


class Detector:
    """텍스트로 탐지 대상을 설정하는 YOLO-World 탐지기."""

    def __init__(self, classes=DEFAULT_CLASSES, model_file=None, device="auto", confidence=DEFAULT_CONFIDENCE):
        path = resolve_model_path(model_file)
        import torch
        from ultralytics import YOLOWorld

        try:
            import clip  # noqa: F401 -- YOLO-World의 텍스트 인코더에 필요
        except ImportError as error:
            raise ImportError(
                'CLIP을 설치하세요: python -m pip install '
                '"https://github.com/ultralytics/CLIP/archive/refs/heads/main.zip"'
            ) from error
        self.device = select_device(torch, device)
        self.confidence = confidence
        self.lock = RLock()
        self.classes = []
        print(f"Using device: {self.device} / PyTorch: {torch.__version__}")
        self.model = YOLOWorld(str(path)).to(self.device)
        self.set_classes(classes)

    def set_classes(self, text):
        """텍스트 특징을 갱신하고 이전 predictor를 초기화합니다. 빈 입력은 추론을 중지합니다."""
        classes = parse_classes(text)
        with self.lock:
            if classes:
                self.model.set_classes(classes)
                self.model.predictor = None
            self.classes = classes
            return list(classes)

    def detect(self, frame):
        """BGR 프레임에 탐지 결과를 그려 반환합니다. 클래스가 비어 있으면 추론하지 않습니다."""
        with self.lock:
            if not self.classes:
                return frame.copy()
            result = self.model.predict(
                frame, device=self.device, conf=self.confidence, verbose=False
            )[0]
            return result.plot()


def camera_frames(detector, camera=0):
    """웹캠 프레임을 좌우 반전하고 탐지·FPS를 표시합니다. 종료 시 카메라를 해제합니다."""
    import cv2

    cap = cv2.VideoCapture(camera)
    try:
        if not cap.isOpened():
            raise RuntimeError(f"웹캠 {camera}번을 열 수 없습니다. 권한, 카메라 번호와 다른 앱의 사용 여부를 확인하세요.")
        previous = time.perf_counter()
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            annotated = detector.detect(frame)
            current = time.perf_counter()
            fps = 1 / max(current - previous, 1e-9)
            previous = current
            cv2.putText(annotated, f"FPS: {fps:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            yield annotated
    finally:
        cap.release()


def build_parser():
    """이 실행 파일에서 사용하는 옵션과 기본값을 정의합니다."""
    parser = argparse.ArgumentParser(description=TITLE,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--classes", default="", help="쉼표로 구분한 영어 객체 이름 (예: person, cup, head)")
    parser.add_argument("--camera", type=int, default=int(os.environ.get("CAMERA_INDEX", "0")),
                        help="카메라 번호 (환경변수: CAMERA_INDEX)")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto",
                        help="auto는 CUDA가 없으면 CPU 사용; cuda는 사용 불가 시 중단")
    parser.add_argument("--model", help="YOLO 가중치 경로 (상대 경로는 이 파일 기준, 기본: 00_Models/yolov8x-worldv2.pt)")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONFIDENCE, help="탐지 신뢰도 임계값 (0~1)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="로컬 웹 서버 포트")
    return parser


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{{ backend }} 객체 탐지</title>
  <style>
    body { font-family: sans-serif; margin: 0; background: #f4f4f4; text-align: center; color: #222; }
    h1 { background: #007bff; color: white; padding: 24px; margin: 0; font-size: 1.6rem; }
    main { max-width: 960px; margin: auto; padding: 20px; }
    form, section { margin: 20px auto; padding: 20px; background: white; border-radius: 8px; }
    input, button { padding: 10px; margin: 5px; max-width: 100%; box-sizing: border-box; }
    input[type="text"] { width: 350px; }
    button { cursor: pointer; }
    img { display: block; margin: 20px auto; max-width: 100%; border-radius: 8px; }
    [hidden] { display: none !important; }
    #status { min-height: 24px; overflow-wrap: anywhere; }
  </style>
</head>
<body>
  <h1>{{ backend }} 객체 탐지</h1>
  <main>
    {% if backend == 'DETR' %}
    <p>DETR-ResNet50 비교 실습: person, cup, car 등 학습된 COCO 클래스만 탐지합니다.</p>
    {% else %}
    <p>영어 객체 이름을 쉼표로 구분해 입력하세요. 예: person, cup, car</p>
    {% endif %}
    <form id="detect-form">
      <label for="desired_classes">탐지할 객체</label><br>
      <input type="text" id="desired_classes" name="desired_classes" placeholder="person, cup">
      <button type="submit">객체 적용</button>
      <p>입력란을 비우고 적용하면 탐지를 중지합니다.</p>
    </form>
    <p id="status" role="status" aria-live="polite">먼저 탐지할 객체를 적용하세요.</p>
    <section>
      <button id="start-camera" type="button">웹캠 시작</button>
      <button id="stop-camera" type="button">웹캠 중지</button>
      <img id="video-feed" alt="웹캠 탐지 화면" hidden>
    </section>
    <form id="upload-form" enctype="multipart/form-data">
      <label for="image">이미지 업로드 (최대 10MB)</label><br>
      <input type="file" id="image" name="image" accept="image/*" required>
      <button type="submit">업로드 및 탐지</button>
      <img id="upload-result" alt="이미지 탐지 결과" hidden>
    </form>
  </main>
  <script>
    const status = document.getElementById('status');
    const feed = document.getElementById('video-feed');
    async function postForm(event, url, onSuccess) {
      event.preventDefault();
      const button = event.currentTarget.querySelector('button');
      button.disabled = true;
      status.textContent = '처리 중입니다. 최초 실행은 모델 다운로드에 시간이 걸릴 수 있습니다.';
      try {
        const response = await fetch(url, { method: 'POST', body: new FormData(event.currentTarget) });
        const data = await response.json();
        if (!response.ok || data.status !== 'success') throw new Error(data.message || '요청 실패');
        onSuccess(data);
      } catch (error) {
        status.textContent = error.message;
      } finally {
        button.disabled = false;
      }
    }
    document.getElementById('detect-form').addEventListener('submit', event => {
      postForm(event, '/update_classes', data => {
        status.textContent = data.classes.length ? '탐지할 객체: ' + data.classes.join(', ') : '탐지를 중지했습니다.';
      });
    });
    document.getElementById('upload-form').addEventListener('submit', event => {
      postForm(event, '/upload_image', data => {
        const result = document.getElementById('upload-result');
        result.src = data.url;
        result.hidden = false;
        status.textContent = '이미지 처리가 완료되었습니다.';
      });
    });
    document.getElementById('start-camera').addEventListener('click', () => {
      if (!feed.hidden) return;
      feed.src = '/video_feed?' + Date.now();
      feed.hidden = false;
    });
    document.getElementById('stop-camera').addEventListener('click', () => {
      feed.removeAttribute('src');
      feed.hidden = true;
    });
    feed.addEventListener('error', () => {
      status.textContent = '웹캠 연결 실패. 터미널 오류, 카메라 권한과 번호를 확인하세요.';
      feed.removeAttribute('src');
      feed.hidden = true;
    });
  </script>
</body>
</html>

"""


def create_app(detector=None, camera=0, model_file=None, device="auto", confidence=DEFAULT_CONFIDENCE):
    """내장 화면과 API를 구성합니다. 미리 주입한 탐지기가 없으면 첫 작업 때 모델을 로드합니다."""
    import base64
    from flask import Flask, Response, jsonify, render_template_string, request

    app = Flask(__name__, static_folder=None)
    app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024
    detector_lock = Lock()
    camera_lock = Lock()

    def get_detector():
        nonlocal detector
        with detector_lock:
            if detector is None:
                detector = Detector("", model_file, device, confidence)
            return detector

    @app.get("/")
    def index():
        return render_template_string(HTML_TEMPLATE, backend=BACKEND)

    @app.post("/update_classes")
    def update_classes():
        try:
            classes = get_detector().set_classes(request.form.get("desired_classes", ""))
            return jsonify(status="success", classes=classes)
        except ValueError as error:
            return jsonify(status="error", message=str(error)), 400
        except Exception as error:
            app.logger.exception("Class update failed")
            return jsonify(status="error", message=str(error)), 500

    @app.get("/video_feed")
    def video_feed():
        try:
            current_detector = get_detector()
        except Exception as error:
            app.logger.exception("Model initialization failed")
            return jsonify(status="error", message=str(error)), 500
        if not camera_lock.acquire(blocking=False):
            return jsonify(status="error", message="웹캠 스트림이 이미 열려 있습니다."), 409
        # generator와 Response 양쪽에서 종료되어도 같은 스트림의 Lock은 한 번만 해제.
        cleanup_lock = Lock()
        released = False

        def cleanup():
            nonlocal released
            with cleanup_lock:
                if not released:
                    released = True
                    camera_lock.release()

        def generate():
            import cv2

            frames = camera_frames(current_detector, camera)
            try:
                for frame in frames:
                    ok, buffer = cv2.imencode(".jpg", frame)
                    if ok:
                        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
            finally:
                frames.close()
                cleanup()

        response = Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")
        response.call_on_close(cleanup)
        return response

    @app.post("/upload_image")
    def upload_image():
        file = request.files.get("image")
        if file is None or not file.filename:
            return jsonify(status="error", message="이미지 파일이 없습니다."), 400
        data = file.read()
        if not data:
            return jsonify(status="error", message="빈 이미지 파일입니다."), 400
        import cv2
        import numpy as np

        try:
            image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                return jsonify(status="error", message="이미지를 읽을 수 없습니다."), 400
            result = get_detector().detect(image)
            ok, buffer = cv2.imencode(".jpg", result)
            if not ok:
                raise RuntimeError("이미지 인코딩 실패")
            url = "data:image/jpeg;base64," + base64.b64encode(buffer.tobytes()).decode("ascii")
            return jsonify(status="success", url=url)
        except Exception as error:
            app.logger.exception("Image detection failed")
            return jsonify(status="error", message=str(error)), 500

    @app.errorhandler(413)
    def upload_too_large(_error):
        return jsonify(status="error", message="업로드 용량은 10MB 이하로 제한됩니다."), 413

    return app


def main():
    """CLI 설정을 적용하고 로컬 Flask 서버를 실행합니다."""
    parser = build_parser()
    args = parser.parse_args()
    if not 0 <= args.conf <= 1:
        parser.error("--conf는 0과 1 사이여야 합니다.")
    detector = None
    if parse_classes(args.classes):
        detector = Detector(args.classes, args.model, args.device, args.conf)
    app = create_app(detector, args.camera, args.model, args.device, args.conf)
    print(f"브라우저에서 http://127.0.0.1:{args.port} 를 여세요.")
    app.run(host="127.0.0.1", port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
