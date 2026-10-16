"""DETR-ResNet50 Flask 웹캠 스트림·이미지 업로드 객체 탐지.

웹캠 프레임은 좌우 반전 후 탐지하며, 박스와 글자는 정상 방향으로 표시합니다.

모델:
    facebook/detr-resnet-50을 최초 실행 시 다운로드해 Hugging Face 캐시에 보관합니다.
    COCO 고정 클래스 모델이며, 입력한 클래스 이름으로 탐지 결과를 필터링합니다.
    지원 클래스 예: person, cup, car.
    --model로 Hugging Face 모델 ID 또는 로컬 from_pretrained 폴더를 지정합니다.

준비 (Conda vision_ai / Python 3.12에서 확인):
    conda activate vision_ai
    PyTorch와 torchvision은 GPU·드라이버에 맞는 CUDA 지원 빌드로 별도 설치합니다.
    CUDA Toolkit 설치만으로 CPU 전용 PyTorch가 GPU를 사용하지는 않습니다.
    python -m pip install transformers timm opencv-python numpy pillow flask
    모델 최초 다운로드에는 인터넷 연결이 필요합니다.

실행 (이 파일이 있는 폴더에서):
    python web_open_vocabulary_detection_ResNet.py
    python web_open_vocabulary_detection_ResNet.py --classes "person, cup" --device cuda
    브라우저: http://127.0.0.1:5001 (포트 변경: --port 5002)
    기본 실행은 클래스가 비어 있으며, 웹에서 객체 적용 후 웹캠 시작 또는 이미지 업로드.
    --classes 지정 시 시작할 때 해당 탐지 대상을 설정합니다.
    입력을 비우고 적용하면 추론을 중지합니다. 웹캠 스트림은 자동 종료되지 않습니다.
    이미지 결과는 화면에 반환하며 파일로 저장하지 않습니다. 요청 용량 제한: 10MiB.
    한 서버의 카메라 스트림은 한 개만 허용합니다. 서버 종료: Ctrl+C.

장치·카메라:
    --device auto(기본): CUDA를 사용할 수 있으면 GPU, 아니면 CPU.
    --device cuda: CUDA를 사용할 수 없으면 진단 메시지와 함께 중단.
    --camera 1: 다른 카메라 선택. 카메라를 사용하는 프로그램은 하나씩 실행하세요.
    --conf: 탐지 신뢰도 임계값(0~1). 전체 옵션: python web_open_vocabulary_detection_ResNet.py --help

DETR 속도 설정:
    기본 입력의 짧은 변: 480px. CUDA에서 FP16 혼합 정밀도 및 지원되는 GPU 전처리 사용.
    python web_open_vocabulary_detection_ResNet.py --device cuda --imgsz 320
    기존 입력 크기·FP32 비교: --imgsz 800 --no-half. CPU에서는 FP16을 사용하지 않습니다.
    입력 크기는 처리 속도와 작은 객체 탐지 정확도에 영향을 줍니다.
    웹캠 JPEG 화질: 기본 80. --jpeg-quality 95로 변경(1~100).

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
from threading import Lock, RLock

DEFAULT_MODEL = "facebook/detr-resnet-50"


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


TITLE = 'Real Time Object Detection (DETR)'
BACKEND = 'DETR'
DEFAULT_CLASSES = 'person, cup'
DEFAULT_CONFIDENCE = 0.5
DEFAULT_PORT = 5001


class Detector:
    """COCO 클래스 필터와 해상도·정밀도 설정을 관리하는 DETR 탐지기."""

    def __init__(self, classes=DEFAULT_CLASSES, model_file=None, device="auto", confidence=DEFAULT_CONFIDENCE, imgsz=480, half=True):
        import importlib.util
        import torch

        if importlib.util.find_spec("timm") is None:
            raise ImportError("DETR 설치 패키지를 확인하세요: python -m pip install transformers timm")
        from transformers import AutoImageProcessor, DetrForObjectDetection, __version__ as transformers_version

        self.torch = torch
        self.device = select_device(torch, device)
        if imgsz < 128:
            raise ValueError("--imgsz는 128 이상이어야 합니다.")
        self.confidence = confidence
        self.half = bool(half and self.device == "cuda")
        self.image_size = {"shortest_edge": imgsz, "longest_edge": round(imgsz * 1333 / 800)}
        if self.device == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.set_float32_matmul_precision("high")
        self.lock = RLock()
        self.classes = []
        name = model_file or DEFAULT_MODEL
        print(f"Using device: {self.device} / PyTorch: {torch.__version__}")
        processor_options = (
            {"backend": "torchvision"} if int(transformers_version.split(".")[0]) >= 5
            else {"use_fast": True}
        )
        self.processor = AutoImageProcessor.from_pretrained(name, **processor_options)
        self.gpu_preprocess = self.device == "cuda" and (
            getattr(self.processor, "backend", None) == "torchvision"
            or getattr(self.processor, "is_fast", False)
        )
        self.model = DetrForObjectDetection.from_pretrained(name).to(self.device).eval()
        self.set_classes(classes)
        print(f"DETR: 입력 짧은 변 {imgsz}px / FP16={self.half} / GPU 전처리={self.gpu_preprocess}")

    def set_classes(self, text):
        """지원하는 COCO 클래스만 적용합니다. 빈 입력은 추론을 중지합니다."""
        classes = parse_classes(text)
        supported = set(self.model.config.id2label.values())
        unknown = [name for name in classes if name not in supported]
        if unknown:
            raise ValueError(
                "DETR의 고정 클래스에 없는 이름: " + ", ".join(unknown)
                + ". person, cup, car 등 학습된 COCO 클래스만 사용할 수 있습니다."
            )
        with self.lock:
            self.classes = classes
            return list(classes)

    def detect(self, frame):
        """BGR 프레임에 탐지 결과를 그려 반환합니다. 클래스가 비어 있으면 추론하지 않습니다."""
        import cv2

        with self.lock:
            if not self.classes:
                return frame.copy()
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            options = {"return_tensors": "pt", "size": self.image_size}
            if self.gpu_preprocess:
                options["device"] = self.device
            inputs = self.processor(images=rgb, **options).to(self.device)
            with self.torch.inference_mode(), self.torch.autocast(
                device_type=self.device, dtype=self.torch.float16, enabled=self.half
            ):
                outputs = self.model(**inputs)
            # 작은 출력 텐서만 FP32로 되돌려 신뢰도와 좌표 계산을 안정화합니다.
            outputs.logits = outputs.logits.float()
            outputs.pred_boxes = outputs.pred_boxes.float()
            # 공식 후처리: no-object 제외, 원본 해상도의 xyxy 좌표로 변환.
            result = self.processor.post_process_object_detection(
                outputs, threshold=self.confidence, target_sizes=[frame.shape[:2]]
            )[0]
            annotated = frame.copy()
            # 개별 box마다 GPU 동기화하지 않고 결과를 한 번씩 묶어서 CPU로 옮깁니다.
            scores = result["scores"].cpu().tolist()
            labels = result["labels"].cpu().tolist()
            boxes = result["boxes"].cpu().tolist()
            for score, label, box in zip(scores, labels, boxes):
                name = self.model.config.id2label[label]
                if name not in self.classes:
                    continue
                x1, y1, x2, y2 = (round(value) for value in box)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(annotated, f"{name}: {score:.2f}", (x1, max(y1 - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            return annotated


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
    parser.add_argument("--classes", default="", help="쉼표로 구분한 COCO 클래스 이름 (예: person, cup)")
    parser.add_argument("--camera", type=int, default=int(os.environ.get("CAMERA_INDEX", "0")),
                        help="카메라 번호 (환경변수: CAMERA_INDEX)")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto",
                        help="auto는 CUDA가 없으면 CPU 사용; cuda는 사용 불가 시 중단")
    parser.add_argument("--model", help="Hugging Face 모델 ID 또는 로컬 모델 폴더 (기본: facebook/detr-resnet-50)")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONFIDENCE, help="탐지 신뢰도 임계값 (0~1)")
    parser.add_argument("--imgsz", type=int, default=480, help="DETR 입력의 짧은 변: 작을수록 빠름 (기존 크기: 800)")
    parser.add_argument("--no-half", action="store_true", help="CUDA FP16 추론 끄기 (정밀도 비교용)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="로컬 웹 서버 포트")
    parser.add_argument("--jpeg-quality", type=int, default=80, help="웹캠 JPEG 화질 (1~100, 높을수록 용량 증가)")
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


def create_app(detector=None, camera=0, model_file=None, device="auto", confidence=DEFAULT_CONFIDENCE,
               imgsz=480, half=True, jpeg_quality=80):
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
                detector = Detector("", model_file, device, confidence, imgsz, half)
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
                    ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
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
    if args.imgsz < 128:
        parser.error("--imgsz는 128 이상이어야 합니다.")
    if not 1 <= args.jpeg_quality <= 100:
        parser.error("--jpeg-quality는 1과 100 사이여야 합니다.")
    detector = None
    if parse_classes(args.classes):
        detector = Detector(args.classes, args.model, args.device, args.conf, args.imgsz, not args.no_half)
    app = create_app(detector, args.camera, args.model, args.device, args.conf,
                     args.imgsz, not args.no_half, args.jpeg_quality)
    print(f"브라우저에서 http://127.0.0.1:{args.port} 를 여세요.")
    app.run(host="127.0.0.1", port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
