"""DETR-ResNet50 OpenCV 웹캠 객체 탐지.

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
    python -m pip install transformers timm opencv-python numpy pillow
    모델 최초 다운로드에는 인터넷 연결이 필요합니다.

실행 (이 파일이 있는 폴더에서):
    python open_vocabulary_detection_ResNet.py
    python open_vocabulary_detection_ResNet.py --classes "person, cup" --device cuda
    기본 탐지 대상: person, cup. --classes ""이면 추론 없이 카메라 화면을 표시합니다.
    종료: 창을 선택한 뒤 q. 종료·오류 발생 시 카메라 자원을 해제합니다.

장치·카메라:
    --device auto(기본): CUDA를 사용할 수 있으면 GPU, 아니면 CPU.
    --device cuda: CUDA를 사용할 수 없으면 진단 메시지와 함께 중단.
    --camera 1: 다른 카메라 선택. 카메라를 사용하는 프로그램은 하나씩 실행하세요.
    --conf: 탐지 신뢰도 임계값(0~1). 전체 옵션: python open_vocabulary_detection_ResNet.py --help

DETR 속도 설정:
    기본 입력의 짧은 변: 480px. CUDA에서 FP16 혼합 정밀도 및 지원되는 GPU 전처리 사용.
    python open_vocabulary_detection_ResNet.py --device cuda --imgsz 320
    기존 입력 크기·FP32 비교: --imgsz 800 --no-half. CPU에서는 FP16을 사용하지 않습니다.
    입력 크기는 처리 속도와 작은 객체 탐지 정확도에 영향을 줍니다.

FPS 표시:
    카메라 처리 루프의 프레임 간격으로 계산합니다.
    캡처·추론·표시/전송 대기의 영향을 받으며, 순수 추론 FPS와 다릅니다.
    첫 프레임과 초기 워밍업 구간은 모델 초기화 때문에 느릴 수 있습니다.

출처:
    원본 실습: https://github.com/Tory-Hwang/05_Open_Vocabulary_Object_Detection
"""
import argparse
import os
import time
from threading import RLock

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
DEFAULT_CLASSES = 'person, cup'
DEFAULT_CONFIDENCE = 0.5


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
    parser.add_argument("--classes", default=DEFAULT_CLASSES, help="쉼표로 구분한 COCO 클래스 이름 (예: person, cup)")
    parser.add_argument("--camera", type=int, default=int(os.environ.get("CAMERA_INDEX", "0")),
                        help="카메라 번호 (환경변수: CAMERA_INDEX)")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto",
                        help="auto는 CUDA가 없으면 CPU 사용; cuda는 사용 불가 시 중단")
    parser.add_argument("--model", help="Hugging Face 모델 ID 또는 로컬 모델 폴더 (기본: facebook/detr-resnet-50)")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONFIDENCE, help="탐지 신뢰도 임계값 (0~1)")
    parser.add_argument("--imgsz", type=int, default=480, help="DETR 입력의 짧은 변: 작을수록 빠름 (기존 크기: 800)")
    parser.add_argument("--no-half", action="store_true", help="CUDA FP16 추론 끄기 (정밀도 비교용)")
    return parser


def main():
    """CLI 설정을 적용하고 웹캠 탐지 창을 실행합니다."""
    parser = build_parser()
    args = parser.parse_args()
    if not 0 <= args.conf <= 1:
        parser.error("--conf는 0과 1 사이여야 합니다.")
    if args.imgsz < 128:
        parser.error("--imgsz는 128 이상이어야 합니다.")
    import cv2

    detector = Detector(args.classes, args.model, args.device, args.conf, args.imgsz, not args.no_half)
    frames = camera_frames(detector, args.camera)
    try:
        for frame in frames:
            cv2.imshow(TITLE, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        frames.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
