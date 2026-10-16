"""YOLO-World OpenCV 웹캠 객체 탐지.

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
    python -m pip install ultralytics opencv-python numpy pillow
    python -m pip install "https://github.com/ultralytics/CLIP/archive/refs/heads/main.zip"
    모델 최초 다운로드에는 인터넷 연결이 필요합니다.

실행 (이 파일이 있는 폴더에서):
    python open_vocabulary_detection_yolo_world.py
    python open_vocabulary_detection_yolo_world.py --classes "person, cup, head" --device cuda
    기본 탐지 대상: person, cup, head. --classes ""이면 추론 없이 카메라 화면을 표시합니다.
    종료: 창을 선택한 뒤 q. 종료·오류 발생 시 카메라 자원을 해제합니다.

장치·카메라:
    --device auto(기본): CUDA를 사용할 수 있으면 GPU, 아니면 CPU.
    --device cuda: CUDA를 사용할 수 없으면 진단 메시지와 함께 중단.
    --camera 1: 다른 카메라 선택. 카메라를 사용하는 프로그램은 하나씩 실행하세요.
    --conf: 탐지 신뢰도 임계값(0~1). 전체 옵션: python open_vocabulary_detection_yolo_world.py --help

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
from pathlib import Path
from threading import RLock

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
DEFAULT_CLASSES = 'person, cup, head'
DEFAULT_CONFIDENCE = 0.25


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
    parser.add_argument("--classes", default=DEFAULT_CLASSES, help="쉼표로 구분한 영어 객체 이름 (예: person, cup, head)")
    parser.add_argument("--camera", type=int, default=int(os.environ.get("CAMERA_INDEX", "0")),
                        help="카메라 번호 (환경변수: CAMERA_INDEX)")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto",
                        help="auto는 CUDA가 없으면 CPU 사용; cuda는 사용 불가 시 중단")
    parser.add_argument("--model", help="YOLO 가중치 경로 (상대 경로는 이 파일 기준, 기본: 00_Models/yolov8x-worldv2.pt)")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONFIDENCE, help="탐지 신뢰도 임계값 (0~1)")
    return parser


def main():
    """CLI 설정을 적용하고 웹캠 탐지 창을 실행합니다."""
    parser = build_parser()
    args = parser.parse_args()
    if not 0 <= args.conf <= 1:
        parser.error("--conf는 0과 1 사이여야 합니다.")
    import cv2

    detector = Detector(args.classes, args.model, args.device, args.conf)
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
