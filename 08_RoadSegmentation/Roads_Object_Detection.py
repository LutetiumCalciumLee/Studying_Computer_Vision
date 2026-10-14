"""640x640 매 프레임 분석과 횡단보도 밖 차도 보행자의 DANGER 경고."""

import argparse
from collections import deque
from pathlib import Path
import time

import cv2
import numpy as np
import torch
from openvino.preprocess import PrePostProcessor
from openvino.runtime import Core, Layout, Model, Type, opset13 as ops
from ultralytics import YOLO

from road_danger import RoadDangerDetector, draw_danger_banner


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
MODELS_DIR = BASE_DIR / "00_Models"
VIDEOS_DIR = BASE_DIR / "00_Sample_Video"
OUTPUT_SIZE = (640, 640)
# 원본 코드와 동일한 BGR 색상: 배경, 도로, 연석, 표시.
PALETTE = np.array(
    [[0, 0, 0], [0, 0, 255], [0, 128, 0], [128, 128, 0]], dtype=np.uint8
)


class RoadSegmenter:
    """OpenVINO에서 클래스 선택까지 수행하고 CUDA 추론과 겹쳐 실행한다."""

    def __init__(self, model_path, cpu_threads=4, device="GPU"):
        model_path = Path(model_path)
        for path in (model_path, model_path.with_suffix(".bin")):
            if not path.is_file():
                raise FileNotFoundError(f"Road segmentation model not found: {path}")

        core = Core()
        if device.startswith("GPU") and not any(
            name.startswith("GPU") for name in core.available_devices
        ):
            raise RuntimeError("OpenVINO Intel GPU is unavailable. Check the Intel GPU driver.")
        model = core.read_model(str(model_path))
        _, channels, self.height, self.width = model.input(0).shape
        # numpy.argmax와 동일하게 동점일 때 첫 번째 클래스를 선택한다.
        top_class = ops.topk(
            model.get_results()[0].input_value(0), np.array(1, dtype=np.int32), axis=1,
            mode="max", sort="value", index_element_type="i32", stable=True,
        ).output(1)
        labels = ops.convert(top_class, Type.u8)
        label_model = Model([labels], model.get_parameters(), "road_class_mask")

        # 연속된 uint8 NHWC 입력을 공유하고 float32/NCHW 변환은 그래프에서 처리한다.
        preprocessing = PrePostProcessor(label_model)
        preprocessing.input().tensor().set_element_type(Type.u8).set_layout(
            Layout("NHWC")
        ).set_shape([1, self.height, self.width, channels])
        preprocessing.input().model().set_layout(Layout("NCHW"))
        preprocessing.input().preprocess().convert_element_type(Type.f32)
        label_model = preprocessing.build()

        config = {"PERFORMANCE_HINT": "LATENCY"}
        if device == "CPU":
            config["INFERENCE_NUM_THREADS"] = cpu_threads
        self.compiled_model = core.compile_model(label_model, device, config)
        self.execution_devices = self.compiled_model.get_property("EXECUTION_DEVICES")
        self.device_names = [
            core.get_property(name, "FULL_DEVICE_NAME") for name in self.execution_devices
        ]
        self.request = self.compiled_model.create_infer_request()
        self.input_data = None

    def start(self, frame):
        resized = cv2.resize(frame, (self.width, self.height))
        # 비동기 추론이 끝날 때까지 공유 입력의 수명을 유지한다.
        self.input_data = resized[np.newaxis]
        self.request.start_async([self.input_data], share_inputs=True)

    def mask(self):
        self.request.wait()
        return self.request.get_output_tensor(0).data[0, 0]

    def overlay(self, frame, mask=None):
        if mask is None:
            mask = cv2.resize(
                self.mask(), (frame.shape[1], frame.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        return cv2.addWeighted(frame, 0.7, PALETTE[mask], 0.3, 0)


class InferencePipeline:
    def __init__(
        self, seg_model, yolo_model, device="0", fp32=False, cpu_threads=4,
        seg_device="GPU", danger_enabled=True,
    ):
        # 전처리와 두 엔진이 CPU 스레드를 과도하게 점유하지 않도록 한다.
        cv2.setNumThreads(1)
        torch.set_num_threads(1)
        self.device = "0" if device == "auto" and torch.cuda.is_available() else device
        if self.device == "auto":
            self.device = "cpu"
        self.torch_device = torch.device(
            f"cuda:{self.device}" if str(self.device).isdigit() else self.device
        )
        if self.torch_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU is unavailable. Run with the vision_ai environment.")
        self.half = self.torch_device.type == "cuda" and not fp32
        self.segmenter = RoadSegmenter(seg_model, cpu_threads, seg_device)
        if not Path(yolo_model).is_file():
            raise FileNotFoundError(f"YOLO model not found: {yolo_model}")
        self.yolo = YOLO(str(yolo_model))
        self.yolo.to(self.torch_device)
        self.yolo_execution_device = next(self.yolo.model.parameters()).device
        self.yolo_device_name = (
            torch.cuda.get_device_name(self.yolo_execution_device)
            if self.yolo_execution_device.type == "cuda" else "CPU"
        )
        self.danger_detector = RoadDangerDetector() if danger_enabled else None
        self.last_assessment = None

    def process(self, frame):
        self.segmenter.start(frame)
        try:
            # 원본 영상에서 탐지한 박스를 도로 색상 오버레이 위에 그린다.
            # Intel GPU 분할은 이 CUDA 추론이 진행되는 동안 함께 실행된다.
            detections = self.yolo.predict(
                frame, imgsz=640, device=self.device, half=self.half, verbose=False
            )[0].cpu()
        except Exception:
            self.segmenter.request.wait()
            raise
        mask = cv2.resize(
            self.segmenter.mask(), (frame.shape[1], frame.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        self.last_assessment = (
            self.danger_detector.assess(frame, mask, detections.boxes.data.numpy())
            if self.danger_detector is not None else None
        )
        overlay = self.segmenter.overlay(frame, mask)
        output = detections.plot(img=overlay)
        if self.last_assessment is not None and self.last_assessment.danger:
            draw_danger_banner(output)
        return output


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("Value must be at least 1")
    return value


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", default=str(VIDEOS_DIR / "input" / "road.mp4"))
    parser.add_argument(
        "--deep_model1",
        default=str(MODELS_DIR / "road-segmentation-adas-0001" / "FP16-INT8"
                    / "road-segmentation-adas-0001.xml"),
    )
    parser.add_argument("--yolo_model", default=str(MODELS_DIR / "yolov8n.pt"))
    parser.add_argument(
        "--output", default=str(VIDEOS_DIR / "output" / "roads_object_danger_output.mp4")
    )
    parser.add_argument(
        "--vidfps", type=positive_int, default=None,
        help="Saved video playback FPS. Default: source FPS. Does not speed up inference.",
    )
    parser.add_argument("--yolo-device", default="0", help="CUDA index (0), cpu, or auto")
    parser.add_argument(
        "--seg-device", choices=("GPU", "GPU.0", "CPU"), default="GPU",
        help="OpenVINO device: GPU uses the Intel GPU, YOLO uses NVIDIA CUDA",
    )
    parser.add_argument("--fp32", action="store_true", help="Use full precision YOLO on CUDA")
    parser.add_argument(
        "--no-danger", action="store_true", help="Disable the DANGER subtitle",
    )
    parser.add_argument(
        "--cpu-threads", type=positive_int, default=4,
        help="Segmentation threads when --seg-device CPU is selected",
    )
    parser.add_argument("--no-display", action="store_true", help="Disable preview window")
    parser.add_argument("--no-save", action="store_true", help="Disable video encoding")
    parser.add_argument(
        "--max-frames", type=positive_int, default=None,
        help="Stop after this many input frames, including warmup frames",
    )
    parser.add_argument(
        "--warmup", type=int, default=10,
        help="Exclude the first N frames from FPS measurement (they are still processed)",
    )
    parser.add_argument(
        "--benchmark", action="store_true",
        help="Process 180 frames without preview or saving; --max-frames overrides length",
    )
    args = parser.parse_args()
    if args.warmup < 0:
        parser.error("--warmup must be nonnegative")
    if args.benchmark:
        args.no_display = args.no_save = True
        args.max_frames = args.max_frames or 180
    if args.max_frames is not None and args.max_frames <= args.warmup:
        parser.error("--max-frames must be greater than --warmup")
    return args


def run(args):
    pipeline = InferencePipeline(
        args.deep_model1, args.yolo_model, args.yolo_device, args.fp32,
        args.cpu_threads, args.seg_device, not args.no_danger,
    )
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open video: {args.video}")
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    output_fps = args.vidfps or (source_fps if source_fps > 0 else 30)
    writer = None
    processed = measured = danger_frames = 0
    measurement_start = measurement_end = None
    recent_times = deque(maxlen=30)
    fps_text = "Processing FPS: warming up"
    try:
        if not args.no_save:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(
                str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), output_fps, OUTPUT_SIZE
            )
            if not writer.isOpened():
                raise RuntimeError(f"Could not create output video: {output_path}")

        print(
            f"Segmentation: {', '.join(pipeline.segmenter.execution_devices)} "
            f"({', '.join(pipeline.segmenter.device_names)})"
        )
        print(
            f"YOLO: {pipeline.yolo_execution_device} ({pipeline.yolo_device_name}), "
            f"FP16: {pipeline.half}"
        )
        while args.max_frames is None or processed < args.max_frames:
            frame_start = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.resize(frame, OUTPUT_SIZE)
            output = pipeline.process(frame)
            if pipeline.last_assessment is not None and pipeline.last_assessment.danger:
                danger_frames += 1
            cv2.putText(
                output, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 0, 0), 2, cv2.LINE_AA,
            )
            if writer is not None:
                writer.write(output)
            stop_requested = False
            if not args.no_display:
                cv2.imshow("Road Segmentation & Object Detection Video", output)
                stop_requested = cv2.waitKey(1) & 0xFF == ord("q")
            frame_end = time.perf_counter()
            if processed >= args.warmup:
                if measurement_start is None:
                    measurement_start = frame_start
                measurement_end = frame_end
                measured += 1
                recent_times.append(frame_end - frame_start)
                fps_text = f"Processing FPS: {len(recent_times) / sum(recent_times):.1f}"
            processed += 1
            if stop_requested:
                break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if not args.no_display:
            cv2.destroyAllWindows()

    if measured:
        elapsed = measurement_end - measurement_start
        print(f"Processed: {processed} frames; measured: {measured} frames")
        print(f"Average processing FPS: {measured / elapsed:.2f} ({elapsed:.3f} seconds)")
    else:
        print(f"Processed: {processed} frames; video ended before FPS warmup completed")
    if writer is not None:
        print(f"Saved: {args.output} (playback FPS: {output_fps:.2f})")
    if pipeline.danger_detector is not None:
        print(f"DANGER frames: {danger_frames}/{processed}; crosswalks use stripe-pattern estimates")


if __name__ == "__main__":
    run(parse_args())
