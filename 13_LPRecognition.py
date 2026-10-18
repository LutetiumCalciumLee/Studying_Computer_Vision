"""사용자 촬영 영상을 입력받아 번호판을 인식한다.

프로젝트 루트에서 실행:
  python 13_Korean_LP_Recognition/13_LPRecognition.py
  python 13_Korean_LP_Recognition/13_LPRecognition.py --input 00_Sample_video/input/my_cars.mp4
  python 13_Korean_LP_Recognition/13_LPRecognition.py --input first.mp4 second.mp4
  python 13_Korean_LP_Recognition/13_LPRecognition.py --input 00_Sample_video/input

--input을 생략하면 00_Sample_video/input/my_cars.mp4를 사용한다.
폴더를 지정하면 그 안의 영상을 파일명 숫자 순으로 처리한다.
"""

import argparse
import math
from pathlib import Path

import cv2
import easyocr
import numpy as np
from PIL import ImageFont, ImageDraw, Image
import torch
import os
import time
import concurrent.futures
from functools import lru_cache
from ultralytics import YOLO
import re

# 경로 설정
script_dir = os.path.dirname(os.path.abspath(__file__))
Models_dir = os.path.abspath(os.path.join(script_dir, "../00_Models"))
Font_dir = os.path.abspath(os.path.join(script_dir, "../00_font"))
Videos_dir = os.path.abspath(os.path.join(script_dir, "../00_Sample_Video"))
DETECTION_INTERVAL = 5
RESULT_MAX_AGE_SECONDS = 0.5
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm", ".wmv"}


def build_argument_parser():
    parser = argparse.ArgumentParser(
        description="직접 촬영한 차량 영상의 번호판을 인식하고 결과를 MP4로 저장합니다."
    )
    parser.add_argument(
        "--input", nargs="+", metavar="VIDEO_OR_FOLDER",
        help="영상 파일 또는 폴더. 여러 개 지정하면 입력한 순서로 처리합니다. "
             "생략하면 00_Sample_video/input/my_cars.mp4를 사용합니다. "
             "상대 경로는 명령을 실행한 폴더 기준이며 파일명만 입력하면 기본 입력 폴더에서도 찾습니다.",
    )
    parser.add_argument(
        "--output", default=str(Path(Videos_dir) / "output" / "lp_output.mp4"),
        metavar="RESULT.mp4", help="결과 MP4 경로. 기본값: 00_Sample_video/output/lp_output.mp4",
    )
    return parser


def find_input_videos(inputs=None):
    default_dir = Path(Videos_dir) / "input"
    if inputs is None:
        default_dir.mkdir(parents=True, exist_ok=True)
        default_video = default_dir / "my_cars.mp4"
        if not default_video.is_file():
            raise FileNotFoundError(
                f"직접 촬영한 영상을 다음 경로에 넣으세요: {default_video}\n"
                "또는 --input으로 다른 영상 파일이나 폴더를 지정하세요."
            )
        inputs = [str(default_video)]

    def natural_key(path):
        return [int(part) if part.isdigit() else part.casefold()
                for part in re.split(r"(\d+)", path.name)]

    video_paths = []
    for item in inputs:
        path = Path(item).expanduser()
        if not path.exists() and not path.is_absolute():
            path = default_dir / path
        if not path.exists():
            raise FileNotFoundError(f"입력 파일 또는 폴더가 없습니다: {item}")
        if path.is_dir():
            videos = sorted(
                (entry for entry in path.iterdir()
                 if entry.is_file() and entry.suffix.lower() in VIDEO_EXTENSIONS),
                key=natural_key,
            )
            if not videos:
                raise ValueError(
                    f"입력 폴더에 영상이 없습니다: {path}\n"
                    "촬영 영상을 넣거나 --input으로 영상 경로를 지정하세요."
                )
            video_paths.extend(video.resolve() for video in videos)
        elif path.suffix.lower() not in VIDEO_EXTENSIONS:
            raise ValueError(f"지원하는 영상 확장자가 아닙니다: {path.name}")
        else:
            video_paths.append(path.resolve())
    return video_paths


def check_video_inputs(video_paths, output_path):
    if output_path.suffix.lower() != ".mp4":
        raise ValueError("--output에는 .mp4 파일 경로를 지정하세요.")

    expected_format = None
    for path in video_paths:
        if path == output_path:
            raise ValueError(f"입력 영상을 덮어쓸 수 없습니다. 다른 출력 경로를 지정하세요: {path}")
        cap = cv2.VideoCapture(str(path))
        try:
            if not cap.isOpened():
                raise ValueError(f"비디오 파일을 열 수 없습니다: {path}")
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if not math.isfinite(fps) or fps <= 0 or width <= 0 or height <= 0:
                raise ValueError(f"비디오의 FPS 또는 해상도가 올바르지 않습니다: {path}")
            current_format = (fps, width, height)
            if expected_format is None:
                expected_format = current_format
            elif (current_format[1:] != expected_format[1:]
                  or not math.isclose(fps, expected_format[0], rel_tol=0, abs_tol=0.001)):
                raise ValueError(
                    f"여러 영상을 연결하려면 FPS와 해상도가 같아야 합니다: {path.name}\n"
                    "영상 형식을 통일하거나 --input으로 영상 하나만 지정하세요."
                )
        finally:
            cap.release()


def check_required_assets():
    # 표준 YOLOv8/CRAFT 모델과 달리 커스텀 모델은 README의 다운로드 단계가 필요하다.
    required = [
        Path(Models_dir) / "lp_det.pt",
        Path(Models_dir) / "lp_models/models/best_acc.pth",
        Path(Models_dir) / "lp_models/user_network/best_acc.py",
        Path(Models_dir) / "lp_models/user_network/best_acc.yaml",
        Path(Models_dir) / "lp_models/user_network/transformation.py",
        Path(Font_dir) / "SpoqaHanSansNeo-Light.ttf",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "필수 모델/설정/폰트가 없습니다. README의 모델 다운로드 명령을 먼저 실행하세요:\n  "
            + "\n  ".join(missing)
        )

def load_model():
    print("모델 로드 중...")
    # 영상 표시 스레드와 AI 스레드가 CPU 전체를 서로 점유하지 않도록 한다.
    cv2.setNumThreads(1)
    torch.set_num_threads(4)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    car_m = YOLO(os.path.join(Models_dir, "yolov8s.pt"))  # YOLOv8 차량 모델 로드
    car_m.to(device)
    lp_m = torch.hub.load('ultralytics/yolov5', 'custom', os.path.join(Models_dir, "lp_det.pt"))  # YOLOv5 번호판 모델 로드
    lp_m.to(device)

    reader = easyocr.Reader(['en'], detect_network='craft', recog_network='best_acc',
                             gpu=torch.cuda.is_available(),
                             user_network_directory=os.path.join(Models_dir, 'lp_models/user_network'),
                             model_storage_directory=os.path.join(Models_dir, 'lp_models/models'))
    print("모델 로드 완료!")
    return car_m, lp_m, reader

def detect_vehicles(car_m, frame):
    results = car_m(frame, classes=[2, 3, 5, 7], verbose=False)
    filtered_boxes = []
    for result in results:  # results는 리스트 형태
        for box in result.boxes.data.cpu().numpy():
            x1, y1, x2, y2, score, class_id = box
            if int(class_id) in [2, 3, 5, 7]:  # 자동차, 버스, 트럭 클래스 필터링
                filtered_boxes.append([x1, y1, x2, y2, score, class_id])
    return filtered_boxes

def process_detected_text(detected_text):
    # 공백 제거
    detected_text = detected_text.replace(" ", "")
    
    # 중복된 텍스트 제거
    if len(detected_text) >= 7 and detected_text[:7] == detected_text[-7:]:
        detected_text = detected_text[:7]
    elif len(detected_text) >= 8 and detected_text[:8] == detected_text[-8:]:
        detected_text = detected_text[:8]
        detected_text = detected_text[:8]
    
    # 한국 번호판 형식 정규식 패턴
    korean_plate_pattern = re.compile(r'^\d{2,3}[가-힣]\d{4}$')
    
    # 한국 번호판 형식과 일치하는지 확인
    if korean_plate_pattern.match(detected_text):
        return detected_text
    else:
        return None

def detect_license_plates(lp_m, reader, frame, car_boxes):
    lp_results = []

    for box in car_boxes:
        x1, y1, x2, y2 = [int(coord) for coord in box[:4]]
        x1, x2 = max(0, x1), min(frame.shape[1], x2)
        y1, y2 = max(0, y1), min(frame.shape[0], y2)
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            continue

        # 번호판 탐지
        results = lp_m(Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)))
        lp_boxes = results.xyxy[0].cpu().numpy()
        for lp_box in lp_boxes:
            lx1, ly1, lx2, ly2 = [int(coord) for coord in lp_box[:4]]
            lx1, lx2 = max(0, lx1), min(roi.shape[1], lx2)
            ly1, ly2 = max(0, ly1), min(roi.shape[0], ly2)
            license_plate = roi[ly1:ly2, lx1:lx2]
            if license_plate.size == 0:
                continue
            try:
                gray_plate = cv2.cvtColor(license_plate, cv2.COLOR_BGR2GRAY)
                text = reader.readtext(gray_plate, detail=0)

                if text:
                    detected_text = "".join(text)
                    processed_text = process_detected_text(detected_text)
                    if processed_text:
                        lp_results.append(((x1 + lx1, y1 + ly1, x1 + lx2, y1 + ly2), processed_text))
            except Exception as e:
                print(f"번호판 인식 오류: {e}")
    return lp_results

@lru_cache(maxsize=1)
def get_plate_font():
    fontpath = os.path.join(Font_dir, "SpoqaHanSansNeo-Light.ttf")
    if not os.path.exists(fontpath):
        fontpath = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    return ImageFont.truetype(fontpath, 40)

def draw_results(frame, car_boxes, lp_results):
    if not lp_results:
        return frame
    font = get_plate_font()
    pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_frame)

    for lp_box, text in lp_results:
        x1, y1, x2, y2 = lp_box
        draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=2)
        draw.text(
            (max(2, x1), max(2, y1 - 48)), text, font=font,
            fill=(0, 255, 0), anchor="lt",
            stroke_width=2, stroke_fill=(0, 255, 0),
        )

    return cv2.cvtColor(np.asarray(pil_frame), cv2.COLOR_RGB2BGR)

def process_frame(car_m, lp_m, reader, frame, frame_counter):
    # 이 함수는 전용 AI 스레드에서만 실행한다. 화면 표시는 결과를 기다리지 않는다.
    with torch.inference_mode():
        car_boxes = detect_vehicles(car_m, frame)
        lp_results = detect_license_plates(lp_m, reader, frame, car_boxes)
    return frame_counter, lp_results

def main(argv=None):
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    try:
        video_paths = find_input_videos(args.input)
        output_path = Path(args.output).expanduser().resolve()
        check_video_inputs(video_paths, output_path)
        check_required_assets()
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))

    print(f"입력 영상 {len(video_paths)}개:")
    for path in video_paths:
        print(f"  {path}")

    # 모든 입력 영상에서 같은 모델을 재사용한다.
    car_m, lp_m, reader = load_model()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    video_out_dir = str(output_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    cap = None
    out = None
    output_format = None
    stop_requested = False

    print("비디오에서 차량 번호판 탐지를 시작합니다. 'q'를 눌러 종료하세요.")
    prev_time = None
    next_frame_time = time.perf_counter()
    frame_count = 0
    last_lp_results = []
    result_frame = -1
    last_submitted_frame = -DETECTION_INTERVAL
    pending = None
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    try:
        for index, video_path in enumerate(video_paths, start=1):
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise RuntimeError(f"비디오 파일을 열 수 없습니다: {video_path}")

            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if fps <= 0 or width <= 0 or height <= 0:
                raise RuntimeError(f"비디오의 FPS 또는 해상도가 올바르지 않습니다: {video_path}")
            current_format = (fps, width, height)
            if out is None:
                output_format = current_format
                out = cv2.VideoWriter(video_out_dir, fourcc, fps, (width, height))
                if not out.isOpened():
                    raise RuntimeError(f"결과 비디오를 저장할 수 없습니다: {video_out_dir}")
                next_frame_time = time.perf_counter()
                prev_time = None
            elif (current_format[1:] != output_format[1:]
                  or not math.isclose(fps, output_format[0], rel_tol=0, abs_tol=0.001)):
                raise RuntimeError(f"분할 영상의 FPS 또는 해상도가 다릅니다: {video_path}")

            # 서로 다른 촬영 영상으로 넘어갈 때 이전 영상의 번호판을 표시하지 않는다.
            clip_start_frame = frame_count
            result_frame = -1
            last_lp_results = []
            last_submitted_frame = frame_count - DETECTION_INTERVAL
            print(f"[{index}/{len(video_paths)}] 재생 중: {os.path.basename(video_path)}")
            frame_duration = 1 / fps
            max_result_age = max(1, round(fps * RESULT_MAX_AGE_SECONDS))
            while True:
                ret, frame = cap.read()
                if not ret:
                    break  # 현재 영상이 끝나면 다음 번호의 영상을 연다.

                current_time = time.perf_counter()
                fps_display = fps if prev_time is None else 1 / max(current_time - prev_time, 1e-6)
                prev_time = current_time

                if pending is not None and pending.done():
                    completed_frame, completed_results = pending.result()
                    if completed_frame >= clip_start_frame:
                        result_frame, last_lp_results = completed_frame, completed_results
                    pending = None

                # AI가 바쁘면 작업을 쌓지 않고 영상 재생을 계속한다.
                if pending is None and frame_count - last_submitted_frame >= DETECTION_INTERVAL:
                    pending = executor.submit(
                        process_frame, car_m, lp_m, reader, frame.copy(), frame_count
                    )
                    last_submitted_frame = frame_count

                visible_results = (
                    last_lp_results if 0 <= frame_count - result_frame <= max_result_age else []
                )
                processed_frame = draw_results(frame, [], visible_results)

                cv2.putText(
                    processed_frame,
                    f"FPS: {fps_display:.2f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 0, 0),
                    2,
                    cv2.LINE_AA,
                )

                out.write(processed_frame)
                cv2.imshow("KOREAN LP Detection", processed_frame)
                frame_count += 1

                # 인식 프레임 사이의 영상이 빠르게 몰아서 표시되지 않도록 재생 속도를 맞춘다.
                next_frame_time += frame_duration
                now = time.perf_counter()
                next_frame_time = max(next_frame_time, now)
                delay_ms = max(1, round((next_frame_time - now) * 1000))
                if cv2.waitKey(delay_ms) & 0xFF == ord('q'):
                    stop_requested = True
                    break

            cap.release()
            cap = None
            if stop_requested:
                break  # q를 누르면 다음 영상으로 넘어가지 않고 전체 종료한다.
        if not stop_requested:
            print("모든 입력 비디오가 끝났습니다.")
    finally:
        if cap is not None:
            cap.release()
        if out is not None:
            out.release()
        cv2.destroyAllWindows()
        if pending is not None:
            pending.cancel()
        executor.shutdown(wait=True, cancel_futures=True)

    print(f"결과가 저장되었습니다: {video_out_dir}")

if __name__ == "__main__":
    main()
