"""YOLO + Ultralytics BoT-SORT 객체 추적 수정본 (원본 파일을 수정하지 않음).

vision_ai 환경에서 실행하세요. 기본 입력은 웹캠 0입니다.
  python 16_ObjectTracking_fixed.py
  python 16_ObjectTracking_fixed.py --source "영상파일.mp4"

화면의 탐지 박스를 클릭하여 추적 대상을 선택합니다.
S: 현재 화면을 멈추고 드래그로 선택 / R: 선택 해제 / Q 또는 Esc: 종료
YOLO가 탐지한 객체만 선택할 수 있습니다. 가림 후 ID가 바뀌면 다시 선택하세요.
"""

import argparse
import importlib.util
import json
import os
import sys
import time
from collections import deque
from pathlib import Path

# 누락된 패키지를 실행 중 자동으로 설치하지 않습니다.
os.environ["YOLO_AUTOINSTALL"] = "false"

PROJECT_DIR = Path(
    r"C:\Users\ljhlu\Desktop\Lee Jae Hyuk\AISW\이재혁\Vision_AI\03.Vision AI 실습\Vision AI"
)
DEFAULT_MODEL = PROJECT_DIR / "00_Models" / "yolov8n.pt"
WINDOW = "YOLO + BoT-SORT - Object Tracking"


def fit_display(image, output_width, output_height):
    """원본 비율을 유지하며 출력 캔버스 중앙에 배치합니다. 작은 영상은 확대하지 않습니다."""
    import cv2
    import numpy as np

    height, width = image.shape[:2]
    scale = min(1.0, output_width / width, output_height / height)
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    left = (output_width - resized_width) // 2
    top = (output_height - resized_height) // 2
    if (resized_width, resized_height) != (width, height):
        image = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((output_height, output_width, 3), dtype=image.dtype)
    canvas[top:top + resized_height, left:left + resized_width] = image
    mapping = (resized_width / width, resized_height / height, left, top)
    return canvas, mapping


def source_point(x, y, mapping, source_size):
    """출력 화면의 클릭 좌표를 원본 영상 좌표로 변환합니다. 검은 여백은 선택하지 않습니다."""
    scale_x, scale_y, left, top = mapping
    source_width, source_height = source_size
    x, y = (x - left) / scale_x, (y - top) / scale_y
    return (x, y) if 0 <= x < source_width and 0 <= y < source_height else None


def source_roi(roi, mapping, source_size):
    scale_x, scale_y, left, top = mapping
    source_width, source_height = source_size
    x, y, width, height = roi
    x1 = max(0, min(source_width, (x - left) / scale_x))
    y1 = max(0, min(source_height, (y - top) / scale_y))
    x2 = max(0, min(source_width, (x + width - left) / scale_x))
    y2 = max(0, min(source_height, (y + height - top) / scale_y))
    return x1, y1, max(0, x2 - x1), max(0, y2 - y1)


def preview_window_size(width, height):
    """창 테두리와 작업 표시줄을 고려하여 모니터 안에 들어가는 미리보기 크기를 구합니다."""
    available_width, available_height = 1280, 720
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        work_area = wintypes.RECT()
        if ctypes.windll.user32.SystemParametersInfoW(0x0030, 0, ctypes.byref(work_area), 0):
            available_width = max(1, work_area.right - work_area.left - 80)
            available_height = max(1, work_area.bottom - work_area.top - 100)
    scale = min(1.0, available_width / width, available_height / height)
    return max(1, round(width * scale)), max(1, round(height * scale))


def select_roi_target(tracks, roi):
    """선택 영역과 실제 검출 박스의 IoU로 ID를 선택합니다. 가짜 검출은 만들지 않습니다."""
    x, y, width, height = roi
    if width <= 0 or height <= 0:
        return None
    roi_area = width * height
    best_id, best_iou = None, 0.05
    for track in tracks:
        x1, y1, x2, y2 = track["box"]
        intersection = max(0, min(x + width, x2) - max(x, x1)) * max(
            0, min(y + height, y2) - max(y, y1)
        )
        union = roi_area + (x2 - x1) * (y2 - y1) - intersection
        iou = intersection / union if union > 0 else 0
        if iou > best_iou:
            best_id, best_iou = track["id"], iou
    return best_id


def select_click_target(tracks, x, y):
    candidates = [
        track for track in tracks
        if track["box"][0] <= x <= track["box"][2]
        and track["box"][1] <= y <= track["box"][3]
    ]
    if not candidates:
        return None
    # 박스가 겹치면 클릭 지점을 포함하는 가장 작은 박스를 선택합니다.
    return min(
        candidates,
        key=lambda t: (t["box"][2] - t["box"][0]) * (t["box"][3] - t["box"][1]),
    )["id"]


def tracked_boxes(result):
    boxes = result.boxes
    if boxes is None or boxes.id is None or len(boxes) == 0:
        return []
    return [
        {"id": int(track_id), "box": box, "class": int(cls), "score": float(score)}
        for box, track_id, cls, score in zip(
            boxes.xyxy.cpu().tolist(), boxes.id.cpu().tolist(),
            boxes.cls.cpu().tolist(), boxes.conf.cpu().tolist(),
        )
    ]


def run(args):
    missing = [name for name in ("cv2", "numpy", "torch", "ultralytics", "lap")
               if importlib.util.find_spec(name) is None]
    if missing:
        raise RuntimeError(
            f"필요한 모듈이 없습니다: {', '.join(missing)}\n"
            "conda activate vision_ai 로 환경을 선택하세요.\n"
            "설치가 필요하면: python -m pip install ultralytics opencv-python lapx"
        )

    import cv2
    import torch
    from ultralytics import YOLO

    model_path = Path(args.model).expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}\n--model 옵션으로 지정하세요.")
    model = YOLO(str(model_path))
    if model.task != "detect":
        raise ValueError("이 수정본은 객체 탐지 모델을 사용합니다. yolov8n.pt를 지정하세요.")
    device = args.device or ("0" if torch.cuda.is_available() else "cpu")
    source = int(args.source) if args.source.isdecimal() else args.source
    cap = cv2.VideoCapture(source)
    history = deque(maxlen=300)
    state = {"tracks": [], "selected_id": None, "notice": "", "notice_until": 0,
             "mapping": None, "source_size": None}
    summary = {"frames": 0, "frames_with_tracks": 0, "selected_frames": 0}
    seen_ids = set()

    def choose(track_id):
        if track_id is None:
            state["notice"] = "No detected object here. Select a YOLO box."
            state["notice_until"] = time.perf_counter() + 3
            print("선택 영역에 탐지된 객체가 없습니다. 실제 YOLO 박스를 선택하세요.")
            return
        state["selected_id"] = track_id
        state["notice"] = ""
        history.clear()
        print(f"선택된 객체 ID: {track_id}")

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and state["mapping"] is not None:
            point = source_point(x, y, state["mapping"], state["source_size"])
            if point is None:
                return
            visible = [t for t in state["tracks"]
                       if state["selected_id"] is None or t["id"] == state["selected_id"]]
            choose(select_click_target(visible, *point))

    try:
        if not cap.isOpened():
            raise RuntimeError(f"영상 입력을 열 수 없습니다: {source}")
        print(f"모델: {model_path}\n입력: {source}\n장치: {device}")
        print("탐지 박스 클릭: 대상 선택 | S: 드래그 선택 | R: 선택 해제 | Q/Esc: 종료")
        print("선택 가능한 대상은 YOLO가 탐지한 객체입니다.")
        if not args.headless:
            cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            cv2.resizeWindow(WINDOW, *preview_window_size(args.width, args.height))
            cv2.setMouseCallback(WINDOW, mouse_callback)

        while True:
            started = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                if summary["frames"] == 0:
                    raise RuntimeError("첫 프레임을 읽을 수 없습니다.")
                if isinstance(source, int):
                    raise RuntimeError("카메라 프레임 입력이 중단되었습니다.")
                break
            # 추적 전에 좌우 반전하여 화면, 박스, 경로, 선택 좌표를 일치시킵니다.
            frame = cv2.flip(frame, 1)
            height, width = frame.shape[:2]
            # YOLO에는 원본 크기의 반전 프레임을 전달합니다. 출력 크기는 표시할 때만 적용합니다.
            state["source_size"] = (width, height)

            # 매 프레임 실제 YOLO 검출을 공급하고 동일 스트림의 추적 상태를 유지합니다.
            result = model.track(
                frame, persist=True, tracker="botsort.yaml", conf=args.conf,
                imgsz=args.imgsz, device=device, verbose=False, save=False,
            )[0]
            tracks = tracked_boxes(result)
            state["tracks"] = tracks
            summary["frames"] += 1
            summary["frames_with_tracks"] += bool(tracks)
            seen_ids.update(t["id"] for t in tracks)
            if args.auto_select and state["selected_id"] is None and tracks:
                choose(max(tracks, key=lambda t: t["score"])["id"])

            selected = next((t for t in tracks if t["id"] == state["selected_id"]), None)
            now = time.perf_counter()
            if selected is not None:
                x1, y1, x2, y2 = selected["box"]
                history.append((now, (round((x1 + x2) / 2), round((y1 + y2) / 2))))
                summary["selected_frames"] += 1
                status, color = f"Tracking ID: {state['selected_id']}", (0, 255, 0)
            elif state["selected_id"] is not None:
                # 다른 객체의 ID로 자동 전환하지 않고 같은 ID의 복귀를 기다립니다.
                if history and history[-1][1] is not None:
                    history.append((now, None))
                status, color = f"ID {state['selected_id']} temporarily lost - R to reselect", (0, 165, 255)
            elif tracks:
                status, color = "Click a box to select / S: drag selection", (255, 200, 0)
            else:
                status, color = "No YOLO detection - show a person, cup or bottle", (0, 165, 255)
            while history and now - history[0][0] > 5:
                history.popleft()

            if not args.headless:
                display = frame.copy()
                for track in tracks:
                    if state["selected_id"] is not None and track["id"] != state["selected_id"]:
                        continue
                    x1, y1, x2, y2 = map(round, track["box"])
                    box_color = (0, 255, 0) if track["id"] == state["selected_id"] else (255, 200, 0)
                    cv2.rectangle(display, (x1, y1), (x2, y2), box_color, 2)
                    label = f"{result.names[track['class']]} ID:{track['id']} {track['score']:.2f}"
                    cv2.putText(display, label, (x1, max(20, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
                points = list(history)
                for previous, current in zip(points, points[1:]):
                    if previous[1] is not None and current[1] is not None:
                        cv2.line(display, previous[1], current[1], (0, 165, 255), 2)
                display, state["mapping"] = fit_display(display, args.width, args.height)
                fps = 1 / max(time.perf_counter() - started, 1e-6)
                cv2.putText(display, f"BoT-SORT | FPS: {fps:.1f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 150, 0), 2)
                cv2.putText(display, status, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.putText(display, "Click/S: select | R: clear | Q/Esc: quit", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)
                if state["notice"] and now < state["notice_until"]:
                    cv2.putText(display, state["notice"], (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
                cv2.imshow(WINDOW, display)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break
                if key == ord("r"):
                    state["selected_id"] = None
                    state["notice"] = ""
                    history.clear()
                elif key == ord("s"):
                    # 같은 프레임의 검출 ID와 선택 영역을 대조합니다.
                    frozen_tracks = [dict(t) for t in tracks]
                    roi_display, roi_mapping = fit_display(result.plot(), args.width, args.height)
                    cv2.namedWindow("Select Object", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
                    cv2.resizeWindow("Select Object", *preview_window_size(args.width, args.height))
                    roi = cv2.selectROI("Select Object", roi_display, fromCenter=False, showCrosshair=True)
                    cv2.destroyWindow("Select Object")
                    if roi[2] > 0 and roi[3] > 0:
                        choose(select_roi_target(frozen_tracks, source_roi(roi, roi_mapping, (width, height))))
            if args.max_frames and summary["frames"] >= args.max_frames:
                break
    finally:
        cap.release()
        if not args.headless:
            cv2.destroyAllWindows()

    summary.update(selected_id=state["selected_id"], seen_ids=sorted(seen_ids))
    print("RESULT " + json.dumps(summary, ensure_ascii=False))
    return summary


def main():
    parser = argparse.ArgumentParser(description="YOLO + BoT-SORT 객체 선택 및 경로 표시")
    parser.add_argument("--source", default="0", help="웹캠 번호 또는 영상 파일 경로")
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="로컬 YOLO 탐지 모델 경로")
    parser.add_argument("--device", default=None, help="cpu 또는 GPU 번호 (기본: 자동 선택)")
    parser.add_argument("--width", type=int, default=960, help="출력 화면 가로 크기 (기본: 960)")
    parser.add_argument("--height", type=int, default=540, help="출력 화면 세로 크기 (기본: 540)")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO 추론 크기")
    parser.add_argument("--conf", type=float, default=0.10, help="저신뢰도 검출도 추적 연관에 사용")
    parser.add_argument("--headless", action="store_true", help="화면 없이 영상으로 실행 확인")
    parser.add_argument("--auto-select", action="store_true", help="첫 검출 중 신뢰도가 가장 높은 대상 선택")
    parser.add_argument("--max-frames", type=int, default=0, help="검증할 프레임 수 (0: 제한 없음)")
    args = parser.parse_args()
    if args.width <= 0 or args.height <= 0 or args.imgsz <= 0 or args.max_frames < 0 or not 0 < args.conf < 1:
        parser.error("width/height/imgsz는 양수, max-frames는 0 이상, conf는 0과 1 사이여야 합니다.")
    try:
        run(args)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"실행 오류: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
