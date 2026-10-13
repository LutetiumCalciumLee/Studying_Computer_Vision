"""앉은 자세·서 있는 자세 이미지를 고유 파일명으로 저장하고 CSV에 누적."""
import csv
import json
from pathlib import Path
from time import monotonic
from uuid import uuid4
from datetime import datetime
import cv2
from ultralytics import YOLO
from pose_display import mirrored_pose_display
from webcam_zoom import digital_zoom
from pose_config import CLASS_NAMES, DATASET_RELATIVE_PATH
from pose_data import (
    FEATURE_COLUMNS, FEATURE_MODE, pose_features, write_image, backup_files,
)

# VS Code에서 직접 수정: "sitting" = 앉음, "standing" = 서 있음
CAPTURE_LABEL = "sitting"
# 표시 화면 배율: 0.5 = 축소, 1.0 = 원본, 1.5 = 확대
ZOOM_FACTOR = 1.0
CAMERA_INDEX = 0
SAVE_INTERVAL_SECONDS = 0.3
MAX_SAMPLES = 1000
PERSON_CONFIDENCE = 0.75


def main():
    if CAPTURE_LABEL not in CLASS_NAMES:
        raise ValueError(f"CAPTURE_LABEL은 {CLASS_NAMES} 중 하나여야 합니다.")
    if SAVE_INTERVAL_SECONDS < 0 or MAX_SAMPLES < 1:
        raise ValueError("SAVE_INTERVAL_SECONDS >= 0, MAX_SAMPLES >= 1이어야 합니다.")
    script_dir = Path(__file__).resolve().parent
    person_dir = script_dir / DATASET_RELATIVE_PATH
    class_dir = person_dir / CAPTURE_LABEL
    class_dir.mkdir(parents=True, exist_ok=True)
    csv_path = person_dir / "keypoints.csv"
    metadata_path = person_dir / "dataset_features.json"
    # 예전 좌표 CSV와 새 좌표를 섞지 않도록 먼저 14번으로 복구
    if csv_path.exists() and csv_path.stat().st_size:
        if not metadata_path.exists():
            raise RuntimeError("먼저 14.2D_Pose_Dataset_Save.py를 실행해 기존 데이터를 복구하세요.")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("preprocessing") != FEATURE_MODE:
            raise RuntimeError("좌표 기준이 다릅니다. 먼저 14번을 실행하세요.")
        with csv_path.open(encoding="utf-8-sig", newline="") as stream:
            header = next(csv.reader(stream))
        if header != ["image_name"] + FEATURE_COLUMNS:
            raise RuntimeError("keypoints.csv 형식이 다릅니다. 먼저 14번을 실행하세요.")

    cap = cv2.VideoCapture(CAMERA_INDEX)
    try:
        if not cap.isOpened():
            raise RuntimeError(f"{CAMERA_INDEX}번 웹캠을 열 수 없습니다.")
        model = YOLO(str(script_dir.parent / "00_Models/yolov8n-pose.pt"))
        backup_files(person_dir, ["keypoints.csv"], "before_capture")
        new_csv = not csv_path.exists() or csv_path.stat().st_size == 0
        session = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:8]
        saved = 0
        last_saved = -float("inf")
        print(f"Capturing {CAPTURE_LABEL} -> {class_dir}", flush=True)
        with csv_path.open("a", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["image_name"] + FEATURE_COLUMNS)
            if new_csv:
                writer.writeheader()
            while saved < MAX_SAMPLES:
                success, frame = cap.read()
                if not success:
                    break
                result = model(frame, verbose=False)[0]
                now = monotonic()
                candidates = [
                    i for i, conf in enumerate(result.boxes.conf.tolist())
                    if conf >= PERSON_CONFIDENCE
                ]
                if candidates and now - last_saved >= SAVE_INTERVAL_SECONDS:
                    # 여러 사람이 있으면 가장 크게 보이는 한 사람만 저장
                    index = max(candidates, key=lambda i: float(
                        (result.boxes.xyxy[i, 2] - result.boxes.xyxy[i, 0]) *
                        (result.boxes.xyxy[i, 3] - result.boxes.xyxy[i, 1])
                    ))
                    x1, y1, x2, y2 = result.boxes.xyxy[index].tolist()
                    height, width = frame.shape[:2]
                    crop = frame[max(0, int(y1)):min(height, int(y2)),
                                 max(0, int(x1)):min(width, int(x2))]
                    if crop.size:
                        filename = f"pose_{CAPTURE_LABEL}_{session}_{saved:04d}.jpg"
                        write_image(class_dir / filename, crop)
                        writer.writerow({
                            "image_name": f"{CAPTURE_LABEL}/{filename}",
                            **pose_features(result, index),
                        })
                        stream.flush()
                        saved += 1
                        last_saved = now
                display = digital_zoom(mirrored_pose_display(result), ZOOM_FACTOR)
                text = f"{CAPTURE_LABEL.title()} | Saved {saved}/{MAX_SAMPLES} | q quit"
                cv2.putText(display, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(display, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(display, "Keep hips and knees visible", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(display, "Keep hips and knees visible", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.imshow("Pose Dataset Capture", display)
                if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
                    break
        if saved:
            metadata_path.write_text(json.dumps({
                "feature_columns": FEATURE_COLUMNS,
                "preprocessing": FEATURE_MODE,
            }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved {saved} images. Run 14 then 15 to rebuild and train.")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
