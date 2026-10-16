"""앉음·서 있음 폴더의 이미지에서 관절 좌표를 추출하여 데이터셋 생성."""
import json
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
from ultralytics import YOLO
from pose_config import CLASS_NAMES, DATASET_RELATIVE_PATH
from pose_data import (
    FEATURE_COLUMNS, FEATURE_MODE, IMAGE_SUFFIXES, pose_features,
    read_image, backup_files,
)

# 클래스 순서는 pose_config.py에서 관리: sitting=0, standing=1
DETECTION_CONFIDENCE = 0.25


def main():
    script_dir = Path(__file__).resolve().parent
    person_dir = script_dir / DATASET_RELATIVE_PATH
    jobs = []
    for label, name in enumerate(CLASS_NAMES):
        folder = person_dir / name
        paths = sorted(path for path in folder.iterdir()
                       if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES) if folder.exists() else []
        if not paths:
            raise RuntimeError(f"{name} 이미지가 없습니다. 13번으로 촬영하세요: {folder}")
        jobs.extend((label, path) for path in paths)
    model = YOLO(str(script_dir.parent / "00_Models/yolov8n-pose.pt"))
    rows, skipped = [], []
    print(f"Rebuilding keypoints from {len(jobs)} images...", flush=True)
    for count, (label, path) in enumerate(jobs, 1):
        relative_path = path.relative_to(person_dir).as_posix()
        try:
            image = read_image(path)
        except (OSError, ValueError):
            image = None
        if image is None:
            skipped.append({"image_name": relative_path, "reason": "image_read_failed"})
        else:
            result = model(image, conf=DETECTION_CONFIDENCE, verbose=False)[0]
            if len(result.boxes) == 0:
                skipped.append({"image_name": relative_path, "reason": "person_not_detected"})
            else:
                areas = (result.boxes.xyxy[:, 2] - result.boxes.xyxy[:, 0]) * (
                    result.boxes.xyxy[:, 3] - result.boxes.xyxy[:, 1])
                index = int(areas.argmax().item())
                features = pose_features(result, index)
                if not any(features.values()):
                    skipped.append({"image_name": relative_path, "reason": "no_visible_keypoints"})
                else:
                    rows.append({"image_name": relative_path, **features, "label": label})
        if count % 25 == 0 or count == len(jobs):
            print(f"Processed {count}/{len(jobs)} | usable {len(rows)} | skipped {len(skipped)}", flush=True)

    df = pd.DataFrame(rows, columns=["image_name"] + FEATURE_COLUMNS + ["label"])
    counts = Counter(row["label"] for row in rows)
    print("Class distribution:", flush=True)
    for label, name in enumerate(CLASS_NAMES):
        print(f"  {name}: {counts[label]}", flush=True)
    insufficient = [name for label, name in enumerate(CLASS_NAMES) if counts[label] < 2]
    if insufficient:
        raise RuntimeError(f"유효 이미지가 2장 미만인 자세: {insufficient}. 기존 CSV는 보존했습니다.")
    if df["image_name"].duplicated().any() or not np.isfinite(df[FEATURE_COLUMNS].to_numpy()).all():
        raise RuntimeError("중복 경로나 잘못된 좌표가 있습니다. 기존 CSV는 보존했습니다.")

    names = ["keypoints.csv", "dataset.csv", "labels.json", "dataset_features.json", "dataset_skipped.csv"]
    backup = backup_files(person_dir, names, "before_rebuild")
    labels = {
        "id_to_name": {str(i): name for i, name in enumerate(CLASS_NAMES)},
        "name_to_id": {name: i for i, name in enumerate(CLASS_NAMES)},
    }
    # 원래 이미지의 이름과 내용은 변경하지 않음. 폴더/파일명으로 고유 식별
    df.drop(columns=["label"]).to_csv(person_dir / "keypoints.csv", index=False)
    df.to_csv(person_dir / "dataset.csv", index=False)
    (person_dir / "labels.json").write_text(json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8")
    (person_dir / "dataset_features.json").write_text(json.dumps({
        "feature_columns": FEATURE_COLUMNS,
        "preprocessing": FEATURE_MODE,
        "num_classes": len(CLASS_NAMES),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(skipped, columns=["image_name", "reason"]).to_csv(person_dir / "dataset_skipped.csv", index=False)
    print(f"Dataset saved: {len(df)} rows / {len(CLASS_NAMES)} classes", flush=True)
    if backup:
        print(f"Previous files backed up: {backup}", flush=True)


if __name__ == "__main__":
    main()
