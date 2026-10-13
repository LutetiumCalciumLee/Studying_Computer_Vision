"""앉은 자세·서 있는 자세의 데이터 검증 후 이진 분류 학습."""
import json
from pathlib import Path
from time import perf_counter
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from pose_data import FEATURE_COLUMNS, FEATURE_MODE, backup_files
from pose_config import CLASS_NAMES, DATASET_RELATIVE_PATH


def main():
    person_dir = Path(__file__).resolve().parent / DATASET_RELATIVE_PATH
    dataset_file = person_dir / "dataset.csv"
    metadata_path = person_dir / "dataset_features.json"
    if not dataset_file.exists() or not metadata_path.exists():
        raise RuntimeError("먼저 14.2D_Pose_Dataset_Save.py를 실행하세요.")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("preprocessing") != FEATURE_MODE:
        raise RuntimeError("좌표 기준이 다릅니다. 14번으로 데이터셋을 다시 생성하세요.")
    df = pd.read_csv(dataset_file)
    required = ["image_name"] + FEATURE_COLUMNS + ["label"]
    missing_columns = [column for column in required if column not in df]
    if missing_columns or df[required].isna().any().any():
        raise ValueError(f"누락 열 또는 빈 값이 있습니다: {missing_columns}")
    if df["image_name"].duplicated().any():
        raise ValueError("데이터셋에 중복 이미지 경로가 있습니다. 14번을 실행하세요.")
    X = df[FEATURE_COLUMNS].astype(float)
    raw_labels = pd.to_numeric(df["label"], errors="raise")
    if not np.isfinite(raw_labels.to_numpy()).all() or (raw_labels % 1 != 0).any():
        raise ValueError("라벨은 정수여야 합니다.")
    y = raw_labels.astype(int)
    if not np.isfinite(X.to_numpy()).all():
        raise ValueError("관절 좌표에 잘못된 숫자가 있습니다.")
    num_classes = len(CLASS_NAMES)
    expected_labels = set(range(num_classes))
    missing_labels = sorted(expected_labels - set(y.unique()))
    extra_labels = sorted(set(y.unique()) - expected_labels)
    if missing_labels or extra_labels:
        raise ValueError(
            f"앉음(0)·서 있음(1) 데이터가 모두 필요합니다. 누락: {missing_labels}, 잘못된 라벨: {extra_labels}. "
            "14번으로 데이터셋을 다시 생성하세요."
        )
    counts = y.value_counts().sort_index()
    print("Class distribution (label: count):", flush=True)
    for label, count in counts.items():
        print(f"  {label} ({CLASS_NAMES[label]}): {count}", flush=True)
    if counts.min() < 2:
        raise ValueError("각 자세에 최소 2개 이상의 데이터가 필요합니다.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=max(0.2, num_classes / len(y)), random_state=42, stratify=y
    )
    model = xgb.XGBClassifier(
        objective="binary:logistic", eval_metric="logloss",
        random_state=42, n_jobs=4,
    )
    started = perf_counter()
    print("Training started...", flush=True)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = float(accuracy_score(y_test, predictions))
    print(f"Training finished in {perf_counter() - started:.2f} seconds", flush=True)
    print(f"Accuracy (random image split): {accuracy:.4f}", flush=True)
    print(classification_report(y_test, predictions, labels=list(range(num_classes)),
                                target_names=CLASS_NAMES, zero_division=0), flush=True)

    backup = backup_files(person_dir, ["model_weights.xgb", "features.json", "training_metrics.json"], "before_training")
    model.save_model(person_dir / "model_weights.xgb")
    (person_dir / "features.json").write_text(json.dumps({
        "feature_columns": FEATURE_COLUMNS,
        "num_classes": num_classes,
        "class_names": list(CLASS_NAMES),
        "preprocessing": FEATURE_MODE,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    (person_dir / "training_metrics.json").write_text(json.dumps({
        "accuracy": accuracy,
        "split": "stratified_random_image_split",
        "train_rows": len(y_train), "test_rows": len(y_test),
        "class_counts": {str(label): int(count) for label, count in counts.items()},
        "class_names": list(CLASS_NAMES),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=list(range(num_classes))).tolist(),
        "note": "연속 촬영 프레임을 무작위 분할한 정확도입니다. 별도 촬영/실시간 정확도는 따로 확인하세요.",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Model and matching features.json saved. Run 16 for live recognition.", flush=True)
    if backup:
        print(f"Previous model backed up: {backup}", flush=True)


if __name__ == "__main__":
    main()
