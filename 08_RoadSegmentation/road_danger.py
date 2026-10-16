"""보행자 발 아래 차도와 추정 횡단보도 영역을 비교한다."""

from dataclasses import dataclass

import cv2
import numpy as np

from crosswalk_detector import CrosswalkDetector


@dataclass
class PersonGround:
    box: tuple
    confidence: float
    state: str
    foot: tuple | None = None
    road_fraction: float | None = None


@dataclass
class DangerAssessment:
    people: list
    crosswalk_polygons: list

    @property
    def danger(self):
        return any(person.state == "road" for person in self.people)


class CrosswalkTracker:
    """잠깐 가려진 줄무늬는 검증된 카메라 움직임으로 짧게 추적한다."""

    def __init__(self, detector=None, max_age=20):
        self.detector = detector or CrosswalkDetector()
        self.max_age = max_age
        self.previous_gray = None
        self.previous_road = None
        self.regions = []

    def _motion(self, gray):
        points = cv2.goodFeaturesToTrack(
            self.previous_gray, maxCorners=120, qualityLevel=0.01,
            minDistance=8, mask=self.previous_road,
        )
        if points is None or len(points) < 8:
            return None
        moved, forward_ok, _ = cv2.calcOpticalFlowPyrLK(
            self.previous_gray, gray, points, None, winSize=(21, 21), maxLevel=3,
        )
        if moved is None or forward_ok is None:
            return None
        returned, backward_ok, _ = cv2.calcOpticalFlowPyrLK(
            gray, self.previous_gray, moved, None, winSize=(21, 21), maxLevel=3,
        )
        if returned is None or backward_ok is None:
            return None
        valid = (forward_ok.ravel() != 0) & (backward_ok.ravel() != 0)
        valid &= np.linalg.norm((returned - points).reshape(-1, 2), axis=1) < 1.5
        if np.count_nonzero(valid) < 8:
            return None
        matrix, inliers = cv2.estimateAffinePartial2D(
            points[valid], moved[valid], method=cv2.RANSAC,
            ransacReprojThreshold=2.5, maxIters=500, confidence=0.98,
        )
        if matrix is None or inliers is None or float(inliers.mean()) < 0.55:
            return None
        scale = float(np.hypot(matrix[0, 0], matrix[0, 1]))
        return matrix if 0.85 <= scale <= 1.15 else None

    @staticmethod
    def _region_mask(shape, polygon):
        mask = np.zeros(shape, np.uint8)
        cv2.fillConvexPoly(mask, np.rint(polygon).astype(np.int32), 1)
        return mask

    def update(self, frame, class_mask):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fresh = [
            np.asarray(polygon, dtype=np.float32)
            for polygon in self.detector.detect(frame, class_mask)
        ]
        fresh_masks = [self._region_mask(gray.shape, polygon) for polygon in fresh]
        regions = [(polygon, 0) for polygon in fresh]
        if self.previous_gray is not None and self.regions:
            matrix = self._motion(gray)
            if matrix is not None:
                for polygon, age in self.regions:
                    if age >= self.max_age:
                        continue
                    projected = cv2.transform(polygon.reshape(1, -1, 2), matrix)[0]
                    mask = self._region_mask(gray.shape, projected)
                    area = int(mask.sum())
                    if area < 40:
                        continue
                    road_support = np.isin(class_mask[mask != 0], (1, 3)).mean()
                    if road_support < 0.4:
                        continue
                    duplicate = any(
                        np.count_nonzero((mask != 0) & (other != 0))
                        / max(1, min(area, int(other.sum()))) > 0.5
                        for other in fresh_masks
                    )
                    if not duplicate:
                        regions.append((projected, age + 1))
        self.regions = regions
        self.previous_gray = gray
        self.previous_road = np.isin(class_mask, (1, 3)).astype(np.uint8) * 255
        crosswalk_mask = np.zeros(gray.shape, np.uint8)
        for polygon, _ in regions:
            cv2.fillConvexPoly(crosswalk_mask, np.rint(polygon).astype(np.int32), 1)
        # 발 위치와 경계 추정 사이의 몇 픽셀 차이만 허용한다.
        crosswalk_mask = cv2.dilate(crosswalk_mask, np.ones((3, 3), np.uint8))
        return crosswalk_mask, [polygon for polygon, _ in regions]


class RoadDangerDetector:
    def __init__(self, person_confidence=0.5, road_fraction=0.6, tracker=None):
        self.person_confidence = person_confidence
        self.road_fraction = road_fraction
        self.tracker = tracker or CrosswalkTracker()

    def assess(self, frame, class_mask, boxes):
        if class_mask.shape != frame.shape[:2]:
            raise ValueError("Frame and road mask sizes must match")
        crosswalk_mask, polygons = self.tracker.update(frame, class_mask)
        height, width = class_mask.shape
        people = []
        for row in np.asarray(boxes).reshape(-1, 6):
            x1, y1, x2, y2, confidence, class_id = map(float, row)
            if not np.isfinite(row).all() or int(class_id) != 0:
                continue
            if confidence < self.person_confidence:
                continue
            person = PersonGround((x1, y1, x2, y2), confidence, "unknown")
            people.append(person)
            # 발이 화면 밖으로 잘렸거나 박스가 너무 작으면 지면 위치를 확정하지 않는다.
            if (x1 <= 0 or x2 >= width - 1 or y1 < 0 or y2 >= height - 2
                    or x2 - x1 < 6 or y2 - y1 < 24 or y2 <= y1):
                continue
            center = int(round((x1 + x2) / 2))
            bottom = int(np.floor(y2)) + 1
            radius = max(2, min(7, int(round((x2 - x1) * 0.12))))
            depth = max(3, min(9, int(round((y2 - y1) * 0.04))))
            left, right = max(0, center - radius), min(width, center + radius + 1)
            lower = min(height, bottom + depth)
            if not (0 <= center < width and 0 <= bottom < height and right > left):
                continue
            ground = class_mask[bottom:lower, left:right]
            if not ground.size:
                continue
            person.foot = (center, bottom)
            # 화살표/차선 같은 일반 도로 표식도 차도 표면으로 취급한다.
            person.road_fraction = float(np.isin(ground, (1, 3)).mean())
            if person.road_fraction < self.road_fraction:
                person.state = "off_road"
                continue
            inside = bool(crosswalk_mask[bottom, center])
            inside |= float(crosswalk_mask[bottom:lower, left:right].mean()) >= 0.5
            person.state = "crosswalk" if inside else "road"
        return DangerAssessment(people, polygons)


def draw_danger_banner(image):
    """횡단보도 밖 차도의 사람이 있을 때만 하단 DANGER 자막을 그린다."""
    text, scale, thickness = "DANGER", 1.8, 4
    (text_width, text_height), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness,
    )
    height, width = image.shape[:2]
    left = (width - text_width) // 2
    bottom = height - 22
    cv2.rectangle(
        image, (left - 16, bottom - text_height - 14),
        (left + text_width + 16, bottom + baseline + 8), (15, 15, 15), -1,
    )
    cv2.putText(image, text, (left, bottom), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (0, 0, 255), thickness, cv2.LINE_AA)
    return image
