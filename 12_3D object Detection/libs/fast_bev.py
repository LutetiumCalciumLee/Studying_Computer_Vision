"""기존 BEV 기하 계산을 사용하는 OpenCV 렌더러."""

from collections import deque

import cv2
import numpy as np

from .Plotting import Plot3DBoxBev


class FastPlot3DBoxBev(Plot3DBoxBev):
    """Matplotlib 캔버스 없이 카메라 영상과 BEV를 합성합니다."""

    def __init__(self, proj_matrix=None):
        self.proj_matrix = proj_matrix
        self.shape = 900
        self.scale = 15
        self.side = 450
        self.camera_width = 1350
        self.object_list = ["car", "pedestrian", "truck", "cyclist", "motorcycle", "bus"]
        self.colors = {
            "car": (255, 0, 0),
            "pedestrian": (0, 128, 0),
            "truck": (0, 255, 255),
            "cyclist": (0, 0, 255),
            "motorcycle": (255, 255, 0),
            "bus": (255, 0, 255),
        }
        self._trajectories = {}
        self._background = self._make_background()
        self.reset()

    def _pixels(self, points):
        points = np.asarray(points, dtype=np.float64)
        pixels = np.rint(points * (self.side - 1) / (self.shape - 1)).astype(np.int32)
        pixels[:, 1] = self.side - 1 - pixels[:, 1]
        return pixels

    def _make_background(self):
        image = np.zeros((self.side, self.side, 3), dtype=np.uint8)
        center = tuple(self._pixels([[self.shape / 2, self.shape * 0.05]])[0])
        for index in range(8):
            radius = round((index + 1) * self.side * 0.1)
            cv2.circle(image, center, radius, (150, 10, 0), 1, cv2.LINE_AA)
            y = center[1] - radius - 4
            if y > 10:
                cv2.putText(
                    image, f"{(index + 1) * 9.6:.1f} m", (5, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1, cv2.LINE_AA,
                )
        for points in [
            [[0, self.shape / 2], [self.shape / 2, 0]],
            [[self.shape / 2, 0], [self.shape, self.shape / 2]],
        ]:
            line = self._pixels(points)
            cv2.line(image, tuple(line[0]), tuple(line[1]), (70, 70, 70), 1, cv2.LINE_AA)
        cv2.drawMarker(image, center, (0, 0, 255), cv2.MARKER_CROSS, 12, 2)
        for index, (name, color) in enumerate(self.colors.items()):
            y = self.side - 96 + index * 15
            cv2.rectangle(image, (self.side - 109, y), (self.side - 101, y + 8), color, -1)
            cv2.putText(
                image, name, (self.side - 96, y + 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1, cv2.LINE_AA,
            )
        return image

    def reset(self):
        self._frame_image = None
        self._bev = self._background.copy()
        self._visible_ids = set()

    def plot(self, img=None, class_object=None, bbox=None, dim=None, loc=None,
             rot_y=None, objId=None):
        if img is not None:
            self._frame_image = img
        if class_object not in self.object_list:
            return
        # 기존 렌더러와 동일한 차원, 위치, 회전 및 compute_bev 함수를 사용합니다.
        corners = self._pixels(self.compute_bev(dim, loc, rot_y))
        color = self.colors[class_object]
        cv2.fillPoly(self._bev, [corners], color, lineType=cv2.LINE_AA)
        cv2.polylines(self._bev, [corners], True, color, 1, cv2.LINE_AA)
        obj_id = objId[0] if isinstance(objId, (list, tuple)) else objId
        text = f"z: {loc[2]:.1f}"
        if obj_id is not None:
            obj_id = int(obj_id)
            text += f" {obj_id}"
            centroid = tuple(np.rint(corners[:-1].mean(axis=0)).astype(int))
            self._trajectories.setdefault(obj_id, deque(maxlen=4)).append(centroid)
            self._visible_ids.add(obj_id)
        x, y = map(int, corners[0])
        if 0 <= x < self.side and 0 <= y < self.side:
            (width, height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.32, 1)
            x = min(x, max(0, self.side - width - 3))
            y = max(height + 2, y)
            cv2.rectangle(self._bev, (x, y - height - 2), (x + width + 2, y + 2), (0, 80, 0), -1)
            cv2.putText(self._bev, text, (x + 1, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.32, (255, 255, 255), 1, cv2.LINE_AA)

    def show_result(self):
        if self._frame_image is None:
            raise ValueError("BEV에 합성할 영상 프레임이 없습니다.")
        # 사라진 객체의 궤적은 제거해 장시간 실행 시 메모리를 제한합니다.
        self._trajectories = {
            key: value for key, value in self._trajectories.items()
            if key in self._visible_ids
        }
        bev = self._bev.copy()
        for key in list(self._trajectories)[-3:]:
            points = list(self._trajectories[key])
            for first, second in zip(points, points[1:]):
                cv2.line(bev, first, second, (0, 255, 0), 1, cv2.LINE_AA)
        canvas = np.full((self.side, self.camera_width + self.side, 3), 255, dtype=np.uint8)
        height, width = self._frame_image.shape[:2]
        ratio = min(self.camera_width / width, self.side / height)
        target = (max(1, round(width * ratio)), max(1, round(height * ratio)))
        camera = cv2.resize(self._frame_image, target, interpolation=cv2.INTER_LINEAR)
        x = (self.camera_width - target[0]) // 2
        y = (self.side - target[1]) // 2
        canvas[y:y + target[1], x:x + target[0]] = camera
        canvas[:, self.camera_width:] = bev
        return canvas
