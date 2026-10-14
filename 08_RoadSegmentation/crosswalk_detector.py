"""Geometry-only crosswalk candidate detector; no learned crosswalk class.

Input is an unannotated BGR frame and cached/live OMZ labels (0 BG,1 road,
2 curb,3 mark). Coordinates/thresholds scale from the current frame size.
"""
import cv2
import numpy as np


class CrosswalkDetector:
    """Return visible validated stripe hulls, without remembering old locations.

    The second input must be the four-class OMZ label image, NOT a bool road
    mask. A hull requires three adequately large, mutually compatible stripes.
    Dark gaps between their corners are included in the returned polygon.
    """

    def __init__(self, min_stripes=3):
        self.min_stripes = max(3, int(min_stripes))
        self.last_mask = None
        self.last_candidates = []

    def detect(self, frame, class_mask):
        self.last_mask, polygons, self.last_candidates = crosswalk_regions(
            frame, class_mask, min_stripes=self.min_stripes,
        )
        return [np.asarray(polygon['points'], dtype=np.int32) for polygon in polygons]


def crosswalk_regions(frame, labels, min_stripes=3, white_value=170):
    height, width = frame.shape[:2]
    labels = np.asarray(labels, dtype=np.uint8)
    if labels.shape != (height, width):
        raise ValueError("Frame and segmentation mask sizes must match")
    road = ((labels == 1) | (labels == 3)).astype(np.uint8)
    near_road = cv2.dilate(road, np.ones((5, 5), np.uint8))
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Yellow road lines and colored objects cannot seed white stripe candidates.
    white = ((hsv[:, :, 1] < 75) & (hsv[:, :, 2] > white_value) & (near_road != 0)).astype(np.uint8)*255
    white = cv2.morphologyEx(white, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    scale = min(width, height)/640
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 20*scale*scale or area > width*height*.06:
            continue
        rectangle = cv2.minAreaRect(contour)
        rw, rh = rectangle[1]
        long, short = max(rw, rh), min(rw, rh)
        if short < 2*scale or long < 32*scale or long/short < 2.2 or long/short > 24:
            continue
        if area/(long*short) < .48:
            continue
        x, y, cw, ch = cv2.boundingRect(contour)
        local = np.zeros((ch, cw), np.uint8)
        cv2.drawContours(local, [contour-np.array([[[x, y]]])], -1, 1, cv2.FILLED)
        inside = local != 0
        local_labels = labels[y:y+ch, x:x+cw]
        mark_fraction = float(np.mean(local_labels[inside] == 3))
        road_fraction = float(np.mean(road[y:y+ch, x:x+cw][inside] != 0))
        if mark_fraction < .15 or road_fraction < .7:
            continue
        box = cv2.boxPoints(rectangle)
        edges = np.roll(box, -1, axis=0)-box
        direction = edges[np.argmax(np.linalg.norm(edges, axis=1))]
        direction /= np.linalg.norm(direction)
        if direction[0] < 0:
            direction *= -1
        candidates.append({'box': box, 'center': np.array(rectangle[0]), 'direction': direction, 'long': long, 'short': short, 'area': area, 'mark_fraction': mark_fraction})
    # Larger real paint components take priority over tiny texture fragments.
    candidates = sorted(candidates, key=lambda item: item['area'], reverse=True)[:60]
    adjacent = [set() for _ in candidates]
    for i, first in enumerate(candidates):
        for j in range(i+1, len(candidates)):
            second = candidates[j]
            cosine = abs(float(np.dot(first['direction'], second['direction'])))
            if cosine < np.cos(np.deg2rad(18)):
                continue
            if min(first['long'], second['long'])/max(first['long'], second['long']) < .55:
                continue
            axis = first['direction']
            a = first['box']@axis
            b = second['box']@axis
            overlap = min(a.max(), b.max())-max(a.min(), b.min())
            if overlap < .12*min(first['long'], second['long']):
                continue
            delta = second['center']-first['center']
            normal = np.array([-axis[1], axis[0]])
            separation = abs(float(np.dot(delta, normal)))
            if separation < max(2*scale, .25*min(first['short'], second['short'])):
                continue
            if separation > .6*max(first['long'], second['long'])+10*scale:
                continue
            # Zebra paint/gaps have comparable widths. Widely separated lane
            # dividers must not exempt the intervening roadway as a crossing.
            if separation > 3.5*max(first['short'], second['short'])+3*scale:
                continue
            if np.linalg.norm(delta) > 1.1*max(first['long'], second['long'])+10*scale:
                continue
            adjacent[i].add(j)
            adjacent[j].add(i)
    mask = np.zeros((height, width), np.uint8)
    polygons = []
    seen = set()
    for seed in range(len(candidates)):
        if seed in seen:
            continue
        stack = [seed]
        component = []
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            component.append(current)
            stack.extend(adjacent[current]-seen)
        if len(component) < min_stripes:
            continue
        boxes = np.concatenate([candidates[index]['box'] for index in component])
        hull = cv2.convexHull(boxes).reshape(-1, 2)
        region = np.zeros_like(mask)
        cv2.fillConvexPoly(region, np.rint(hull).astype(np.int32), 1)
        if cv2.contourArea(hull) > width*height*.35:
            continue
        # Fill dark spaces BETWEEN stripes, but avoid exempting sidewalks/curbs.
        region &= road
        mask |= region
        polygons.append({'points': np.rint(hull).astype(int).tolist(), 'stripe_count': len(component)})
    details = [{'center': np.rint(item['center']).astype(int).tolist(), 'long': round(item['long'], 2), 'short': round(item['short'], 2), 'mark_fraction': round(item['mark_fraction'], 3), 'box': np.rint(item['box']).astype(int).tolist()} for item in candidates]
    return mask, polygons, details
