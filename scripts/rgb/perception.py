"""Fixed-camera RGB perception for the three known Task E objects.

The detector deliberately uses only image pixels.  Calibration files may have
been fitted offline from the demonstration metadata, but no GT is required at
runtime.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

import cv2
import numpy as np


@dataclass
class Detection:
    object_id: str
    visible: bool
    pixel_xy: list[float]
    world_xy: list[float] | None
    confidence: float
    bbox_xywh: list[int]
    area_px: float

    def to_dict(self) -> dict:
        return asdict(self)


def _components(mask: np.ndarray, min_area: float = 120.0) -> list[dict]:
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    output = []
    for idx in range(1, n):
        x, y, w, h, area = stats[idx].tolist()
        if area < min_area:
            continue
        component = (labels == idx).astype(np.uint8)
        contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        perimeter = float(cv2.arcLength(max(contours, key=cv2.contourArea), True)) if contours else 0.0
        hull_area = float(cv2.contourArea(cv2.convexHull(max(contours, key=cv2.contourArea)))) if contours else 0.0
        output.append({
            "xy": [float(centroids[idx][0]), float(centroids[idx][1])],
            "bbox": [int(x), int(y), int(w), int(h)],
            "area": float(area),
            "aspect": float(max(w, h) / max(min(w, h), 1)),
            "fill": float(area / max(w * h, 1)),
            "solidity": float(area / max(hull_area, 1.0)),
            "perimeter": perimeter,
        })
    return output


def _sugar_full_component(image: np.ndarray) -> dict | None:
    """Find the complete pale sugar-box footprint, including its white side.

    The yellow label is only a small top strip and its centroid is displaced
    from the collision/root center.  A low-saturation, bright connected
    component in the table ROI captures the full rectangular asset instead.
    """
    hsv = cv2.cvtColor(np.asarray(image)[..., :3].astype(np.uint8), cv2.COLOR_RGB2HSV)
    mask = ((hsv[..., 1] <= 105) & (hsv[..., 2] >= 80)).astype(np.uint8) * 255
    mask[:190] = 0
    mask[320:] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    candidates = []
    for component in _components(mask, min_area=1000.0):
        x, y, w, h = component["bbox"]
        if x < 270 and w >= 55 and h >= 25 and w <= 135 and h <= 80:
            candidates.append(component)
    return max(candidates, key=lambda item: item["area"], default=None)


def _merge_nearby_vertical(parts: list[dict], x_min: float = 205.0, x_max: float = 270.0) -> dict | None:
    """Merge highlight fragments belonging to the upright mustard bottle."""
    selected = [c for c in parts if x_min <= c["xy"][0] <= x_max and c["area"] >= 100.0]
    if len(selected) < 2:
        return None
    selected.sort(key=lambda c: c["bbox"][1])
    x0 = min(c["bbox"][0] for c in selected)
    y0 = min(c["bbox"][1] for c in selected)
    x1 = max(c["bbox"][0] + c["bbox"][2] for c in selected)
    y1 = max(c["bbox"][1] + c["bbox"][3] for c in selected)
    if y1 - y0 < 45 or x1 - x0 < 10:
        return None
    area = sum(float(c["area"]) for c in selected)
    xy = [sum(float(c["xy"][j]) * float(c["area"]) for c in selected) / area for j in range(2)]
    return {
        "xy": xy,
        "bbox": [int(x0), int(y0), int(x1 - x0), int(y1 - y0)],
        "area": area,
        "aspect": float(max(x1 - x0, y1 - y0) / max(min(x1 - x0, y1 - y0), 1)),
        "fill": area / max((x1 - x0) * (y1 - y0), 1),
        "solidity": max(float(c["solidity"]) for c in selected),
        "perimeter": sum(float(c["perimeter"]) for c in selected),
    }


def detect_pixels(rgb: np.ndarray) -> dict[str, dict]:
    """Detect object centers in an RGB uint8 image without world calibration."""
    image = np.asarray(rgb)
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError(f"Expected HxWx3 RGB image, got {image.shape}")
    image = image[..., :3].astype(np.uint8, copy=False)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    yellow = cv2.inRange(hsv, np.array([15, 70, 70], np.uint8), np.array([45, 255, 255], np.uint8))
    # The box has pale printed yellow while mustard and banana retain a much
    # higher saturation.  This second mask is useful when the pale box and
    # bottle touch and become one connected component in the broad mask.
    saturated_yellow = cv2.inRange(hsv, np.array([15, 180, 70], np.uint8), np.array([45, 255, 255], np.uint8))
    pink = cv2.inRange(hsv, np.array([135, 35, 45], np.uint8), np.array([179, 255, 255], np.uint8))
    kernel = np.ones((3, 3), np.uint8)
    yellow = cv2.morphologyEx(yellow, cv2.MORPH_OPEN, kernel)
    yellow = cv2.morphologyEx(yellow, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    saturated_yellow = cv2.morphologyEx(saturated_yellow, cv2.MORPH_OPEN, kernel)
    # Mustard highlights can be split into upper/lower blobs by a specular
    # stripe.  This close joins only nearby pieces of the same bottle.
    saturated_yellow = cv2.morphologyEx(saturated_yellow, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    pink = cv2.morphologyEx(pink, cv2.MORPH_OPEN, kernel)
    pink = cv2.morphologyEx(pink, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

    yellow_parts = _components(yellow, min_area=180.0)
    saturated_parts = _components(saturated_yellow, min_area=100.0)
    # The fixed assets have distinct projected shapes: sugar is broad, mustard
    # is tall, and banana is the remaining medium-size curved component.
    sugar = [c for c in yellow_parts if c["bbox"][2] >= c["bbox"][3] * 0.75]
    tall = [c for c in yellow_parts if c["bbox"][3] > c["bbox"][2] * 2.15]
    banana = [c for c in yellow_parts if c not in sugar and c not in tall]

    def best(items: list[dict], fallback: list[dict] | None = None) -> dict | None:
        pool = items or (fallback or [])
        return max(pool, key=lambda c: c["area"], default=None)

    # Prefer the high-saturation shape classes.  They remain separate even
    # when the broad yellow mask has merged sugar and mustard.
    mustard_sat = [c for c in saturated_parts if c["bbox"][3] > c["bbox"][2] * 2.0 and c["area"] >= 400.0]
    banana_sat = [c for c in saturated_parts if c["area"] >= 500.0 and c["bbox"][3] <= c["bbox"][2] * 2.2]
    sat_mustard = best(mustard_sat) or _merge_nearby_vertical(saturated_parts)
    sat_banana = best(banana_sat)

    # Recover the sugar pixels after erasing the two high-saturation object
    # boxes.  A small margin accounts for anti-aliased edges at contact.
    sugar_mask = yellow.copy()
    for candidate in (sat_mustard, sat_banana):
        if candidate is None:
            continue
        x, y, w, h = candidate["bbox"]
        margin = 5
        x0, y0 = max(0, x - margin), max(0, y - margin)
        x1, y1 = min(sugar_mask.shape[1], x + w + margin), min(sugar_mask.shape[0], y + h + margin)
        sugar_mask[y0:y1, x0:x1] = 0
    sugar_remaining = [c for c in _components(sugar_mask, min_area=100.0)
                       if c["bbox"][2] >= c["bbox"][3] * 0.45 and c["area"] >= 100.0]
    # In this fixed view the three spawn bands project roughly left-to-right
    # as sugar, mustard, banana.  These broad x guards only disambiguate a
    # residual fragment after mask subtraction; they do not encode its world
    # position or use simulation state.
    sugar_left = [c for c in sugar_remaining if c["xy"][0] < 260.0 and c["xy"][1] > 180.0]
    banana_right = [c for c in banana_sat if c["xy"][0] >= 260.0]

    components = {
        "object_1": _sugar_full_component(image) or best(sugar_left) or best(sugar),
        "object_2": sat_mustard or best(tall),
        "object_3": best(banana_right) or sat_banana or best(banana),
    }
    # A small amount of overlap between masks can make the first shape rule
    # ambiguous.  Fill missing labels from the largest unused components.
    unused = sorted(yellow_parts, key=lambda c: c["area"], reverse=True)
    used_ids = {id(c) for c in components.values() if c is not None}
    for key in ("object_1", "object_2", "object_3"):
        if components[key] is None:
            for candidate in unused:
                if id(candidate) not in used_ids:
                    components[key] = candidate
                    used_ids.add(id(candidate))
                    break

    basket = max(_components(pink, min_area=1000.0), key=lambda c: c["area"], default=None)
    components["basket"] = basket
    return components


class RGBPerception:
    """Perception wrapper with calibration and short-term target hold."""

    def __init__(self, calibration_path: str | Path, hold_frames: int = 10, smoothing: float = 0.35):
        self.calibration_path = Path(calibration_path)
        if not self.calibration_path.is_file():
            raise FileNotFoundError(
                f"RGB calibration file not found: {self.calibration_path}. "
                "Run scripts/rgb/calibrate_task_e.py first."
            )
        payload = json.loads(self.calibration_path.read_text(encoding="utf-8"))
        matrix = np.asarray(payload.get("pixel_to_world_xy"), dtype=np.float64)
        if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
            raise ValueError("Calibration pixel_to_world_xy must be a finite 3x3 matrix")
        self.pixel_to_world = matrix
        raw_object_matrices = payload.get("object_pixel_to_world_xy", {})
        self.object_pixel_to_world = {}
        for name, value in raw_object_matrices.items():
            object_matrix = np.asarray(value, dtype=np.float64)
            if object_matrix.shape == (3, 3) and np.isfinite(object_matrix).all():
                self.object_pixel_to_world[str(name)] = object_matrix
        raw_offsets = payload.get("object_world_offsets_m", {})
        self.object_world_offsets = {
            str(name): np.asarray(value, dtype=np.float64)
            for name, value in raw_offsets.items()
            if np.asarray(value).shape == (2,) and np.isfinite(np.asarray(value)).all()
        }
        self.hold_frames = int(hold_frames)
        self.smoothing = float(smoothing)
        self._last: dict[str, tuple[np.ndarray, int, dict]] = {}

    def image_to_world(self, pixel_xy: list[float] | np.ndarray, object_name: str | None = None) -> np.ndarray:
        p = np.asarray([float(pixel_xy[0]), float(pixel_xy[1]), 1.0], dtype=np.float64)
        matrix = self.object_pixel_to_world.get(object_name, self.pixel_to_world)
        out = matrix @ p
        return out[:2] / max(float(out[2]), 1e-9)

    def reset(self) -> None:
        """Clear temporal smoothing/hold state at an episode boundary."""
        self._last.clear()

    def detect(self, rgb: np.ndarray) -> dict[str, Detection]:
        raw = detect_pixels(rgb)
        output: dict[str, Detection] = {}
        for name, component in raw.items():
            previous = self._last.get(name)
            if component is None:
                if previous is None or previous[1] >= self.hold_frames:
                    output[name] = Detection(name, False, [float("nan")] * 2, None, 0.0, [], 0.0)
                    continue
                pixel = previous[0]
                age = previous[1] + 1
                data = previous[2]
                self._last[name] = (pixel, age, data)
                output[name] = self._make_detection(name, data, confidence=0.35 / max(age, 1))
                continue

            pixel = np.asarray(component["xy"], dtype=np.float64)
            if previous is not None:
                pixel = self.smoothing * pixel + (1.0 - self.smoothing) * previous[0]
            data = dict(component)
            data["xy"] = pixel.tolist()
            self._last[name] = (pixel, 0, data)
            confidence = self._confidence(name, component)
            output[name] = self._make_detection(name, data, confidence)
        return output

    def _make_detection(self, name: str, component: Mapping, confidence: float) -> Detection:
        pixel = [float(x) for x in component["xy"]]
        # New calibration files contain an object-specific mapping fitted to
        # the visible footprint of each asset.  The legacy global mapping and
        # offsets remain a fallback for older calibration artifacts.
        world = self.image_to_world(pixel, name)
        if name not in self.object_pixel_to_world:
            world = world + self.object_world_offsets.get(name, np.zeros(2, dtype=np.float64))
        return Detection(
            object_id=name,
            visible=True,
            pixel_xy=pixel,
            world_xy=[float(world[0]), float(world[1])],
            confidence=float(confidence),
            bbox_xywh=[int(x) for x in component["bbox"]],
            area_px=float(component["area"]),
        )

    @staticmethod
    def _confidence(name: str, component: Mapping) -> float:
        area = float(component["area"])
        shape = float(component["solidity"])
        area_score = min(1.0, area / (500.0 if name == "object_1" else 1000.0))
        return float(np.clip(0.55 * area_score + 0.45 * shape, 0.0, 1.0))
