from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FACE_LANDMARKER_MODEL_CANDIDATES = (
    PROJECT_ROOT / "assets" / "mediapipe" / "face_landmarker.task",
    PROJECT_ROOT / "assets" / "mediapipe" / "face_landmarker_v2.task",
)


@dataclass(slots=True)
class MediaPipeTasksContext:
    mp: object
    face_landmarker_module: object
    base_options_cls: object
    model_path: Path


def resolve_face_landmarker_model_path() -> Path | None:
    env_value = os.getenv("MEDIAPIPE_FACE_LANDMARKER_MODEL")
    candidates = []
    if env_value:
        candidates.append(Path(env_value).expanduser())
    candidates.extend(DEFAULT_FACE_LANDMARKER_MODEL_CANDIDATES)

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def load_mediapipe_tasks_context() -> MediaPipeTasksContext | None:
    model_path = resolve_face_landmarker_model_path()
    if model_path is None:
        return None

    try:
        import mediapipe as mp
        from mediapipe.tasks.python.core.base_options import BaseOptions
        from mediapipe.tasks.python.vision import face_landmarker
    except Exception:
        return None

    return MediaPipeTasksContext(
        mp=mp,
        face_landmarker_module=face_landmarker,
        base_options_cls=BaseOptions,
        model_path=model_path,
    )


def create_mp_image(mp_module, image_rgb: np.ndarray):
    return mp_module.Image(
        image_format=mp_module.ImageFormat.SRGB,
        data=np.ascontiguousarray(image_rgb),
    )


def landmarks_to_xy_array(landmarks: list[object], landmark_dim: int) -> np.ndarray:
    coords: list[float] = []
    for landmark in landmarks:
        coords.append(float(landmark.x))
        coords.append(float(landmark.y))

    arr = np.asarray(coords, dtype=np.float32)
    if arr.shape[0] >= landmark_dim:
        return arr[:landmark_dim]

    padded = np.zeros(landmark_dim, dtype=np.float32)
    padded[: arr.shape[0]] = arr
    return padded


def bbox_from_landmarks(
    landmarks: list[object],
    image_shape: tuple[int, ...],
    padding_ratio: float = 0.1,
) -> tuple[int, int, int, int] | None:
    if not landmarks:
        return None

    height, width = image_shape[:2]
    xs = np.asarray([float(landmark.x) for landmark in landmarks], dtype=np.float32)
    ys = np.asarray([float(landmark.y) for landmark in landmarks], dtype=np.float32)
    xs = np.clip(xs, 0.0, 1.0)
    ys = np.clip(ys, 0.0, 1.0)

    min_x = float(xs.min())
    max_x = float(xs.max())
    min_y = float(ys.min())
    max_y = float(ys.max())

    box_width = max_x - min_x
    box_height = max_y - min_y
    if box_width <= 0 or box_height <= 0:
        return None

    pad_x = box_width * padding_ratio
    pad_y = box_height * padding_ratio
    x1 = max(0, int(np.floor((min_x - pad_x) * width)))
    y1 = max(0, int(np.floor((min_y - pad_y) * height)))
    x2 = min(width, int(np.ceil((max_x + pad_x) * width)))
    y2 = min(height, int(np.ceil((max_y + pad_y) * height)))

    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)
