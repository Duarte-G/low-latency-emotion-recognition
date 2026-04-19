from __future__ import annotations

from pathlib import Path
import json

import cv2
import numpy as np
import pandas as pd

from .data import row_to_rgb
from .face_detection import FaceCropper
from .mediapipe_runtime import create_mp_image, landmarks_to_xy_array, load_mediapipe_tasks_context


class FaceLandmarkExtractor:
    def __init__(self, landmark_dim: int = 936, min_detection_confidence: float = 0.5) -> None:
        self.landmark_dim = landmark_dim
        self.min_detection_confidence = min_detection_confidence
        self._backend = None
        self._extractor = None
        self._warned_unavailable = False

    def _init_backend(self) -> None:
        if self._backend is not None:
            return

        try:
            import mediapipe as mp

            solutions = getattr(mp, "solutions", None)
            if solutions is not None and hasattr(solutions, "face_mesh"):
                self._extractor = solutions.face_mesh.FaceMesh(
                    static_image_mode=True,
                    refine_landmarks=False,
                    max_num_faces=1,
                    min_detection_confidence=self.min_detection_confidence,
                    min_tracking_confidence=0.5,
                )
                self._backend = "mediapipe_face_mesh"
                return
        except Exception:
            pass

        try:
            from mediapipe.python.solutions import face_mesh as mp_face_mesh

            self._extractor = mp_face_mesh.FaceMesh(
                static_image_mode=True,
                refine_landmarks=False,
                max_num_faces=1,
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=0.5,
            )
            self._backend = "mediapipe_face_mesh"
            return
        except Exception:
            pass

        tasks_context = load_mediapipe_tasks_context()
        if tasks_context is not None:
            try:
                options = tasks_context.face_landmarker_module.FaceLandmarkerOptions(
                    base_options=tasks_context.base_options_cls(model_asset_path=str(tasks_context.model_path)),
                    num_faces=1,
                    min_face_detection_confidence=self.min_detection_confidence,
                    min_face_presence_confidence=self.min_detection_confidence,
                    min_tracking_confidence=0.5,
                    output_face_blendshapes=False,
                    output_facial_transformation_matrixes=False,
                )
                self._extractor = {
                    "context": tasks_context,
                    "landmarker": tasks_context.face_landmarker_module.FaceLandmarker.create_from_options(options),
                }
                self._backend = "mediapipe_tasks_face_landmarker"
                return
            except Exception:
                self._extractor = None

        self._backend = "none"

    @property
    def backend(self) -> str:
        self._init_backend()
        return self._backend

    def extract(self, image_rgb: np.ndarray) -> np.ndarray:
        self._init_backend()

        if self._backend == "none":
            if not self._warned_unavailable:
                print("Landmarks do MediaPipe indisponiveis neste ambiente. Usando vetores zerados.")
                self._warned_unavailable = True
            return np.zeros(self.landmark_dim, dtype=np.float32)

        resized = cv2.resize(image_rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
        if self._backend == "mediapipe_face_mesh":
            results = self._extractor.process(resized)
            if not results.multi_face_landmarks:
                return np.zeros(self.landmark_dim, dtype=np.float32)
            return landmarks_to_xy_array(results.multi_face_landmarks[0].landmark, self.landmark_dim)

        mp_image = create_mp_image(self._extractor["context"].mp, resized)
        results = self._extractor["landmarker"].detect(mp_image)
        if not results.face_landmarks:
            return np.zeros(self.landmark_dim, dtype=np.float32)
        return landmarks_to_xy_array(results.face_landmarks[0], self.landmark_dim)

    def __del__(self) -> None:
        extractor = self._extractor
        if hasattr(extractor, "close"):
            extractor.close()
        elif isinstance(extractor, dict):
            landmarker = extractor.get("landmarker")
            if hasattr(landmarker, "close"):
                landmarker.close()

def compute_landmarks_for_dataframe(
    dataframe: pd.DataFrame,
    cache_path: Path,
    landmark_dim: int,
    use_face_crop: bool,
) -> dict[str, object]:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cropper = FaceCropper(min_detection_confidence=0.5)
    extractor = FaceLandmarkExtractor(landmark_dim=landmark_dim)

    landmarks = np.zeros((len(dataframe), landmark_dim), dtype=np.float32)
    valid_count = 0
    for idx, (_, row) in enumerate(dataframe.iterrows()):
        image_rgb = row_to_rgb(row)
        if use_face_crop:
            image_rgb = cropper.crop_face(image_rgb)

        vector = extractor.extract(image_rgb)
        if np.any(vector):
            valid_count += 1
        landmarks[idx] = vector

    np.save(cache_path, landmarks)
    metadata = {
        "cache_path": str(cache_path),
        "num_samples": int(len(dataframe)),
        "landmark_dim": int(landmark_dim),
        "landmark_backend": extractor.backend,
        "face_crop_backend": cropper.backend if use_face_crop else "disabled",
        "nonzero_landmarks": int(valid_count),
    }
    cache_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata
