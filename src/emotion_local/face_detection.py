from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .mediapipe_runtime import bbox_from_landmarks, create_mp_image, load_mediapipe_tasks_context, mediapipe_import_supported


@dataclass(slots=True)
class FaceDetectionResult:
    face_rgb: np.ndarray
    bbox_xyxy: tuple[int, int, int, int] | None = None


class FaceCropper:
    """Tenta MediaPipe; se nao estiver disponivel, cai para OpenCV Haar Cascade."""

    def __init__(self, min_detection_confidence: float = 0.5) -> None:
        self.min_detection_confidence = min_detection_confidence
        self._backend = None
        self._detector = None

    def _init_backend(self) -> None:
        if self._backend is not None:
            return

        if mediapipe_import_supported():
            try:
                import mediapipe as mp

                solutions = getattr(mp, "solutions", None)
                if solutions is not None and hasattr(solutions, "face_detection"):
                    self._detector = solutions.face_detection.FaceDetection(
                        model_selection=0,
                        min_detection_confidence=self.min_detection_confidence,
                    )
                    self._backend = "mediapipe"
                    return
            except Exception:
                pass

            try:
                from mediapipe.python.solutions import face_detection as mp_face_detection

                self._detector = mp_face_detection.FaceDetection(
                    model_selection=0,
                    min_detection_confidence=self.min_detection_confidence,
                )
                self._backend = "mediapipe"
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
                self._detector = {
                    "context": tasks_context,
                    "landmarker": tasks_context.face_landmarker_module.FaceLandmarker.create_from_options(options),
                }
                self._backend = "mediapipe_tasks_face_landmarker"
                return
            except Exception:
                self._detector = None

        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        if cascade_path.exists():
            cascade = cv2.CascadeClassifier(str(cascade_path))
            if not cascade.empty():
                self._detector = cascade
                self._backend = "opencv"
                return

        self._backend = "none"

    def detect(self, image_rgb: np.ndarray) -> FaceDetectionResult:
        self._init_backend()

        if self._backend == "mediapipe":
            results = self._detector.process(image_rgb)
            if not results.detections:
                return FaceDetectionResult(face_rgb=image_rgb, bbox_xyxy=None)

            detection = results.detections[0]
            bbox = detection.location_data.relative_bounding_box
            height, width = image_rgb.shape[:2]

            x1 = max(0, int(bbox.xmin * width))
            y1 = max(0, int(bbox.ymin * height))
            x2 = min(width, int((bbox.xmin + bbox.width) * width))
            y2 = min(height, int((bbox.ymin + bbox.height) * height))
            roi = image_rgb[y1:y2, x1:x2]
            if roi.size:
                return FaceDetectionResult(face_rgb=roi, bbox_xyxy=(x1, y1, x2, y2))
            return FaceDetectionResult(face_rgb=image_rgb, bbox_xyxy=None)

        if self._backend == "mediapipe_tasks_face_landmarker":
            mp_image = create_mp_image(self._detector["context"].mp, image_rgb)
            results = self._detector["landmarker"].detect(mp_image)
            if not results.face_landmarks:
                return FaceDetectionResult(face_rgb=image_rgb, bbox_xyxy=None)

            bbox_xyxy = bbox_from_landmarks(results.face_landmarks[0], image_rgb.shape)
            if bbox_xyxy is None:
                return FaceDetectionResult(face_rgb=image_rgb, bbox_xyxy=None)

            x1, y1, x2, y2 = bbox_xyxy
            roi = image_rgb[y1:y2, x1:x2]
            if roi.size:
                return FaceDetectionResult(face_rgb=roi, bbox_xyxy=bbox_xyxy)
            return FaceDetectionResult(face_rgb=image_rgb, bbox_xyxy=None)

        if self._backend == "opencv":
            gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
            faces = self._detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))
            if len(faces) == 0:
                return FaceDetectionResult(face_rgb=image_rgb, bbox_xyxy=None)

            x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
            roi = image_rgb[y : y + h, x : x + w]
            if roi.size:
                return FaceDetectionResult(face_rgb=roi, bbox_xyxy=(x, y, x + w, y + h))
            return FaceDetectionResult(face_rgb=image_rgb, bbox_xyxy=None)

        return FaceDetectionResult(face_rgb=image_rgb, bbox_xyxy=None)

    def crop_face(self, image_rgb: np.ndarray) -> np.ndarray:
        return self.detect(image_rgb).face_rgb

    @property
    def backend(self) -> str:
        self._init_backend()
        return self._backend

    def __del__(self) -> None:
        try:
            detector = self._detector
            if hasattr(detector, "close"):
                detector.close()
            elif isinstance(detector, dict):
                landmarker = detector.get("landmarker")
                if hasattr(landmarker, "close"):
                    landmarker.close()
        except Exception:
            pass
