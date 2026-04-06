from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


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
