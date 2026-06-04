"""Inferencia em imagem/quadro e modo webcam.

A classe EmotionPredictor reconstroi o modelo a partir do checkpoint, detecta
a face, extrai landmarks (se o modelo usar) e retorna emocao, confianca e
probabilidades. run_webcam executa a inferencia ao vivo via OpenCV.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from .dataset import IMAGENET_MEAN, IMAGENET_STD
from .face_detection import FaceCropper
from .model import EmotionClassifier, resolve_device


class EmotionPredictor:
    def __init__(self, checkpoint_path: Path, device: str = "auto") -> None:
        self.device = resolve_device(device)
        checkpoint_path = checkpoint_path.resolve()
        if checkpoint_path.is_dir():
            raise IsADirectoryError(
                "O argumento --checkpoint precisa apontar para um arquivo .pt/.pth. "
                f"Foi recebido um diretorio: {checkpoint_path}"
            )
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint nao encontrado: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.labels = checkpoint.get("emotion_labels", ["Angry", "Happy", "Sad", "Neutral"])
        model_metadata = checkpoint.get("model_metadata", {})
        self.use_landmarks = model_metadata.get("use_landmarks", False)
        self.landmark_dim = model_metadata.get("landmark_dim", 936)
        self.model = EmotionClassifier(
            num_classes=len(self.labels),
            pretrained=False,
            use_landmarks=self.use_landmarks,
            landmark_dim=self.landmark_dim,
            landmark_hidden_dim=model_metadata.get("landmark_hidden_dim", 128),
            fusion_hidden_dim=model_metadata.get("fusion_hidden_dim", 256),
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
        self._face_cropper = FaceCropper(min_detection_confidence=0.5)
        self._landmark_extractor = None

    def _detect_face(self, image_bgr: np.ndarray):
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        return self._face_cropper.detect(image_rgb)

    def _extract_landmarks(self, face_rgb: np.ndarray) -> torch.Tensor | None:
        if not self.use_landmarks:
            return None

        if self._landmark_extractor is None:
            from .landmarks import FaceLandmarkExtractor

            self._landmark_extractor = FaceLandmarkExtractor(landmark_dim=self.landmark_dim)

        vector = self._landmark_extractor.extract(face_rgb)
        return torch.tensor(vector, dtype=torch.float32, device=self.device).unsqueeze(0)

    def _predict_face_rgb(self, face_rgb: np.ndarray) -> dict[str, object]:
        landmarks = self._extract_landmarks(face_rgb)
        tensor = self.transform(Image.fromarray(face_rgb)).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor, landmarks)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

        predicted_idx = int(np.argmax(probs))
        return {
            "label": self.labels[predicted_idx],
            "confidence": float(probs[predicted_idx]),
            "probabilities": {label: float(probs[idx]) for idx, label in enumerate(self.labels)},
        }

    def analyze_frame(self, image_bgr: np.ndarray) -> dict[str, object]:
        detection = self._detect_face(image_bgr)
        prediction = self._predict_face_rgb(detection.face_rgb)
        prediction["face_detected"] = detection.bbox_xyxy is not None
        prediction["bbox_xyxy"] = detection.bbox_xyxy
        return prediction

    def predict_image(self, image_path: Path) -> dict[str, object]:
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            raise FileNotFoundError(f"Nao foi possivel abrir a imagem: {image_path}")

        return self.analyze_frame(image_bgr)

    def predict_frame(self, image_bgr: np.ndarray) -> dict[str, object]:
        return self.analyze_frame(image_bgr)


def run_webcam(checkpoint_path: Path, camera_index: int = 0, device: str = "auto") -> None:
    predictor = EmotionPredictor(checkpoint_path=checkpoint_path, device=device)
    capture = cv2.VideoCapture(camera_index)

    if not capture.isOpened():
        raise RuntimeError(f"Nao foi possivel abrir a webcam no indice {camera_index}.")

    window_name = "Emotion Detection Webcam"
    print("Pressione 'q' para sair.")
    gui_available = True
    warned_headless = False

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                print("Falha ao capturar frame da webcam.")
                break

            result = predictor.predict_frame(frame)
            text = f"{result['label']} ({result['confidence']:.2%})"

            cv2.putText(
                frame,
                text,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            if gui_available:
                try:
                    cv2.imshow(window_name, frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                except cv2.error:
                    gui_available = False
                    if not warned_headless:
                        print(
                            "OpenCV sem suporte a janelas detectado. "
                            "Continuando webcam sem preview; interrompa com Ctrl+C.",
                            flush=True,
                        )
                        warned_headless = True
            else:
                if not warned_headless:
                    print(
                        "OpenCV sem suporte a janelas detectado. "
                        "Continuando webcam sem preview; interrompa com Ctrl+C.",
                        flush=True,
                    )
                    warned_headless = True
    finally:
        capture.release()
        if gui_available:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass
