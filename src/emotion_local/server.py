"""Servidor HTTP (Flask) para integracao com a Unity.

Expoe os endpoints /health (status), /emotion (ultima emocao processada) e
/predict (recebe uma imagem e retorna emocao, confianca, bbox e probabilidades).
"""

from __future__ import annotations

from pathlib import Path
import threading

import cv2
import numpy as np
from flask import Flask, jsonify, request

from .inference import EmotionPredictor


def _to_python_bbox(bbox_xyxy):
    if bbox_xyxy is None:
        return None
    return [int(value) for value in bbox_xyxy]


def _to_python_probabilities(probabilities: dict[str, object]) -> dict[str, float]:
    return {str(label): float(value) for label, value in probabilities.items()}


def create_app(checkpoint_path: Path, device: str = "auto") -> Flask:
    app = Flask(__name__)
    predictor = EmotionPredictor(checkpoint_path=checkpoint_path, device=device)

    current_emotion = {
        "emotion": "Neutral",
        "confidence": 0.0,
        "face_detected": False,
    }
    emotion_lock = threading.Lock()

    @app.get("/health")
    def healthcheck():
        return jsonify(
            {
                "status": "ok",
                "device": str(predictor.device),
                "use_landmarks": predictor.use_landmarks,
                "labels": predictor.labels,
            }
        )

    @app.get("/emotion")
    def get_emotion():
        with emotion_lock:
            return jsonify(dict(current_emotion))

    @app.post("/predict")
    def predict_emotion_from_image():
        nonlocal current_emotion

        if "image" not in request.files:
            return jsonify({"error": "Nenhuma imagem fornecida no campo 'image'."}), 400

        file = request.files["image"]
        img_bytes = file.read()
        if not img_bytes:
            return jsonify({"error": "Arquivo de imagem vazio."}), 400

        nparr = np.frombuffer(img_bytes, np.uint8)
        image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image_bgr is None:
            return jsonify({"error": "Falha ao decodificar imagem."}), 400

        try:
            result = predictor.predict_frame(image_bgr)
        except Exception as exc:
            return jsonify({"error": f"Erro interno no processamento: {exc}"}), 500

        if not result.get("face_detected", False):
            payload = {
                "emotion": "Neutral",
                "confidence": 0.0,
                "face_detected": False,
                "bbox_xyxy": None,
            }
        else:
            payload = {
                "emotion": result["label"],
                "confidence": round(float(result["confidence"]), 4),
                "face_detected": True,
                "bbox_xyxy": _to_python_bbox(result.get("bbox_xyxy")),
                "probabilities": _to_python_probabilities(result.get("probabilities", {})),
            }

        with emotion_lock:
            current_emotion = payload

        return jsonify(payload)

    return app


def run_server(
    checkpoint_path: Path,
    host: str = "0.0.0.0",
    port: int = 5000,
    device: str = "auto",
    debug: bool = False,
) -> None:
    app = create_app(checkpoint_path=checkpoint_path, device=device)
    app.run(host=host, port=port, debug=debug)
