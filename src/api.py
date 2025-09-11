from flask import Flask, jsonify, request
import cv2
import numpy as np
from teste_webcam import LocalEmotionDetector
import threading
import copy

app = Flask(__name__)

# Criar detector
detector = LocalEmotionDetector()

# Variáveis globais com lock para thread safety
current_emotion = {"emotion": "Neutral", "confidence": 0.0}
emotion_lock = threading.Lock()

@app.route('/emotion', methods=['GET'])
def get_emotion():
    """Endpoint legado para compatibilidade com a Unity"""
    with emotion_lock:
        return jsonify(copy.deepcopy(current_emotion))

@app.route('/predict', methods=['POST'])
def predict_emotion_from_image():
    """Novo endpoint que processa imagens enviadas"""
    global current_emotion
    
    try:
        # Verificar se a imagem foi enviada
        if 'image' not in request.files:
            return jsonify({'error': 'Nenhuma imagem fornecida'}), 400
        
        # Ler a imagem
        file = request.files['image']
        img_bytes = file.read()
        
        # Converter bytes para imagem OpenCV
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({'error': 'Falha ao decodificar imagem'}), 400
        
        # Converter BGR para RGB (MediaPipe requer RGB)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Detectar faces
        faces = detector.detect_faces(img_rgb)
        
        if len(faces) > 0:
            x, y, w, h = faces[0]
            face_roi = img[y:y+h, x:x+w]
            
            if face_roi.size > 0:
                emotion, confidence, _ = detector.predict_emotion(face_roi)
                
                # Atualizar a emoção atual
                with emotion_lock:
                    current_emotion = {
                        'emotion': emotion, 
                        'confidence': round(confidence, 3)
                    }
                
                return jsonify(current_emotion)
        
        # Se nenhuma face for detectada
        with emotion_lock:
            current_emotion = {"emotion": "Neutral", "confidence": 0.0}
        
        return jsonify(current_emotion)
        
    except Exception as e:
        print(f"Erro no processamento: {e}")
        return jsonify({'error': 'Erro interno no processamento'}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)