import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image
import torchvision.transforms as transforms
import os

class EmotionClassifier(nn.Module):
    """Mesmo modelo usado no treinamento"""
    
    def __init__(self, num_classes=4, dropout=0.3):
        super(EmotionClassifier, self).__init__()
        
        # EfficientNet-B0
        self.backbone = efficientnet_b0(pretrained=False)  # Não precisamos do pretrained agora
        
        # Substituir classificador
        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_features, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout/2),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

def load_trained_model(model_path='emotion_model_complete.pth'):
    """Carrega modelo treinado com todas as informações"""
    
    print(f"🔄 Carregando modelo de: {model_path}")
    
    # Verificar se arquivo existe
    if not os.path.exists(model_path):
        print(f"❌ Arquivo não encontrado: {model_path}")
        print("💡 Certifique-se de ter baixado o modelo do Colab")
        return None, None
    
    # Carregar modelo completo
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Extrair informações
    model_state = checkpoint['model_state_dict']
    emotion_labels = checkpoint['emotion_labels']
    normalization = checkpoint['normalization']
    
    print(f"✅ Informações do modelo:")
    print(f"  • Arquitetura: {checkpoint['model_architecture']}")
    print(f"  • Classes: {emotion_labels}")
    print(f"  • Input size: {checkpoint['input_size']}")
    
    # Recriar modelo
    model = EmotionClassifier(num_classes=len(emotion_labels))
    model.load_state_dict(model_state)
    model.eval()
    
    # Criar transformações
    transform = transforms.Compose([
        transforms.Resize(checkpoint['input_size']),
        transforms.ToTensor(),
        transforms.Normalize(mean=normalization['mean'], 
                           std=normalization['std'])
    ])
    
    print("✅ Modelo carregado com sucesso!")
    
    return model, {
        'labels': emotion_labels,
        'transform': transform,
        'normalization': normalization
    }

class LocalEmotionDetector:
    """Sistema completo de detecção de emoções local"""
    
    def __init__(self, model_path='emotion_model_complete.pth'):
        print("🚀 Inicializando sistema local...")
        
        # Carregar modelo
        self.model, self.model_info = load_trained_model(model_path)
        
        if self.model is None:
            raise Exception("Falha ao carregar modelo")
        
        self.emotion_labels = self.model_info['labels']
        self.transform = self.model_info['transform']
        
        # Configurar MediaPipe
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.5
        )

        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
)
        
        print("✅ Sistema inicializado!")
        print(f"📋 Emoções detectáveis: {self.emotion_labels}")
    
    def detect_faces(self, image):
        """Detecta faces usando MediaPipe"""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_image)
        
        faces = []
        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                h, w, _ = image.shape
                
                x = max(0, int(bbox.xmin * w))
                y = max(0, int(bbox.ymin * h))
                width = min(w - x, int(bbox.width * w))
                height = min(h - y, int(bbox.height * h))
                
                faces.append((x, y, width, height))
        
        return faces
    
    def extract_landmarks(self, image):
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_image)
        
        landmarks = []
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                h, w, _ = image.shape
                for landmark in face_landmarks.landmark:
                    x, y = int(landmark.x * w), int(landmark.y * h)
                    landmarks.append((x, y))
        
        return landmarks
    
    def predict_emotion(self, face_image):
        """Prediz emoção de uma face"""
        # Converter BGR para RGB
        if len(face_image.shape) == 3:
            face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        else:
            face_rgb = face_image
        
        # Converter para PIL e depois grayscale
        pil_image = Image.fromarray(face_rgb)
        gray_image = pil_image.convert('L')
        rgb_image = Image.merge('RGB', (gray_image, gray_image, gray_image))
        
        # Aplicar transformações
        tensor = self.transform(rgb_image).unsqueeze(0)
        
        # Predição
        with torch.no_grad():
            output = self.model(tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        emotion = self.emotion_labels[predicted.item()]
        conf_score = confidence.item()
        all_probs = probabilities[0].numpy()
        
        return emotion, conf_score, all_probs
    
    def run_webcam(self):
        """Executa detecção em tempo real via webcam"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Erro: Webcam não acessível")
            return
        
        print("🎥 Webcam iniciada! Pressione 'q' para sair")
        
        frame_count = 0
        skip_frames = 10  # Processar a cada 10 frames (~3 FPS se webcam = 30fps)
        
        current_emotion = "Neutral"
        current_confidence = 0.0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)  # Espelhar
            
            # Detectar faces
            faces = self.detect_faces(frame)
            
            # Processar emoção a cada N frames
            if frame_count % skip_frames == 0 and len(faces) > 0:
                x, y, w, h = faces[0]  # Primeira face
                face_roi = frame[y:y+h, x:x+w]
                
                if face_roi.size > 0:
                    try:
                        emotion, confidence, _ = self.predict_emotion(face_roi)
                        current_emotion = emotion
                        current_confidence = confidence
                    except Exception as e:
                        print(f"Erro na predição: {e}")
            
            # Desenhar resultados
            for (x, y, w, h) in faces:
                # Cor baseada na confiança
                color = (0, 255, 0) if current_confidence > 0.6 else (0, 255, 255)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                
                # Texto
                text = f"{current_emotion}: {current_confidence:.2f}"
                cv2.rectangle(frame, (x, y-30), (x+200, y), color, -1)
                cv2.putText(frame, text, (x+5, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.6, (0, 0, 0), 2)
            
            # Info na tela
            cv2.putText(frame, "Pressione 'q' para sair", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow('Detecção Local de Emoções', frame)
            
            frame_count += 1
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Webcam finalizada")

if __name__ == "__main__":
    detector = LocalEmotionDetector()
    detector.run_webcam()