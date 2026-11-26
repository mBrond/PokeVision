import json
import sys
from pathlib import Path

import cv2
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]  # raiz do repositório
sys.path.insert(0, str(ROOT))

from classifier.treinar_resnet import CLASSES_JSON, PESOS, criar_modelo, eval_tf

# 1. Classes e pesos do modelo mais recente (os mesmos que a Pokédex usa)
classes = json.loads(CLASSES_JSON.read_text(encoding="utf-8"))

# 2. Recriar a estrutura do modelo e carregar os pesos salvos
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = criar_modelo(len(classes), device)
model.load_state_dict(torch.load(PESOS, map_location=device))
model.eval()

# 3. Pipeline de transformação idêntico ao de validação/teste (sem data augmentation)
transform = eval_tf

# 4. Inicializar a captura da Webcam (0 costuma ser a câmera padrão)
cap = cv2.VideoCapture(0)

print("Pressione 'q' na janela de vídeo para encerrar.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Erro ao acessar a webcam.")
        break
        
    # O OpenCV captura em BGR, precisamos converter para RGB para o PyTorch
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_frame)
    
    # Aplicar as transformações e adicionar a dimensão de batch (unsqueeze)
    tensor_img = transform(pil_img).unsqueeze(0).to(device)
    
    # Fazer a predição sem calcular gradientes
    with torch.no_grad():
        output = model(tensor_img)
        probabilities = torch.softmax(output, dim=1)
        confidence, prediction_idx = torch.max(probabilities, 1)
        
    class_name = classes[prediction_idx.item()]
    prob_value = confidence.item() * 100
    
    # Definir um threshold de confiança para evitar falsos positivos constantes
    if prob_value > 60.0:
        text = f"{class_name} ({prob_value:.1f}%)"
        color = (0, 255, 0) # Verde para alta confiança
    else:
        text = "Procurando Pokemon..."
        color = (0, 0, 255) # Vermelho se não tiver certeza
        
    # Escrever o resultado na tela do frame
    cv2.putText(frame, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
    
    # Mostrar o frame atualizado na janela
    cv2.imshow("Webcam - Classificador Pokemon", frame)
    
    # Pressione a tecla 'q' para sair do loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()