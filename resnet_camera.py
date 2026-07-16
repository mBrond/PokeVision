import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18
from PIL import Image

# 1. Definir as mesmas classes do seu treinamento
classes = [
    'Abra', 'Aerodactyl', 'Alakazam', 'Arbok', 'Arcanine', 'Articuno', 'Beedrill', 'Bellsprout', 
    'Blastoise', 'Bulbasaur', 'Butterfree', 'Caterpie', 'Chansey', 'Charizard', 'Charmander', 
    'Charmeleon', 'Clefable', 'Clefairy', 'Cloyster', 'Cubone', 'Dewgong', 'Diglett', 'Ditto', 
    'Dodrio', 'Doduo', 'Dragonair', 'Dragonite', 'Dratini', 'Drowzee', 'Dugtrio', 'Eevee', 
    'Ekans', 'Electabuzz', 'Electrode', 'Exeggcute', 'Exeggutor', 'Farfetchd', 'Fearow', 
    'Flareon', 'Gastly', 'Gengar', 'Geodude', 'Gloom', 'Golbat', 'Goldeen', 'Golduck', 
    'Graveler', 'Grimer', 'Growlithe', 'Gyarados', 'Haunter', 'Hitmonchan', 'Hitmonlee', 
    'Horsea', 'Hypno', 'Ivysaur', 'Jigglypuff', 'Jolteon', 'Jynx', 'Kabutops', 'Kadabra', 
    'Kakuna', 'Kangaskhan', 'Kingler', 'Koffing', 'Lapras', 'Lickitung', 'Machamp', 'Machoke', 
    'Machop', 'Magikarp', 'Magmar'
]

# 2. Recriar a estrutura do modelo e carregar os pesos salvos
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = resnet18()
model.fc = nn.Linear(model.fc.in_features, len(classes))

# Substitua pelo nome do seu arquivo de pesos salvo durante o treino
model.load_state_dict(torch.load("model_epoch_5.pth", map_location=device)) 
model.to(device)
model.eval()

# 3. Pipeline de transformação idêntico ao de validação/teste (sem data augmentation)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  
])

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