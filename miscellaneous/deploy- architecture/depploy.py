import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import time

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 225 * 475, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = self.fc_layer(x)
        return x

# Definir las transformaciones
transform = transforms.Compose([
    transforms.Resize((950, 450)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Función para cargar y predecir la clase de una imagen
def predict_image(image_path, model):
    # Cargar y transformar la imagen
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    image = image.unsqueeze(0)

    # Poner el modelo en modo evaluación y hacer la predicción
    model.eval()
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)

    return predicted.item()

# Configurar el dispositivo para CPU
device = torch.device('cpu')

start_model = time.time()

# Cargar el modelo en CPU
model_path = 'model.pth'
model = SimpleCNN()
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)  # Asegurar que el modelo esté en CPU

# Lista de imágenes para predecir
images = ['vacia.jpg', 'ac2.jpg', 'pie3.jpg']
clases = ['cama_vacia', 'vaca_acostada', 'vaca_de_pie']

# Medir el tiempo de inicio
start_time = time.time()

# Predecir la clase de cada imagen
for image in images:
    predicted_class = predict_image(image, model)
    print(f"Predicción: {clases[predicted_class]}")

# Medir el tiempo de fin
end_time = time.time()
execution_time = end_time - start_time
execution_model = end_time - start_model

print(f"Tiempo de cargar el modelo: {execution_model:.4f} segundos")
print(f"Tiempo de ejecución: {execution_time:.4f} segundos")
