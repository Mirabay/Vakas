import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# make a cofusion matrix
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Definir transformaciones para las imágenes
transform = transforms.Compose([
    transforms.Resize((450, 950)),  # Resize the images
    transforms.RandomHorizontalFlip(p=0.5),  # Apply horizontal flip with a 50% chance
    transforms.RandomVerticalFlip(p=0.5),    # Apply vertical flip with a 50% chance (optional)
    transforms.RandomRotation(degrees=30),   # Randomly rotate the image by up to 30 degrees
    transforms.ToTensor(),                   # Convert image to tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize the values
])

# Cargar los conjuntos de datos desde las carpetas separadas
train_dataset = ImageFolder(root='dataset_split\\train', transform=transform)
validation_dataset = ImageFolder(root='dataset_split\\validation', transform=transform)
test_dataset = ImageFolder(root='dataset_split\\test', transform=transform)

# Verifica que las clases están correctamente identificadas
print(f"Clases encontradas: {train_dataset.classes}")

# Crear los DataLoader para cada conjunto de datos
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Modelo simple (CNN) para clasificación de las tres clases
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
            nn.Linear(16 * 225 * 475, 128),
            nn.ReLU(),
            nn.Linear(128, 3)  # 3 clases: vaca_de_pie, vaca_acostada, cama_vacia
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = self.fc_layer(x)
        return x
    
    # Inicializar el modelo, criterio (loss) y optimizador
model = SimpleCNN()
criterion = nn.CrossEntropyLoss()  # Para clasificación multiclase
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Entrenar con Gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
print(f'Usando el dispositivo: {device}')

# Hiperparámetros
num_epochs = 25

# Entrenamiento del modelo
for epoch in range(num_epochs):
    model.train()  # Modo de entrenamiento
    running_loss = 0.0

    # Usar tqdm para la barra de progreso
    for images, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch'):
        images, labels = images.to(device), labels.to(device)  # Move to the same device
        optimizer.zero_grad()  # Resetear gradientes
        outputs = model(images)  # Forward
        loss = criterion(outputs, labels)  # Calcular pérdida
        loss.backward()  # Backpropagation
        optimizer.step()  # Actualizar pesos

        running_loss += loss.item()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}')

    # Evaluación en el conjunto de validación
    model.eval()  # Modo de evaluación
    correct = 0
    total = 0

    with torch.no_grad():  # No se calculan gradientes en evaluación
        for images, labels in validation_loader:
            images, labels = images.to(device), labels.to(device)  # Move to the same device
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)  # Predicción
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    validation_accuracy = 100 * correct / total
    print(f'Validation Accuracy: {validation_accuracy}%')

    # Guardar el modelo cada 3 epochs
    if (epoch + 1) % 3 == 0:
        torch.save(model.state_dict(), f'model_epoch_{epoch+1}.pth')
        print(f'Model saved at epoch {epoch+1}')
        


model.eval()  # Modo de evaluación
y_true = []
y_pred = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)  # Move to the same device
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        y_true += labels.cpu().numpy().tolist()
        y_pred += predicted.cpu().numpy().tolist()

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=train_dataset.classes, yticklabels=train_dataset.classes)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

