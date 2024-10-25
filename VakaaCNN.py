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

# Make a sound when the training is done
import winsound
import os

# Crear la carpeta 'Modelos' si no existe
if not os.path.exists('Modelos'):
    os.makedirs('Modelos')

# Definir transformaciones para las imágenes
transform = transforms.Compose([
    transforms.Resize((950, 450)),  # Resize the images
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
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

# Modelo simple (CNN) para clasificación de las tres clases
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1), # 3 canales de entrada, 16 filtros de 3x3
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 225 * 475, 32), 
            nn.ReLU(),

            # nn.Linear(16, 32),
            # nn.ReLU(),

            nn.Linear(32, 64),
            nn.ReLU(),

            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 3),  # 3 clases: vaca_de_pie, vaca_acostada, cama_vacia
            # nn.Softmax(dim=1) # Softmax para clasificación multiclase
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
AcurracyTarget = 95

# Crear la carpeta 'TrainingHistory' si no existe
if not os.path.exists('TrainingHistory'):
    os.makedirs('TrainingHistory')

# Inicializar listas para guardar el historial de entrenamiento
train_losses = []
validation_accuracies = []

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

    epoch_loss = running_loss / len(train_loader)
    train_losses.append(epoch_loss)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss}')

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
    validation_accuracies.append(validation_accuracy)
    print(f'Validation Accuracy: {validation_accuracy}%')

    if validation_accuracy > AcurracyTarget:
        torch.save(model.state_dict(), f'Modelos/model_acc_{validation_accuracy:.2f}.pth')
        AcurracyTarget = validation_accuracy
        print(f'Model saved at epoch {epoch+1}')        

# Graficar el historial de entrenamiento
plt.figure(figsize=(12, 5))

# Pérdida de entrenamiento
plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss History')
plt.legend()
plt.grid(True)

# Precisión de validación
plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs + 1), validation_accuracies, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.title('Validation Accuracy History')
plt.legend()
plt.grid(True)

# Guardar las figuras
plt.savefig('TrainingHistory/training_history.png')
# plt.show()

frequency = 2500  # Set Frequency To 2500 Hertz
duration = 1000  # Set Duration To 1000 ms == 1 second
winsound.Beep(frequency, duration)

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

