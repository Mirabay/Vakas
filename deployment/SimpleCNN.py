"This file contains the model for the classification of the images"
import torch.nn as nn


# Modelo simple (CNN) para clasificación de las tres clases
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(
                3, 16, kernel_size=3, padding=1
            ),  # 3 canales de entrada, 16 filtros de 3x3
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.fc_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 225 * 475, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 3),  # 3 clases: vaca_de_pie, vaca_acostada, cama_vacia
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = self.fc_layer(x)
        return x
