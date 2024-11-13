import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deployment.functions import *


def predict_images_in_folder(folder_path, model):
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(folder_path, filename)
            predict_and_save(
                model,
                image_path,
                "/home/oskar/Documents/ITC/IA/reto_vacas/modelo/Vakas/data_analysis/coordinates.json",
                "/home/oskar/Documents/ITC/IA/reto_vacas/modelo/Vakas/data_analysis/predictions.csv",
            )


folder_path = "/home/oskar/Documents/ITC/IA/reto_vacas/clasificador/fotos_originales"
model = load_model(
    "/home/oskar/Documents/ITC/IA/reto_vacas/modelo/Vakas/data_analysis/model2_acc_98.14.pth"
)
predict_images_in_folder(folder_path, model)
