"This script generates a CSV file with the predictions of the model for a folder of images"
import sys
import os
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deployment.functions import *


def predict_images_in_folder(folder_path, model):
    image_files = [
        f for f in os.listdir(folder_path) if f.endswith(".jpg") or f.endswith(".png")
    ]
    for filename in tqdm(image_files, desc="Processing images"):
        image_path = os.path.join(folder_path, filename)
        predict_and_save(
            model,
            image_path,
            "C:/Users/urigo/Documents/Vakas/data_analysis/coordinates.json",
            "C:/Users/urigo/Documents/Vakas/data_analysis/predictionsFull-Dataset.csv",
        )


# folder_path = "C:/Users/urigo/Downloads/Bed/Bed"
folder_path = "C:\\Users\\urigo\\Downloads\\Bed\\Bed"
model = load_model("C:\\Users\\urigo\\Documents\\Vakas\\Modelos\\model_acc_95.94.pth")
predict_images_in_folder(folder_path, model)
