import torch
from torchvision import transforms
from PIL import Image
from .SimpleCNN import SimpleCNN
import json
from PIL import Image
import csv
import os
from datetime import datetime
import cv2
import numpy as np


# Load the model. Returns the model
def load_model(model_path, device="cpu"):
    model = SimpleCNN()
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    model.eval()
    return model


# Predict the class of an image from a PIL image. Returns the class index
def predict_from_image(model, image):
    transform = transforms.Compose(
        [
            transforms.Resize((950, 450)),  # Corrected to use a tuple
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    image = transform(image).unsqueeze(0)  # Add batch dimension
    clases = ["cama_vacia", "vaca_acostada", "vaca_parada"]
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        predictedClass = clases[predicted.item()]
    return predictedClass


# Crop images from a JSON file with coordinates. Returns a list of PIL images
def crop_images_from_json(image_path, json_path):
    # Load the image
    image = Image.open(image_path)

    # Load the JSON file
    with open(json_path, "r") as file:
        crop_data = json.load(file)

    images = []

    # Iterate through the list of crop coordinates
    for _, crop in enumerate(crop_data):
        x = crop["x"]
        y = crop["y"]
        width = crop["width"]
        height = crop["height"]
        # Crop the image
        cropped_image = image.crop((x, y, x + width, y + height))
        images.append(cropped_image)
    return images


# Predict the class of an image from a full image and a JSON file with coordinates. Stores the prediction in a csv file
def predict_and_save(model, image_path, coordinates_path, output_path):
    # Read the image
    img = cv2.imread(image_path)

    # Perspective transformation
    pts1 = np.float32([[0, 0], [0, 1920], [1080, 0], [1920, 1080]])
    pts2 = np.float32([[0, 0], [0, 1850], [1080, 0], [1850, 1080]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    img = cv2.warpPerspective(img, M, (1680, 950))

    # Save the transformed image to a temporary path
    temp_image_path = "/tmp/transformed_image.jpg"
    cv2.imwrite(temp_image_path, img)

    # Use the transformed image for further processing
    images = crop_images_from_json(temp_image_path, coordinates_path)

    predictions = []

    for image in images:
        prediction = predict_from_image(model, image)
        predictions.append(prediction)

    # Get the timestamp from the image path
    filename = os.path.basename(image_path)
    timestamp_str = filename.split(".")[0]

    # Convert the timestamp to a standardized format
    timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d-%H-%M-%S").strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    # Check if the CSV file exists
    if not os.path.exists(output_path):
        with open(output_path, mode="w", newline="") as file:
            writer = csv.writer(file)
            # Create column names from A to Z and then AA to ZZ if needed
            column_names = ["Timestamp"]
            for i in range(len(predictions)):
                if i < 26:
                    column_names.append(chr(65 + i))
                else:
                    column_names.append(chr(65 + (i // 26) - 1) + chr(65 + (i % 26)))
            writer.writerow(column_names)

    # Append the predictions to the CSV file
    with open(output_path, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([timestamp] + predictions)
