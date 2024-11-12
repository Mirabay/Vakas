import torch
from torchvision import transforms
from PIL import Image
from SimpleCNN import SimpleCNN
import json
from PIL import Image
import csv
import os


# Load the model. Returns the model
def load_model(model_path):
    model = SimpleCNN()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


# Predict the class of an image from a PIL image. Returns the class index
def predict_from_image(model, image):
    transform = transforms.Compose(
        [
            transforms.Resize((950, 450)),  # Corrected to use a tuple
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Corrected mean and std
        ]
    )

    image = transform(image).unsqueeze(0)  # Add batch dimension
    clases = ['cama_vacia','vaca_acostada','vaca_parada']
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output.data, 1)
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
    images = crop_images_from_json(image_path, coordinates_path)
    predictions = []

    for image in images:
        prediction = predict_from_image(model, image)
        predictions.append(prediction)

    # Check if the CSV file exists
    if not os.path.exists(output_path):
        with open(output_path, mode="w", newline="") as file:
            writer = csv.writer(file)
            # Create column names from A to Z and then AA to ZZ if needed
            column_names = []
            for i in range(len(predictions)):
                if i < 26:
                    column_names.append(chr(65 + i))
                else:
                    column_names.append(chr(65 + (i // 26) - 1) + chr(65 + (i % 26)))
            writer.writerow(column_names)

    # Append the predictions to the CSV file
    with open(output_path, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(predictions)
