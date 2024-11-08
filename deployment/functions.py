import torch
from torchvision import transforms
from PIL import Image
from VakaaCNN import SimpleCNN  # TODO: Fix the import


def load_model(model_path):
    model = SimpleCNN()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def predict(model, image_path):
    transform = transforms.Compose(
        [
            transforms.Resize(950, 450),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)

    return predicted.item()
