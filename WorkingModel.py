from PIL import Image

def crop_image(image_path, coordinates, output_path):
    """
    Crop the image at the given path using the provided coordinates and save it to the output path.
    
    :param image_path: Path to the input image.
    :param coordinates: Tuple of (left, upper, right, lower) coordinates for cropping.
    :param output_path: Path to save the cropped image.
    """
    with Image.open(image_path) as img:
        cropped_img = img.crop(coordinates)
        return cropped_img.save(output_path)

# Example usage:
# crop_image("path/to/image.jpg", (100, 100, 400, 400), "path/to/output.jpg")
