from bed_classifier.functions import predict_and_save, load_model


def main():
    path_to_model_weights = "path/to/model_weights.pth"
    image_path = "path/to/image.jpg"
    coordinates_path = "path/to/coordinates.json"
    path_to_database = (
        "path/to/database.csv"  # This will be created if it does not exist
    )

    model = load_model(path_to_model_weights)
    predict_and_save(
        model,
        image_path,
        coordinates_path,
        path_to_database,
    )


if __name__ == "__main__":
    main()
