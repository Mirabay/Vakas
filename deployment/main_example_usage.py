from bed_classifier.functions import predict_and_save, load_model

# Import the functions from functions.py


def main():
    model = load_model(
        "/home/oskar/Documents/ITC/IA/reto_vacas/modelo/Vakas/deployment/model_acc_95.94.pth"
    )
    image_path = "/home/oskar/Documents/ITC/IA/reto_vacas/clasificador/fotos_originales/2024-02-07-06-45-03.jpg"
    json_path = "/home/oskar/Documents/ITC/IA/reto_vacas/modelo/Vakas/data_analysis/coordinates.json"
    predict_and_save(
        model,
        image_path,
        json_path,
        "/home/oskar/Documents/ITC/IA/reto_vacas/modelo/Vakas/deployment/predictions.csv",
    )


if __name__ == "__main__":
    main()
