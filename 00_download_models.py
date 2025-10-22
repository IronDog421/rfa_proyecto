

"https://github.com/YapaLab/yolo-face/releases/download/v0.0.0/yolov11s-face.pt"


from pathlib import Path


def main():

    script_dir = Path(__file__).resolve().parent

    models_dir = script_dir / "models"
    models_dir.mkdir(parents=False, exist_ok=True)

    # Download the YOLOv11s-face model if it doesn't exist
    face_model_path = models_dir / "yolov11s-face.pt"
    if not face_model_path.exists():
        import requests

        url = "https://github.com/YapaLab/yolo-face/releases/download/v0.0.0/yolov11s-face.pt"
        print(f"Downloading YOLOv11s-face model from {url}...")
        response = requests.get(url)
        with open(face_model_path, "wb") as f:
            f.write(response.content)
        print(f"Model downloaded and saved to {face_model_path}")
    else:
        print(f"Model already exists at {face_model_path}")

if __name__ == "__main__":
    main()