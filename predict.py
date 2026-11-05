from pathlib import Path
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras

MODEL_PATH = Path("helmet_mask_cnn.keras")
DEFAULT_IMAGE = Path("dataset/test/with_helmet_mask/with_helmet_mask_11.jpg")
IMAGE_SIZE = (128, 128)
CLASS_NAMES = ["Wearing helmet/mask", "Not wearing helmet/mask"]

def main():
    image_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_IMAGE
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}. Run train.py first.")

    # Load model and run inference
    model = keras.models.load_model(MODEL_PATH)
    image = keras.utils.load_img(image_path, target_size=IMAGE_SIZE)
    array = keras.utils.img_to_array(image)
    array = np.expand_dims(array, axis=0)

    probability_class_1 = float(model.predict(array, verbose=0)[0][0])
    predicted_index = 1 if probability_class_1 >= 0.5 else 0
    confidence = probability_class_1 if predicted_index == 1 else (1.0 - probability_class_1)

    print("=" * 45)
    print("HELMET / MASK DETECTION RESULT")
    print("=" * 45)
    print("Image     :", image_path.name)
    print("Prediction:", CLASS_NAMES[predicted_index])
    print(f"Confidence: {confidence * 100:.2f}%")
    print(f"Raw Score : {probability_class_1:.4f}")
    print("=" * 45)

if __name__ == "__main__":
    main()
