"""CNN-based Helmet and Mask Detection Classifier."""

from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 15
DATA_DIR = Path("dataset")
MODEL_PATH = "helmet_mask_cnn.keras"

def load_split(split, shuffle):
    folder = DATA_DIR / split
    if not folder.exists():
        raise FileNotFoundError(f"Missing folder: {folder.resolve()}")
    return keras.utils.image_dataset_from_directory(
        folder,
        labels="inferred",
        label_mode="binary",
        class_names=["with_helmet_mask", "without_helmet_mask"],
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        seed=42
    )

def main():
    train_ds = load_split("train", True)
    val_ds = load_split("validation", False)
    test_ds = load_split("test", False)

    print("Class mapping: with_helmet_mask = 0, without_helmet_mask = 1")
    print("Training batches:", tf.data.experimental.cardinality(train_ds).numpy())

    # CNN Architecture
    model = keras.Sequential([
        layers.Input(shape=(128, 128, 3)),
        layers.Rescaling(1.0 / 255),
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(MODEL_PATH, monitor="val_loss", save_best_only=True)
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks
    )
    model.save(MODEL_PATH)

    # Plot training metrics
    plt.figure()
    plt.plot(history.history["accuracy"], label="Training accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig("accuracy_curve.png", dpi=150)
    plt.close()

    plt.figure()
    plt.plot(history.history["loss"], label="Training loss")
    plt.plot(history.history["val_loss"], label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("loss_curve.png", dpi=150)
    plt.close()

    # Evaluation on test set
    test_loss, test_accuracy = model.evaluate(test_ds, verbose=1)
    print(f"\nTEST LOSS: {test_loss:.4f}")
    print(f"TEST ACCURACY: {test_accuracy * 100:.2f}%")

    y_true = np.concatenate([labels.numpy().reshape(-1) for _, labels in test_ds])
    probabilities = model.predict(test_ds).reshape(-1)
    y_pred = (probabilities >= 0.5).astype(int)

    print("\nClassification report:")
    print(classification_report(
        y_true, y_pred,
        target_names=["with_helmet_mask", "without_helmet_mask"],
        zero_division=0
    ))
    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred))

    with open("results.txt", "w", encoding="utf-8") as f:
        f.write("CNN HELMET/MASK CLASSIFIER RESULTS\n")
        f.write(f"Test loss: {test_loss:.4f}\n")
        f.write(f"Test accuracy: {test_accuracy * 100:.2f}%\n")
        f.write("Class 0: with_helmet_mask\n")
        f.write("Class 1: without_helmet_mask\n")
        f.write("\nClassification report:\n")
        f.write(classification_report(
            y_true, y_pred,
            target_names=["with_helmet_mask", "without_helmet_mask"],
            zero_division=0
        ))
        f.write("\nConfusion matrix:\n")
        f.write(str(confusion_matrix(y_true, y_pred)))

    print("\nSaved: helmet_mask_cnn.keras, accuracy_curve.png, loss_curve.png, results.txt")

if __name__ == "__main__":
    main()
