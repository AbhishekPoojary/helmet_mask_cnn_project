import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

MODEL_FILE = Path("helmet_mask_cnn.keras")
DEFAULT_IMG = Path("dataset/test/with_helmet_mask/with_helmet_mask_11.jpg")
IMAGE_SIZE = (128, 128)
CLASSES = ["Wearing helmet/mask", "Not wearing helmet/mask"]

def find_file(path_candidate):
    p = Path(path_candidate)
    if p.exists():
        return p
    sub_p = Path("helmet_mask_cnn_project") / p
    if sub_p.exists():
        return sub_p
    return p

def main():
    img_path = Path(sys.argv[1]) if len(sys.argv) > 1 else find_file(DEFAULT_IMG)
    model_path = find_file(MODEL_FILE)

    if not model_path.exists():
        print(f"Error: Model not found at {model_path}. Please train first using train.py.")
        return
    if not img_path.exists():
        print(f"Error: Image not found at {img_path}.")
        return

    print("=" * 65)
    print("CNN VISUAL FEATURE EXTRACTION DEMONSTRATION")
    print("=" * 65)
    print(f"Input image : {img_path}")
    print(f"Model file  : {model_path}\n")

    model = keras.models.load_model(model_path)
    raw_img = keras.utils.load_img(img_path, target_size=IMAGE_SIZE)
    img_array = keras.utils.img_to_array(raw_img)
    x = np.expand_dims(img_array, axis=0)

    rescaling_layer = model.get_layer("rescaling")
    conv1_layer = model.get_layer("conv2d")
    W, b = conv1_layer.get_weights()

    x_norm = rescaling_layer(x)
    conv_all = (tf.nn.conv2d(x_norm, W, strides=[1, 1, 1, 1], padding="SAME") + b).numpy()[0]

    scores = []
    for i in range(W.shape[-1]):
        c = conv_all[:, :, i]
        pos_ratio = (c > 0).mean()
        scores.append(abs(pos_ratio - 0.35) - 0.5 * np.var(c))
    best_filter_idx = int(np.argmin(scores))

    kernel_weights = np.mean(W[:, :, :, best_filter_idx], axis=-1)
    conv_response = conv_all[:, :, best_filter_idx]
    feature_map = (conv_response - conv_response.min()) / (conv_response.max() - conv_response.min() + 1e-6)
    relu_response = np.maximum(0, conv_response)
    pool_response = tf.nn.max_pool2d(
        relu_response[np.newaxis, :, :, np.newaxis],
        ksize=2, strides=2, padding="SAME"
    )[0, :, :, 0].numpy()

    prob_class1 = float(model.predict(x, verbose=0)[0][0])
    pred_idx = 1 if prob_class1 >= 0.5 else 0
    confidence = prob_class1 if pred_idx == 1 else (1.0 - prob_class1)
    label = CLASSES[pred_idx]

    print("STEP-BY-STEP PROGRESSION:")
    print(f"1. Original Image : Size = {IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}x3")
    print(f"2. Kernel (Filter): Filter #{best_filter_idx}, Size = 3x3")
    print(f"3. Convolution    : Linear dot product (I * K + b), Value range = [{conv_response.min():.2f}, {conv_response.max():.2f}]")
    print(f"4. Feature Map    : Spatial response of detected patterns")
    print(f"5. ReLU Activation: f(x) = max(0, x), Negative values set to 0 (Kept range: [0, {relu_response.max():.2f}])")
    print(f"6. Max Pooling    : 2x2 window downsamples resolution to 64x64")
    print(f"7. Classification : Output score = {prob_class1:.4f} -> Prediction: {label} ({confidence * 100:.1f}%)")

    # 1. Visual Progression: Original Image -> Kernel -> Conv -> Feature Map -> ReLU -> Pooling -> Classification
    fig, axes = plt.subplots(1, 7, figsize=(21, 3.8))
    fig.suptitle(
        "Visual Progression: Original Image -> Kernel -> Convolution -> Feature Map -> ReLU -> Pooling -> Classification",
        fontsize=12, fontweight="bold", y=0.98
    )

    axes[0].imshow(raw_img)
    axes[0].set_title("1. Original Image\n(128x128x3)", fontsize=10, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(kernel_weights, cmap="coolwarm")
    axes[1].set_title(f"2. Kernel (3x3)\n(Filter #{best_filter_idx} Weights)", fontsize=10, fontweight="bold")
    for r in range(3):
        for c in range(3):
            val = kernel_weights[r, c]
            txt_color = "white" if abs(val) > 0.08 else "black"
            axes[1].text(c, r, f"{val:.2f}", ha="center", va="center", color=txt_color, fontsize=9, fontweight="bold")
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    axes[2].imshow(conv_response, cmap="viridis")
    axes[2].set_title("3. Convolution\n(I * K + b)", fontsize=10, fontweight="bold")
    axes[2].axis("off")

    axes[3].imshow(feature_map, cmap="plasma")
    axes[3].set_title("4. Feature Map\n(Linear response)", fontsize=10, fontweight="bold")
    axes[3].axis("off")

    axes[4].imshow(relu_response, cmap="inferno")
    axes[4].set_title("5. ReLU\nmax(0, x)", fontsize=10, fontweight="bold")
    axes[4].axis("off")

    axes[5].imshow(pool_response, cmap="inferno")
    axes[5].set_title("6. Pooling (2x2)\n(Downsampled 64x64)", fontsize=10, fontweight="bold")
    axes[5].axis("off")

    axes[6].axis("off")
    is_helmet = (pred_idx == 0)
    bg_color = "#e8f5e9" if is_helmet else "#ffebee"
    border_color = "#2e7d32" if is_helmet else "#c62828"
    text_color = "#1b5e20" if is_helmet else "#b71c1c"

    axes[6].text(
        0.5, 0.65,
        f"PREDICTION:\n{label}",
        ha="center", va="center", fontsize=11, fontweight="bold",
        color=text_color,
        bbox=dict(boxstyle="round,pad=0.6", facecolor=bg_color, edgecolor=border_color, lw=2)
    )
    axes[6].text(
        0.5, 0.28,
        f"Confidence: {confidence * 100:.1f}%\nSigmoid P: {prob_class1:.3f}",
        ha="center", va="center", fontsize=9.5
    )
    axes[6].set_title("7. Classification\n(Dense + Sigmoid)", fontsize=10, fontweight="bold")

    plt.tight_layout()
    prog_fig_path = Path("visual_progression.png")
    plt.savefig(prog_fig_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"\nSaved progression diagram to: {prog_fig_path.resolve()}")

    # 2. Progressive Feature Extraction Across Conv Layers
    extractor = keras.Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer("conv2d").output,
            model.get_layer("conv2d_1").output,
            model.get_layer("conv2d_2").output,
        ]
    )
    layer_acts = extractor.predict(x, verbose=0)

    fig, axes = plt.subplots(3, 4, figsize=(13, 9))
    fig.suptitle("Progressive Visual Feature Extraction Across CNN Layers", fontsize=14, fontweight="bold", y=0.99)
    layer_labels = [
        "Conv Layer 1 (128x128)\n[Low-Level: Edges & Textures]",
        "Conv Layer 2 (64x64)\n[Mid-Level: Curves & Contours]",
        "Conv Layer 3 (32x32)\n[High-Level: Abstract Patterns]"
    ]

    for r in range(3):
        acts = layer_acts[r][0]
        num_filters = acts.shape[-1]
        variances = [np.var(acts[:, :, ch]) for ch in range(num_filters)]
        top4 = np.argsort(variances)[-4:][::-1]

        for c_idx, ch in enumerate(top4):
            ax = axes[r, c_idx]
            ax.imshow(acts[:, :, ch], cmap="viridis")
            ax.set_title(f"Filter #{ch}", fontsize=9)
            ax.axis("off")

        axes[r, 0].text(-0.25, 0.5, layer_labels[r], transform=axes[r, 0].transAxes,
                        fontsize=10, fontweight="bold", va="center", ha="right")

    plt.tight_layout(rect=[0.14, 0, 1, 0.96])
    feat_fig_path = Path("progressive_features.png")
    plt.savefig(feat_fig_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved progressive features diagram to: {feat_fig_path.resolve()}")
    print("=" * 65)

if __name__ == "__main__":
    main()
