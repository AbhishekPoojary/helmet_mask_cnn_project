from pathlib import Path
import random
import shutil

DOWNLOADS = Path.home() / "Downloads"
HELMET_SOURCE = DOWNLOADS / "helmet" / "JPEGImages"
NO_HELMET_SOURCE = DOWNLOADS / "nohelmet"
DATASET = Path("dataset")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
random.seed(42)

def get_images(folder):
    return [
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]

def copy_split(images, class_name):
    random.shuffle(images)
    total = len(images)
    train_end = int(total * 0.70)
    val_end = int(total * 0.85)

    splits = {
        "train": images[:train_end],
        "validation": images[train_end:val_end],
        "test": images[val_end:]
    }

    for split, files in splits.items():
        destination = DATASET / split / class_name
        destination.mkdir(parents=True, exist_ok=True)
        for i, source in enumerate(files):
            shutil.copy2(source, destination / f"{class_name}_{i}.jpg")
        print(f"{split}/{class_name}: {len(files)} images")

def main():
    if not HELMET_SOURCE.exists():
        print("Helmet folder not found:", HELMET_SOURCE)
        return

    if not NO_HELMET_SOURCE.exists():
        print("No-helmet folder not found:", NO_HELMET_SOURCE)
        return

    helmet_images = get_images(HELMET_SOURCE)
    nohelmet_images = get_images(NO_HELMET_SOURCE)

    print("Helmet images found:", len(helmet_images))
    print("No-helmet images found:", len(nohelmet_images))

    count = min(len(helmet_images), len(nohelmet_images))
    if count < 20:
        print("Not enough images. Please check the source folders.")
        return

    # Balance classes and split into train (70%), validation (15%), test (15%)
    helmet_images = random.sample(helmet_images, count)
    nohelmet_images = random.sample(nohelmet_images, count)

    copy_split(helmet_images, "with_helmet_mask")
    copy_split(nohelmet_images, "without_helmet_mask")

    print("\nDataset preparation completed!")

if __name__ == "__main__":
    main()
