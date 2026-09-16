import os
from PIL import Image
from collections import defaultdict

# Change this if your dataset folder is somewhere else
DATASET_DIR = "dataset"

splits = ["train", "validate", "test"]
classes = ["normal", "glaucoma"]

valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

summary = defaultdict(dict)
corrupt_images = []
image_sizes = defaultdict(list)

print("\n==============================")
print("DATASET VERIFICATION REPORT")
print("==============================\n")

for split in splits:
    print(f"\nSplit: {split}")
    print("-" * 30)

    for class_name in classes:
        folder_path = os.path.join(DATASET_DIR, split, class_name)

        if not os.path.exists(folder_path):
            print(f"Missing folder: {folder_path}")
            continue

        image_files = [
            f for f in os.listdir(folder_path)
            if f.lower().endswith(valid_extensions)
        ]

        summary[split][class_name] = len(image_files)

        print(f"{class_name}: {len(image_files)} images")

        for img_name in image_files:
            img_path = os.path.join(folder_path, img_name)

            try:
                with Image.open(img_path) as img:
                    image_sizes[split].append(img.size)
            except Exception:
                corrupt_images.append(img_path)

print("\n==============================")
print("FINAL COUNT SUMMARY")
print("==============================")

total_images = 0

for split in splits:
    split_total = 0
    print(f"\n{split.upper()}")

    for class_name in classes:
        count = summary[split].get(class_name, 0)
        split_total += count
        print(f"{class_name}: {count}")

    total_images += split_total
    print(f"Total {split}: {split_total}")

print(f"\nTotal images in dataset: {total_images}")

print("\n==============================")
print("IMAGE SIZE CHECK")
print("==============================")

for split in splits:
    sizes = image_sizes[split]

    if sizes:
        unique_sizes = set(sizes)
        print(f"\n{split}:")
        print(f"Total readable images: {len(sizes)}")
        print(f"Number of unique image sizes: {len(unique_sizes)}")
        print(f"Sample sizes: {list(unique_sizes)[:10]}")
    else:
        print(f"\n{split}: No readable images found")

print("\n==============================")
print("CORRUPT IMAGE CHECK")
print("==============================")

if corrupt_images:
    print(f"Found {len(corrupt_images)} corrupt/unreadable images:\n")
    for path in corrupt_images:
        print(path)
else:
    print("No corrupt images found. Dataset is clean.")

print("\nVerification completed.")