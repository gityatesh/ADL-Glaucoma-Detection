from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "dataset"

splits = ["train", "validate", "test"]
classes = ["normal", "glaucoma"]

valid_extensions = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff"
}

bad_images = []

print("\n======================================")
print("STRICT IMAGE INTEGRITY CHECK")
print("======================================\n")

total_checked = 0

for split in splits:
    for class_name in classes:

        folder = DATASET_DIR / split / class_name

        print(f"Checking: {split}/{class_name}")

        for path in folder.iterdir():

            if path.suffix.lower() not in valid_extensions:
                continue

            total_checked += 1

            try:
                # Actually decode the entire image
                with Image.open(path) as img:
                    img.load()

                # Also verify RGB conversion used by ImageFolder
                with Image.open(path) as img:
                    img.convert("RGB").load()

            except Exception as e:
                bad_images.append((path, str(e)))

                print("\nBAD IMAGE FOUND:")
                print(path)
                print("Error:", e)
                print()

print("\n======================================")
print("RESULT")
print("======================================")

print(f"Images checked: {total_checked}")
print(f"Bad images: {len(bad_images)}")

if bad_images:
    print("\nProblem files:\n")

    for path, error in bad_images:
        print(path)
        print(" ->", error)
else:
    print("\nAll images passed strict decoding.")