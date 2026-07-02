import argparse
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".ppm", ".bmp"}


def parse_args():
    parser = argparse.ArgumentParser(description="Inspect a CAML/raw image dataset.")
    parser.add_argument("--data_root", required=True, help="Dataset root to scan recursively.")
    parser.add_argument("--sample_per_folder", type=int, default=5, help="Number of dimensions to sample per folder.")
    return parser.parse_args()


def label_from_path(path):
    folder = path.parent.name.lower()
    name = path.name.lower()
    if folder in {"traina", "traina_img", "testa", "testa_img", "normal", "no"}:
        return "normal"
    if folder in {"trainb", "trainb_img", "testb", "testb_img", "abnormal", "yes", "tumor"}:
        return "abnormal"
    if name.startswith("no"):
        return "normal"
    if name.startswith("yes") or name.startswith("y"):
        return "abnormal"
    return "unknown"


def read_label_files(data_root):
    labels = {}
    for label_file in data_root.glob("*img-name_label.txt"):
        with open(label_file, "r", encoding="utf-8") as file:
            for line in file:
                parts = line.strip().split()
                if len(parts) >= 2:
                    labels[parts[0]] = "normal" if parts[1] == "0" else "abnormal"
    return labels


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {data_root}")

    label_file_labels = read_label_files(data_root)
    image_paths = [
        path
        for path in sorted(data_root.rglob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]

    folder_counts = Counter()
    extension_counts = Counter()
    inferred_labels = Counter()
    label_file_counts = Counter()
    dimensions = defaultdict(Counter)
    corrupt = []

    for path in image_paths:
        folder_counts[str(path.parent.relative_to(data_root))] += 1
        extension_counts[path.suffix.lower()] += 1
        inferred_labels[label_from_path(path)] += 1
        if path.name in label_file_labels:
            label_file_counts[label_file_labels[path.name]] += 1

        try:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                dimensions[str(path.parent.relative_to(data_root))][image.size] += 1
        except Exception as exc:
            corrupt.append((str(path), str(exc)))

    print(f"Dataset root: {data_root}")
    print(f"Total images: {len(image_paths)}")
    print(f"Image formats: {dict(extension_counts)}")
    print(f"Folders:")
    for folder, count in sorted(folder_counts.items()):
        print(f"  {folder}: {count}")
    print(f"Inferred class distribution: {dict(inferred_labels)}")
    if label_file_labels:
        print(f"Label-file class distribution: {dict(label_file_counts)}")
        print(f"Label entries: {len(label_file_labels)}")
    print("Sample dimensions by folder:")
    for folder, counts in sorted(dimensions.items()):
        shown = list(counts.items())[: args.sample_per_folder]
        formatted = ", ".join([f"{size[0]}x{size[1]}={count}" for size, count in shown])
        print(f"  {folder}: {formatted}")
    print(f"Corrupt images: {len(corrupt)}")
    for path, error in corrupt[:20]:
        print(f"  {path}: {error}")


if __name__ == "__main__":
    main()
