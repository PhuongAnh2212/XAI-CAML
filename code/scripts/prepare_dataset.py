import argparse
import random
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".ppm", ".bmp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare user-provided normal/abnormal images in the CAML data layout."
    )
    parser.add_argument("--normal_dir", required=True, help="Folder with normal images.")
    parser.add_argument("--abnormal_dir", required=True, help="Folder with abnormal images.")
    parser.add_argument("--output_dir", required=True, help="Output CAML dataset root.")
    parser.add_argument("--test_ratio", type=float, default=0.2, help="Fraction of each class used for test.")
    parser.add_argument("--limit_per_class", type=int, default=None, help="Optional cap per class for smoke tests.")
    parser.add_argument("--seed", type=int, default=42, help="Random split seed.")
    parser.add_argument(
        "--copy_mode",
        choices=["copy", "symlink"],
        default="copy",
        help="Copy images or create symlinks into the CAML folder structure.",
    )
    return parser.parse_args()


def list_images(image_dir):
    image_dir = Path(image_dir)
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")
    images = [
        path
        for path in sorted(image_dir.rglob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if not images:
        raise ValueError(f"No supported image files found in: {image_dir}")
    return images


def split_images(images, test_ratio, limit_per_class, rng):
    images = list(images)
    rng.shuffle(images)
    if limit_per_class is not None:
        images = images[:limit_per_class]
    test_count = int(round(len(images) * test_ratio))
    if len(images) > 1:
        test_count = min(max(test_count, 1), len(images) - 1)
    else:
        test_count = 0
    return images[test_count:], images[:test_count]


def clean_output_dirs(output_dir):
    for folder_name in ["trainA_img", "trainB_img", "testA_img", "testB_img"]:
        folder = output_dir / folder_name
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir(parents=True, exist_ok=True)


def unique_name(destination_dir, prefix, source_path, used_names):
    stem = source_path.stem
    suffix = source_path.suffix.lower()
    candidate = f"{prefix}_{stem}{suffix}"
    counter = 1
    while candidate in used_names or (destination_dir / candidate).exists():
        candidate = f"{prefix}_{stem}_{counter}{suffix}"
        counter += 1
    used_names.add(candidate)
    return candidate


def place_images(images, destination_dir, label, prefix, copy_mode, used_names):
    label_rows = []
    for source_path in images:
        image_name = unique_name(destination_dir, prefix, source_path, used_names)
        destination_path = destination_dir / image_name
        if copy_mode == "symlink":
            destination_path.symlink_to(source_path.resolve())
        else:
            shutil.copy2(source_path, destination_path)
        label_rows.append((image_name, label))
    return label_rows


def write_labels(path, rows):
    with open(path, "w", encoding="utf-8") as file:
        for image_name, label in rows:
            file.write(f"{image_name} {label}\n")


def main():
    args = parse_args()
    if not 0 <= args.test_ratio < 1:
        raise ValueError("--test_ratio must be in [0, 1).")

    rng = random.Random(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    clean_output_dirs(output_dir)

    normal_images = list_images(args.normal_dir)
    abnormal_images = list_images(args.abnormal_dir)

    normal_train, normal_test = split_images(normal_images, args.test_ratio, args.limit_per_class, rng)
    abnormal_train, abnormal_test = split_images(abnormal_images, args.test_ratio, args.limit_per_class, rng)

    train_rows = []
    test_rows = []
    used_names = set()
    train_rows.extend(
        place_images(normal_train, output_dir / "trainA_img", 0, "normal", args.copy_mode, used_names)
    )
    train_rows.extend(
        place_images(abnormal_train, output_dir / "trainB_img", 1, "abnormal", args.copy_mode, used_names)
    )
    test_rows.extend(
        place_images(normal_test, output_dir / "testA_img", 0, "normal", args.copy_mode, used_names)
    )
    test_rows.extend(
        place_images(abnormal_test, output_dir / "testB_img", 1, "abnormal", args.copy_mode, used_names)
    )

    write_labels(output_dir / "trainAB_img-name_label.txt", train_rows)
    write_labels(output_dir / "testAB_img-name_label.txt", test_rows)

    print(f"Prepared CAML dataset at: {output_dir}")
    print(f"trainA_img normal: {len(normal_train)}")
    print(f"trainB_img abnormal: {len(abnormal_train)}")
    print(f"testA_img normal: {len(normal_test)}")
    print(f"testB_img abnormal: {len(abnormal_test)}")
    print(f"train labels: {output_dir / 'trainAB_img-name_label.txt'}")
    print(f"test labels: {output_dir / 'testAB_img-name_label.txt'}")


if __name__ == "__main__":
    main()
