import argparse
import csv
import os
from collections import Counter


IMG_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".ppm",
    ".bmp",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", type=str, default="True", help="Use gpu or not")
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Root folder containing CAML test image folders and label txt files",
    )
    parser.add_argument("--output_dir", type=str, default="CL_Analysis/results", help="Output directory for latent CSV")
    parser.add_argument(
        "--image_path",
        type=str,
        default=None,
        help="Optional combined image folder. If set, A_img_path and B_img_path are ignored.",
    )
    parser.add_argument("--A_img_path", "--testA_path", dest="A_img_path", type=str, default=None)
    parser.add_argument("--B_img_path", "--testB_path", dest="B_img_path", type=str, default=None)
    parser.add_argument(
        "--AB_image_name_label_path",
        "--label_path",
        "--test_label_file",
        dest="AB_image_name_label_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--latentAB_save_path",
        "--output_csv",
        dest="latentAB_save_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--CAML_trained_gen_model_path",
        "--checkpoint",
        "--checkpoint_path",
        dest="CAML_trained_gen_model_path",
        type=str,
        default="trained_models/CAML_brain_trained_model.pt",
    )
    parser.add_argument("--style_dim", type=int, default=8)
    parser.add_argument("--train_is", type=str, default="False")
    parser.add_argument(
        "--print_paths",
        type=str,
        default="True",
        help="Print each image path loaded for latent extraction.",
    )
    parser.add_argument("--print_limit", type=int, default=20, help="Number of image paths to print.")
    parser.add_argument(
        "--validate_only",
        type=str,
        default="False",
        help="Only resolve paths and labels; do not load PyTorch/checkpoint or write latents.",
    )
    return parser.parse_args()


def resolve_data_root(data_root):
    if data_root is None:
        return None
    if os.path.isdir(data_root):
        return data_root
    parent_candidate = os.path.join("..", data_root)
    if os.path.isdir(parent_candidate):
        return parent_candidate
    return data_root


def resolve_data_subdir(data_root, explicit_path, preferred_name, fallback_name):
    if explicit_path:
        return explicit_path
    preferred_path = os.path.join(data_root, preferred_name)
    if os.path.isdir(preferred_path):
        return preferred_path
    return os.path.join(data_root, fallback_name)


def resolve_options(opts):
    if opts.data_root is None and (opts.image_path is None or opts.AB_image_name_label_path is None):
        raise ValueError(
            "Provide --data_root, or provide both --image_path and --label_path/--test_label_file."
        )
    if opts.data_root is not None:
        opts.data_root = resolve_data_root(opts.data_root)
        opts.A_img_path = resolve_data_subdir(opts.data_root, opts.A_img_path, "testA_img", "testA")
        opts.B_img_path = resolve_data_subdir(opts.data_root, opts.B_img_path, "testB_img", "testB")
        opts.AB_image_name_label_path = opts.AB_image_name_label_path or os.path.join(
            opts.data_root, "testAB_img-name_label.txt"
        )
    opts.latentAB_save_path = opts.latentAB_save_path or os.path.join(
        opts.output_dir, "testAB_CL_codes_extraction_results.csv"
    )
    return opts


def read_labels(label_path):
    if not os.path.exists(label_path):
        raise FileNotFoundError("Label file does not exist: " + label_path)

    labels = {}
    with open(label_path, encoding="utf-8") as file_name_label:
        for line_number, line in enumerate(file_name_label, start=1):
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) < 2:
                raise ValueError(f"Invalid label line {line_number} in {label_path}: {line.strip()}")
            labels[parts[0]] = parts[1]
    return labels


def is_image_file(file_name):
    return os.path.splitext(file_name)[1].lower() in IMG_EXTENSIONS


def collect_image_paths(opts, labels):
    if opts.image_path:
        image_dirs = [opts.image_path]
    else:
        image_dirs = [opts.A_img_path, opts.B_img_path]

    records = []
    for image_dir in image_dirs:
        if not os.path.isdir(image_dir):
            raise NotADirectoryError("Image directory does not exist: " + image_dir)
        for image_name in sorted(os.listdir(image_dir)):
            if not is_image_file(image_name):
                continue
            if image_name not in labels:
                raise KeyError("Image is missing from label file: " + image_name)
            records.append(
                {
                    "image_name": image_name,
                    "image_path": os.path.join(image_dir, image_name),
                    "label": labels[image_name],
                }
            )
    return records


def print_path_summary(opts, records):
    label_counts = Counter(record["label"] for record in records)
    print("resolved_data_root: " + str(opts.data_root))
    print("resolved_testA_path: " + opts.A_img_path)
    print("resolved_testB_path: " + opts.B_img_path)
    print("resolved_label_file: " + opts.AB_image_name_label_path)
    print("resolved_output_csv: " + opts.latentAB_save_path)
    print("resolved_checkpoint_path: " + opts.CAML_trained_gen_model_path)
    print("images_to_extract: " + str(len(records)))
    print("input_class_counts: " + dict(label_counts).__repr__())
    if opts.print_paths == "True":
        for record in records[: opts.print_limit]:
            print("loading_image_path: " + record["image_path"] + " label=" + record["label"])


def verify_output_csv(latent_csv, input_records, style_dim):
    expected_names = [record["image_name"] for record in input_records]
    expected_labels = Counter(record["label"] for record in input_records)
    with open(latent_csv, newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        fieldnames = reader.fieldnames or []

    latent_columns = [column for column in fieldnames if column not in ["image_name", "label"]]
    output_names = [row["image_name"] for row in rows]
    output_labels = Counter(row["label"] for row in rows)

    missing_names = sorted(set(expected_names) - set(output_names))
    extra_names = sorted(set(output_names) - set(expected_names))
    print("verify_csv_exists: " + str(os.path.exists(latent_csv)))
    print("verify_csv_rows: " + str(len(rows)))
    print("verify_latent_dimensions: " + str(len(latent_columns)))
    print("verify_expected_latent_dimensions: " + str(style_dim))
    print("verify_output_class_counts: " + dict(output_labels).__repr__())
    print("verify_expected_class_counts: " + dict(expected_labels).__repr__())
    print("verify_missing_image_names: " + missing_names.__repr__())
    print("verify_extra_image_names: " + extra_names.__repr__())
    if len(latent_columns) != style_dim:
        raise ValueError("Latent dimension count does not match --style_dim.")
    if missing_names or extra_names:
        raise ValueError("Latent CSV image names do not match resolved input image names.")
    if output_labels != expected_labels:
        raise ValueError("Latent CSV class counts do not match resolved input labels.")


def main():
    opts = resolve_options(parse_args())
    labels = read_labels(opts.AB_image_name_label_path)
    records = collect_image_paths(opts, labels)
    print_path_summary(opts, records)

    if opts.validate_only == "True":
        print("Validation finished without latent extraction.")
        return

    import torch
    from PIL import Image
    from torch.autograd import Variable
    from torchvision import transforms

    from trainer_exchange import trainer

    if opts.cuda == "True" and torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    caml_trainer = trainer(
        device=device,
        style_dim=opts.style_dim,
        optim_para=None,
        gen_loss_weight_para=None,
        dis_loss_weight_para=None,
        train_is=opts.train_is,
    )
    caml_trainer.to(device)
    state_dict_gen = torch.load(opts.CAML_trained_gen_model_path, map_location=device)
    caml_trainer.gen.load_state_dict(state_dict_gen["ab"])
    caml_trainer.eval()
    encode = caml_trainer.gen.encode

    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    os.makedirs(os.path.dirname(os.path.abspath(opts.latentAB_save_path)), exist_ok=True)
    output_label_counts = Counter()
    with open(opts.latentAB_save_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        columns_name_list = ["image_name"]
        for latent_ids in range(opts.style_dim):
            columns_name_list.append(str(latent_ids))
        columns_name_list.append("label")
        writer.writerow(columns_name_list)

        with torch.no_grad():
            for index, record in enumerate(records, start=1):
                example_img = Variable(
                    transform(Image.open(record["image_path"]).convert("RGB")).unsqueeze(0).to(device)
                )
                _, style_code = encode(example_img)

                row_list = [record["image_name"]]
                for latent_index in range(style_code.size(1)):
                    row_list.append(str(style_code[0][latent_index].item()))
                row_list.append(record["label"])
                writer.writerow(row_list)
                output_label_counts[record["label"]] += 1

                print(str(index) + "  " + record["image_name"] + " label=" + record["label"])

    print("Latents(class-associated codes) extraction finished")
    print("latent_csv: " + opts.latentAB_save_path)
    print("latent_rows: " + str(len(records)))
    print("latent_dimensions: " + str(opts.style_dim))
    print("output_class_counts: " + dict(output_label_counts).__repr__())
    verify_output_csv(opts.latentAB_save_path, records, opts.style_dim)


if __name__ == "__main__":
    main()
