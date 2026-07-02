import argparse
import json
import os
import sys

import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate local counterfactual explanations along a hypergraph shortest path."
    )
    parser.add_argument("--cuda", type=str, default="True", help="Use cuda:0 if available.")
    parser.add_argument("--image_dir", required=True, help="Directory containing source/target images.")
    parser.add_argument("--hypergraph", required=True, help="Hypergraph JSON from CL_Analysis/hypergraph_builder.py.")
    parser.add_argument("--path_file", required=True, help="CSV from hypergraph_shortest_path.py.")
    parser.add_argument("--checkpoint", required=True, help="Trained CAML generator checkpoint.")
    parser.add_argument("--classifier", required=True, help="Black-box classifier checkpoint to explain.")
    parser.add_argument("--source_image", default=None, help="Source image filename. Defaults to first path row.")
    parser.add_argument("--target_image", default=None, help="Target image filename. Defaults to first path row.")
    parser.add_argument("--style_dim", type=int, default=8)
    parser.add_argument("--output_dir", required=True, help="Directory for generated images and saliency maps.")
    return parser.parse_args()


def select_device(cuda_flag):
    import torch

    if cuda_flag == "True" and torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def heatmap_show(image, difference_map):
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("OpenCV is required for saliency heatmap generation. Install opencv-python.") from exc

    image = image.data.cpu().numpy()
    saliency_map = difference_map.data.cpu().numpy()
    saliency_map = saliency_map - saliency_map.min()
    max_value = saliency_map.max()
    if max_value > 0:
        saliency_map = saliency_map / max_value
    saliency_map = saliency_map.clip(0, 1)
    saliency_map = np.uint8(saliency_map * 255).transpose(1, 2, 0)
    image = np.uint8(image * 255).transpose(1, 2, 0)
    color_heatmap = cv2.applyColorMap(saliency_map, cv2.COLORMAP_JET)
    image_with_heatmap = np.float32(color_heatmap) + np.float32(image)
    image_with_heatmap = image_with_heatmap / np.max(image_with_heatmap)
    return color_heatmap, np.uint8(255 * image_with_heatmap)


def load_path(path_file, source_image, target_image):
    path_data = pd.read_csv(path_file)
    if source_image and target_image:
        rows = path_data.loc[
            (path_data["source_image"] == source_image) & (path_data["target_image"] == target_image)
        ]
        if rows.empty:
            raise ValueError(f"Path from {source_image} to {target_image} not found in {path_file}")
        row = rows.iloc[0]
    else:
        row = path_data.iloc[0]
        source_image = row["source_image"]
        target_image = row["target_image"]

    hyperedge_path = [edge_id for edge_id in str(row["hyperedge_path"]).split(",") if edge_id]
    centers = json.loads(row["hyperedge_centers_json"])
    return source_image, target_image, hyperedge_path, centers


def build_transform():
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )


def main():
    args = parse_args()

    import torch
    import torchvision.utils as vutils
    from PIL import Image
    from torch.nn import functional

    from trainer_exchange import trainer

    device = select_device(args.cuda)
    source_image, target_image, hyperedge_path, centers = load_path(
        args.path_file, args.source_image, args.target_image
    )
    if not hyperedge_path:
        raise ValueError("The path file contains no hyperedge path.")

    os.makedirs(args.output_dir, exist_ok=True)

    caml_trainer = trainer(
        device=device,
        style_dim=args.style_dim,
        optim_para=None,
        gen_loss_weight_para=None,
        dis_loss_weight_para=None,
        train_is="False",
    )
    caml_trainer.to(device)
    state_dict_gen = torch.load(args.checkpoint, map_location=device)
    caml_trainer.gen.load_state_dict(state_dict_gen["ab"])
    caml_trainer.eval()
    encode = caml_trainer.gen.encode
    decode = caml_trainer.gen.decode

    classifier = torch.load(args.classifier, map_location=device)
    classifier = classifier.eval().to(device)

    transform = build_transform()
    source_path = os.path.join(args.image_dir, source_image)
    if not os.path.exists(source_path):
        print(f"Source image does not exist: {source_path}")
        sys.exit(1)

    source_tensor = transform(Image.open(source_path).convert("RGB")).unsqueeze(0).to(device)
    content_code, _ = encode(source_tensor)

    source_prediction = functional.softmax(classifier(source_tensor), dim=1)
    _, source_class = source_prediction.max(dim=1)
    print(f"source image is predicted as: {source_class.item()}")

    for index, (edge_id, center) in enumerate(zip(hyperedge_path, centers)):
        style_tensor = torch.tensor(center, dtype=torch.float32, device=device).unsqueeze(0)
        generated_tensor = decode(content_code, style_tensor)
        generated_path = os.path.join(
            args.output_dir,
            f"ex_{source_image}_ref_{target_image}_hyperedge_{index}_{edge_id}_gen.png",
        )
        vutils.save_image((generated_tensor.data + 1) / 2, generated_path, padding=0, normalize=False)

        generated_prediction = functional.softmax(classifier(generated_tensor), dim=1)
        _, generated_class = generated_prediction.max(dim=1)
        print(f"generated image at hyperedge {edge_id} is predicted as: {generated_class.item()}")

        if generated_class.item() != source_class.item() or index == len(hyperedge_path) - 1:
            difference = torch.abs(((generated_tensor + 1) / 2) - ((source_tensor + 1) / 2))
            _, heatmap_with_image = heatmap_show((source_tensor.squeeze(0) + 1) / 2, difference.squeeze(0))
            heatmap_path = os.path.join(
                args.output_dir,
                f"saliency_map_with_img_hyperedge_{index}_{edge_id}.jpg",
            )
            try:
                import cv2
            except ImportError as exc:
                raise ImportError("OpenCV is required for saliency heatmap generation. Install opencv-python.") from exc
            cv2.imwrite(heatmap_path, heatmap_with_image)
            print(f"saved local explanation to {heatmap_path}")
            break


if __name__ == "__main__":
    main()
