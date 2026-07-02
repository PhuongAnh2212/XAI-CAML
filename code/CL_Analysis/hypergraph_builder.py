import argparse
import itertools
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a higher-order hypergraph from CAML latent CSV files."
    )
    parser.add_argument("--latent_csv", required=True, help="CSV with image_name, latent columns, label.")
    parser.add_argument(
        "--method",
        default="knn",
        help="Hyperedge method: knn, label, mapper, local_overlap, rips, or comma-separated combinations.",
    )
    parser.add_argument("--output", default=None, help="Output hypergraph JSON path.")
    parser.add_argument("--output_dir", default=None, help="Output directory. Writes hypergraph.json by default.")
    parser.add_argument("--plot_path", default=None, help="Optional PNG path for latent scatter plot colored by label.")
    parser.add_argument(
        "--membership_csv",
        default=None,
        help="Optional CSV path recording hyperedge-to-image membership.",
    )
    parser.add_argument("--k", type=int, default=10, help="Number of neighbors for knn/local_overlap.")
    parser.add_argument("--overlap_threshold", type=float, default=0.5, help="Jaccard threshold for local_overlap.")
    parser.add_argument("--rips_epsilon", type=float, default=None, help="Distance threshold for Vietoris-Rips edges.")
    parser.add_argument("--rips_max_order", type=int, default=2, help="Max simplex order. 2 means pair/triple hyperedges.")
    parser.add_argument("--max_rips_edges", type=int, default=50000, help="Safety cap for generated Rips hyperedges.")
    parser.add_argument("--mapper_cover", type=int, default=7, help="Kepler Mapper cover intervals.")
    parser.add_argument("--mapper_overlap", type=float, default=0.49, help="Kepler Mapper cover overlap.")
    parser.add_argument("--dbscan_eps", type=float, default=0.3, help="DBSCAN eps for mapper clusters.")
    parser.add_argument("--dbscan_min_samples", type=int, default=15, help="DBSCAN min_samples for mapper clusters.")
    return parser.parse_args()


def load_latents(latent_csv):
    df = pd.read_csv(latent_csv)
    required = {"image_name", "label"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError("Missing required latent CSV columns: " + ", ".join(sorted(missing)))

    feature_names = [column for column in df.columns if column not in ["image_name", "label"]]
    if not feature_names:
        raise ValueError("No latent feature columns found.")

    images = df["image_name"].astype(str).tolist()
    labels = df["label"].astype(str).tolist()
    features = df[feature_names].astype(float).to_numpy()
    return df, feature_names, images, labels, features


def pairwise_distances(features):
    diff = features[:, None, :] - features[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2))


def add_hyperedge(hyperedges, edge_type, member_indices, images, labels, features, metadata=None):
    member_indices = sorted(set(int(index) for index in member_indices))
    if len(member_indices) < 2:
        return

    center = features[member_indices].mean(axis=0)
    spread = float(np.mean(np.sqrt(np.sum((features[member_indices] - center) ** 2, axis=1))))
    edge_id = f"HE_{len(hyperedges):06d}"
    hyperedges.append(
        {
            "id": edge_id,
            "type": edge_type,
            "vertices": [images[index] for index in member_indices],
            "labels": sorted(set(labels[index] for index in member_indices)),
            "center": center.tolist(),
            "weight": spread if spread > 0 else 1e-12,
            "metadata": metadata or {},
        }
    )


def build_knn_edges(hyperedges, images, labels, features, k):
    distances = pairwise_distances(features)
    neighbor_count = min(k + 1, len(images))
    for index in range(len(images)):
        neighbors = np.argsort(distances[index])[:neighbor_count]
        add_hyperedge(
            hyperedges,
            "knn",
            neighbors,
            images,
            labels,
            features,
            {"anchor": images[index], "k": k},
        )


def build_label_edges(hyperedges, images, labels, features):
    label_to_indices = defaultdict(list)
    for index, label in enumerate(labels):
        label_to_indices[label].append(index)
    for label, indices in sorted(label_to_indices.items()):
        add_hyperedge(
            hyperedges,
            "label",
            indices,
            images,
            labels,
            features,
            {"label": label},
        )


def build_local_overlap_edges(hyperedges, images, labels, features, k, overlap_threshold):
    distances = pairwise_distances(features)
    neighbor_count = min(k + 1, len(images))
    neighborhoods = [
        set(np.argsort(distances[index])[:neighbor_count].tolist()) for index in range(len(images))
    ]
    for left in range(len(neighborhoods)):
        for right in range(left + 1, len(neighborhoods)):
            intersection = neighborhoods[left].intersection(neighborhoods[right])
            union = neighborhoods[left].union(neighborhoods[right])
            jaccard = len(intersection) / len(union) if union else 0
            if jaccard >= overlap_threshold:
                add_hyperedge(
                    hyperedges,
                    "local_overlap",
                    union,
                    images,
                    labels,
                    features,
                    {
                        "anchors": [images[left], images[right]],
                        "jaccard": jaccard,
                        "k": k,
                    },
                )


def build_rips_edges(hyperedges, images, labels, features, epsilon, max_order, max_edges):
    if epsilon is None:
        raise ValueError("--rips_epsilon is required when method includes rips.")

    distances = pairwise_distances(features)
    generated = 0
    max_size = max_order + 1
    for size in range(2, max_size + 1):
        for combination in itertools.combinations(range(len(images)), size):
            if all(distances[left, right] <= epsilon for left, right in itertools.combinations(combination, 2)):
                add_hyperedge(
                    hyperedges,
                    "rips",
                    combination,
                    images,
                    labels,
                    features,
                    {"epsilon": epsilon, "order": size - 1},
                )
                generated += 1
                if generated >= max_edges:
                    return


def build_mapper_edges(hyperedges, images, labels, features, args):
    try:
        import kmapper as km
        import sklearn.cluster
        import sklearn.manifold
    except ImportError as exc:
        raise ImportError("Mapper hyperedges require kmapper and scikit-learn.") from exc

    mapper = km.KeplerMapper(verbose=0)
    projected_data = mapper.fit_transform(features, projection=sklearn.manifold.TSNE(n_iter=500))
    graph = mapper.map(
        projected_data,
        clusterer=sklearn.cluster.DBSCAN(eps=args.dbscan_eps, min_samples=args.dbscan_min_samples),
        cover=km.Cover(args.mapper_cover, args.mapper_overlap),
    )
    for node_id, member_indices in sorted(graph["nodes"].items()):
        add_hyperedge(
            hyperedges,
            "mapper",
            member_indices,
            images,
            labels,
            features,
            {"mapper_node": node_id},
        )


def deduplicate_hyperedges(hyperedges):
    seen = set()
    unique_edges = []
    for edge in hyperedges:
        key = (edge["type"], tuple(sorted(edge["vertices"])))
        if key in seen:
            continue
        seen.add(key)
        edge["id"] = f"HE_{len(unique_edges):06d}"
        unique_edges.append(edge)
    return unique_edges


def write_membership_csv(hyperedges, membership_csv):
    rows = []
    for edge in hyperedges:
        for image_name in edge["vertices"]:
            rows.append(
                {
                    "hyperedge_id": edge["id"],
                    "hyperedge_type": edge["type"],
                    "image_name": image_name,
                    "hyperedge_weight": edge["weight"],
                }
            )
    pd.DataFrame(rows).to_csv(membership_csv, index=False)


def write_hypergraph_plot(vertices, plot_path):
    if not vertices:
        return
    features = np.array([vertex["latent"] for vertex in vertices], dtype=float)
    labels = [str(vertex["label"]) for vertex in vertices]
    centered = features - features.mean(axis=0, keepdims=True)
    if centered.shape[1] >= 2:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        points = centered @ vh[:2].T
    else:
        points = np.column_stack([centered[:, 0], np.zeros(centered.shape[0])])

    os.makedirs(os.path.dirname(os.path.abspath(plot_path)), exist_ok=True)

    try:
        import matplotlib.pyplot as plt

        colors = {"0": "tab:blue", "1": "tab:red"}
        plt.figure(figsize=(8, 6))
        for label in sorted(set(labels)):
            indices = [index for index, item in enumerate(labels) if item == label]
            label_name = "normal (0)" if label == "0" else "abnormal (" + label + ")"
            plt.scatter(
                points[indices, 0],
                points[indices, 1],
                s=24,
                alpha=0.8,
                c=colors.get(label, "tab:gray"),
                label=label_name,
            )
        plt.title("CAML hypergraph vertices projected from latent space")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_path, dpi=160)
        plt.close()
    except ImportError:
        from PIL import Image, ImageDraw

        width, height, margin = 900, 700, 50
        image = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(image)
        x_min, y_min = points.min(axis=0)
        x_max, y_max = points.max(axis=0)
        x_span = x_max - x_min if x_max > x_min else 1.0
        y_span = y_max - y_min if y_max > y_min else 1.0
        color_map = {"0": (31, 119, 180), "1": (214, 39, 40)}
        draw.text((margin, 15), "CAML hypergraph vertices: blue=normal(0), red=abnormal(1)", fill=(0, 0, 0))
        for point, label in zip(points, labels):
            x = margin + int((point[0] - x_min) / x_span * (width - 2 * margin))
            y = height - margin - int((point[1] - y_min) / y_span * (height - 2 * margin))
            color = color_map.get(label, (127, 127, 127))
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color, outline=color)
        image.save(plot_path)
    print("Saved hypergraph plot to " + plot_path)


def main():
    args = parse_args()
    df, feature_names, images, labels, features = load_latents(args.latent_csv)
    if args.output is None:
        if args.output_dir is None:
            raise ValueError("Either --output or --output_dir is required.")
        args.output = os.path.join(args.output_dir, "hypergraph.json")
    if args.membership_csv is None and args.output_dir is not None:
        args.membership_csv = os.path.join(args.output_dir, "hypergraph_membership.csv")
    if args.plot_path is None and args.output_dir is not None:
        args.plot_path = os.path.join(args.output_dir, "hypergraph_plot.png")

    methods = [method.strip() for method in args.method.split(",") if method.strip()]
    hyperedges = []
    for method in methods:
        if method == "knn":
            build_knn_edges(hyperedges, images, labels, features, args.k)
        elif method == "label":
            build_label_edges(hyperedges, images, labels, features)
        elif method == "mapper":
            build_mapper_edges(hyperedges, images, labels, features, args)
        elif method == "local_overlap":
            build_local_overlap_edges(hyperedges, images, labels, features, args.k, args.overlap_threshold)
        elif method == "rips":
            build_rips_edges(
                hyperedges,
                images,
                labels,
                features,
                args.rips_epsilon,
                args.rips_max_order,
                args.max_rips_edges,
            )
        else:
            raise ValueError(f"Unsupported method: {method}")

    hyperedges = deduplicate_hyperedges(hyperedges)
    vertices = [
        {
            "id": image_name,
            "image_name": image_name,
            "label": label,
            "latent": features[index].tolist(),
        }
        for index, (image_name, label) in enumerate(zip(images, labels))
    ]
    hypergraph = {
        "schema": "caml_hypergraph_v1",
        "latent_csv": os.path.abspath(args.latent_csv),
        "feature_names": feature_names,
        "vertex_type": "image",
        "vertices": vertices,
        "hyperedges": hyperedges,
        "metadata": {
            "methods": methods,
            "num_vertices": len(vertices),
            "num_hyperedges": len(hyperedges),
        },
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as file:
        json.dump(hypergraph, file, indent=2)

    if args.membership_csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.membership_csv)), exist_ok=True)
        write_membership_csv(hyperedges, args.membership_csv)

    if args.plot_path:
        write_hypergraph_plot(vertices, args.plot_path)

    label_counts = pd.Series([vertex["label"] for vertex in vertices]).value_counts().sort_index().to_dict()
    print(f"Saved hypergraph with {len(vertices)} vertices and {len(hyperedges)} hyperedges to {args.output}")
    print("Label distribution: " + repr(label_counts))


if __name__ == "__main__":
    main()
