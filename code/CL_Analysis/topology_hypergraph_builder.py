"""Build and regularize a hypergraph whose hyperedges are Mapper nodes."""

import argparse
import json
import os
from collections import Counter

import numpy as np
import pandas as pd


EPSILON = 1e-12


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a topology-derived hypergraph from CAML latent embeddings."
    )
    parser.add_argument("--latent_csv", required=True, help="CSV with image_name, latent columns, and label.")
    parser.add_argument(
        "--mapper_json",
        default=None,
        help="Optional Mapper graph JSON. If omitted, Mapper is run from the latent CSV.",
    )
    parser.add_argument("--output_dir", required=True, help="Directory for topology-hypergraph outputs.")
    parser.add_argument(
        "--lambda_reg",
        type=float,
        default=0.1,
        help="Non-negative hypergraph smoothing strength.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Save an original-versus-smoothed PCA scatter plot.",
    )
    return parser.parse_args()


def load_latents(path):
    if not os.path.isfile(path):
        raise FileNotFoundError("Latent CSV does not exist: " + path)

    frame = pd.read_csv(path)
    required = {"image_name", "label"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError("Missing required latent CSV columns: " + ", ".join(missing))
    if frame.empty:
        raise ValueError("Latent CSV contains no samples: " + path)

    feature_names = [column for column in frame.columns if column not in required]
    if not feature_names:
        raise ValueError("Latent CSV contains no latent feature columns.")
    try:
        features = frame[feature_names].astype(float).to_numpy()
    except (TypeError, ValueError) as exc:
        raise ValueError("All latent feature columns must be numeric.") from exc
    if not np.isfinite(features).all():
        raise ValueError("Latent features contain NaN or infinite values.")

    images = frame["image_name"].astype(str).tolist()
    if len(images) != len(set(images)):
        raise ValueError("image_name values must be unique.")
    labels = frame["label"].astype(str).tolist()
    return frame, feature_names, images, labels, features


def load_mapper_nodes(path):
    if not os.path.isfile(path):
        raise FileNotFoundError("Mapper JSON does not exist: " + path)
    with open(path, "r", encoding="utf-8") as file:
        graph = json.load(file)

    # Accept a raw Kepler Mapper graph or a graph nested under common wrapper keys.
    for key in ("graph", "mapper_graph"):
        if isinstance(graph, dict) and isinstance(graph.get(key), dict):
            graph = graph[key]
            break
    nodes = graph.get("nodes") if isinstance(graph, dict) else None
    if not isinstance(nodes, dict) or not nodes:
        raise ValueError("Mapper JSON contains an empty or missing 'nodes' mapping.")
    return nodes


def run_mapper(features):
    try:
        import kmapper as km
        from sklearn.cluster import DBSCAN
        from sklearn.manifold import TSNE
    except ImportError as exc:
        raise ImportError(
            "Direct Mapper construction requires kmapper and scikit-learn. "
            "Install them or provide --mapper_json."
        ) from exc

    if len(features) < 2:
        raise ValueError("Mapper requires at least two latent samples.")
    perplexity = min(30.0, max(1.0, float(len(features) - 1)))
    mapper = km.KeplerMapper(verbose=0)
    projected = mapper.fit_transform(
        features,
        projection=TSNE(
            n_components=2,
            max_iter=500,
            perplexity=perplexity,
            random_state=42,
            init="pca",
            learning_rate="auto",
        ),
    )
    graph = mapper.map(
        projected,
        clusterer=DBSCAN(eps=0.3, min_samples=15),
        cover=km.Cover(n_cubes=7, perc_overlap=0.49),
    )
    nodes = graph.get("nodes", {})
    if not nodes:
        raise ValueError(
            "Mapper produced an empty graph. Adjust the Mapper/DBSCAN settings or provide a Mapper JSON."
        )
    return nodes


def normalize_member_indices(members, images, node_id):
    image_to_index = {image: index for index, image in enumerate(images)}
    normalized = []
    for member in members:
        if isinstance(member, (int, np.integer)):
            index = int(member)
        elif isinstance(member, str) and member in image_to_index:
            index = image_to_index[member]
        elif isinstance(member, str):
            try:
                index = int(member)
            except ValueError as exc:
                raise ValueError(
                    f"Mapper node {node_id!r} has unknown image/sample member {member!r}."
                ) from exc
        else:
            raise ValueError(f"Mapper node {node_id!r} has unsupported member {member!r}.")
        if index < 0 or index >= len(images):
            raise IndexError(
                f"Mapper node {node_id!r} references sample index {index}, "
                f"but valid indices are 0..{len(images) - 1}."
            )
        normalized.append(index)
    return sorted(set(normalized))


def build_topology_hyperedges(nodes, images, labels):
    unique_edges = []
    members_to_edge = {}
    for node_id in sorted(nodes, key=str):
        members = nodes[node_id]
        if not isinstance(members, (list, tuple, set)):
            raise ValueError(f"Mapper node {node_id!r} members must be a list-like collection.")
        indices = normalize_member_indices(members, images, node_id)
        if len(indices) < 2:
            continue

        key = tuple(indices)
        if key in members_to_edge:
            members_to_edge[key]["metadata"]["duplicate_mapper_node_ids"].append(str(node_id))
            continue

        edge = {
            "id": f"HE_{len(unique_edges):06d}",
            "type": "mapper_topology",
            "member_indices": indices,
            "vertices": [images[index] for index in indices],
            "labels": sorted(set(labels[index] for index in indices)),
            "metadata": {
                "mapper_node_id": str(node_id),
                "duplicate_mapper_node_ids": [],
            },
        }
        unique_edges.append(edge)
        members_to_edge[key] = edge

    if not unique_edges:
        raise ValueError("Mapper graph contains no valid hyperedges with at least two samples.")
    return unique_edges


def build_incidence(num_vertices, hyperedges):
    incidence = np.zeros((num_vertices, len(hyperedges)), dtype=float)
    for edge_index, edge in enumerate(hyperedges):
        incidence[edge["member_indices"], edge_index] = 1.0
    return incidence


def zhou_laplacian(incidence, epsilon=EPSILON):
    """Return L = I - Dv^-1/2 H W De^-1 H.T Dv^-1/2, with W = I."""
    edge_degree = incidence.sum(axis=0)
    vertex_degree = incidence.sum(axis=1)
    inv_edge_degree = 1.0 / np.maximum(edge_degree, epsilon)
    inv_sqrt_vertex_degree = 1.0 / np.sqrt(np.maximum(vertex_degree, epsilon))

    normalized_incidence = inv_sqrt_vertex_degree[:, None] * incidence
    propagation = (normalized_incidence * inv_edge_degree[None, :]) @ normalized_incidence.T
    laplacian = np.eye(incidence.shape[0], dtype=float) - propagation
    # Remove harmless floating-point asymmetry before saving/solving.
    return 0.5 * (laplacian + laplacian.T), vertex_degree, edge_degree


def smooth_latents(features, laplacian, lambda_reg):
    if not np.isfinite(lambda_reg) or lambda_reg < 0:
        raise ValueError("--lambda_reg must be a finite, non-negative number.")
    system = np.eye(len(features), dtype=float) + lambda_reg * laplacian
    condition_number = float("inf")
    try:
        condition_number = float(np.linalg.cond(system))
        if not np.isfinite(condition_number) or condition_number > 1e12:
            raise np.linalg.LinAlgError(
                f"regularization system is ill-conditioned (condition number {condition_number:.3e})"
            )
        return np.linalg.solve(system, features), condition_number, "solve"
    except np.linalg.LinAlgError as exc:
        try:
            solution, _, rank, _ = np.linalg.lstsq(system, features, rcond=None)
        except np.linalg.LinAlgError as fallback_exc:
            raise RuntimeError(
                "Hypergraph smoothing failed because the regularization matrix is singular."
            ) from fallback_exc
        if rank < system.shape[0]:
            raise RuntimeError(
                "Hypergraph smoothing failed because the regularization matrix is singular "
                f"(rank {rank}/{system.shape[0]})."
            ) from exc
        print("Warning: np.linalg.solve was unstable; used np.linalg.lstsq.")
        return solution, condition_number, "lstsq"


def add_hyperedge_geometry(hyperedges, smoothed_features):
    for edge in hyperedges:
        indices = edge.pop("member_indices")
        member_features = smoothed_features[indices]
        center = member_features.mean(axis=0)
        spread = float(np.linalg.norm(member_features - center, axis=1).mean())
        edge["center"] = center.tolist()
        edge["weight"] = max(spread, EPSILON)
        edge["metadata"]["size"] = len(indices)


def write_membership_csv(hyperedges, path):
    rows = []
    for edge in hyperedges:
        for image_name in edge["vertices"]:
            rows.append(
                {
                    "hyperedge_id": edge["id"],
                    "hyperedge_type": edge["type"],
                    "mapper_node_id": edge["metadata"]["mapper_node_id"],
                    "image_name": image_name,
                    "hyperedge_weight": edge["weight"],
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)


def pca_projection(original, smoothed):
    combined = np.vstack([original, smoothed])
    centered = combined - combined.mean(axis=0, keepdims=True)
    if centered.shape[1] == 1:
        projected = np.column_stack([centered[:, 0], np.zeros(len(centered))])
    else:
        _, _, right_vectors = np.linalg.svd(centered, full_matrices=False)
        projected = centered @ right_vectors[:2].T
        if projected.shape[1] == 1:
            projected = np.column_stack([projected[:, 0], np.zeros(len(projected))])
    return projected[: len(original)], projected[len(original) :]


def write_plot(original, smoothed, labels, path):
    original_points, smoothed_points = pca_projection(original, smoothed)
    try:
        import matplotlib.pyplot as plt

        figure, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
        color_values = pd.Categorical(labels).codes
        for axis, points, title in zip(
            axes,
            (original_points, smoothed_points),
            ("Original latent space", "Hypergraph-smoothed latent space"),
        ):
            scatter = axis.scatter(points[:, 0], points[:, 1], c=color_values, cmap="tab10", s=20, alpha=0.8)
            axis.set_title(title)
            axis.set_xlabel("PCA 1")
            axis.set_ylabel("PCA 2")
        handles, _ = scatter.legend_elements()
        figure.legend(handles, sorted(set(labels)), title="label", loc="upper center", ncol=5)
        figure.tight_layout(rect=(0, 0, 1, 0.92))
        figure.savefig(path, dpi=160)
        plt.close(figure)
        return
    except ImportError:
        pass

    # Keep --plot useful in lightweight environments without matplotlib.
    from PIL import Image, ImageDraw

    width, height, margin = 1200, 520, 45
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    all_points = np.vstack([original_points, smoothed_points])
    minima, maxima = all_points.min(axis=0), all_points.max(axis=0)
    spans = np.maximum(maxima - minima, EPSILON)
    palette = [
        (31, 119, 180),
        (214, 39, 40),
        (44, 160, 44),
        (255, 127, 14),
        (148, 103, 189),
    ]
    label_colors = {label: palette[index % len(palette)] for index, label in enumerate(sorted(set(labels)))}
    panel_width = width // 2
    for panel, (points, title) in enumerate(
        ((original_points, "Original latent space"), (smoothed_points, "Hypergraph-smoothed latent space"))
    ):
        x_offset = panel * panel_width
        draw.text((x_offset + margin, 15), title + " (shared PCA)", fill=(0, 0, 0))
        for point, label in zip(points, labels):
            x = x_offset + margin + int((point[0] - minima[0]) / spans[0] * (panel_width - 2 * margin))
            y = height - margin - int((point[1] - minima[1]) / spans[1] * (height - 2 * margin))
            color = label_colors[label]
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color, outline=color)
    canvas.save(path)


def write_summary(path, summary):
    with open(path, "w", encoding="utf-8") as file:
        for key, value in summary.items():
            file.write(f"{key}: {value}\n")


def main():
    args = parse_args()
    _, feature_names, images, labels, features = load_latents(args.latent_csv)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.mapper_json:
        mapper_nodes = load_mapper_nodes(args.mapper_json)
        mapper_source = os.path.abspath(args.mapper_json)
    else:
        mapper_nodes = run_mapper(features)
        mapper_source = "generated_from_latent_csv"

    hyperedges = build_topology_hyperedges(mapper_nodes, images, labels)
    incidence = build_incidence(len(images), hyperedges)
    laplacian, vertex_degree, edge_degree = zhou_laplacian(incidence)
    smoothed, condition_number, solver = smooth_latents(features, laplacian, args.lambda_reg)
    add_hyperedge_geometry(hyperedges, smoothed)

    vertices = [
        {
            "id": image,
            "image_name": image,
            "label": labels[index],
            "latent": smoothed[index].tolist(),
            "original_latent": features[index].tolist(),
        }
        for index, image in enumerate(images)
    ]
    label_distribution = dict(sorted(Counter(labels).items()))
    average_size = float(edge_degree.mean())
    isolated_vertices = int(np.count_nonzero(vertex_degree == 0))
    hypergraph = {
        "schema": "caml_hypergraph_v1",
        "latent_csv": os.path.abspath(args.latent_csv),
        "feature_names": feature_names,
        "vertex_type": "image",
        "vertices": vertices,
        "hyperedges": hyperedges,
        "metadata": {
            "method": "mapper_topology",
            "mapper_source": mapper_source,
            "num_mapper_nodes": len(mapper_nodes),
            "num_vertices": len(vertices),
            "num_hyperedges": len(hyperedges),
            "average_hyperedge_size": average_size,
            "isolated_vertices": isolated_vertices,
            "lambda_reg": args.lambda_reg,
            "laplacian": "zhou_normalized",
            "hyperedge_weights": "identity_for_laplacian",
            "smoothing_solver": solver,
            "smoothing_condition_number": condition_number,
            "vertex_latent_field": "smoothed",
        },
    }

    paths = {
        "hypergraph": os.path.join(args.output_dir, "hypergraph.json"),
        "membership": os.path.join(args.output_dir, "hypergraph_membership.csv"),
        "laplacian": os.path.join(args.output_dir, "hypergraph_laplacian.npy"),
        "smoothed_latent": os.path.join(args.output_dir, "smoothed_latent.csv"),
        "summary": os.path.join(args.output_dir, "topology_hypergraph_summary.txt"),
    }
    with open(paths["hypergraph"], "w", encoding="utf-8") as file:
        json.dump(hypergraph, file, indent=2)
    write_membership_csv(hyperedges, paths["membership"])
    np.save(paths["laplacian"], laplacian)
    smoothed_frame = pd.DataFrame(smoothed, columns=feature_names)
    smoothed_frame.insert(0, "image_name", images)
    smoothed_frame["label"] = labels
    smoothed_frame.to_csv(paths["smoothed_latent"], index=False)

    summary = {
        "num_vertices": len(vertices),
        "num_mapper_nodes": len(mapper_nodes),
        "num_hyperedges": len(hyperedges),
        "average_hyperedge_size": f"{average_size:.6f}",
        "isolated_vertices": isolated_vertices,
        "label_distribution": label_distribution,
        "lambda_reg": args.lambda_reg,
        "smoothing_solver": solver,
        "smoothing_condition_number": f"{condition_number:.6e}",
        **{f"output_{name}": os.path.abspath(path) for name, path in paths.items()},
    }
    if args.plot:
        paths["plot"] = os.path.join(args.output_dir, "original_vs_smoothed_pca.png")
        write_plot(features, smoothed, labels, paths["plot"])
        summary["output_plot"] = os.path.abspath(paths["plot"])
    write_summary(paths["summary"], summary)

    print(f"Number of vertices: {len(vertices)}")
    print(f"Number of Mapper nodes: {len(mapper_nodes)}")
    print(f"Number of hyperedges: {len(hyperedges)}")
    print(f"Average hyperedge size: {average_size:.4f}")
    print(f"Label distribution: {label_distribution}")
    if isolated_vertices:
        print(f"Isolated vertices: {isolated_vertices}")
    print("Output paths:")
    for name, path in paths.items():
        print(f"  {name}: {os.path.abspath(path)}")


if __name__ == "__main__":
    main()
