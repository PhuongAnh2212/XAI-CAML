import argparse
import csv
import heapq
import json
import math
import os
from collections import defaultdict

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Find a higher-order shortest path through hyperedges.")
    parser.add_argument("--hypergraph", required=True, help="Hypergraph JSON from CL_Analysis/hypergraph_builder.py.")
    parser.add_argument("--source_image", required=True, help="Start image filename.")
    parser.add_argument("--target_image", required=True, help="Target image filename.")
    parser.add_argument("--output", required=True, help="Output CSV path.")
    parser.add_argument(
        "--overlap_weight",
        type=float,
        default=1.0,
        help="Penalty multiplier for weak hyperedge overlap.",
    )
    return parser.parse_args()


def load_hypergraph(path):
    with open(path, "r", encoding="utf-8") as file:
        hypergraph = json.load(file)
    if hypergraph.get("schema") != "caml_hypergraph_v1":
        raise ValueError("Unsupported hypergraph schema.")
    return hypergraph


def euclidean(left, right):
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    return float(np.sqrt(np.sum((left - right) ** 2)))


def build_hyperedge_graph(hypergraph, overlap_weight):
    hyperedges = {edge["id"]: edge for edge in hypergraph["hyperedges"]}
    incident = defaultdict(list)
    for edge in hypergraph["hyperedges"]:
        for image_name in edge["vertices"]:
            incident[image_name].append(edge["id"])

    graph = defaultdict(dict)
    image_to_edges = defaultdict(set)
    for edge in hypergraph["hyperedges"]:
        for image_name in edge["vertices"]:
            image_to_edges[image_name].add(edge["id"])

    edge_ids = sorted(hyperedges)
    for index, left_id in enumerate(edge_ids):
        left_vertices = set(hyperedges[left_id]["vertices"])
        for right_id in edge_ids[index + 1 :]:
            right_vertices = set(hyperedges[right_id]["vertices"])
            overlap = left_vertices.intersection(right_vertices)
            if not overlap:
                continue
            center_distance = euclidean(hyperedges[left_id]["center"], hyperedges[right_id]["center"])
            penalty = overlap_weight / len(overlap)
            weight = center_distance + penalty
            graph[left_id][right_id] = weight
            graph[right_id][left_id] = weight

    return hyperedges, incident, graph


def dijkstra(graph, start_nodes, end_nodes):
    distances = {}
    previous = {}
    queue = []
    end_nodes = set(end_nodes)

    for node, initial_distance in start_nodes.items():
        distances[node] = initial_distance
        previous[node] = None
        heapq.heappush(queue, (initial_distance, node))

    while queue:
        current_distance, current_node = heapq.heappop(queue)
        if current_distance > distances.get(current_node, math.inf):
            continue
        if current_node in end_nodes:
            return current_distance, reconstruct_path(previous, current_node)
        for neighbor, weight in graph[current_node].items():
            candidate = current_distance + weight
            if candidate < distances.get(neighbor, math.inf):
                distances[neighbor] = candidate
                previous[neighbor] = current_node
                heapq.heappush(queue, (candidate, neighbor))

    return math.inf, []


def reconstruct_path(previous, end_node):
    path = []
    current = end_node
    while current is not None:
        path.append(current)
        current = previous[current]
    path.reverse()
    return path


def vertex_lookup(hypergraph):
    return {vertex["image_name"]: vertex for vertex in hypergraph["vertices"]}


def main():
    args = parse_args()
    hypergraph = load_hypergraph(args.hypergraph)
    vertices = vertex_lookup(hypergraph)
    if args.source_image not in vertices:
        raise ValueError(f"Source image not found in hypergraph vertices: {args.source_image}")
    if args.target_image not in vertices:
        raise ValueError(f"Target image not found in hypergraph vertices: {args.target_image}")

    hyperedges, incident, graph = build_hyperedge_graph(hypergraph, args.overlap_weight)
    source_edges = incident.get(args.source_image, [])
    target_edges = incident.get(args.target_image, [])
    if not source_edges:
        raise ValueError(f"Source image has no incident hyperedges: {args.source_image}")
    if not target_edges:
        raise ValueError(f"Target image has no incident hyperedges: {args.target_image}")

    source_latent = vertices[args.source_image]["latent"]
    target_latent = vertices[args.target_image]["latent"]
    start_nodes = {
        edge_id: euclidean(source_latent, hyperedges[edge_id]["center"]) for edge_id in source_edges
    }
    target_costs = {
        edge_id: euclidean(target_latent, hyperedges[edge_id]["center"]) for edge_id in target_edges
    }

    distance, path = dijkstra(graph, start_nodes, target_edges)
    if path:
        distance += target_costs[path[-1]]

    centers = [hyperedges[edge_id]["center"] for edge_id in path]
    edge_types = [hyperedges[edge_id]["type"] for edge_id in path]

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "source_image",
                "target_image",
                "hyperedge_path",
                "hyperedge_types",
                "hyperedge_centers_json",
                "distance",
            ]
        )
        writer.writerow(
            [
                args.source_image,
                args.target_image,
                ",".join(path),
                ",".join(edge_types),
                json.dumps(centers),
                distance,
            ]
        )

    print(f"Saved hypergraph shortest path to {args.output}")


if __name__ == "__main__":
    main()
