import argparse
import csv
import heapq
import itertools
import json
import math
import os
from collections import defaultdict

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Compute distances over a CAML hypergraph.")
    parser.add_argument("--hypergraph", required=True, help="Hypergraph JSON from hypergraph_builder.py.")
    parser.add_argument("--output", required=True, help="Output CSV path.")
    parser.add_argument(
        "--mode",
        choices=["projected_edges", "source", "all_pairs"],
        default="projected_edges",
        help="Distance output type.",
    )
    parser.add_argument("--source_image", default=None, help="Required for --mode source.")
    return parser.parse_args()


def load_hypergraph(path):
    with open(path, "r", encoding="utf-8") as file:
        hypergraph = json.load(file)
    if hypergraph.get("schema") != "caml_hypergraph_v1":
        raise ValueError("Unsupported hypergraph schema.")
    return hypergraph


def latent_distance(left, right):
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    return float(np.sqrt(np.sum((left - right) ** 2)))


def build_projected_graph(hypergraph):
    vertices = {vertex["image_name"]: vertex for vertex in hypergraph["vertices"]}
    graph = defaultdict(dict)
    shared_edges = defaultdict(list)

    for edge in hypergraph["hyperedges"]:
        members = edge["vertices"]
        if len(members) < 2:
            continue
        traversal_penalty = float(edge.get("weight", 1.0)) / max(len(members) - 1, 1)
        for left, right in itertools.combinations(members, 2):
            distance = latent_distance(vertices[left]["latent"], vertices[right]["latent"])
            weight = distance + traversal_penalty
            if right not in graph[left] or weight < graph[left][right]:
                graph[left][right] = weight
                graph[right][left] = weight
            shared_edges[(left, right)].append(edge["id"])
            shared_edges[(right, left)].append(edge["id"])

    return graph, shared_edges


def dijkstra(graph, source):
    distances = {vertex: math.inf for vertex in graph}
    previous = {vertex: None for vertex in graph}
    distances[source] = 0.0
    queue = [(0.0, source)]

    while queue:
        current_distance, current_vertex = heapq.heappop(queue)
        if current_distance > distances[current_vertex]:
            continue
        for neighbor, weight in graph[current_vertex].items():
            candidate = current_distance + weight
            if candidate < distances.get(neighbor, math.inf):
                distances[neighbor] = candidate
                previous[neighbor] = current_vertex
                heapq.heappush(queue, (candidate, neighbor))
    return distances, previous


def write_projected_edges(output, graph, shared_edges):
    with open(output, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["source_image", "target_image", "weight", "shared_hyperedges"])
        written = set()
        for source, neighbors in sorted(graph.items()):
            for target, weight in sorted(neighbors.items()):
                key = tuple(sorted([source, target]))
                if key in written:
                    continue
                written.add(key)
                writer.writerow([source, target, weight, "|".join(shared_edges[(source, target)])])


def write_source_distances(output, graph, source):
    distances, previous = dijkstra(graph, source)
    with open(output, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["source_image", "target_image", "distance", "previous_image"])
        for target, distance in sorted(distances.items()):
            writer.writerow([source, target, distance, previous[target] or ""])


def write_all_pairs(output, graph):
    vertices = sorted(graph)
    with open(output, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["source_image", "target_image", "distance"])
        for source in vertices:
            distances, _ = dijkstra(graph, source)
            for target in vertices:
                writer.writerow([source, target, distances.get(target, math.inf)])


def main():
    args = parse_args()
    hypergraph = load_hypergraph(args.hypergraph)
    graph, shared_edges = build_projected_graph(hypergraph)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    if args.mode == "projected_edges":
        write_projected_edges(args.output, graph, shared_edges)
    elif args.mode == "source":
        if not args.source_image:
            raise ValueError("--source_image is required for --mode source.")
        if args.source_image not in graph:
            raise ValueError(f"Source image is not connected in hypergraph: {args.source_image}")
        write_source_distances(args.output, graph, args.source_image)
    else:
        write_all_pairs(args.output, graph)

    print(f"Saved hypergraph distance output to {args.output}")


if __name__ == "__main__":
    main()
