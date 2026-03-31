from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

from rl_adn.network.topology import get_active_edges

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
NETWORK_ROOT = PACKAGE_ROOT / "data_sources" / "network_data"


def _resolve_baseline_edges(node_count: int) -> list[tuple[int, int]]:
    line_info = pd.read_csv(NETWORK_ROOT / f"node_{node_count}" / f"Lines_{node_count}.csv")
    return get_active_edges(line_info)


def _build_tree_layout(node_count: int, edges: list[tuple[int, int]]) -> dict[int, tuple[float, float]]:
    adjacency: dict[int, list[int]] = {node: [] for node in range(1, node_count + 1)}
    for left, right in edges:
        adjacency[left].append(right)
        adjacency[right].append(left)

    depth = {1: 0}
    children: dict[int, list[int]] = {}
    order = [1]
    for node in order:
        child_nodes = [candidate for candidate in sorted(adjacency[node]) if candidate not in depth]
        children[node] = child_nodes
        for child in child_nodes:
            depth[child] = depth[node] + 1
            order.append(child)

    x_map: dict[int, float] = {}
    cursor = 0.0

    def assign(node: int) -> float:
        nonlocal cursor
        child_nodes = children.get(node, [])
        if not child_nodes:
            x_map[node] = cursor
            cursor += 1.0
            return x_map[node]
        child_positions = [assign(child) for child in child_nodes]
        x_map[node] = sum(child_positions) / len(child_positions)
        return x_map[node]

    assign(1)
    max_depth = max(depth.values()) if depth else 0
    leaf_slots = max(cursor - 1.0, 1.0)

    positions: dict[int, tuple[float, float]] = {}
    for node in range(1, node_count + 1):
        normalized_x = (x_map.get(node, 0.0) / leaf_slots) if leaf_slots else 0.5
        normalized_y = (depth.get(node, 0) / max_depth) if max_depth else 0.0
        positions[node] = (0.08 + normalized_x * 0.84, 0.10 + normalized_y * 0.80)
    return positions


@lru_cache(maxsize=None)
def get_feeder_layout(node_count: int) -> dict[str, object]:
    base_edges = _resolve_baseline_edges(node_count)
    positions = _build_tree_layout(node_count, base_edges)
    return {
        "node_count": node_count,
        "base_edges": [[left, right] for left, right in base_edges],
        "positions": {str(node): {"x": x_coord, "y": y_coord} for node, (x_coord, y_coord) in positions.items()},
    }
