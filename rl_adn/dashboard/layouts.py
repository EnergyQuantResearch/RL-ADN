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


def _set_path(positions: dict[int, tuple[float, float]], nodes: list[int], *, start_x: float, start_y: float, dx: float, dy: float) -> None:
    for index, node in enumerate(nodes):
        positions[node] = (start_x + dx * index, start_y + dy * index)


def _build_34_single_line_layout() -> dict[int, tuple[float, float]]:
    positions: dict[int, tuple[float, float]] = {}
    _set_path(positions, list(range(1, 13)), start_x=0.08, start_y=0.16, dx=0.065, dy=0.0)
    _set_path(positions, [13, 14, 15, 16], start_x=positions[3][0], start_y=0.31, dx=0.0, dy=0.12)
    _set_path(positions, list(range(17, 28)), start_x=positions[6][0], start_y=0.28, dx=0.0, dy=0.055)
    _set_path(positions, [28, 29, 30], start_x=positions[7][0], start_y=0.33, dx=0.0, dy=0.12)
    _set_path(positions, [31, 32, 33, 34], start_x=positions[10][0], start_y=0.30, dx=0.0, dy=0.11)
    return positions


def _build_69_single_line_layout() -> dict[int, tuple[float, float]]:
    positions: dict[int, tuple[float, float]] = {}
    _set_path(positions, list(range(1, 29)), start_x=0.05, start_y=0.14, dx=0.032, dy=0.0)
    _set_path(positions, [29, 30, 31, 32, 33, 34, 35, 36], start_x=positions[4][0] - 0.028, start_y=0.27, dx=0.0, dy=0.065)
    _set_path(positions, [37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47], start_x=positions[4][0] + 0.03, start_y=0.27, dx=0.0, dy=0.047)
    _set_path(positions, [48, 49, 50, 51], start_x=positions[5][0] + 0.015, start_y=0.27, dx=0.0, dy=0.095)
    _set_path(positions, [52, 53], start_x=positions[9][0] - 0.012, start_y=0.28, dx=0.0, dy=0.16)
    _set_path(positions, list(range(54, 67)), start_x=positions[10][0] + 0.018, start_y=0.25, dx=0.0, dy=0.044)
    _set_path(positions, [67, 68], start_x=positions[12][0] - 0.01, start_y=0.28, dx=0.0, dy=0.16)
    _set_path(positions, [69], start_x=positions[13][0] + 0.02, start_y=0.30, dx=0.0, dy=0.0)
    return positions


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
    tree_positions = _build_tree_layout(node_count, base_edges)
    if node_count == 34:
        single_line_positions = _build_34_single_line_layout()
    elif node_count == 69:
        single_line_positions = _build_69_single_line_layout()
    else:
        single_line_positions = tree_positions

    return {
        "view": "single_line",
        "node_count": node_count,
        "base_edges": [[left, right] for left, right in base_edges],
        "positions": {str(node): {"x": x_coord, "y": y_coord} for node, (x_coord, y_coord) in single_line_positions.items()},
        "graph_positions": {str(node): {"x": x_coord, "y": y_coord} for node, (x_coord, y_coord) in tree_positions.items()},
    }
