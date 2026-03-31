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


def _default_node_meta(node_count: int) -> dict[int, dict[str, float | str | bool]]:
    return {
        node: {
            "anchor": "middle",
            "label_dx": 0.0,
            "label_dy": -18.0,
            "metric_dx": 0.0,
            "metric_dy": 24.0,
            "hover_dx": 18.0,
            "hover_dy": -18.0,
            "hover_anchor": "start",
            "battery_icon_dx": 12.0,
            "battery_icon_dy": -28.0,
            "battery_card_dx": 12.0,
            "battery_card_dy": 42.0,
            "battery_card_anchor": "start",
            "show_voltage": False,
        }
        for node in range(1, node_count + 1)
    }


def _apply_side(meta: dict[int, dict[str, float | str | bool]], nodes: list[int], *, side: str) -> None:
    if side == "right":
        label_dx, metric_dx, anchor = 12.0, 12.0, "start"
        hover_dx, hover_anchor = 20.0, "start"
        icon_dx, card_dx, card_anchor = 14.0, 14.0, "start"
    elif side == "left":
        label_dx, metric_dx, anchor = -12.0, -12.0, "end"
        hover_dx, hover_anchor = -18.0, "end"
        icon_dx, card_dx, card_anchor = -42.0, -16.0, "end"
    else:
        label_dx, metric_dx, anchor = 0.0, 0.0, "middle"
        hover_dx, hover_anchor = 18.0, "start"
        icon_dx, card_dx, card_anchor = 14.0, 14.0, "start"

    for node in nodes:
        meta[node]["anchor"] = anchor
        meta[node]["label_dx"] = label_dx
        meta[node]["metric_dx"] = metric_dx
        meta[node]["hover_dx"] = hover_dx
        meta[node]["hover_anchor"] = hover_anchor
        meta[node]["battery_icon_dx"] = icon_dx
        meta[node]["battery_card_dx"] = card_dx
        meta[node]["battery_card_anchor"] = card_anchor


def _mark_voltage(meta: dict[int, dict[str, float | str | bool]], nodes: list[int]) -> None:
    for node in nodes:
        meta[node]["show_voltage"] = True


def _build_34_single_line_layout() -> tuple[dict[int, tuple[float, float]], dict[int, dict[str, float | str]]]:
    positions: dict[int, tuple[float, float]] = {}
    meta = _default_node_meta(34)

    _set_path(positions, list(range(1, 13)), start_x=0.12, start_y=0.14, dx=0.058, dy=0.0)
    _set_path(positions, [13, 14, 15, 16], start_x=positions[3][0] + 0.015, start_y=0.30, dx=0.0, dy=0.13)
    _set_path(positions, list(range(17, 28)), start_x=positions[6][0] - 0.02, start_y=0.28, dx=0.0, dy=0.055)
    _set_path(positions, [28, 29, 30], start_x=positions[7][0] + 0.11, start_y=0.40, dx=0.0, dy=0.13)
    _set_path(positions, [31, 32, 33, 34], start_x=positions[10][0] + 0.11, start_y=0.36, dx=0.0, dy=0.11)

    _apply_side(meta, list(range(1, 13)), side="center")
    _apply_side(meta, [13, 14, 15, 16], side="right")
    _apply_side(meta, [17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27], side="right")
    _apply_side(meta, [28, 29, 30], side="left")
    _apply_side(meta, [31, 32, 33, 34], side="left")

    for node in [17, 19, 21, 23, 25, 27]:
        _apply_side(meta, [node], side="left")
    for node in [28, 30]:
        _apply_side(meta, [node], side="right")
    for node in [31, 33]:
        _apply_side(meta, [node], side="right")
    _mark_voltage(meta, [1, 3, 4, 6, 7, 8, 10, 11, 13, 15, 17, 19, 21, 23, 25, 26, 28, 29, 31, 33])

    return positions, meta


def _build_69_single_line_layout() -> tuple[dict[int, tuple[float, float]], dict[int, dict[str, float | str]]]:
    positions: dict[int, tuple[float, float]] = {}
    meta = _default_node_meta(69)

    _set_path(positions, list(range(1, 29)), start_x=0.07, start_y=0.12, dx=0.028, dy=0.0)
    _set_path(positions, [29, 30, 31, 32, 33, 34, 35, 36], start_x=positions[4][0] - 0.02, start_y=0.27, dx=0.0, dy=0.062)
    _set_path(positions, [37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47], start_x=positions[4][0] + 0.06, start_y=0.26, dx=0.0, dy=0.043)
    _set_path(positions, [48, 49, 50, 51], start_x=positions[5][0] + 0.11, start_y=0.28, dx=0.0, dy=0.12)
    _set_path(positions, [52, 53], start_x=positions[9][0] + 0.08, start_y=0.30, dx=0.0, dy=0.17)
    _set_path(positions, list(range(54, 67)), start_x=positions[10][0] + 0.16, start_y=0.24, dx=0.0, dy=0.043)
    _set_path(positions, [67, 68], start_x=positions[12][0] + 0.08, start_y=0.30, dx=0.0, dy=0.17)
    _set_path(positions, [69], start_x=positions[13][0] + 0.12, start_y=0.30, dx=0.0, dy=0.0)

    _apply_side(meta, list(range(1, 29)), side="center")
    _apply_side(meta, [29, 30, 31, 32, 33, 34, 35, 36], side="left")
    _apply_side(meta, [37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47], side="right")
    _apply_side(meta, [48, 49, 50, 51], side="right")
    _apply_side(meta, [52, 53], side="right")
    _apply_side(meta, list(range(54, 67)), side="right")
    _apply_side(meta, [67, 68, 69], side="right")

    for node in [30, 32, 34, 36]:
        _apply_side(meta, [node], side="right")
    for node in [38, 40, 42, 44, 46]:
        _apply_side(meta, [node], side="left")
    for node in [55, 57, 59, 61, 63, 65]:
        _apply_side(meta, [node], side="left")
    _mark_voltage(meta, [1, 4, 8, 12, 18, 24, 28, 29, 33, 37, 43, 47, 50, 52, 54, 58, 62, 66, 67, 69])

    return positions, meta


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
        single_line_positions, node_meta = _build_34_single_line_layout()
    elif node_count == 69:
        single_line_positions, node_meta = _build_69_single_line_layout()
    else:
        single_line_positions = tree_positions
        node_meta = _default_node_meta(node_count)

    return {
        "view": "single_line",
        "node_count": node_count,
        "base_edges": [[left, right] for left, right in base_edges],
        "positions": {str(node): {"x": x_coord, "y": y_coord} for node, (x_coord, y_coord) in single_line_positions.items()},
        "node_meta": {str(node): meta for node, meta in node_meta.items()},
        "graph_positions": {str(node): {"x": x_coord, "y": y_coord} for node, (x_coord, y_coord) in tree_positions.items()},
    }
