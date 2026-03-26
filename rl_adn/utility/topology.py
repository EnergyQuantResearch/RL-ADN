from typing import Dict, Iterable, List, Tuple

import networkx as nx
import numpy as np
import pandas as pd

from rl_adn.environments.topology_scenarios import TopologyScenario


Edge = Tuple[int, int]


def _normalize_edge(edge: Edge) -> Edge:
    return tuple(sorted((int(edge[0]), int(edge[1]))))


def _find_line_index(line_info: pd.DataFrame, edge: Edge) -> int:
    normalized = _normalize_edge(edge)
    for idx, row in line_info.iterrows():
        row_edge = _normalize_edge((int(row["FROM"]), int(row["TO"])))
        if row_edge == normalized:
            return idx
    raise KeyError(f"Edge {edge} does not exist in the baseline topology")


def apply_topology_scenario(line_info: pd.DataFrame, scenario: TopologyScenario) -> pd.DataFrame:
    scenario_lines = line_info.copy(deep=True)
    original_dtypes = scenario_lines.dtypes.to_dict()
    scenario_lines["STATUS"] = scenario_lines["STATUS"].astype(int)

    for old_edge, new_edge in scenario.rewires:
        old_idx = _find_line_index(scenario_lines, old_edge)
        inherited = scenario_lines.loc[old_idx].copy()
        inherited["FROM"] = int(new_edge[0])
        inherited["TO"] = int(new_edge[1])
        inherited["STATUS"] = 1
        scenario_lines.loc[old_idx, "STATUS"] = 0
        scenario_lines = pd.concat([scenario_lines, inherited.to_frame().T], ignore_index=True)

    for column, dtype in original_dtypes.items():
        scenario_lines[column] = scenario_lines[column].astype(dtype)
    return scenario_lines


def get_active_edges(line_info: pd.DataFrame) -> List[Edge]:
    active = line_info[line_info["STATUS"].astype(int) == 1]
    return [(int(row["FROM"]), int(row["TO"])) for _, row in active.iterrows()]


def validate_radial_topology(bus_info: pd.DataFrame, line_info: pd.DataFrame) -> Dict[str, object]:
    node_ids = [int(node) for node in bus_info["NODES"].tolist()]
    slack_nodes = bus_info.loc[bus_info["Tb"] == 1, "NODES"].tolist()
    if len(slack_nodes) != 1:
        raise ValueError("Exactly one slack node is required")

    active_edges = get_active_edges(line_info)
    active_graph = nx.Graph()
    active_graph.add_nodes_from(node_ids)
    active_graph.add_edges_from(active_edges)

    is_connected = nx.is_connected(active_graph)
    is_radial = active_graph.number_of_edges() == len(node_ids) - 1 and nx.is_tree(active_graph)
    slack = int(slack_nodes[0])
    slack_reaches_all = len(nx.node_connected_component(active_graph, slack)) == len(node_ids) if is_connected else False

    return {
        "is_connected": is_connected,
        "is_radial": is_radial,
        "active_edge_count": active_graph.number_of_edges(),
        "slack_reaches_all": slack_reaches_all,
    }


def build_adjacency_matrix(node_count: int, line_info: pd.DataFrame) -> np.ndarray:
    adjacency = np.zeros((node_count, node_count), dtype=np.int64)
    for from_node, to_node in get_active_edges(line_info):
        adjacency[from_node - 1, to_node - 1] = 1
        adjacency[to_node - 1, from_node - 1] = 1
    return adjacency


def build_edge_index(line_info: pd.DataFrame) -> np.ndarray:
    active_edges = get_active_edges(line_info)
    bidirectional_edges: List[Edge] = []
    for from_node, to_node in active_edges:
        bidirectional_edges.append((from_node - 1, to_node - 1))
        bidirectional_edges.append((to_node - 1, from_node - 1))
    return np.asarray(bidirectional_edges, dtype=np.int64).T
