from __future__ import annotations

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


def _require_pandapower():
    try:
        import pandapower as pp
        import pandapower.topology as pandapower_topology
    except ImportError as exc:
        raise ImportError("pandapower helpers require the optional dependency 'pandapower'.") from exc
    return pp, pandapower_topology


def generate_network(nodes: int, child: int = 3, plot_graph: bool = False, load_factor: int = 2, line_factor: int = 3):
    """Generate a synthetic radial feeder and return node/line DataFrames."""
    line_count = nodes - 1
    graph = nx.full_rary_tree(child, nodes)

    if plot_graph:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        nx.draw_kamada_kawai(graph, node_size=100, with_labels=True, font_size="medium", ax=ax)

    assert nodes == len(graph.nodes)
    assert line_count == len(graph.edges)

    pct, ict, zct = 1, 0, 0
    nodes_frame = pd.DataFrame(list(graph.nodes), columns=["NODES"]) + 1
    active_power = np.random.normal(50 * load_factor, scale=50, size=nodes).round(3)
    reactive_power = (active_power * 0.1).round(3)
    bus_properties = pd.DataFrame(
        {
            "Tb": np.full(nodes, 0, dtype=int),
            "PD": active_power,
            "QD": reactive_power,
            "Pct": np.full(nodes, pct, dtype=int),
            "Ict": np.full(nodes, ict, dtype=int),
            "Zct": np.full(nodes, zct, dtype=int),
        }
    )
    bus_properties.loc[0] = [1, 0.0, 0.0, pct, ict, zct]

    resistance, reactance = 0.3144 / line_factor, 0.054 / line_factor
    lines = pd.DataFrame.from_records(list(graph.edges), columns=["FROM", "TO"]) + 1
    line_properties = pd.DataFrame(
        np.tile([[resistance, reactance, 0, 1, 1]], (line_count, 1)),
        columns=["R", "X", "B", "STATUS", "TAP"],
    ).astype({"R": float, "X": float, "B": int, "STATUS": int, "TAP": int})

    return pd.concat([nodes_frame, bus_properties], axis=1), pd.concat([lines, line_properties], axis=1)


def create_pandapower_net(network_info: dict, branch_info: pd.DataFrame | None = None, bus_info: pd.DataFrame | None = None):
    """Create a pandapower network from packaged network metadata and optional DataFrames."""
    vm_pu = network_info["vm_pu"]
    branch_info_file = network_info["branch_info_file"]
    bus_info_file = network_info["bus_info_file"]
    pp, _ = _require_pandapower()

    branch_frame = pd.read_csv(branch_info_file, encoding="utf-8") if branch_info is None else branch_info.copy(deep=True)
    bus_frame = pd.read_csv(bus_info_file, encoding="utf-8") if bus_info is None else bus_info.copy(deep=True)

    net = pp.create_empty_network()
    bus_lookup = {bus_name: pp.create_bus(net, vn_kv=11.0, name=f"Bus {bus_name}") for bus_name in bus_frame["NODES"]}

    slack_bus = bus_frame[bus_frame["Tb"] == 1]["NODES"].values
    if len(slack_bus) != 1:
        raise ValueError("Exactly one slack bus is required for pandapower conversion")
    pp.create_ext_grid(net, bus=bus_lookup[slack_bus.item()], vm_pu=vm_pu, name="Grid Connection")

    active_branches = branch_frame[branch_frame["STATUS"].astype(float) != 0].reset_index(drop=True)
    for line_index, (_row_index, values) in enumerate(active_branches[["FROM", "TO", "R", "X", "B"]].iterrows(), start=1):
        from_bus, to_bus, resistance, reactance, susceptance = values
        pp.create_line_from_parameters(
            net,
            from_bus=bus_lookup[from_bus],
            to_bus=bus_lookup[to_bus],
            length_km=1,
            r_ohm_per_km=resistance,
            x_ohm_per_km=reactance,
            c_nf_per_km=susceptance,
            max_i_ka=10,
            name=f"Line {line_index}",
        )

    for node in bus_frame["NODES"]:
        pp.create_load(net, bus=bus_lookup[node], p_mw=0.02, q_mvar=0.0, name="Load")
    return net


def plot_pandapower_net(net) -> None:
    """Plot a pandapower network using the topology graph helper."""
    _, pandapower_topology = _require_pandapower()
    graph = pandapower_topology.create_nxgraph(net, respect_switches=False)
    pos = {bus: (net.bus_geodata.at[bus, "x"], net.bus_geodata.at[bus, "y"]) for bus in graph.nodes}

    buses = [bus for bus in graph.nodes if net.bus.at[bus, "type"] == "b"]
    loads = [bus for bus in graph.nodes if net.bus.at[bus, "type"] == "l"]
    pv_generations = [bus for bus in graph.nodes if net.bus.at[bus, "type"] == "s"]

    nx.draw_networkx_nodes(graph, pos, nodelist=buses, node_color="red", node_size=200, label="Buses")
    nx.draw_networkx_nodes(graph, pos, nodelist=loads, node_color="blue", node_size=200, label="Loads")
    nx.draw_networkx_nodes(graph, pos, nodelist=pv_generations, node_color="green", node_size=200, label="PV Generations")
    nx.draw_networkx_edges(graph, pos, width=1.0, alpha=0.5)
    nx.draw_networkx_labels(graph, pos, labels={bus: str(bus).split(" ")[-1] for bus in graph.nodes}, font_size=8)

    plt.legend()
    plt.axis("off")
    plt.show()
