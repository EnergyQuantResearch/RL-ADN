from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rl_adn.config import make_env_config
from rl_adn.environments.topology_scenarios import (
    get_topology_scenario,
    list_topology_scenario_ids,
)
from rl_adn.network.grid import GridTensor
from rl_adn.network.topology import (
    apply_topology_scenario,
    build_adjacency_matrix,
    build_edge_index,
    get_active_edges,
    validate_radial_topology,
)

ROOT = Path(__file__).resolve().parents[1]
NETWORK_ROOT = ROOT / "rl_adn" / "data_sources" / "network_data"


def _load_network_tables(node: int):
    node_dir = NETWORK_ROOT / f"node_{node}"
    bus_info = pd.read_csv(node_dir / f"Nodes_{node}.csv")
    line_info = pd.read_csv(node_dir / f"Lines_{node}.csv")
    return bus_info, line_info


def _write_synthetic_timeseries(path: Path, node_count: int, periods: int = 96 * 2) -> None:
    index = pd.date_range("2021-01-01", periods=periods, freq="15min", tz="UTC")
    data = {"date_time": index}

    for idx in range(1, node_count + 1):
        data[f"active_power_node_{idx}"] = np.full(periods, float(idx))
    for idx in range(1, node_count + 1):
        data[f"renewable_active_power_node_{idx}"] = np.zeros(periods)
    data["price"] = np.linspace(10.0, 20.0, periods)

    pd.DataFrame(data).to_csv(path, index=False)


def test_paper_scenario_registry_includes_tp1_to_tp7_for_34_and_69():
    assert list_topology_scenario_ids(34) == ["TP1", "TP2", "TP3", "TP4", "TP5", "TP6", "TP7"]
    assert list_topology_scenario_ids(69) == ["TP1", "TP2", "TP3", "TP4", "TP5", "TP6", "TP7"]


@pytest.mark.parametrize("node", [34, 69])
@pytest.mark.parametrize("scenario_id", ["TP1", "TP2", "TP3", "TP4", "TP5", "TP6", "TP7"])
def test_topology_scenarios_generate_connected_radial_graphs(node, scenario_id):
    bus_info, line_info = _load_network_tables(node)
    scenario = get_topology_scenario(node, scenario_id)

    scenario_lines = apply_topology_scenario(line_info, scenario)
    validation = validate_radial_topology(bus_info, scenario_lines)

    assert validation["is_connected"] is True
    assert validation["is_radial"] is True
    assert validation["active_edge_count"] == len(bus_info) - 1


@pytest.mark.parametrize("node", [34, 69])
@pytest.mark.parametrize("scenario_id", ["TP1", "TP2", "TP3", "TP4", "TP5", "TP6", "TP7"])
def test_each_topology_scenario_supports_laurent_initialization(node, scenario_id):
    bus_info, line_info = _load_network_tables(node)
    scenario = get_topology_scenario(node, scenario_id)
    scenario_lines = apply_topology_scenario(line_info, scenario)

    grid = GridTensor(node_file_path="", lines_file_path="", from_file=False, nodes_frame=bus_info, lines_frame=scenario_lines)
    grid.Q_file = np.zeros(len(bus_info) - 1)
    solution = grid.run_pf(active_power=np.zeros(len(bus_info) - 1))

    assert solution["convergence"] is True


def test_make_env_config_preserves_old_behavior_when_topology_fields_omitted():
    config = make_env_config()

    assert config.algorithm == "Laurent"
    assert config.topology.mode == "fixed"
    assert config.topology.scenario_id == "TP1"
    assert config.topology.scenario_pool == ()
    assert config.topology.return_graph is False


def test_make_env_config_accepts_fixed_topology_scenario():
    config = make_env_config(node=34, topology_scenario="TP3")

    assert config.feeder_id == "34-bus"
    assert config.topology.mode == "fixed"
    assert config.topology.scenario_id == "TP3"


def test_make_env_config_provides_curated_defaults_for_supported_feeders():
    config_34 = make_env_config(node=34)
    config_69 = make_env_config(
        node=69,
        topology_scenario="TP2",
        time_series_data_path="synthetic.csv",
    )

    assert config_34.battery_nodes == (11, 15, 26, 29, 33)
    assert config_69.battery_nodes == (13, 15, 17, 19, 21, 23, 25, 26, 64)


def test_make_env_config_rejects_empty_scenario_pool():
    with pytest.raises(ValueError):
        make_env_config(node=34, topology_mode="scenario_pool", topology_pool=[])


def test_fixed_topology_reset_is_reproducible_and_exposes_metadata():
    pytest.importorskip("gymnasium")
    from rl_adn import PowerNetEnv

    env = PowerNetEnv(make_env_config(node=34, topology_scenario="TP4", return_graph=True))

    state_a, info_a = env.reset(seed=2026)
    state_b, info_b = env.reset(seed=2026)

    assert state_a.shape == state_b.shape
    assert info_a["topology_scenario"] == "TP4"
    assert info_b["topology_scenario"] == "TP4"
    assert info_a["feeder_id"] == "34-bus"
    assert info_a["active_edges_count"] == info_b["active_edges_count"]


def test_scenario_pool_sampling_is_deterministic_under_fixed_seed():
    pytest.importorskip("gymnasium")
    from rl_adn import PowerNetEnv

    env_a = PowerNetEnv(make_env_config(node=34, topology_mode="scenario_pool", topology_pool=["TP2", "TP3", "TP4"]))
    seq_a = [env_a.reset(seed=2026 + index)[1]["topology_scenario"] for index in range(3)]

    env_b = PowerNetEnv(make_env_config(node=34, topology_mode="scenario_pool", topology_pool=["TP2", "TP3", "TP4"]))
    seq_b = [env_b.reset(seed=2026 + index)[1]["topology_scenario"] for index in range(3)]

    assert seq_a == seq_b


def test_graph_exports_match_active_topology():
    pytest.importorskip("gymnasium")
    from rl_adn import PowerNetEnv

    env = PowerNetEnv(make_env_config(node=34, topology_scenario="TP2", return_graph=True))
    env.reset()

    metadata = env.get_topology_metadata()
    graph_data = env.get_graph_data()

    assert metadata["scenario_id"] == "TP2"
    assert graph_data["adjacency"].shape == (34, 34)
    assert graph_data["edge_index"].shape[0] == 2
    assert len(graph_data["node_ids"]) == 34
    assert metadata["edge_count"] == len(metadata["active_edges"])

    expected_adjacency = build_adjacency_matrix(34, env.active_line_info)
    expected_edge_index = build_edge_index(env.active_line_info)
    assert np.array_equal(graph_data["adjacency"], expected_adjacency)
    assert np.array_equal(graph_data["edge_index"], expected_edge_index)


def test_69_bus_env_supports_custom_timeseries_with_fixed_topology(tmp_path: Path):
    pytest.importorskip("gymnasium")
    from rl_adn import PowerNetEnv

    synthetic_path = tmp_path / "69_node_time_series.csv"
    _write_synthetic_timeseries(synthetic_path, node_count=69)

    env = PowerNetEnv(
        make_env_config(
            node=69,
            topology_scenario="TP2",
            time_series_data_path=str(synthetic_path),
        )
    )

    state, info = env.reset(seed=2026)
    assert state is not None
    assert info["topology_scenario"] == "TP2"
    assert info["feeder_id"] == "69-bus"


def test_get_active_edges_reflects_applied_scenario():
    _, line_info = _load_network_tables(34)
    scenario = get_topology_scenario(34, "TP2")
    scenario_lines = apply_topology_scenario(line_info, scenario)

    active_edges = get_active_edges(scenario_lines)
    assert (24, 26) in active_edges or (26, 24) in active_edges
    assert (25, 26) not in active_edges and (26, 25) not in active_edges
