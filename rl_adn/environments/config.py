from pathlib import Path
from typing import Dict, List, Optional

from rl_adn.environments.topology_scenarios import get_topology_scenario

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PACKAGE_ROOT / "data_sources"
NETWORK_ROOT = DATA_ROOT / "network_data"
TIME_SERIES_ROOT = DATA_ROOT / "time_series_data"

DEFAULT_NODE = 34
DEFAULT_BATTERY_LISTS = {
    # Paper battery placements are reported with 1-based node labels.
    # RL-ADN stores controllable battery nodes as zero-based bus indices.
    34: [11, 15, 26, 29, 33],  # paper nodes {12, 16, 27, 30, 34}
    69: [13, 15, 17, 19, 21, 23, 25, 26, 64],  # paper nodes {14, 16, 18, 20, 22, 24, 26, 27, 65}
}


def _resolve_network_info(node: int, vm_pu: float, s_base: float) -> Dict[str, object]:
    node_dir = NETWORK_ROOT / f"node_{node}"
    bus_info_file = node_dir / f"Nodes_{node}.csv"
    branch_info_file = node_dir / f"Lines_{node}.csv"

    if not bus_info_file.exists() or not branch_info_file.exists():
        raise FileNotFoundError(f"Missing packaged network data for node_{node}")

    return {
        "vm_pu": vm_pu,
        "s_base": s_base,
        "bus_info_file": str(bus_info_file),
        "branch_info_file": str(branch_info_file),
    }


def _resolve_time_series_data_path(node: int, override: Optional[str]) -> str:
    if override is not None:
        return str(Path(override))

    candidate = TIME_SERIES_ROOT / f"{node}_node_time_series.csv"
    if candidate.exists():
        return str(candidate)

    if node == DEFAULT_NODE:
        raise FileNotFoundError(f"Missing packaged time-series data for node_{node}")

    raise ValueError(
        f"No packaged time-series data is available for node_{node}; please provide time_series_data_path explicitly."
    )


def make_env_config(
    node: int = DEFAULT_NODE,
    algorithm: str = "Laurent",
    train: bool = True,
    battery_list: Optional[List[int]] = None,
    year: int = 2020,
    month: int = 1,
    day: int = 1,
    state_pattern: str = "default",
    vm_pu: float = 1.0,
    s_base: float = 1000,
    time_series_data_path: Optional[str] = None,
    topology_mode: str = "fixed",
    topology_scenario: Optional[str] = None,
    topology_pool: Optional[List[str]] = None,
    return_graph: bool = False,
) -> Dict[str, object]:
    if topology_mode not in {"fixed", "scenario_pool"}:
        raise ValueError("topology_mode must be either 'fixed' or 'scenario_pool'")

    if topology_pool is not None and len(topology_pool) == 0:
        raise ValueError("topology_pool must not be empty when provided")

    if topology_scenario is not None:
        get_topology_scenario(node, topology_scenario)

    if topology_pool is not None:
        for scenario_id in topology_pool:
            get_topology_scenario(node, scenario_id)

    if battery_list is None:
        default_battery_list = DEFAULT_BATTERY_LISTS.get(node)
        if default_battery_list is None:
            raise ValueError(
                f"battery_list must be provided for node counts without a curated default (received node={node})"
            )
        battery_list = list(default_battery_list)

    return {
        "voltage_limits": [0.95, 1.05],
        "algorithm": algorithm,
        "battery_list": list(battery_list),
        "year": year,
        "month": month,
        "day": day,
        "train": train,
        "state_pattern": state_pattern,
        "network_info": _resolve_network_info(node=node, vm_pu=vm_pu, s_base=s_base),
        "time_series_data_path": _resolve_time_series_data_path(node=node, override=time_series_data_path),
        "feeder_id": f"{node}-bus",
        "topology_mode": topology_mode,
        "topology_scenario": topology_scenario,
        "topology_pool": list(topology_pool) if topology_pool is not None else None,
        "return_graph": return_graph,
    }


env_config = make_env_config()
