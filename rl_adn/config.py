from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

from rl_adn.environments.topology_scenarios import get_topology_scenario

PACKAGE_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PACKAGE_ROOT / "data_sources"
NETWORK_ROOT = DATA_ROOT / "network_data"
TIME_SERIES_ROOT = DATA_ROOT / "time_series_data"

DEFAULT_NODE_COUNT = 34
CURATED_BATTERY_NODES = {
    34: (11, 15, 26, 29, 33),  # paper nodes {12, 16, 27, 30, 34}
    69: (13, 15, 17, 19, 21, 23, 25, 26, 64),  # paper nodes {14, 16, 18, 20, 22, 24, 26, 27, 65}
}


@dataclass(frozen=True)
class BatteryConfig:
    capacity_kwh: float = 300.0
    max_charge_kw: float = 50.0
    max_discharge_kw: float = 50.0
    efficiency: float = 1.0
    degradation_eur_per_kw: float = 0.0
    max_soc: float = 0.8
    min_soc: float = 0.2
    initial_soc: float = 0.4
    time_interval_minutes: float = 15.0

    def __post_init__(self) -> None:
        if self.capacity_kwh <= 0:
            raise ValueError("capacity_kwh must be positive")
        if self.max_charge_kw <= 0 or self.max_discharge_kw <= 0:
            raise ValueError("Battery charge and discharge limits must be positive")
        if self.time_interval_minutes <= 0:
            raise ValueError("time_interval_minutes must be positive")
        if not 0 <= self.min_soc < self.max_soc <= 1:
            raise ValueError("Battery SOC bounds must satisfy 0 <= min_soc < max_soc <= 1")
        if not self.min_soc <= self.initial_soc <= self.max_soc:
            raise ValueError("initial_soc must lie within [min_soc, max_soc]")

    @classmethod
    def default(cls, *, time_interval_minutes: float = 15.0) -> "BatteryConfig":
        return cls(time_interval_minutes=time_interval_minutes)


@dataclass(frozen=True)
class TopologyConfig:
    mode: Literal["fixed", "scenario_pool"] = "fixed"
    scenario_id: str = "TP1"
    scenario_pool: tuple[str, ...] = ()
    return_graph: bool = False

    def __post_init__(self) -> None:
        if self.mode not in {"fixed", "scenario_pool"}:
            raise ValueError("TopologyConfig.mode must be 'fixed' or 'scenario_pool'")
        if self.mode == "scenario_pool" and not self.scenario_pool:
            raise ValueError("TopologyConfig.scenario_pool must not be empty when mode='scenario_pool'")


@dataclass(frozen=True)
class EnvConfig:
    node_count: int
    algorithm: Literal["Laurent", "PandaPower"]
    battery_nodes: tuple[int, ...]
    battery: BatteryConfig
    year: int
    month: int
    day: int
    train: bool
    state_pattern: str
    voltage_limits: tuple[float, float]
    vm_pu: float
    s_base: float
    bus_info_file: str
    branch_info_file: str
    time_series_data_path: str
    feeder_id: str
    topology: TopologyConfig = field(default_factory=TopologyConfig)

    def __post_init__(self) -> None:
        if self.algorithm not in {"Laurent", "PandaPower"}:
            raise ValueError("algorithm must be 'Laurent' or 'PandaPower'")
        if len(self.battery_nodes) == 0:
            raise ValueError("battery_nodes must not be empty")
        if self.state_pattern != "default":
            raise ValueError("Only the 'default' state pattern is currently supported")
        if self.voltage_limits[0] >= self.voltage_limits[1]:
            raise ValueError("voltage_limits must be ordered as (low, high)")
        if not Path(self.bus_info_file).exists():
            raise FileNotFoundError(f"Bus data file does not exist: {self.bus_info_file}")
        if not Path(self.branch_info_file).exists():
            raise FileNotFoundError(f"Branch data file does not exist: {self.branch_info_file}")

    @property
    def network_info(self) -> dict[str, object]:
        return {
            "vm_pu": self.vm_pu,
            "s_base": self.s_base,
            "bus_info_file": self.bus_info_file,
            "branch_info_file": self.branch_info_file,
        }


def _resolve_network_files(node_count: int) -> tuple[str, str]:
    node_dir = NETWORK_ROOT / f"node_{node_count}"
    bus_info_file = node_dir / f"Nodes_{node_count}.csv"
    branch_info_file = node_dir / f"Lines_{node_count}.csv"
    if not bus_info_file.exists() or not branch_info_file.exists():
        raise FileNotFoundError(f"Missing packaged network data for node_{node_count}")
    return str(bus_info_file), str(branch_info_file)


def _resolve_time_series_data_path(node_count: int, override: Optional[str]) -> str:
    if override is not None:
        return str(Path(override))

    candidate = TIME_SERIES_ROOT / f"{node_count}_node_time_series.csv"
    if candidate.exists():
        return str(candidate)

    if node_count == DEFAULT_NODE_COUNT:
        raise FileNotFoundError(f"Missing packaged time-series data for node_{node_count}")

    raise ValueError(f"No packaged time-series data is available for node_{node_count}; please provide time_series_data_path explicitly.")


def _resolve_battery_nodes(node_count: int, override: Optional[tuple[int, ...]]) -> tuple[int, ...]:
    if override is not None:
        return tuple(override)
    try:
        return CURATED_BATTERY_NODES[node_count]
    except KeyError as exc:
        raise ValueError(f"battery_nodes must be provided for node counts without a curated default (received node_count={node_count})") from exc


def make_env_config(
    *,
    node: int = DEFAULT_NODE_COUNT,
    algorithm: Literal["Laurent", "PandaPower"] = "Laurent",
    train: bool = True,
    battery_nodes: Optional[tuple[int, ...]] = None,
    battery: Optional[BatteryConfig] = None,
    year: int = 2020,
    month: int = 1,
    day: int = 1,
    state_pattern: str = "default",
    vm_pu: float = 1.0,
    s_base: float = 1000.0,
    time_series_data_path: Optional[str] = None,
    topology_mode: Literal["fixed", "scenario_pool"] = "fixed",
    topology_scenario: Optional[str] = None,
    topology_pool: Optional[list[str]] = None,
    return_graph: bool = False,
) -> EnvConfig:
    scenario_id = topology_scenario or "TP1"
    get_topology_scenario(node, scenario_id)
    if topology_pool is not None:
        for topology_case in topology_pool:
            get_topology_scenario(node, topology_case)

    bus_info_file, branch_info_file = _resolve_network_files(node)
    resolved_time_series_path = _resolve_time_series_data_path(node, time_series_data_path)
    resolved_battery_nodes = _resolve_battery_nodes(node, battery_nodes)

    resolved_battery = battery or BatteryConfig.default()
    resolved_battery = BatteryConfig(
        capacity_kwh=resolved_battery.capacity_kwh,
        max_charge_kw=resolved_battery.max_charge_kw,
        max_discharge_kw=resolved_battery.max_discharge_kw,
        efficiency=resolved_battery.efficiency,
        degradation_eur_per_kw=resolved_battery.degradation_eur_per_kw,
        max_soc=resolved_battery.max_soc,
        min_soc=resolved_battery.min_soc,
        initial_soc=resolved_battery.initial_soc,
        time_interval_minutes=resolved_battery.time_interval_minutes,
    )

    topology = TopologyConfig(
        mode=topology_mode,
        scenario_id=scenario_id,
        scenario_pool=tuple(topology_pool or ()),
        return_graph=return_graph,
    )

    return EnvConfig(
        node_count=node,
        algorithm=algorithm,
        battery_nodes=resolved_battery_nodes,
        battery=resolved_battery,
        year=year,
        month=month,
        day=day,
        train=train,
        state_pattern=state_pattern,
        voltage_limits=(0.95, 1.05),
        vm_pu=vm_pu,
        s_base=s_base,
        bus_info_file=bus_info_file,
        branch_info_file=branch_info_file,
        time_series_data_path=resolved_time_series_path,
        feeder_id=f"{node}-bus",
        topology=topology,
    )


DEFAULT_ENV_CONFIG = make_env_config()
