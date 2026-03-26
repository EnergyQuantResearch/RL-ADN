from __future__ import annotations

from dataclasses import replace
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from rl_adn.config import DEFAULT_ENV_CONFIG, EnvConfig
from rl_adn.data import GeneralPowerDataManager
from rl_adn.environments.battery import Battery
from rl_adn.environments.observation import ObservationSnapshot, SlotFeatures, StateScaler, build_default_state
from rl_adn.environments.reward import RewardBreakdown, compute_default_reward
from rl_adn.environments.solvers import LaurentSolverAdapter, PandaPowerSolverAdapter, PowerFlowSnapshot
from rl_adn.environments.topology_scenarios import get_topology_scenario
from rl_adn.network.topology import (
    apply_topology_scenario,
    build_adjacency_matrix,
    build_edge_index,
    get_active_edges,
    validate_radial_topology,
)


class PowerNetEnv(gym.Env[np.ndarray, np.ndarray]):
    """Gymnasium environment for ESS dispatch in active distribution networks."""

    metadata = {"render_modes": []}

    def __init__(self, config: EnvConfig = DEFAULT_ENV_CONFIG) -> None:
        super().__init__()
        self.config = config
        self.algorithm = config.algorithm
        self.train = config.train
        self.state_pattern = config.state_pattern
        self.feeder_id = config.feeder_id
        self.node_count = config.node_count
        self.battery_nodes = tuple(config.battery_nodes)
        self.topology_config = config.topology
        self.voltage_low_boundary, self.voltage_high_boundary = config.voltage_limits
        self.year = config.year
        self.month = config.month
        self.day = config.day

        self._rng = np.random.default_rng()
        self._episode_done = False

        self._baseline_bus_info = pd.read_csv(config.bus_info_file)
        self._baseline_line_info = pd.read_csv(config.branch_info_file)
        if len(self._baseline_bus_info) != self.node_count:
            raise ValueError("EnvConfig.node_count does not match the packaged bus data")

        self.data_manager = GeneralPowerDataManager(config.time_series_data_path)
        self.battery_config = replace(config.battery, time_interval_minutes=self.data_manager.time_interval)
        self.batteries = {node_index: Battery(self.battery_config) for node_index in self.battery_nodes}
        self.episode_length = int(24 * 60 / self.data_manager.time_interval)
        self.state_scaler = StateScaler(
            node_count=self.node_count,
            battery_count=len(self.battery_nodes),
            active_power_min=self.data_manager.active_power_min,
            active_power_max=self.data_manager.active_power_max,
            price_min=self.data_manager.price_min,
            price_max=self.data_manager.price_max,
            episode_length=self.episode_length,
            min_soc=self.battery_config.min_soc,
            max_soc=self.battery_config.max_soc,
        )

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(len(self.battery_nodes),),
            dtype=np.float32,
        )
        state_dim = self.node_count + 2 * len(self.battery_nodes) + 2
        self.observation_space = spaces.Box(
            low=-2.0,
            high=2.0,
            shape=(state_dim,),
            dtype=np.float32,
        )

        self.active_power_indices = [self.data_manager.df.columns.get_loc(col) for col in self.data_manager.active_power_cols]
        self.renewable_active_power_indices = [self.data_manager.df.columns.get_loc(col) for col in self.data_manager.renewable_active_power_cols]
        self.price_index = self.data_manager.df.columns.get_loc(self.data_manager.price_col[0])

        self.current_time = 0
        self.current_scenario = None
        self.active_line_info: pd.DataFrame | None = None
        self.active_bus_info: pd.DataFrame | None = None
        self.active_edges: list[tuple[int, int]] = []
        self.adjacency_matrix: np.ndarray | None = None
        self.edge_index: np.ndarray | None = None
        self.solver = None

        self.current_slot_features: SlotFeatures | None = None
        self.current_observation: ObservationSnapshot | None = None
        self.current_precontrol_snapshot: PowerFlowSnapshot | None = None
        self.last_reward_breakdown: RewardBreakdown | None = None

        self._apply_topology(self.topology_config.scenario_id)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the environment and return the initial observation and metadata."""
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        scenario_id = self._sample_topology_scenario()
        self._apply_topology(scenario_id)
        self._select_episode_date()
        self.current_time = 0
        self._episode_done = False
        self._reset_batteries()
        self.current_observation, self.current_slot_features, self.current_precontrol_snapshot = self._observe_current_slot()
        return self.current_observation.normalized_state.copy(), self._build_info()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Advance one environment step using a flat action vector in ``[-1, 1]``."""
        if self.current_observation is None or self.current_slot_features is None or self.current_precontrol_snapshot is None:
            raise RuntimeError("reset() must be called before step()")
        if self._episode_done:
            raise RuntimeError("Episode has finished; call reset() before step() again")

        action_vector = np.asarray(action, dtype=np.float32).reshape(-1)
        if action_vector.shape != (len(self.battery_nodes),):
            raise ValueError(f"Expected action shape {(len(self.battery_nodes),)}, received {tuple(action_vector.shape)}")

        battery_dispatch_kw = np.array(
            [self.batteries[node_index].step(action_component) for node_index, action_component in zip(self.battery_nodes, action_vector)],
            dtype=np.float32,
        )

        net_load_kw = self.current_slot_features.active_power_kw - self.current_slot_features.renewable_active_power_kw
        post_control_snapshot = self.solver.dispatch(net_load_kw, self.battery_nodes, battery_dispatch_kw)
        reward_breakdown = compute_default_reward(
            price=self.current_slot_features.price,
            saved_power_kw=self.current_precontrol_snapshot.import_power_kw - post_control_snapshot.import_power_kw,
            battery_voltages_pu=post_control_snapshot.node_voltages_pu[list(self.battery_nodes)],
        )
        self.last_reward_breakdown = reward_breakdown

        truncated = self.current_time >= self.episode_length - 1
        terminated = False
        info = self._build_info(post_control_snapshot=post_control_snapshot, battery_dispatch_kw=battery_dispatch_kw)

        if truncated:
            self._episode_done = True
            return self.current_observation.normalized_state.copy(), reward_breakdown.total, terminated, True, info

        self.current_time += 1
        self.current_observation, self.current_slot_features, self.current_precontrol_snapshot = self._observe_current_slot()
        return self.current_observation.normalized_state.copy(), reward_breakdown.total, terminated, False, info

    def get_topology_metadata(self) -> dict[str, Any]:
        return {
            "feeder_id": self.feeder_id,
            "scenario_id": self.current_scenario.scenario_id,
            "node_count": self.node_count,
            "edge_count": len(self.active_edges),
            "active_edges": list(self.active_edges),
        }

    def get_graph_data(self) -> dict[str, Any]:
        if self.adjacency_matrix is None or self.edge_index is None or self.active_line_info is None:
            raise RuntimeError("Topology has not been initialized")
        return {
            "adjacency": self.adjacency_matrix.copy(),
            "edge_index": self.edge_index.copy(),
            "node_ids": np.arange(1, self.node_count + 1, dtype=np.int64),
            "active_line_data": self.active_line_info[["FROM", "TO", "R", "X", "B", "STATUS", "TAP"]].to_dict("records"),
        }

    def render(self) -> None:
        return None

    def _sample_topology_scenario(self) -> str:
        if self.topology_config.mode == "fixed":
            return self.topology_config.scenario_id
        if self.topology_config.mode == "scenario_pool":
            return str(self._rng.choice(self.topology_config.scenario_pool))
        raise ValueError("Unsupported topology mode")

    def _apply_topology(self, scenario_id: str) -> None:
        scenario = get_topology_scenario(self.node_count, scenario_id)
        active_line_info = apply_topology_scenario(self._baseline_line_info, scenario)
        validation = validate_radial_topology(self._baseline_bus_info, active_line_info)
        if not (validation["is_connected"] and validation["is_radial"] and validation["slack_reaches_all"]):
            raise ValueError(f"Topology scenario {scenario_id} is not a valid radial topology")

        self.current_scenario = scenario
        self.active_line_info = active_line_info
        self.active_bus_info = self._baseline_bus_info.copy(deep=True)
        self.active_edges = get_active_edges(active_line_info)
        self.adjacency_matrix = build_adjacency_matrix(self.node_count, active_line_info)
        self.edge_index = build_edge_index(active_line_info)

        if self.algorithm == "Laurent":
            self.solver = LaurentSolverAdapter(
                bus_info=self.active_bus_info,
                line_info=self.active_line_info,
                s_base=self.config.s_base,
            )
        elif self.algorithm == "PandaPower":
            self.solver = PandaPowerSolverAdapter(
                network_info=self.config.network_info,
                bus_info=self.active_bus_info,
                line_info=self.active_line_info,
                s_base=self.config.s_base,
            )
        else:
            raise ValueError("Unsupported algorithm")

    def _select_episode_date(self) -> None:
        candidate_dates = self.data_manager.train_dates if self.train else self.data_manager.test_dates
        if not candidate_dates:
            raise ValueError("No episode dates are available for the requested split")
        date_index = int(self._rng.integers(0, len(candidate_dates)))
        self.year, self.month, self.day = candidate_dates[date_index]

    def _reset_batteries(self) -> None:
        for battery in self.batteries.values():
            battery.reset()

    def _extract_slot_features(self, timeslot: int) -> SlotFeatures:
        one_slot_data = self.data_manager.select_timeslot_data(self.year, self.month, self.day, timeslot)
        active_power = np.asarray(one_slot_data[self.active_power_indices], dtype=np.float32)
        renewable_active_power = np.zeros_like(active_power)
        if self.renewable_active_power_indices:
            renewable_active_power = np.asarray(one_slot_data[self.renewable_active_power_indices], dtype=np.float32)
        price = float(one_slot_data[self.price_index])

        if active_power.shape[0] != self.node_count:
            raise ValueError(f"Time-series active power dimension {active_power.shape[0]} does not match feeder node count {self.node_count}")
        if renewable_active_power.shape[0] != self.node_count:
            raise ValueError(f"Time-series renewable power dimension {renewable_active_power.shape[0]} does not match feeder node count {self.node_count}")

        return SlotFeatures(
            active_power_kw=active_power,
            renewable_active_power_kw=renewable_active_power,
            price=price,
        )

    def _observe_current_slot(self) -> tuple[ObservationSnapshot, SlotFeatures, PowerFlowSnapshot]:
        slot_features = self._extract_slot_features(self.current_time)
        net_load_kw = slot_features.active_power_kw - slot_features.renewable_active_power_kw
        precontrol_snapshot = self.solver.observe(net_load_kw)
        battery_soc = np.array([self.batteries[node_index].SOC() for node_index in self.battery_nodes], dtype=np.float32)
        observation = build_default_state(
            slot_features=slot_features,
            battery_soc=battery_soc,
            current_time=self.current_time,
            node_voltages_pu=precontrol_snapshot.node_voltages_pu,
            battery_nodes=self.battery_nodes,
            scaler=self.state_scaler,
        )
        return observation, slot_features, precontrol_snapshot

    def _build_info(
        self,
        *,
        post_control_snapshot: PowerFlowSnapshot | None = None,
        battery_dispatch_kw: np.ndarray | None = None,
    ) -> dict[str, Any]:
        info: dict[str, Any] = {
            "feeder_id": self.feeder_id,
            "topology_scenario": self.current_scenario.scenario_id,
            "active_edges_count": len(self.active_edges),
            "current_time": self.current_time,
        }
        if self.current_observation is not None:
            info["current_normalized_obs"] = self.current_observation.normalized_state.copy()
        if battery_dispatch_kw is not None:
            info["battery_dispatch_kw"] = np.asarray(battery_dispatch_kw, dtype=np.float32).copy()
        if post_control_snapshot is not None and self.last_reward_breakdown is not None:
            info["post_control_voltage_pu"] = post_control_snapshot.node_voltages_pu.copy()
            info["reward_breakdown"] = {
                "economic": self.last_reward_breakdown.economic,
                "voltage_penalty": self.last_reward_breakdown.voltage_penalty,
                "saved_money": self.last_reward_breakdown.saved_money,
            }
        return info
