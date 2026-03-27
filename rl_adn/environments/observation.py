from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SlotFeatures:
    active_power_kw: np.ndarray
    renewable_active_power_kw: np.ndarray
    price: float


@dataclass(frozen=True)
class ObservationSnapshot:
    active_power_kw: np.ndarray
    renewable_active_power_kw: np.ndarray
    price: float
    node_voltages_pu: np.ndarray
    battery_soc: np.ndarray
    raw_state: np.ndarray
    normalized_state: np.ndarray


@dataclass(frozen=True)
class StateScaler:
    node_count: int
    battery_count: int
    active_power_min: float
    active_power_max: float
    price_min: float
    price_max: float
    episode_length: int
    min_soc: float
    max_soc: float

    def normalize(self, raw_state: np.ndarray) -> np.ndarray:
        state = raw_state.astype(np.float32, copy=True)
        state[: self.node_count] = (state[: self.node_count] - self.active_power_min) / (self.active_power_max - self.active_power_min)
        state[self.node_count : self.node_count + self.battery_count] = (state[self.node_count : self.node_count + self.battery_count] - self.min_soc) / (self.max_soc - self.min_soc)
        price_index = self.node_count + self.battery_count
        state[price_index] = (state[price_index] - self.price_min) / (self.price_max - self.price_min)
        time_index = price_index + 1
        state[time_index] = state[time_index] / max(self.episode_length - 1, 1)
        return state

    def denormalize(self, normalized_state: np.ndarray) -> np.ndarray:
        state = normalized_state.astype(np.float32, copy=True)
        state[: self.node_count] = (state[: self.node_count] * (self.active_power_max - self.active_power_min)) + self.active_power_min
        state[self.node_count : self.node_count + self.battery_count] = (state[self.node_count : self.node_count + self.battery_count] * (self.max_soc - self.min_soc)) + self.min_soc
        price_index = self.node_count + self.battery_count
        state[price_index] = state[price_index] * (self.price_max - self.price_min) + self.price_min
        time_index = price_index + 1
        state[time_index] = state[time_index] * max(self.episode_length - 1, 1)
        return state


def build_default_state(
    *,
    slot_features: SlotFeatures,
    battery_soc: np.ndarray,
    current_time: int,
    node_voltages_pu: np.ndarray,
    battery_nodes: tuple[int, ...],
    scaler: StateScaler,
) -> ObservationSnapshot:
    raw_state = np.concatenate(
        (
            slot_features.active_power_kw.astype(np.float32, copy=False),
            battery_soc.astype(np.float32, copy=False),
            np.array([slot_features.price, float(current_time)], dtype=np.float32),
            node_voltages_pu[list(battery_nodes)].astype(np.float32, copy=False),
        )
    )
    normalized_state = scaler.normalize(raw_state)
    return ObservationSnapshot(
        active_power_kw=slot_features.active_power_kw.astype(np.float32, copy=True),
        renewable_active_power_kw=slot_features.renewable_active_power_kw.astype(np.float32, copy=True),
        price=float(slot_features.price),
        node_voltages_pu=node_voltages_pu.astype(np.float32, copy=True),
        battery_soc=battery_soc.astype(np.float32, copy=True),
        raw_state=raw_state,
        normalized_state=normalized_state,
    )
