from __future__ import annotations

from dataclasses import replace

import numpy as np

from rl_adn.config import BatteryConfig


class Battery:
    """Simple single-battery state tracker for ESS dispatch."""

    def __init__(self, config: BatteryConfig) -> None:
        self.config = config
        self.last_power_kw = 0.0
        self.reset()

    @property
    def capacity_kwh(self) -> float:
        return self.config.capacity_kwh

    @property
    def time_interval_minutes(self) -> float:
        return self.config.time_interval_minutes

    def with_time_interval(self, time_interval_minutes: float) -> "Battery":
        return Battery(replace(self.config, time_interval_minutes=time_interval_minutes))

    def step(self, action: float | np.ndarray) -> float:
        """Apply a normalized action in [-1, 1] and return realized battery power in kW."""
        action_array = np.asarray(action, dtype=np.float32).reshape(-1)
        if action_array.size != 1:
            raise ValueError("Battery.step expects a scalar action or a size-1 array")

        normalized_action = float(np.clip(action_array[0], -1.0, 1.0))
        rated_power_kw = self.config.max_discharge_kw if normalized_action >= 0 else self.config.max_charge_kw
        requested_power_kw = normalized_action * rated_power_kw
        interval_hours = self.config.time_interval_minutes / 60.0

        current_energy_kwh = self.current_soc * self.capacity_kwh
        requested_energy_kwh = requested_power_kw * interval_hours
        updated_soc = np.clip(
            (current_energy_kwh + requested_energy_kwh) / self.capacity_kwh,
            self.config.min_soc,
            self.config.max_soc,
        )

        realized_energy_kwh = (updated_soc - self.current_soc) * self.capacity_kwh
        self.last_power_kw = realized_energy_kwh / interval_hours
        self.current_soc = float(updated_soc)
        return self.last_power_kw

    def SOC(self) -> float:
        return self.current_soc

    def reset(self) -> None:
        self.current_soc = float(self.config.initial_soc)
        self.last_power_kw = 0.0
