from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RewardBreakdown:
    total: float
    economic: float
    voltage_penalty: float
    saved_money: float


def compute_default_reward(
    *,
    price: float,
    saved_power_kw: float,
    battery_voltages_pu: np.ndarray,
    voltage_target_pu: float = 1.0,
    voltage_band_pu: float = 0.05,
    penalty_scale: float = 100.0,
) -> RewardBreakdown:
    economic = float(price * saved_power_kw)
    voltage_penalty = 0.0
    for voltage in battery_voltages_pu:
        voltage_penalty += min(0.0, penalty_scale * (voltage_band_pu - abs(voltage_target_pu - float(voltage))))
    total = economic + voltage_penalty
    return RewardBreakdown(
        total=float(total),
        economic=float(economic),
        voltage_penalty=float(voltage_penalty),
        saved_money=float(-economic),
    )
