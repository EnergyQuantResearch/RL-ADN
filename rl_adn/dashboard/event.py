from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np


def _to_serializable(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return [_to_serializable(item) for item in value]
    if isinstance(value, list):
        return [_to_serializable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _to_serializable(item) for key, item in value.items()}
    return value


@dataclass(frozen=True)
class DashboardEvent:
    event_type: str
    episode_id: int
    step_index: int
    feeder_id: str
    topology_scenario: str
    active_edges: list[list[int]]
    baseline_edges: list[list[int]]
    battery_nodes: list[int]
    node_voltages_pu: list[float]
    battery_soc: list[float]
    battery_dispatch_kw: list[float] | None
    active_power_kw: list[float]
    renewable_active_power_kw: list[float]
    net_load_kw: list[float]
    price: float
    reward: float | None
    reward_breakdown: dict[str, float] | None
    terminated: bool
    truncated: bool
    current_time: int
    timestamp_utc: str

    @classmethod
    def from_snapshot(cls, *, event_type: str, episode_id: int, snapshot: dict[str, Any]) -> "DashboardEvent":
        return cls(
            event_type=event_type,
            episode_id=episode_id,
            step_index=int(snapshot["current_time"]),
            feeder_id=str(snapshot["feeder_id"]),
            topology_scenario=str(snapshot["topology_scenario"]),
            active_edges=_to_serializable(snapshot["active_edges"]),
            baseline_edges=_to_serializable(snapshot["baseline_edges"]),
            battery_nodes=_to_serializable(snapshot["battery_nodes"]),
            node_voltages_pu=_to_serializable(snapshot["node_voltages_pu"]),
            battery_soc=_to_serializable(snapshot["battery_soc"]),
            battery_dispatch_kw=_to_serializable(snapshot.get("battery_dispatch_kw")),
            active_power_kw=_to_serializable(snapshot["active_power_kw"]),
            renewable_active_power_kw=_to_serializable(snapshot["renewable_active_power_kw"]),
            net_load_kw=_to_serializable(snapshot["net_load_kw"]),
            price=float(snapshot["price"]),
            reward=None if snapshot.get("reward") is None else float(snapshot["reward"]),
            reward_breakdown=_to_serializable(snapshot.get("reward_breakdown")),
            terminated=bool(snapshot.get("terminated", False)),
            truncated=bool(snapshot.get("truncated", False)),
            current_time=int(snapshot["current_time"]),
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
        )

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)
