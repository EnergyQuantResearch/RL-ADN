from __future__ import annotations

from collections import deque
from threading import Lock
from typing import Any

from rl_adn.dashboard.layouts import get_feeder_layout


class DashboardStateStore:
    def __init__(self, history_limit: int = 500) -> None:
        self.history_limit = history_limit
        self._events: deque[dict[str, Any]] = deque(maxlen=history_limit)
        self._latest: dict[str, Any] | None = None
        self._lock = Lock()

    def record(self, payload: dict[str, Any]) -> None:
        with self._lock:
            self._latest = payload
            self._events.append(payload)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            latest = dict(self._latest) if self._latest is not None else None
            events = list(self._events)

        history = {
            "steps": [event["step_index"] for event in events],
            "reward": [event["reward"] for event in events],
            "price": [event["price"] for event in events],
            "voltage_min": [min(event["node_voltages_pu"]) if event["node_voltages_pu"] else None for event in events],
            "voltage_max": [max(event["node_voltages_pu"]) if event["node_voltages_pu"] else None for event in events],
            "total_dispatch_kw": [sum(event["battery_dispatch_kw"] or []) for event in events],
            "soc_by_battery": {},
        }

        if latest is not None:
            for node_index, battery_node in enumerate(latest["battery_nodes"]):
                history["soc_by_battery"][str(battery_node)] = [
                    event["battery_soc"][node_index] if node_index < len(event["battery_soc"]) else None for event in events
                ]

        return {
            "status": "idle" if latest is None else ("finished" if latest["terminated"] or latest["truncated"] else "running"),
            "latest": latest,
            "layout": None if latest is None else get_feeder_layout(int(latest["feeder_id"].split("-")[0])),
            "history": history,
            "history_limit": self.history_limit,
        }
