from __future__ import annotations

from typing import Any

from rl_adn.dashboard.event import DashboardEvent
from rl_adn.dashboard.server import DashboardServer, launch_dashboard


class DashboardCallback:
    def __init__(self, server: DashboardServer | None = None, *, port: int = 8787, open_browser: bool = True, history_limit: int = 500) -> None:
        self.server = server or launch_dashboard(port=port, open_browser=open_browser, history_limit=history_limit)
        self.episode_id = -1

    def on_reset(self, obs, info: dict[str, Any], env) -> None:  # noqa: ANN001
        del obs, info
        self.episode_id += 1
        event = DashboardEvent.from_snapshot(event_type="reset", episode_id=self.episode_id, snapshot=env.get_dashboard_snapshot())
        self.server.store.record(event.to_payload())

    def on_step(self, obs, reward: float, terminated: bool, truncated: bool, info: dict[str, Any], env, action) -> None:  # noqa: ANN001
        del obs, reward, terminated, truncated, info, action
        event = DashboardEvent.from_snapshot(event_type="step", episode_id=self.episode_id, snapshot=env.get_dashboard_snapshot())
        self.server.store.record(event.to_payload())

    def close(self) -> None:
        self.server.close()
