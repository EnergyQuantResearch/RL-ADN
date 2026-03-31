from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from urllib.request import urlopen

from rl_adn import PowerNetEnv, make_env_config
from rl_adn.dashboard import DashboardCallback, launch_dashboard
from rl_adn.dashboard.layouts import get_feeder_layout


def test_dashboard_snapshot_is_available_after_reset_and_step():
    env = PowerNetEnv(make_env_config())
    obs, _info = env.reset(seed=2026)
    snapshot = env.get_dashboard_snapshot()
    assert snapshot["feeder_id"] == "34-bus"
    assert snapshot["battery_dispatch_kw"] is None

    action = env.action_space.sample()
    next_obs, reward, terminated, truncated, _info = env.step(action)
    step_snapshot = env.get_dashboard_snapshot()
    assert obs is not None
    assert next_obs is not None
    assert isinstance(reward, float)
    assert step_snapshot["battery_dispatch_kw"] is not None
    assert len(step_snapshot["node_voltages_pu"]) == env.node_count
    assert step_snapshot["terminated"] is terminated
    assert step_snapshot["truncated"] is truncated


def test_dashboard_callback_collects_events():
    env = PowerNetEnv(make_env_config())
    server = launch_dashboard(port=0, open_browser=False, history_limit=16)
    callback = DashboardCallback(server=server)
    try:
        obs, info = env.reset(seed=2026)
        callback.on_reset(obs, info, env)
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        callback.on_step(obs, reward, terminated, truncated, info, env, action)
        payload = server.store.snapshot()
        assert payload["latest"] is not None
        assert payload["latest"]["event_type"] == "step"
        assert payload["history"]["steps"]
    finally:
        callback.close()


def test_dashboard_api_returns_idle_then_live_state():
    server = launch_dashboard(port=0, open_browser=False, history_limit=8)
    try:
        with urlopen(f"{server.url}api/state") as response:
            payload = json.loads(response.read().decode("utf-8"))
        assert payload["status"] == "idle"

        env = PowerNetEnv(make_env_config())
        callback = DashboardCallback(server=server)
        obs, info = env.reset(seed=2026)
        callback.on_reset(obs, info, env)
        with urlopen(f"{server.url}api/state") as response:
            payload = json.loads(response.read().decode("utf-8"))
        assert payload["latest"]["topology_scenario"] == "TP1"
        assert payload["layout"]["node_count"] == 34
    finally:
        server.close()


def test_feeder_layout_counts_match_nodes():
    layout_34 = get_feeder_layout(34)
    layout_69 = get_feeder_layout(69)
    assert len(layout_34["positions"]) == 34
    assert len(layout_69["positions"]) == 69
    assert len(layout_34["base_edges"]) == 33
    assert len(layout_69["base_edges"]) == 68


def test_live_dashboard_example_runs_without_open_browser():
    example_script = Path(__file__).resolve().parents[1] / "examples" / "live_dashboard_rollout.py"
    result = subprocess.run(
        [
            sys.executable,
            str(example_script),
            "--steps",
            "3",
            "--no-open-browser",
            "--port",
            "0",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
