from pathlib import Path

import pytest

import rl_adn


def test_make_env_config_returns_existing_packaged_paths():
    config = rl_adn.make_env_config()

    assert config.bus_info_file
    assert config.branch_info_file
    assert Path(config.bus_info_file).exists()
    assert Path(config.branch_info_file).exists()
    assert Path(config.time_series_data_path).exists()


def test_powernet_env_is_available_when_gymnasium_is_installed():
    pytest.importorskip("gymnasium")

    env = rl_adn.PowerNetEnv(rl_adn.make_env_config())
    state, info = env.reset(seed=2026)
    assert state is not None
    assert info["feeder_id"] == "34-bus"
