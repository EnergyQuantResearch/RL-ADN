from pathlib import Path

import pytest

import rl_adn


def test_make_env_config_returns_existing_packaged_paths():
    config = rl_adn.make_env_config()

    network_info = config["network_info"]
    assert network_info["bus_info_file"]
    assert network_info["branch_info_file"]
    assert Path(network_info["bus_info_file"]).exists()
    assert Path(network_info["branch_info_file"]).exists()
    assert Path(config["time_series_data_path"]).exists()


def test_powernet_env_is_available_when_gym_is_installed():
    pytest.importorskip("gym")

    env = rl_adn.PowerNetEnv(rl_adn.make_env_config())
    state = env.reset()
    assert state is not None
