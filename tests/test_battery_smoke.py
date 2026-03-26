import numpy as np
import pytest

from rl_adn.environments.battery import Battery, battery_parameters
from rl_adn.environments.config import make_env_config


def test_battery_soc_remains_scalar_after_vector_action():
    battery = Battery(battery_parameters)
    battery.reset()

    battery.step(np.array([0.0], dtype=np.float32))

    assert np.isscalar(battery.SOC())


def test_battery_rejects_multi_value_action():
    battery = Battery(battery_parameters)
    battery.reset()

    try:
        battery.step(np.array([0.0, 1.0], dtype=np.float32))
    except ValueError:
        pass
    else:
        raise AssertionError("Battery.step should reject actions with more than one value")


def test_battery_uses_15_minute_interval_by_default():
    battery = Battery(battery_parameters)
    battery.reset()

    battery.step(np.array([1.0], dtype=np.float32))

    assert np.isclose(battery.SOC(), 0.4 + (50.0 * 0.25) / 300.0)
    assert np.isclose(battery.energy_change, 50.0)


def test_battery_respects_custom_time_interval():
    battery = Battery({**battery_parameters, "time_interval_minutes": 5})
    battery.reset()

    battery.step(np.array([1.0], dtype=np.float32))

    assert np.isclose(battery.SOC(), 0.4 + (50.0 * (5.0 / 60.0)) / 300.0)
    assert np.isclose(battery.energy_change, 50.0)


def test_environment_batteries_inherit_dataset_time_interval():
    pytest.importorskip("gym")
    from rl_adn.environments.env import PowerNetEnv

    env = PowerNetEnv(make_env_config())

    try:
        battery = getattr(env, f"battery_{env.battery_list[0]}")
        assert np.isclose(battery.time_interval_minutes, env.data_manager.time_interval)
    finally:
        del env
