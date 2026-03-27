import numpy as np
import pytest

from rl_adn import BatteryConfig, make_env_config
from rl_adn.environments.battery import Battery


def test_battery_soc_remains_scalar_after_vector_action():
    battery = Battery(BatteryConfig.default())

    battery.step(np.array([0.0], dtype=np.float32))

    assert np.isscalar(battery.SOC())


def test_battery_rejects_multi_value_action():
    battery = Battery(BatteryConfig.default())

    try:
        battery.step(np.array([0.0, 1.0], dtype=np.float32))
    except ValueError:
        pass
    else:
        raise AssertionError("Battery.step should reject actions with more than one value")


def test_battery_uses_15_minute_interval_by_default():
    battery = Battery(BatteryConfig.default())

    battery.step(np.array([1.0], dtype=np.float32))

    assert np.isclose(battery.SOC(), 0.4 + (50.0 * 0.25) / 300.0)
    assert np.isclose(battery.last_power_kw, 50.0)


def test_battery_respects_custom_time_interval():
    battery = Battery(BatteryConfig.default(time_interval_minutes=5.0))

    battery.step(np.array([1.0], dtype=np.float32))

    assert np.isclose(battery.SOC(), 0.4 + (50.0 * (5.0 / 60.0)) / 300.0)
    assert np.isclose(battery.last_power_kw, 50.0)


def test_environment_batteries_inherit_dataset_time_interval():
    pytest.importorskip("gymnasium")
    from rl_adn import PowerNetEnv

    env = PowerNetEnv(make_env_config())

    try:
        battery = env.batteries[env.battery_nodes[0]]
        assert np.isclose(battery.time_interval_minutes, env.data_manager.time_interval)
    finally:
        del env
