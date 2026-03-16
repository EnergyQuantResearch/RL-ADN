import numpy as np

from rl_adn.environments.battery import Battery, battery_parameters


def test_battery_soc_remains_scalar_after_vector_action():
    battery = Battery(battery_parameters)
    battery.reset()

    battery.step(np.array([0.0], dtype=np.float32))

    assert np.isscalar(battery.SOC())
