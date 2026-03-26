import importlib

import pytest

pytest.importorskip("gymnasium")
pytest.importorskip("torch")


@pytest.mark.parametrize(
    "module_name",
    [
        "rl_adn.algorithms.DDPG",
        "rl_adn.algorithms.PPO",
        "rl_adn.algorithms.SAC",
        "rl_adn.algorithms.TD3",
    ],
)
def test_drl_algorithm_modules_import(module_name):
    importlib.import_module(module_name)
