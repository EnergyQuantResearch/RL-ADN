import importlib

import pytest


pytest.importorskip("gym")
pytest.importorskip("torch")


@pytest.mark.parametrize(
    "module_name",
    [
        "rl_adn.DRL_algorithms.DDPG",
        "rl_adn.DRL_algorithms.PPO",
        "rl_adn.DRL_algorithms.SAC",
        "rl_adn.DRL_algorithms.TD3",
    ],
)
def test_drl_algorithm_modules_import(module_name):
    importlib.import_module(module_name)
