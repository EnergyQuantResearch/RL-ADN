"""Top-level package for RL-ADN."""

from importlib import import_module

__version__ = "0.1.3"

__all__ = [
    "__version__",
    "Battery",
    "battery_parameters",
    "GeneralPowerDataManager",
    "PowerNetEnv",
    "env_config",
    "make_env_config",
]

_LAZY_EXPORTS = {
    "Battery": ("rl_adn.environments.battery", "Battery"),
    "battery_parameters": ("rl_adn.environments.battery", "battery_parameters"),
    "GeneralPowerDataManager": ("rl_adn.data_manager", "GeneralPowerDataManager"),
    "PowerNetEnv": ("rl_adn.environments", "PowerNetEnv"),
    "env_config": ("rl_adn.environments", "env_config"),
    "make_env_config": ("rl_adn.environments", "make_env_config"),
}


def __getattr__(name):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module 'rl_adn' has no attribute {name!r}")

    module_name, attr_name = _LAZY_EXPORTS[name]
    module = import_module(module_name)
    return getattr(module, attr_name)
