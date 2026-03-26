"""Public package surface for RL-ADN."""

from importlib import import_module

__version__ = "0.2.0"

__all__ = [
    "__version__",
    "Battery",
    "BatteryConfig",
    "EnvConfig",
    "GeneralPowerDataManager",
    "PowerNetEnv",
    "TopologyConfig",
    "make_env_config",
]

_LAZY_EXPORTS = {
    "Battery": ("rl_adn.environments", "Battery"),
    "BatteryConfig": ("rl_adn.config", "BatteryConfig"),
    "EnvConfig": ("rl_adn.config", "EnvConfig"),
    "GeneralPowerDataManager": ("rl_adn.data", "GeneralPowerDataManager"),
    "PowerNetEnv": ("rl_adn.environments", "PowerNetEnv"),
    "TopologyConfig": ("rl_adn.config", "TopologyConfig"),
    "make_env_config": ("rl_adn.config", "make_env_config"),
}


def __getattr__(name: str):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module 'rl_adn' has no attribute {name!r}")

    module_name, attr_name = _LAZY_EXPORTS[name]
    module = import_module(module_name)
    return getattr(module, attr_name)
