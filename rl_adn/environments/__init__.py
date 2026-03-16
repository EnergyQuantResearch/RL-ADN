"""Stable environment exports for RL-ADN."""

from importlib import import_module

__all__ = [
    "Battery",
    "battery_parameters",
    "PowerNetEnv",
    "env_config",
    "make_env_config",
]

_LAZY_EXPORTS = {
    "Battery": ("rl_adn.environments.battery", "Battery"),
    "battery_parameters": ("rl_adn.environments.battery", "battery_parameters"),
    "PowerNetEnv": ("rl_adn.environments.env", "PowerNetEnv"),
    "env_config": ("rl_adn.environments.config", "env_config"),
    "make_env_config": ("rl_adn.environments.config", "make_env_config"),
}


def __getattr__(name):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module 'rl_adn.environments' has no attribute {name!r}")

    module_name, attr_name = _LAZY_EXPORTS[name]
    module = import_module(module_name)
    return getattr(module, attr_name)
