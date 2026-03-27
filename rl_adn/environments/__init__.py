"""Environment-facing exports for RL-ADN."""

from importlib import import_module

__all__ = [
    "Battery",
    "ObservationSnapshot",
    "PowerNetEnv",
    "RewardBreakdown",
    "StateScaler",
]

_LAZY_EXPORTS = {
    "Battery": ("rl_adn.environments.battery", "Battery"),
    "ObservationSnapshot": ("rl_adn.environments.observation", "ObservationSnapshot"),
    "PowerNetEnv": ("rl_adn.environments.env", "PowerNetEnv"),
    "RewardBreakdown": ("rl_adn.environments.reward", "RewardBreakdown"),
    "StateScaler": ("rl_adn.environments.observation", "StateScaler"),
}


def __getattr__(name: str):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module 'rl_adn.environments' has no attribute {name!r}")

    module_name, attr_name = _LAZY_EXPORTS[name]
    module = import_module(module_name)
    return getattr(module, attr_name)
