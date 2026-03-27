"""Legacy compatibility facade for algorithm utilities."""

from importlib import import_module

__all__ = [
    "Config",
    "ReplayBuffer",
    "SumTree",
    "build_mlp",
    "get_episode_return",
    "get_optim_param",
]

_LAZY_EXPORTS = {
    "Config": ("rl_adn.algorithms.training_config", "Config"),
    "ReplayBuffer": ("rl_adn.algorithms.replay", "ReplayBuffer"),
    "SumTree": ("rl_adn.algorithms.replay", "SumTree"),
    "build_mlp": ("rl_adn.algorithms.torch_utils", "build_mlp"),
    "get_episode_return": ("rl_adn.algorithms.evaluation", "get_episode_return"),
    "get_optim_param": ("rl_adn.algorithms.torch_utils", "get_optim_param"),
}


def __getattr__(name: str):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module 'rl_adn.algorithms.utility' has no attribute {name!r}")

    module_name, attr_name = _LAZY_EXPORTS[name]
    module = import_module(module_name)
    return getattr(module, attr_name)
