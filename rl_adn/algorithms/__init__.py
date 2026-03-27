"""RL algorithm exports for RL-ADN."""

from importlib import import_module

__all__ = [
    "AgentDDPG",
    "AgentPPO",
    "AgentSAC",
    "AgentTD3",
]

_LAZY_EXPORTS = {
    "AgentDDPG": ("rl_adn.algorithms.DDPG", "AgentDDPG"),
    "AgentPPO": ("rl_adn.algorithms.PPO", "AgentPPO"),
    "AgentSAC": ("rl_adn.algorithms.SAC", "AgentSAC"),
    "AgentTD3": ("rl_adn.algorithms.TD3", "AgentTD3"),
}


def __getattr__(name: str):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module 'rl_adn.algorithms' has no attribute {name!r}")

    module_name, attr_name = _LAZY_EXPORTS[name]
    module = import_module(module_name)
    return getattr(module, attr_name)
