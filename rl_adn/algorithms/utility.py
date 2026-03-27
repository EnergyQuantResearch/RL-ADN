"""Legacy compatibility facade for algorithm utilities.

Prefer importing from the narrower modules:
- ``rl_adn.algorithms.training_config``
- ``rl_adn.algorithms.torch_utils``
- ``rl_adn.algorithms.replay``
- ``rl_adn.algorithms.evaluation``
"""

from rl_adn.algorithms.evaluation import get_episode_return
from rl_adn.algorithms.replay import ReplayBuffer, SumTree
from rl_adn.algorithms.torch_utils import build_mlp, get_optim_param
from rl_adn.algorithms.training_config import Config

__all__ = [
    "Config",
    "ReplayBuffer",
    "SumTree",
    "build_mlp",
    "get_episode_return",
    "get_optim_param",
]
