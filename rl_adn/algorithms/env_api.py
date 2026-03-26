from __future__ import annotations

from typing import Any

import numpy as np


def reset_env(env) -> np.ndarray:
    reset_result = env.reset()
    if isinstance(reset_result, tuple) and len(reset_result) == 2:
        observation, _info = reset_result
        return observation
    return reset_result


def step_env(env, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
    step_result = env.step(action)
    if len(step_result) == 5:
        observation, reward, terminated, truncated, info = step_result
        return observation, float(reward), bool(terminated or truncated), info
    observation, reward, done, info = step_result
    return observation, float(reward), bool(done), info
