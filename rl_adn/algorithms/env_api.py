from __future__ import annotations

from typing import Any

import numpy as np


def reset_env(env) -> np.ndarray:
    observation, _info = reset_env_with_info(env)
    return observation


def reset_env_with_info(env) -> tuple[np.ndarray, dict[str, Any]]:
    reset_result = env.reset()
    if isinstance(reset_result, tuple) and len(reset_result) == 2:
        observation, info = reset_result
        return observation, dict(info)
    return reset_result, {}


def step_env(env, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
    observation, reward, terminated, truncated, info = step_env_with_info(env, action)
    return observation, reward, bool(terminated or truncated), info


def step_env_with_info(env, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
    step_result = env.step(action)
    if len(step_result) == 5:
        observation, reward, terminated, truncated, info = step_result
        return observation, float(reward), bool(terminated), bool(truncated), dict(info)
    observation, reward, done, info = step_result
    return observation, float(reward), bool(done), False, dict(info)
