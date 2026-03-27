from __future__ import annotations

from typing import Any

import torch

from rl_adn.algorithms.env_api import reset_env, step_env


def get_episode_return(env, act, device) -> tuple[float, int, float, float, float, float, list[Any]]:
    """Evaluate one episode with the current policy and return aggregate metrics."""
    env.train = False
    episode_return = 0.0
    violation_time = 0
    reward_for_power = 0.0
    reward_for_good_action = 0.0
    reward_for_penalty = 0.0
    violation_value = 0.0
    state_list = []

    state = reset_env(env)
    for _ in range(env.episode_length):
        state_tensor = torch.as_tensor((state,), device=device, dtype=torch.float32)
        action_tensor = act(state_tensor)
        action = action_tensor.detach().cpu().numpy()[0]
        next_state, reward, done, info = step_env(env, action)
        state_list.append(state)

        post_control_voltage = info["post_control_voltage_pu"]
        for node_index in env.battery_nodes:
            violation = min(0.0, 0.05 - abs(1.0 - post_control_voltage[node_index]))
            if violation < 0:
                violation_time += 1
            violation_value += violation

        reward_breakdown = info["reward_breakdown"]
        reward_for_power += reward_breakdown["economic"]
        reward_for_penalty += reward_breakdown["voltage_penalty"]
        episode_return += reward
        state = next_state
        if done:
            break

    return (
        episode_return,
        violation_time,
        violation_value,
        reward_for_power,
        reward_for_good_action,
        reward_for_penalty,
        state_list,
    )
