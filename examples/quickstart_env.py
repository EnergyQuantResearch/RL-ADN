import numpy as np

from rl_adn import PowerNetEnv, make_env_config


def main() -> None:
    config = make_env_config()
    env = PowerNetEnv(config)
    state, info = env.reset(seed=2026)

    action = np.zeros(len(config.battery_nodes), dtype=np.float32)
    next_state, reward, terminated, truncated, step_info = env.step(action)

    print("Initial state shape:", state.shape)
    print("Topology scenario:", info["topology_scenario"])
    print("Next state shape:", next_state.shape)
    print("Reward:", reward)
    print("Terminated:", terminated)
    print("Truncated:", truncated)
    print("Saved money:", step_info["reward_breakdown"]["saved_money"])


if __name__ == "__main__":
    main()
