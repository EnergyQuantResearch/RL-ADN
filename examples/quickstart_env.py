import numpy as np

from rl_adn import PowerNetEnv, make_env_config


def main() -> None:
    config = make_env_config()
    env = PowerNetEnv(config)
    state = env.reset()

    action = np.zeros((len(config["battery_list"]), 1), dtype=np.float32)
    next_state, reward, done, _ = env.step(action)

    print("Initial state shape:", state.shape)
    print("Next state shape:", next_state.shape)
    print("Reward:", reward)
    print("Done:", done)


if __name__ == "__main__":
    main()
