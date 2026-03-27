from __future__ import annotations

from rl_adn import PowerNetEnv, make_env_config
from rl_adn.algorithms.DDPG import AgentDDPG
from rl_adn.algorithms.utility import Config


def main() -> None:
    config = make_env_config()
    env = PowerNetEnv(config)
    args = Config()
    args.num_envs = 1
    args.if_off_policy = True
    args.batch_size = 8
    args.repeat_times = 1.0

    agent = AgentDDPG(
        net_dims=[32, 32],
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        gpu_id=-1,
        args=args,
    )
    states, actions, rewards, undones = agent.explore_one_env(env, horizon_len=2, if_random=True)

    print("Collected states:", tuple(states.shape))
    print("Collected actions:", tuple(actions.shape))
    print("Collected rewards:", tuple(rewards.shape))
    print("Collected undones:", tuple(undones.shape))


if __name__ == "__main__":
    main()
