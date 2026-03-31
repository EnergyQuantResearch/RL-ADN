from __future__ import annotations

import argparse
import time

from rl_adn import PowerNetEnv, make_env_config
from rl_adn.dashboard import DashboardCallback


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a live RL-ADN dashboard rollout.")
    parser.add_argument("--node", type=int, choices=[34, 69], default=34)
    parser.add_argument("--scenario", type=str, default="TP1")
    parser.add_argument("--scenario-pool", type=str, default="")
    parser.add_argument("--algorithm", choices=["Laurent", "PandaPower"], default="Laurent")
    parser.add_argument("--policy", choices=["random", "ddpg-smoke", "td3-smoke"], default="random")
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--open-browser", dest="open_browser", action="store_true", default=True)
    parser.add_argument("--no-open-browser", dest="open_browser", action="store_false")
    return parser.parse_args()


def run_random_rollout(env: PowerNetEnv, steps: int, callback: DashboardCallback) -> None:
    observation, info = env.reset(seed=2026)
    callback.on_reset(observation, info, env)
    for _ in range(steps):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        callback.on_step(observation, reward, terminated, truncated, info, env, action)
        if terminated or truncated:
            observation, info = env.reset(seed=2026)
            callback.on_reset(observation, info, env)


def run_agent_rollout(env: PowerNetEnv, steps: int, policy: str, callback: DashboardCallback) -> None:
    from rl_adn.algorithms.DDPG import AgentDDPG
    from rl_adn.algorithms.TD3 import AgentTD3
    from rl_adn.algorithms.utility import Config

    args = Config()
    args.num_envs = 1
    args.if_off_policy = True
    args.batch_size = 8
    args.repeat_times = 1.0

    agent_cls = AgentDDPG if policy == "ddpg-smoke" else AgentTD3
    agent = agent_cls(
        net_dims=[32, 32],
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        gpu_id=-1,
        args=args,
    )
    agent.explore_one_env(env, horizon_len=steps, if_random=True, callback=callback)


def main() -> None:
    args = parse_args()
    topology_pool = [item.strip() for item in args.scenario_pool.split(",") if item.strip()]
    config = make_env_config(
        node=args.node,
        algorithm=args.algorithm,
        topology_mode="scenario_pool" if topology_pool else "fixed",
        topology_scenario=args.scenario,
        topology_pool=topology_pool or None,
    )
    env = PowerNetEnv(config)
    callback = DashboardCallback(port=args.port, open_browser=args.open_browser)

    try:
        if args.policy == "random":
            run_random_rollout(env, args.steps, callback)
        else:
            run_agent_rollout(env, args.steps, args.policy, callback)
        if args.open_browser:
            print(f"Dashboard running at {callback.server.url} . Press Ctrl+C to stop.")
            while True:
                time.sleep(1.0)
    finally:
        callback.close()


if __name__ == "__main__":
    main()
