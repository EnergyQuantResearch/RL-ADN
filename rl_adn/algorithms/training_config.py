from __future__ import annotations

import os
from pprint import pprint
from typing import Any

import numpy as np
import torch


class Config:
    """Configuration container for RL-ADN training workflows."""

    def __init__(self, agent_class=None, env_class=None, env_args: dict[str, Any] | None = None):
        self.agent_class = agent_class
        self.env_class = env_class
        self.env_args = dict(env_args or {})
        self.if_off_policy = self.get_if_off_policy()

        default_env_args = {
            "env_name": None,
            "num_envs": 1,
            "max_step": 96,
            "state_dim": None,
            "action_dim": None,
            "if_discrete": None,
        }
        default_env_args.update(self.env_args)
        self.env_args = default_env_args
        self.env_name = self.env_args["env_name"]
        self.num_envs = self.env_args["num_envs"]
        self.max_step = self.env_args["max_step"]
        self.state_dim = self.env_args["state_dim"]
        self.action_dim = self.env_args["action_dim"]
        self.if_discrete = self.env_args["if_discrete"]

        self.gamma = 0.99
        self.reward_scale = 1.0
        self.net_dims = (64, 32)
        self.learning_rate = 6e-5
        self.clip_grad_norm = 3.0
        self.state_value_tau = 0.0
        self.soft_update_tau = 5e-3

        if self.if_off_policy:
            self.batch_size = 64
            self.target_step = 512
            self.buffer_size = int(1e6)
            self.repeat_times = 1.0
            self.if_use_per = False
        else:
            self.batch_size = 128
            self.target_step = 2048
            self.buffer_size = None
            self.repeat_times = 8.0
            self.if_use_vtrace = False

        self.random_seed = 0
        self.num_episode = 2000
        self.gpu_id = 0
        self.num_workers = 2
        self.num_threads = 8
        self.learner_gpus = 0

        self.run_name = None
        self.cwd = None
        self.if_remove = True
        self.train = True

    def init_before_training(self) -> None:
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        torch.set_num_threads(self.num_threads)
        torch.set_default_dtype(torch.float32)

        if self.cwd is None:
            agent_name = self.agent_class.__name__[5:] if self.agent_class else "agent"
            run_name = self.run_name or "default"
            self.cwd = f"./{agent_name}/{run_name}"

        if self.if_remove:
            import shutil

            shutil.rmtree(self.cwd, ignore_errors=True)
        os.makedirs(self.cwd, exist_ok=True)

    def get_if_off_policy(self) -> bool:
        agent_name = self.agent_class.__name__ if self.agent_class else ""
        on_policy_names = ("SARSA", "VPG", "A2C", "A3C", "TRPO", "PPO", "MPO")
        return all(agent_name.find(name) == -1 for name in on_policy_names)

    def print(self) -> None:
        pprint(vars(self))

    def to_dict(self) -> dict[str, Any]:
        return vars(self)
